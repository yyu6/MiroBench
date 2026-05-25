"""Export OASIS SQLite trace → discussion.json + discussion.md."""
from __future__ import annotations

import json
import os
import sqlite3
import warnings
from datetime import datetime


def export_discussion(
    db_path: str,
    profiles_path: str,
    output_dir: str,
    meta: dict,
) -> tuple[str, str]:
    """Return (json_path, md_path)."""
    profiles = _load_profiles(profiles_path)
    posts, comments = _load_from_db(db_path)
    thread = _build_thread(posts, comments, profiles)

    discussion = {"meta": meta, "posts": thread}

    json_path = os.path.join(output_dir, "discussion.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(discussion, f, ensure_ascii=False, indent=2)

    md_path = os.path.join(output_dir, "discussion.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(_render_markdown(discussion))

    return json_path, md_path


def _load_profiles(profiles_path: str) -> dict[int, dict]:
    with open(profiles_path, encoding="utf-8") as f:
        profiles = json.load(f)
    return {p["user_id"]: p for p in profiles}


def _load_from_db(db_path: str) -> tuple[list, list]:
    conn = sqlite3.connect(db_path)
    try:
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()

        if _has_table(cur, "post") and _has_table(cur, "comment"):
            comment_columns = _table_columns(cur, "comment")
            cur.execute("""
                SELECT post_id, user_id, content, created_at, num_likes, num_dislikes
                FROM post
                ORDER BY created_at ASC, post_id ASC
            """)
            posts = [dict(r) for r in cur.fetchall()]

            comment_select = [
                "comment_id",
                "post_id",
                "user_id",
                "content",
                "created_at",
                "num_likes",
                "num_dislikes",
            ]
            if "parent_comment_id" in comment_columns:
                comment_select.append("parent_comment_id")
            if "depth" in comment_columns:
                comment_select.append("depth")
            cur.execute(
                f"SELECT {', '.join(comment_select)} FROM comment "
                "ORDER BY created_at ASC, comment_id ASC"
            )
            comments = [dict(r) for r in cur.fetchall()]
            return posts, comments

        # Case-insensitive match: OASIS versions differ (create_post vs CREATE_POST)
        cur.execute("""
            SELECT user_id, info, created_at FROM trace
            WHERE LOWER(action) = 'create_post'
            ORDER BY created_at ASC
        """)
        posts = [dict(r) for r in cur.fetchall()]

        cur.execute("""
            SELECT user_id, info, created_at FROM trace
            WHERE LOWER(action) = 'create_comment'
            ORDER BY created_at ASC
        """)
        comments = [dict(r) for r in cur.fetchall()]

        return posts, comments
    finally:
        conn.close()


def _has_table(cur: sqlite3.Cursor, table_name: str) -> bool:
    cur.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ? LIMIT 1",
        (table_name,),
    )
    return cur.fetchone() is not None


def _table_columns(cur: sqlite3.Cursor, table_name: str) -> set[str]:
    cur.execute(f"PRAGMA table_info({table_name})")
    return {str(row[1]) for row in cur.fetchall()}


def _first_present(info: dict, *keys: str) -> str:
    """Return the first non-None value from info for any of the given keys."""
    for k in keys:
        if k in info and info[k] is not None:
            return str(info[k])
    return str(info)


def _build_thread(
    posts: list[dict], comments: list[dict], profiles: dict[int, dict]
) -> list[dict]:
    post_list: list[dict] = []
    # Map from OASIS post_id → index in post_list for comment attachment
    oasis_post_id_map: dict[int, int] = {}

    for i, row in enumerate(posts):
        info = _normalize_post_info(row)
        profile = profiles.get(row["user_id"], {})
        post = {
            "post_id": i + 1,
            "author": profile.get("username", f"user_{row['user_id']}"),
            "author_karma": profile.get("karma", 0),
            "content": _first_present(info, "content", "post_content"),
            "timestamp": row["created_at"],
            "likes": int(info.get("num_likes", 0) or 0),
            "dislikes": int(info.get("num_dislikes", 0) or 0),
            "comments": [],
        }
        if "post_id" in info:
            oasis_post_id_map[info["post_id"]] = i
        post_list.append(post)

    for j, row in enumerate(comments):
        info = _normalize_comment_info(row)
        profile = profiles.get(row["user_id"], {})
        exported_comment_id = info.get("comment_id") or (j + 1)
        comment = {
            "comment_id": exported_comment_id,
            "author": profile.get("username", f"user_{row['user_id']}"),
            "author_karma": profile.get("karma", 0),
            "content": _first_present(info, "content", "comment"),
            "timestamp": row["created_at"],
            "likes": int(info.get("num_likes", 0) or 0),
            "dislikes": int(info.get("num_dislikes", 0) or 0),
            "parent_comment_id": info.get("parent_comment_id"),
            "depth": int(info.get("depth", 0) or 0),
            "replies": [],
        }
        target_idx = oasis_post_id_map.get(info.get("post_id"))
        if target_idx is not None:
            post_list[target_idx]["comments"].append(comment)
        elif post_list:
            warnings.warn(
                f"Comment {j + 1} references unknown post_id {info.get('post_id')!r}; "
                "attaching to last post as fallback."
            )
            post_list[-1]["comments"].append(comment)
        else:
            warnings.warn(f"Comment {j + 1} dropped: no posts exist to attach it to.")

    for post in post_list:
        post["comments"] = _nest_export_comments(post["comments"])

    return post_list


def _parse_info(info_raw) -> dict:
    if not info_raw:
        return {}
    if isinstance(info_raw, dict):
        return info_raw
    try:
        return json.loads(info_raw)
    except (json.JSONDecodeError, TypeError):
        return {"content": str(info_raw)}


def _normalize_post_info(row: dict) -> dict:
    if "info" in row:
        return _parse_info(row["info"])
    return {
        "post_id": row.get("post_id"),
        "content": row.get("content"),
        "num_likes": row.get("num_likes", 0),
        "num_dislikes": row.get("num_dislikes", 0),
    }


def _normalize_comment_info(row: dict) -> dict:
    if "info" in row:
        return _parse_info(row["info"])
    return {
        "comment_id": row.get("comment_id"),
        "post_id": row.get("post_id"),
        "content": row.get("content"),
        "num_likes": row.get("num_likes", 0),
        "num_dislikes": row.get("num_dislikes", 0),
        "parent_comment_id": row.get("parent_comment_id"),
        "depth": row.get("depth", 0),
    }


def _render_markdown(discussion: dict) -> str:
    meta = discussion["meta"]
    category = meta.get("product_category", "products")
    sub = category.replace(" ", "_")
    lines = [
        f"# r/{sub} simulation — {category}",
        f"*Hint: {meta.get('hint', 'none')} | "
        f"Agents: {meta.get('agent_count')} | "
        f"Simulated: {meta.get('simulated_hours')}h | "
        f"Run: {meta.get('run_id')}*",
        "",
        "---",
        "",
    ]

    for post in discussion["posts"]:
        ts = _fmt_ts(post.get("timestamp"))
        content = post["content"]
        preview = content[:100].replace("\n", " ")
        lines += [
            f"## [{post['likes']}↑] {preview}{'...' if len(content) > 100 else ''}",
            f"**u/{post['author']}** (karma: {post['author_karma']:,}) · {ts}",
            "",
            content,
            "",
        ]
        lines += _render_comment_markdown(post["comments"])
        lines += ["---", ""]

    return "\n".join(lines)


def _fmt_ts(ts: str | None) -> str:
    if not ts:
        return ""
    try:
        dt = datetime.fromisoformat(str(ts))
        return dt.strftime("%I:%M %p").lstrip("0")
    except ValueError:
        return str(ts)


def _nest_export_comments(flat_comments: list[dict]) -> list[dict]:
    """Convert flat exported comments into a nested reply tree."""

    comment_map: dict[int, dict] = {}
    ordered_comments: list[dict] = []
    for comment in flat_comments:
        copied = dict(comment)
        copied["replies"] = list(comment.get("replies") or [])
        comment_map[int(copied["comment_id"])] = copied
        ordered_comments.append(copied)

    roots: list[dict] = []
    for comment in ordered_comments:
        parent_comment_id = comment.get("parent_comment_id")
        if parent_comment_id in (None, "", 0):
            roots.append(comment)
            continue
        parent = comment_map.get(int(parent_comment_id))
        if parent is None:
            roots.append(comment)
            continue
        parent.setdefault("replies", []).append(comment)
    return roots


def _render_comment_markdown(
    comments: list[dict],
    level: int = 0,
) -> list[str]:
    """Render nested comments into blockquoted Markdown."""

    lines: list[str] = []
    prefix = ">" * (level + 1)
    for comment in comments:
        cts = _fmt_ts(comment.get("timestamp"))
        lines.extend(
            [
                f"{prefix} **u/{comment['author']}** (karma: {comment['author_karma']:,}) · {cts} [{comment['likes']}↑]",
                prefix,
                *(f"{prefix} {line}" for line in comment["content"].splitlines()),
                "",
            ]
        )
        lines.extend(_render_comment_markdown(comment.get("replies") or [], level=level + 1))
    return lines
