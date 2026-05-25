"""Thread-loading helpers extracted from the GEO LLM-judge module.

This module contains only the loader/normalisation functions needed by
``mirobench.generation.prompt_examples``. The full LLM-judge logic was
intentionally left behind to keep ``mirobench.generation`` focused on
generating discussion threads (not evaluating them).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

@dataclass
class DiscussionComment:
    """A visible comment snippet attached to a discussion thread.

    The judge does not need the full Reddit object model. It only needs the
    parts that affect realism decisions: author identity, text, score, nesting
    depth, and rough time metadata.
    """

    author: str
    content: str
    comment_id: Optional[str] = None
    score: Optional[int] = None
    depth: int = 0
    timestamp: Optional[str] = None


@dataclass
class DiscussionThread:
    """A normalized thread representation consumed by the judge.

    Both actual Reddit data and generated simulation outputs are converted into
    this shape so the rest of the pipeline can stay source-agnostic.
    """

    source_kind: str
    thread_id: str
    title: str
    body: str
    author: str
    score: Optional[int]
    comment_count: int
    comments: list[DiscussionComment]
    subreddit: Optional[str] = None
    topic: Optional[str] = None
    url: Optional[str] = None


def sanitize_text(text: Any) -> str:
    """Remove control characters and invalid surrogates from arbitrary text.

    This helper is used aggressively across the judge pipeline because both raw
    product data and raw Reddit dumps can contain malformed characters that
    break prompt construction or JSON serialization.
    """

    sanitized_chars: list[str] = []
    for ch in str(text or ""):
        codepoint = ord(ch)
        if codepoint in (9, 10, 13):
            sanitized_chars.append(ch)
            continue
        if codepoint < 32:
            sanitized_chars.append(" ")
            continue
        if 0xD800 <= codepoint <= 0xDFFF:
            continue
        sanitized_chars.append(ch)
    return "".join(sanitized_chars)


def auto_detect_target_kind(path: str | Path) -> str:
    """Infer which loader should be used for a target path.

    Returns one of:
    - `generated`: simulation output from this repo
    - `actual_bundle`: one scraped Reddit bundle for one product/card
    - `actual_manifest`: a manifest pointing to multiple scraped bundles
    """

    path = Path(path)
    if path.is_dir():
        if (path / "discussion.json").exists():
            return "generated"
        raise ValueError(f"Cannot detect target kind from directory: {path}")

    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if isinstance(data, dict) and {"meta", "posts"}.issubset(data.keys()):
        return "generated"
    if isinstance(data, dict) and {"topic", "posts"}.issubset(data.keys()):
        return "actual_bundle"
    if isinstance(data, list) and data and isinstance(data[0], dict) and "files" in data[0]:
        return "actual_manifest"
    raise ValueError(f"Cannot detect target kind for {path}")


def load_generated_threads(
    path: str | Path,
    max_threads: int = 10,
    max_comments: int = 8,
) -> list[DiscussionThread]:
    """Load simulated discussion threads from a `discussion.json` file.

    The generated discussion format is flatter than real Reddit data, so this
    loader mostly does field renaming plus defensive sanitation.
    """

    path = Path(path)
    if path.is_dir():
        path = path / "discussion.json"

    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    posts = data.get("posts") or []
    threads: list[DiscussionThread] = []
    for post in posts[:max_threads]:
        comments = _flatten_generated_comments(
            post.get("comments") or [],
            max_comments=max_comments,
        )
        threads.append(
            DiscussionThread(
                source_kind="generated",
                thread_id=str(post.get("post_id")),
                title=_truncate_text(sanitize_text(post.get("content") or ""), 120),
                body=sanitize_text(post.get("content") or ""),
                author=sanitize_text(post.get("author") or "unknown"),
                score=_safe_int(post.get("likes")),
                comment_count=len(post.get("comments") or []),
                comments=comments,
                subreddit=sanitize_text((data.get("meta") or {}).get("product_category") or "generated_discussion"),
                topic=sanitize_text((data.get("meta") or {}).get("product_category") or ""),
                url=None,
            )
        )
    return threads


def load_actual_threads(
    bundle_json_path: str | Path,
    comments_jsonl_path: str | Path | None = None,
    max_threads: int = 10,
    max_comments: int = 8,
) -> list[DiscussionThread]:
    """Load one real Reddit bundle and its comments into judge threads.

    The bundle JSON stores post-level metadata, while the `.comments.jsonl`
    sidecar stores flattened comments. This function joins them and ranks posts
    so the judge sees the most relevant/high-signal threads first.
    """

    bundle_json_path = Path(bundle_json_path)
    if comments_jsonl_path is None:
        comments_jsonl_path = bundle_json_path.with_name(
            bundle_json_path.stem + ".comments.jsonl"
        )
    comments_jsonl_path = Path(comments_jsonl_path)

    with bundle_json_path.open("r", encoding="utf-8") as handle:
        bundle = json.load(handle)

    comments_by_post = _load_actual_comments(comments_jsonl_path)
    posts = bundle.get("posts") or []
    ranked_posts = sorted(
        posts,
        key=lambda post: (
            not bool(post.get("is_relevant", True)),
            _safe_int(post.get("search_rank"), default=10**9),
            -_safe_int(post.get("num_comments"), default=0),
            -_safe_int(post.get("score"), default=0),
        ),
    )

    threads: list[DiscussionThread] = []
    for post in ranked_posts[:max_threads]:
        post_id = str(post.get("id"))
        raw_comments = comments_by_post.get(post_id, [])
        selected_comments = sorted(
            raw_comments,
            key=lambda comment: (
                _safe_int(comment.get("depth"), default=0),
                -_safe_int(comment.get("score"), default=0),
                comment.get("created_iso") or "",
            ),
        )[:max_comments]
        comments = [
            DiscussionComment(
                author=sanitize_text(comment.get("author") or "unknown"),
                content=sanitize_text(comment.get("body") or ""),
                comment_id=sanitize_text(comment.get("comment_id") or "") or None,
                score=_safe_int(comment.get("score")),
                depth=_safe_int(comment.get("depth"), default=0),
                timestamp=sanitize_text(comment.get("created_iso") or ""),
            )
            for comment in selected_comments
        ]

        title = sanitize_text(post.get("title") or post.get("post_title") or "")
        body = sanitize_text(post.get("selftext") or "")
        if not body:
            body = title

        threads.append(
            DiscussionThread(
                source_kind="actual",
                thread_id=post_id,
                title=title or _truncate_text(body, 120),
                body=body,
                author=sanitize_text(post.get("author") or "unknown"),
                score=_safe_int(post.get("score")),
                comment_count=len(raw_comments) if raw_comments else _safe_int(post.get("num_comments"), default=0),
                comments=comments,
                subreddit=sanitize_text(post.get("subreddit") or ""),
                topic=sanitize_text(bundle.get("topic") or ""),
                url=sanitize_text(post.get("permalink") or post.get("url") or ""),
            )
        )
    return threads


def load_threads_from_manifest(
    manifest_path: str | Path,
    max_items: int = 10,
    threads_per_item: int = 2,
    max_comments: int = 8,
) -> list[DiscussionThread]:
    """Load representative actual threads from a manifest of many cards.

    This is mainly used when:
    - building a batch evaluation set
    - retrieving a few actual reference threads for the judge
    """

    with Path(manifest_path).open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    threads: list[DiscussionThread] = []
    for item in manifest[:max_items]:
        files = item.get("files") or {}
        if item.get("discussion_bundle_status") != "json_bundle":
            continue
        bundle_path = _resolve_repo_path(files.get("json"))
        comments_path = _resolve_repo_path(files.get("comments_jsonl"))
        if not bundle_path or not comments_path:
            continue
        threads.extend(
            load_actual_threads(
                bundle_json_path=bundle_path,
                comments_jsonl_path=comments_path,
                max_threads=threads_per_item,
                max_comments=max_comments,
            )
        )
    return threads


def _load_actual_comments(comments_jsonl_path: Path) -> dict[str, list[dict[str, Any]]]:
    """Load flattened Reddit comments and group them by post id."""

    comments_by_post: dict[str, list[dict[str, Any]]] = {}
    if not comments_jsonl_path.exists():
        return comments_by_post

    with comments_jsonl_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            post_id = str(record.get("post_id"))
            comments_by_post.setdefault(post_id, []).append(record)
    return comments_by_post


def _flatten_generated_comments(
    comments: list[dict[str, Any]],
    max_comments: int,
    depth: int = 0,
) -> list[DiscussionComment]:
    """Flatten nested generated comments into judge-friendly comment objects."""

    flattened: list[DiscussionComment] = []
    for comment in comments:
        flattened.append(
            DiscussionComment(
                author=sanitize_text(comment.get("author") or "unknown"),
                content=sanitize_text(comment.get("content") or ""),
                comment_id=sanitize_text(comment.get("comment_id") or "") or None,
                score=_safe_int(comment.get("likes")),
                depth=_safe_int(comment.get("depth"), default=depth) or depth,
                timestamp=sanitize_text(comment.get("timestamp") or ""),
            )
        )
        if len(flattened) >= max_comments:
            return flattened[:max_comments]
        flattened.extend(
            _flatten_generated_comments(
                comment.get("replies") or [],
                max_comments=max_comments - len(flattened),
                depth=depth + 1,
            )
        )
        if len(flattened) >= max_comments:
            return flattened[:max_comments]
    return flattened[:max_comments]


def _resolve_repo_path(path: str | Path | None) -> Path | None:
    """Resolve a manifest/file path relative to the repo root when needed."""

    if not path:
        return None
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return REPO_ROOT / candidate


def _safe_int(value: Any, default: Optional[int] = None) -> Optional[int]:
    """Best-effort integer conversion used for heterogeneous raw inputs."""

    try:
        if value is None or value == "":
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _truncate_text(text: str, limit: int) -> str:
    """Normalize whitespace and trim a string to a bounded prompt length."""

    text = sanitize_text(text)
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 1].rstrip() + "…"
