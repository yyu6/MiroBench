#!/usr/bin/env python3
"""Create deterministic matched seed pools and real-reference discussions.

Each selected Reddit root post becomes a fixed seed for both SynthPAI and
OASIS.  Its scraped comments are also converted to GEO's ``discussion.json``
format, so evaluation compares generated and real threads on the exact same
post sample.
"""
from __future__ import annotations

import argparse
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

from common import REPO_ROOT, count_comments, load_jsonl, read_json, write_json


DEFAULT_DATA_ROOT = REPO_ROOT / "data" / "reddit_domain_posts 2"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "artifacts" / "reddit_multidomain_baselines" / "inputs"
PORTABLE_INPUT_ROOT = REPO_ROOT / "experiments" / "reddit_multidomain_baselines" / "portable_inputs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--domains", nargs="*", help="Defaults to every domain directory.")
    parser.add_argument("--max-seeds", type=int, default=150)
    parser.add_argument("--posts-per-run", type=int, default=5)
    parser.add_argument("--min-real-comments", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def available_domains(data_root: Path) -> list[str]:
    if data_root.is_dir():
        domains = sorted(
            path.name
            for path in data_root.iterdir()
            if path.is_dir() and (path / f"{path.name}.jsonl").exists()
        )
        if domains:
            return domains
    return sorted(path.stem for path in (PORTABLE_INPUT_ROOT / "seed_pools").glob("*.json"))


def main() -> None:
    args = parse_args()
    data_root = args.data_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    domains = args.domains or available_domains(data_root)
    if not domains:
        raise SystemExit(f"No domain directories found under {data_root}")
    for domain in domains:
        build_domain(
            data_root=data_root,
            output_root=output_root,
            domain=domain,
            max_seeds=args.max_seeds,
            posts_per_run=args.posts_per_run,
            min_real_comments=args.min_real_comments,
            seed=args.seed,
            force=args.force,
        )


def build_domain(
    *,
    data_root: Path,
    output_root: Path,
    domain: str,
    max_seeds: int,
    posts_per_run: int,
    min_real_comments: int,
    seed: int,
    force: bool,
) -> Path:
    """Build one domain's seed pool and same-seed real-reference artifacts."""

    if max_seeds < 1 or posts_per_run < 1:
        raise ValueError("--max-seeds and --posts-per-run must be positive")
    pool_path = output_root / "seed_pools" / f"{domain}.json"
    reference_root = output_root / "real_reference" / domain
    if pool_path.exists() and (reference_root / "reference_manifest.json").exists() and not force:
        print(f"[skip] domain={domain} -> {pool_path}")
        return pool_path

    domain_dir = data_root / domain
    posts_path = domain_dir / f"{domain}.jsonl"
    comments_path = domain_dir / f"{domain}.comments.jsonl"
    manifest_path = domain_dir / f"{domain}.comments_manifest.json"
    if not posts_path.exists() or not comments_path.exists():
        raise FileNotFoundError(
            f"Missing posts/comments JSONL for domain={domain}: {domain_dir}. "
            "On a fresh clone, run experiments/reddit_multidomain_baselines/setup.sh "
            "to install the committed portable inputs first."
        )
    if force and reference_root.exists():
        # The caller explicitly requested replacement. Keeping stale higher run
        # numbers would silently add old threads to later evaluation.
        shutil.rmtree(reference_root)

    posts = load_jsonl(posts_path)
    comments = load_jsonl(comments_path)
    comments_by_post: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for comment in comments:
        post_id = str(comment.get("post_id") or "").strip()
        if post_id:
            comments_by_post[post_id].append(comment)

    eligible = [
        post
        for post in posts
        if _post_content(post)
        and len(comments_by_post.get(str(post.get("id") or ""), [])) >= min_real_comments
    ]
    if len(eligible) < max_seeds:
        raise ValueError(
            f"domain={domain} has only {len(eligible)} eligible posts, need {max_seeds}; "
            "lower --max-seeds or --min-real-comments"
        )
    rng = random.Random(f"{seed}:{domain}")
    selected = rng.sample(sorted(eligible, key=lambda row: str(row.get("id") or "")), max_seeds)
    selected.sort(key=lambda row: str(row.get("id") or ""))

    seed_posts = [_to_seed_post(post, index) for index, post in enumerate(selected)]
    manifest = _load_manifest(manifest_path)
    pool = {
        "meta": {
            "builder": "experiments/reddit_multidomain_baselines/scripts/build_seed_pools.py",
            "domain": domain,
            "data_root": str(data_root),
            "posts_jsonl": str(posts_path),
            "comments_jsonl": str(comments_path),
            "max_seeds": max_seeds,
            "posts_per_run": posts_per_run,
            "min_real_comments": min_real_comments,
            "random_seed": seed,
            "available_posts": len(posts),
            "eligible_posts": len(eligible),
            "source_comment_count": len(comments),
            "source_manifest": manifest,
        },
        "seed_posts": seed_posts,
    }
    write_json(pool_path, pool)
    _write_real_reference(
        reference_root=reference_root,
        domain=domain,
        seed_posts=seed_posts,
        comments_by_post=comments_by_post,
        posts_per_run=posts_per_run,
        pool_path=pool_path,
    )
    total_comments = sum(len(comments_by_post[str(post.get("id") or "")]) for post in selected)
    print(
        f"[done] domain={domain} seeds={len(seed_posts)} "
        f"real_comments={total_comments} pool={pool_path}"
    )
    return pool_path


def _load_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = read_json(path)
    return data if isinstance(data, dict) else {}


def _to_seed_post(post: dict[str, Any], index: int) -> dict[str, Any]:
    return {
        "poster_agent_id": index,
        "source_raw_post_id": str(post.get("id") or ""),
        "source_product": str(post.get("subreddit") or ""),
        "source_product_dir": str(post.get("subreddit") or ""),
        "source_file": str(post.get("permalink") or ""),
        "post_type": str(post.get("link_flair_text") or post.get("post_hint") or "reddit_post"),
        "topic_domain": str(post.get("topic_domain") or ""),
        "subreddit": str(post.get("subreddit") or ""),
        "title": _text(post.get("title")),
        "body": _text(post.get("selftext")),
        "content": _post_content(post),
        "source_created_utc": post.get("created_utc"),
        "source_num_comments": int(post.get("num_comments") or 0),
    }


def _write_real_reference(
    *,
    reference_root: Path,
    domain: str,
    seed_posts: list[dict[str, Any]],
    comments_by_post: dict[str, list[dict[str, Any]]],
    posts_per_run: int,
    pool_path: Path,
) -> None:
    total_comment_count = 0
    for run_id, start in enumerate(range(0, len(seed_posts), posts_per_run)):
        batch = seed_posts[start : start + posts_per_run]
        posts = []
        for post_slot, seed in enumerate(batch):
            flat_comments = comments_by_post.get(str(seed.get("source_raw_post_id") or ""), [])
            tree = _nest_comments(flat_comments)
            total_comment_count += count_comments(tree)
            posts.append(
                {
                    "post_id": f"real_run{run_id:03d}_post{post_slot:02d}_seed{start + post_slot:03d}",
                    "post_slot": post_slot,
                    "seed_index": start + post_slot,
                    "source_raw_post_id": seed.get("source_raw_post_id"),
                    "source_product_dir": seed.get("source_product_dir"),
                    "source_file": seed.get("source_file"),
                    "post_type": seed.get("post_type"),
                    "title": seed.get("title"),
                    "author": "real_reddit_op",
                    "author_karma": 0,
                    "content": seed.get("content"),
                    "timestamp": "",
                    "likes": 0,
                    "dislikes": 0,
                    "comments": tree,
                }
            )
        write_json(
            reference_root / f"run_{run_id:03d}_sampled_reddit" / "discussion.json",
            {
                "meta": {
                    "product_category": domain,
                    "baseline": "real_reddit_reference",
                    "run_id": f"run_{run_id:03d}_sampled_reddit",
                },
                "posts": posts,
            },
        )
    write_json(
        reference_root / "reference_manifest.json",
        {
            "domain": domain,
            "seed_pool": str(pool_path),
            "seed_count": len(seed_posts),
            "posts_per_run": posts_per_run,
            "run_count": (len(seed_posts) + posts_per_run - 1) // posts_per_run,
            "real_comment_count": total_comment_count,
        },
    )


def _nest_comments(comments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    nodes: dict[str, dict[str, Any]] = {}
    ordered_ids: list[str] = []
    for index, comment in enumerate(comments, start=1):
        comment_id = str(comment.get("id") or f"comment_{index}")
        nodes[comment_id] = {
            "comment_id": comment_id,
            "author": _text(comment.get("author")) or "[deleted]",
            "author_karma": 0,
            "content": _text(comment.get("body")),
            "timestamp": _text(comment.get("created_iso")),
            "likes": int(comment.get("score") or 0),
            "dislikes": 0,
            "parent_comment_id": comment.get("parent_comment_id"),
            "depth": 0,
            "replies": [],
        }
        ordered_ids.append(comment_id)

    roots: list[dict[str, Any]] = []
    for comment_id in ordered_ids:
        node = nodes[comment_id]
        parent_id = str(node.get("parent_comment_id") or "")
        parent = nodes.get(parent_id)
        if parent is None or parent_id == comment_id:
            roots.append(node)
        else:
            parent["replies"].append(node)
    _set_depths(roots, depth=0)
    return roots


def _set_depths(nodes: list[dict[str, Any]], *, depth: int) -> None:
    for node in nodes:
        node["depth"] = depth
        _set_depths(list(node.get("replies") or []), depth=depth + 1)


def _post_content(post: dict[str, Any]) -> str:
    title = _text(post.get("title")).strip()
    body = _text(post.get("selftext")).strip()
    return "\n\n".join(part for part in (title, body) if part).strip()


def _text(value: Any) -> str:
    return "" if value is None else str(value)


if __name__ == "__main__":
    main()
