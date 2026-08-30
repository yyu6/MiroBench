#!/usr/bin/env python3
"""Import the legacy credit-card benchmark into the multidomain input layout.

The original credit-card crawl predates ``build_seed_pools.py`` and uses
``comment_id`` plus Reddit-prefixed parent ids (``t1_``/``t3_``).  This
maintainer utility keeps the benchmark's first 150 seeds, deduplicates the
scraped comments, normalizes their parent links, and writes the same seed-pool
and real-reference layout used by the other domains.

Fresh machines do not need the legacy crawl or this import step.  They consume
the already anonymized result committed under ``portable_inputs``.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

from build_seed_pools import _write_real_reference
from common import REPO_ROOT, load_jsonl, read_json, write_json


DEFAULT_SEED_POOL = (
    REPO_ROOT
    / "artifacts"
    / "seed_posts"
    / "credit_cards_test_real_distribution_seed_pool_154_20260609.json"
)
DEFAULT_RAW_ROOT = REPO_ROOT / "data" / "raw" / "discussions" / "credit_cards"
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT / "artifacts" / "reddit_multidomain_baselines" / "inputs"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-pool-json", type=Path, default=DEFAULT_SEED_POOL)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--max-seeds", type=int, default=150)
    parser.add_argument("--posts-per-run", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_pool_path = args.seed_pool_json.expanduser().resolve()
    raw_root = args.raw_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    source_pool = read_json(source_pool_path)
    source_seeds = list(source_pool.get("seed_posts") or [])
    if len(source_seeds) < args.max_seeds:
        raise ValueError(
            f"Legacy pool has {len(source_seeds)} seeds; need {args.max_seeds}"
        )

    selected = [
        normalize_seed(seed, index=index)
        for index, seed in enumerate(source_seeds[: args.max_seeds])
    ]
    selected_ids = {str(seed["source_raw_post_id"]) for seed in selected}
    comments_by_post = load_legacy_comments(raw_root, selected_ids)

    pool_path = output_root / "seed_pools" / "credit_cards.json"
    write_json(
        pool_path,
        {
            "meta": {
                "builder": (
                    "experiments/reddit_multidomain_baselines/scripts/"
                    "import_legacy_credit_cards.py"
                ),
                "domain": "credit_cards",
                "source_seed_pool_json": str(source_pool_path),
                "raw_root": str(raw_root),
                "source_seed_count": len(source_seeds),
                "max_seeds": len(selected),
                "posts_per_run": args.posts_per_run,
                "selection": "first max_seeds records from the legacy matched pool",
                "source_comment_count": sum(map(len, comments_by_post.values())),
            },
            "seed_posts": selected,
        },
    )
    reference_root = output_root / "real_reference" / "credit_cards"
    _write_real_reference(
        reference_root=reference_root,
        domain="credit_cards",
        seed_posts=selected,
        comments_by_post=comments_by_post,
        posts_per_run=args.posts_per_run,
        pool_path=pool_path,
    )
    with_comments = sum(
        bool(comments_by_post[str(seed["source_raw_post_id"])]) for seed in selected
    )
    print(
        f"[done] domain=credit_cards seeds={len(selected)} "
        f"threads_with_comments={with_comments} "
        f"real_comments={sum(map(len, comments_by_post.values()))} pool={pool_path}"
    )


def normalize_seed(seed: dict[str, Any], *, index: int) -> dict[str, Any]:
    post_id = str(seed.get("source_raw_post_id") or "").strip()
    if not post_id:
        raise ValueError(f"Legacy seed {index} has no source_raw_post_id")
    title = text(seed.get("title")).strip()
    body = text(seed.get("body")).strip()
    content = text(seed.get("content")).strip()
    if not content:
        content = "\n\n".join(part for part in (title, body) if part)
    return {
        "poster_agent_id": index,
        "source_raw_post_id": post_id,
        "source_product": text(seed.get("source_product") or "CreditCards"),
        "source_product_dir": text(
            seed.get("source_product_dir") or seed.get("source_product") or "CreditCards"
        ),
        "source_file": f"reddit_post:{post_id}",
        "post_type": text(seed.get("post_type") or "reddit_post"),
        "topic_domain": "credit_cards",
        "subreddit": text(seed.get("real_subreddit") or "CreditCards"),
        "title": title,
        "body": body,
        "content": content,
        "source_created_utc": seed.get("source_created_utc"),
        "source_num_comments": int(seed.get("real_num_comments") or 0),
    }


def load_legacy_comments(
    raw_root: Path, selected_ids: set[str]
) -> dict[str, list[dict[str, Any]]]:
    deduplicated: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for path in sorted(raw_root.glob("*/*.comments.jsonl")):
        for row in load_jsonl(path):
            post_id = str(row.get("post_id") or "").strip()
            if post_id not in selected_ids:
                continue
            comment_id = str(row.get("comment_id") or row.get("id") or "").strip()
            if not comment_id:
                continue
            parent_id = str(
                row.get("parent_comment_id") or row.get("parent_id") or ""
            ).strip()
            parent_comment_id = parent_id[3:] if parent_id.startswith("t1_") else None
            deduplicated[post_id][comment_id] = {
                "id": comment_id,
                "post_id": post_id,
                "parent_comment_id": parent_comment_id,
                "author": text(row.get("author")) or "[deleted]",
                "body": text(row.get("body")),
                "created_iso": text(row.get("created_iso")),
                "score": int(row.get("score") or 0),
            }
    return {
        post_id: list(deduplicated.get(post_id, {}).values())
        for post_id in selected_ids
    }


def text(value: Any) -> str:
    return "" if value is None else str(value)


if __name__ == "__main__":
    main()
