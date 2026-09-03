#!/usr/bin/env python3
"""Prepare one 150-row real score CSV per MiroBench domain.

The fixed real references are scored as one combined discussion per domain so
learned metric models are loaded once rather than once per five-thread run.
When a compatible full real-score cache already exists, the selected reference
rows are imported by Reddit post ID and no model inference is performed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
PACKAGE_ROOT = REPO_ROOT / "generalized_card"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from generalized_card.thread_metric_suite import (  # noqa: E402
    load_thread_metrics,
    score_thread_metric_suite,
)


DEFAULT_RUN_ROOT = REPO_ROOT / "artifacts" / "reddit_multidomain_baselines"
DEFAULT_DOMAINS = (
    "camera",
    "celebrity",
    "cellphone",
    "credit_cards",
    "game",
    "headphones",
    "health_issue",
    "laptop",
    "movies",
    "news",
    "sports",
    "tv_series",
)
CORE_METRICS = (
    "self_bleu_4",
    "self_bertscore_mean_f1",
    "semantic_mean_cosine",
    "mean_story_probability",
    "emotion_entropy",
    "polite_rate",
    "neutral_rate",
    "impolite_rate",
    "avg_depth",
    "hard_disagree_rate",
    "structural_virality",
    "length_cv",
)

# These caches use the same metric output schema and contain the source post IDs
# selected by the current fixed references. Missing or incompatible caches are
# ignored and the reference is scored directly.
CACHE_CANDIDATES = {
    "celebrity": REPO_ROOT / "artifacts/baselines/celebrity_geo/real/thread_scores.csv",
    "credit_cards": REPO_ROOT
    / "artifacts/baselines/credit_cards_gpt4omini/real/thread_scores.csv",
    "game": REPO_ROOT / "artifacts/baselines/game_geo/real/thread_scores.csv",
    "news": REPO_ROOT / "artifacts/baselines/news_geo/real/thread_scores.csv",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--domains", nargs="*", default=list(DEFAULT_DOMAINS))
    parser.add_argument("--device", default="auto", choices=("cpu", "mps", "cuda", "auto"))
    parser.add_argument("--metric-parallel", type=int, default=2)
    parser.add_argument("--expected-threads", type=int, default=150)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def load_reference_posts(reference_root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for run_dir in sorted(reference_root.glob("run_*_sampled_reddit")):
        match = re.search(r"run_(\d+)_", run_dir.name)
        if not match:
            continue
        run_id = int(match.group(1))
        payload = json.loads((run_dir / "discussion.json").read_text(encoding="utf-8"))
        for post in payload.get("posts", []):
            copied = dict(post)
            copied["_run_id"] = run_id
            copied["_source_run_dir"] = str(run_dir)
            records.append(copied)
    return records


def output_is_complete(path: Path, expected_threads: int) -> bool:
    if not path.exists():
        return False
    try:
        frame = pd.read_csv(path)
    except Exception:
        return False
    return len(frame) == expected_threads and all(metric in frame for metric in CORE_METRICS)


def select_cached_rows(
    cache_path: Path,
    posts: list[dict[str, Any]],
) -> list[pd.Series | None] | None:
    if not cache_path.exists():
        return None
    cache = pd.read_csv(cache_path)
    if "thread_id" not in cache or any(metric not in cache for metric in CORE_METRICS):
        return None
    by_id: dict[str, pd.DataFrame] = {
        thread_id: group
        for thread_id, group in cache.groupby(cache["thread_id"].astype(str), sort=False)
    }
    selected: list[pd.Series | None] = []
    for post in posts:
        source_id = str(post.get("source_raw_post_id") or "")
        candidates = by_id.get(source_id)
        if candidates is None or candidates.empty:
            selected.append(None)
            continue
        if len(candidates) > 1 and "product" in candidates:
            product = str(post.get("source_product_dir") or "")
            exact = candidates[candidates["product"].astype(str) == product]
            if len(exact) == 1:
                candidates = exact
        selected.append(candidates.iloc[0].copy())
    return selected


def attach_reference_metadata(
    frame: pd.DataFrame,
    posts: list[dict[str, Any]],
    *,
    metric_id_column: str,
) -> pd.DataFrame:
    post_by_metric_id = {str(post.get("post_id") or ""): post for post in posts}
    if metric_id_column == "thread_id" and all(
        str(value) in post_by_metric_id for value in frame[metric_id_column].astype(str)
    ):
        ordered_posts = [post_by_metric_id[str(value)] for value in frame[metric_id_column]]
    else:
        # Cached full-score rows are already selected in reference order.
        if len(frame) != len(posts):
            raise ValueError("Score/reference row count mismatch")
        ordered_posts = posts
    frame = frame.copy()
    frame["_metric_thread_id"] = frame[metric_id_column].astype(str)
    frame["_run_id"] = [int(post["_run_id"]) for post in ordered_posts]
    frame["post_slot"] = [int(post.get("post_slot") or 0) for post in ordered_posts]
    frame["seed_index"] = [post.get("seed_index") for post in ordered_posts]
    frame["source_raw_post_id"] = [post.get("source_raw_post_id") for post in ordered_posts]
    frame["source_product_dir"] = [post.get("source_product_dir") for post in ordered_posts]
    frame["source_file"] = [post.get("source_file") for post in ordered_posts]
    return frame


def score_combined_reference(
    *,
    domain: str,
    posts: list[dict[str, Any]],
    output_dir: Path,
    device: str,
    metric_parallel: int,
) -> pd.DataFrame:
    source_ids = "\n".join(str(post.get("source_raw_post_id") or "") for post in posts)
    input_hash = hashlib.sha256(source_ids.encode("utf-8")).hexdigest()[:12]
    combined_dir = output_dir / f"sanity_combined_reference_{input_hash}"
    combined_dir.mkdir(parents=True, exist_ok=True)
    combined_path = combined_dir / "discussion.json"
    combined_payload = {
        "meta": {
            "product_category": domain,
            "baseline": "real_reddit_reference_combined_for_sanity",
            "thread_count": len(posts),
        },
        "posts": [
            {key: value for key, value in post.items() if not key.startswith("_")}
            for post in posts
        ],
    }
    combined_path.write_text(
        json.dumps(combined_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    score_thread_metric_suite(
        combined_dir,
        python=sys.executable,
        repo_root=REPO_ROOT,
        device=device,
        metric_parallel=metric_parallel,
    )
    frame = load_thread_metrics(combined_dir).copy()
    return attach_reference_metadata(frame, posts, metric_id_column="thread_id")


def main() -> None:
    args = parse_args()
    run_root = args.run_root.expanduser().resolve()
    if args.expected_threads < 1 or args.metric_parallel < 1:
        raise ValueError("--expected-threads and --metric-parallel must be positive")

    for domain in args.domains:
        reference_root = run_root / "inputs" / "real_reference" / domain
        output_dir = run_root / "evaluation" / "real_reference" / domain
        output_path = output_dir / "revised_generated_thread_scores.csv"
        if output_is_complete(output_path, args.expected_threads) and not args.force:
            print(f"[skip] domain={domain} cached={output_path}")
            continue
        posts = load_reference_posts(reference_root)
        if len(posts) != args.expected_threads:
            raise ValueError(
                f"domain={domain} has {len(posts)} reference threads; "
                f"expected {args.expected_threads}"
            )
        output_dir.mkdir(parents=True, exist_ok=True)
        cache_path = CACHE_CANDIDATES.get(domain)
        cached_rows = select_cached_rows(cache_path, posts) if cache_path else None
        if cached_rows is None:
            cached_rows = [None] * len(posts)
        missing_positions = [index for index, row in enumerate(cached_rows) if row is None]
        missing_posts = [posts[index] for index in missing_positions]
        source = "compatible_full_score_cache"
        newly_scored = pd.DataFrame()
        if missing_posts:
            source = (
                "combined_reference_metric_scoring"
                if len(missing_posts) == len(posts)
                else "compatible_cache_plus_missing_reference_scoring"
            )
            print(
                f"[score] domain={domain} missing_threads={len(missing_posts)}/"
                f"{len(posts)} device={args.device}"
            )
            newly_scored = score_combined_reference(
                domain=domain,
                posts=missing_posts,
                output_dir=output_dir,
                device=args.device,
                metric_parallel=args.metric_parallel,
            )
        else:
            print(f"[reuse] domain={domain} source={cache_path}")
        new_by_id = {
            str(row["source_raw_post_id"]): row
            for _, row in newly_scored.iterrows()
        }
        final_rows: list[pd.Series] = []
        for post, cached_row in zip(posts, cached_rows):
            if cached_row is not None:
                final_rows.append(cached_row)
                continue
            source_id = str(post.get("source_raw_post_id") or "")
            if source_id not in new_by_id:
                raise RuntimeError(f"domain={domain} missing score for source post {source_id}")
            final_rows.append(new_by_id[source_id])
        frame = attach_reference_metadata(
            pd.DataFrame(final_rows).reset_index(drop=True),
            posts,
            metric_id_column="thread_id",
        )
        if len(frame) != args.expected_threads:
            raise RuntimeError(f"domain={domain} produced {len(frame)} score rows")
        frame.to_csv(output_path, index=False)
        manifest = {
            "domain": domain,
            "reference_root": str(reference_root),
            "output_csv": str(output_path),
            "thread_count": len(frame),
            "core_metrics": list(CORE_METRICS),
            "source": source,
            "cache_source": str(cache_path) if cache_path and cache_path.exists() else None,
            "cached_thread_count": len(posts) - len(missing_posts),
            "newly_scored_thread_count": len(missing_posts),
        }
        (output_dir / "sanity_score_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"[done] domain={domain} rows={len(frame)} output={output_path}")


if __name__ == "__main__":
    main()
