#!/usr/bin/env python3
"""Score sampled-generator runs and merge them for matched-seed evaluation.

The sampled generator stores generated post ids like
``sampled_run00_post05_seed005``.  The matched-seed evaluator expects each row
to have:

* ``_run_id``: zero-based simulation/run id
* ``thread_id``: one-based post slot inside that run

This script runs the standard thread metric suite for every
``run_*_sampled_reddit`` directory, then rewrites ``thread_id`` back to the
original post slot so interrupted/resumed runs with missing slots still match
the correct real seed post.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "generalized_card"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from generalized_card.thread_metric_suite import (  # noqa: E402
    load_thread_metrics,
    score_thread_metric_suite,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score sampled generated Reddit runs and merge thread metrics."
    )
    parser.add_argument(
        "artifact_dir",
        type=Path,
        help="Directory containing run_*_sampled_reddit subdirectories.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for merged generated score CSV.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device passed to metric scorers. Default: cpu.",
    )
    parser.add_argument(
        "--metric-parallel",
        type=int,
        default=2,
        help="Maximum metric scorers to run concurrently per run. Default: 2.",
    )
    parser.add_argument(
        "--expected-seeds",
        type=int,
        default=54,
        help="Expected seed-post pool size used only for missing-index reporting.",
    )
    return parser.parse_args()


def run_id_from_dir(run_dir: Path) -> int:
    match = re.search(r"run_(\d+)_", run_dir.name)
    if not match:
        raise ValueError(f"Cannot parse run id from {run_dir.name}")
    return int(match.group(1))


def load_post_mapping(run_dir: Path) -> dict[str, dict[str, object]]:
    discussion_path = run_dir / "discussion.json"
    data = json.loads(discussion_path.read_text(encoding="utf-8"))
    mapping: dict[str, dict[str, object]] = {}
    for post in data.get("posts", []):
        post_id = str(post.get("post_id") or "")
        post_slot = post.get("post_slot")
        if not post_id or post_slot is None:
            raise ValueError(f"Post missing post_id/post_slot in {discussion_path}")
        mapping[post_id] = {
            "thread_id": int(post_slot) + 1,
            "post_slot": int(post_slot),
            "seed_index": post.get("seed_index"),
            "source_raw_post_id": post.get("source_raw_post_id"),
            "source_product_dir": post.get("source_product_dir"),
            "source_file": post.get("source_file"),
        }
    return mapping


def score_and_load_run(
    run_dir: Path, *, repo_root: Path, args: argparse.Namespace
) -> pd.DataFrame:
    run_id = run_id_from_dir(run_dir)
    print(f"[score] {run_dir.name}", flush=True)
    score_thread_metric_suite(
        run_dir,
        python=sys.executable,
        repo_root=repo_root,
        device=args.device,
        metric_parallel=args.metric_parallel,
    )

    df = load_thread_metrics(run_dir).copy()
    post_mapping = load_post_mapping(run_dir)
    metric_thread_ids = df["thread_id"].astype(str).tolist()
    missing = [
        thread_id for thread_id in metric_thread_ids if thread_id not in post_mapping
    ]
    if missing:
        sample = ", ".join(missing[:5])
        raise RuntimeError(
            f"{run_dir}: metric thread ids not found in discussion.json: {sample}"
        )

    df["_metric_thread_id"] = metric_thread_ids
    df["_run_id"] = run_id
    df["_source_sim_dir"] = str(run_dir)
    df["thread_id"] = [
        post_mapping[thread_id]["thread_id"] for thread_id in metric_thread_ids
    ]
    df["post_slot"] = [
        post_mapping[thread_id]["post_slot"] for thread_id in metric_thread_ids
    ]
    df["seed_index"] = [
        post_mapping[thread_id]["seed_index"] for thread_id in metric_thread_ids
    ]
    df["source_raw_post_id"] = [
        post_mapping[thread_id]["source_raw_post_id"] for thread_id in metric_thread_ids
    ]
    df["source_product_dir"] = [
        post_mapping[thread_id]["source_product_dir"] for thread_id in metric_thread_ids
    ]
    df["source_file"] = [
        post_mapping[thread_id]["source_file"] for thread_id in metric_thread_ids
    ]
    return df


def main() -> None:
    args = parse_args()
    artifact_dir = args.artifact_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    repo_root = Path.cwd().resolve()

    run_dirs = sorted(artifact_dir.glob("run_*_sampled_reddit"))
    if not run_dirs:
        raise FileNotFoundError(
            f"No run_*_sampled_reddit directories under {artifact_dir}"
        )

    frames = [
        score_and_load_run(run_dir, repo_root=repo_root, args=args)
        for run_dir in run_dirs
    ]
    merged = pd.concat(frames, ignore_index=True)
    merged = merged.sort_values(["_run_id", "thread_id"]).reset_index(drop=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "revised_generated_thread_scores.csv"
    merged.to_csv(csv_path, index=False)

    observed_seed_indices = {
        int(value)
        for value in merged["seed_index"].dropna().tolist()
        if str(value).strip()
    }
    missing_seed_indices = sorted(
        set(range(args.expected_seeds)) - observed_seed_indices
    )
    summary = {
        "source_artifact": str(artifact_dir),
        "generated_scores_csv": str(csv_path),
        "rows": int(len(merged)),
        "run_count": int(merged["_run_id"].nunique()),
        "missing_seed_indices": missing_seed_indices,
    }
    (output_dir / "score_merge_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[done] {csv_path}")
    print(f"Rows: {summary['rows']} / runs: {summary['run_count']}")
    print(f"Missing seed indices: {missing_seed_indices}")


if __name__ == "__main__":
    main()
