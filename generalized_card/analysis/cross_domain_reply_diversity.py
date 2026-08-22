#!/usr/bin/env python3
"""Does real Reddit writing diversify with reply depth in every domain, or
only in camera?

`root_reply_diversity.py` checked one property (`docs/DECISIONS.md` G3) on
camera alone: real `reply_reply` pairs are less similar than real `root_root`
pairs, and this generalizes past the 10 v103-matched threads (82% of 247
evaluation-excluded camera threads, Wilcoxon p~=0). Before treating that as a
property of Reddit writing worth building a new mechanism against (rather
than a camera-specific accident), this checks the same two things -- the
root/reply split, and a depth-binned trend -- on the real corpus of all four
registered domains: camera, cell_phone, headphone, laptop.

Uses the cheap sentence-embedding scorer (`all-mpnet-base-v2`,
`semantic_mean_cosine`'s model, not BERTScore) -- this is a test of a text
property's *direction* across domains, not a reproduction of
`self_bertscore_mean_f1` itself, and no domain has ever had a paid generation
run (`docs/DECISIONS.md` D3), so there is no generated side to compare
against here regardless. Real data only. No API call.

    python3 generalized_card/analysis/cross_domain_reply_diversity.py
    python3 generalized_card/analysis/cross_domain_reply_diversity.py --domain camera
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
SCORER_DIR = REPO / "scripts" / "evaluation"
ANALYSIS_DIR = Path(__file__).resolve().parent
for path in (SCORER_DIR, ANALYSIS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from score_thread_semantic_uniformity import (  # noqa: E402
    CommentEmbedder,
    DEFAULT_MODEL,
    ThreadComment,
    load_real_comments,
)
from bertscore_pair_diagnosis import _is_root, _parent_map  # noqa: E402

# One canonical seed pool per domain -- the one each domain's own
# `--prepare-only`/profile-build path actually uses (headphone has two pool
# files on disk; 150 is the current one, matching camera's convention).
DOMAINS: dict[str, dict[str, Any]] = {
    "camera": {
        "real_dir": REPO / "data/raw/discussions/camera_product",
        "seed_pool": REPO / "artifacts/generalized_card/seed_pools/camera_product_150_seed42.json",
    },
    "cell_phone": {
        "real_dir": REPO / "data/raw/discussions/cell_phone_product",
        "seed_pool": REPO / "artifacts/generalized_card/seed_pools/cell_phone_product_100_seed42.json",
    },
    "headphone": {
        "real_dir": REPO / "data/raw/discussions/headphone_product",
        "seed_pool": REPO / "artifacts/generalized_card/seed_pools/headphone_product_150_seed42.json",
    },
    "laptop": {
        "real_dir": REPO / "data/raw/discussions/laptop_product",
        "seed_pool": REPO / "artifacts/generalized_card/seed_pools/laptop_product_100_seed42.json",
    },
}
DEPTH_BINS = ((0, 1), (1, 2), (2, 4), (4, 7), (7, 999))
MIN_PAIRS_PER_BUCKET = 2


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def mean(values: list[float]) -> float:
    return statistics.mean(values) if values else float("nan")


def load_excluded_threads(real_dir: Path, seed_pool: Path) -> dict[str, list[ThreadComment]]:
    pool = _load_json(seed_pool)
    seed_ids = {str(row["source_raw_post_id"]) for row in pool["seed_posts"]}

    out: dict[str, list[ThreadComment]] = {}
    for product_dir in sorted(p for p in real_dir.iterdir() if p.is_dir()):
        try:
            comments_by_thread, _ = load_real_comments(product_dir)
        except FileNotFoundError:
            continue
        for thread_id, comments in comments_by_thread.items():
            if thread_id in seed_ids or thread_id in out:
                continue
            out[thread_id] = comments
    return out


def analyze_domain(name: str, real_dir: Path, seed_pool: Path, batch_size: int) -> None:
    print(f"\n{'=' * 78}\n{name}\n{'=' * 78}")
    if not real_dir.exists():
        print(f"  no real data dir at {real_dir}, skipping")
        return
    if not seed_pool.exists():
        print(f"  no seed pool at {seed_pool}, skipping")
        return

    comments_by_thread = load_excluded_threads(real_dir, seed_pool)
    print(f"excluded real threads loaded: {len(comments_by_thread)}")
    if not comments_by_thread:
        return

    embedder = CommentEmbedder(model_name=DEFAULT_MODEL, device="cpu", max_length=256)

    thread_ids = list(comments_by_thread)
    all_texts, owner = [], []
    for thread_id in thread_ids:
        for comment in comments_by_thread[thread_id]:
            all_texts.append(comment.text)
            owner.append(thread_id)
    print(f"embedding {len(all_texts)} comments across {len(thread_ids)} threads ...")
    import numpy as np

    vectors = embedder.encode(all_texts, batch_size=batch_size)

    by_thread_vectors: dict[str, list[np.ndarray]] = defaultdict(list)
    by_thread_comments: dict[str, list[ThreadComment]] = defaultdict(list)
    for vec, thread_id, comment in zip(
        vectors, owner, [c for cs in comments_by_thread.values() for c in cs]
    ):
        by_thread_vectors[thread_id].append(vec)
        by_thread_comments[thread_id].append(comment)

    per_thread_rows = []
    depth_bucket_sims: dict[tuple[int, int], list[float]] = defaultdict(list)

    for thread_id in thread_ids:
        comments = by_thread_comments[thread_id]
        vecs = by_thread_vectors[thread_id]
        parent = _parent_map(comments)
        root_idx = [i for i, c in enumerate(comments) if _is_root(c.comment_id, parent)]
        reply_idx = [i for i, c in enumerate(comments) if not _is_root(c.comment_id, parent)]

        def bucket_mean(idx: list[int]) -> tuple[float, int]:
            sims = [float(np.dot(vecs[i], vecs[j])) for a, i in enumerate(idx) for j in idx[a + 1 :]]
            return (mean(sims), len(sims))

        root_mean, root_n = bucket_mean(root_idx)
        reply_mean, reply_n = bucket_mean(reply_idx)
        per_thread_rows.append(
            {
                "thread_id": thread_id,
                "root_root_pairs": root_n,
                "reply_reply_pairs": reply_n,
                "root_root_mean_cos": root_mean,
                "reply_reply_mean_cos": reply_mean,
            }
        )

        n = len(comments)
        for i in range(n):
            for j in range(i + 1, n):
                deepest = max(comments[i].depth, comments[j].depth)
                sim = float(np.dot(vecs[i], vecs[j]))
                for low, high in DEPTH_BINS:
                    if low <= deepest < high:
                        depth_bucket_sims[(low, high)].append(sim)
                        break

    usable = [
        row
        for row in per_thread_rows
        if row["root_root_pairs"] >= MIN_PAIRS_PER_BUCKET and row["reply_reply_pairs"] >= MIN_PAIRS_PER_BUCKET
    ]
    diffs = [row["reply_reply_mean_cos"] - row["root_root_mean_cos"] for row in usable]
    lower = sum(1 for d in diffs if d < 0)
    print(f"\nthreads with >= {MIN_PAIRS_PER_BUCKET} pairs in both buckets: {len(usable)} of {len(per_thread_rows)}")
    print(f"{'mean root_root cosine':32s} {mean([r['root_root_mean_cos'] for r in usable]):.4f}")
    print(f"{'mean reply_reply cosine':32s} {mean([r['reply_reply_mean_cos'] for r in usable]):.4f}")
    print(f"{'mean (reply_reply - root_root)':32s} {mean(diffs):+.4f}")
    if usable:
        print(
            f"{'threads where reply_reply < root_root':32s} "
            f"{lower}/{len(usable)} ({lower / len(usable):.3f})"
        )
    try:
        from scipy import stats as ss

        if len(diffs) >= 2:
            w = ss.wilcoxon(diffs)
            print(f"{'Wilcoxon p (reply_reply != root_root)':32s} {w.pvalue:.6f}")
    except ImportError:
        pass

    print("\ndepth range      n pairs   mean cosine")
    for low, high in DEPTH_BINS:
        sims = depth_bucket_sims[(low, high)]
        label = f"[{low},{high if high < 999 else '+'})"
        print(f"{label:12s} {len(sims):9d}   {mean(sims):.4f}" if sims else f"{label:12s} {0:9d}   n/a")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--domain", choices=sorted(DOMAINS), default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    domains = [args.domain] if args.domain else sorted(DOMAINS)
    for name in domains:
        cfg = DOMAINS[name]
        analyze_domain(name, cfg["real_dir"], cfg["seed_pool"], args.batch_size)


if __name__ == "__main__":
    main()
