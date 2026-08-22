#!/usr/bin/env python3
"""Falsify-before-build check for a depth-conditioned reply-diversity guard.

G3/G17 diagnosed `self_bertscore_mean_f1`'s excess as a broad, depth-growing
convergence between replies (not concentrated in any one tree-relation
bucket, not fixed by three targeted surface patches -- G16). The candidate
next mechanism is a real-time, text-level guard: after the Writer drafts a
comment, check its embedding similarity against the *whole thread so far*
(not just its ancestor chain -- G11 already showed an ancestor-only check is
insufficient), and require a rewrite if it exceeds a depth-conditioned
ceiling derived from real data's own measured curve.

Before writing any generation code, this measures -- on already-generated
artifacts, with the exact cheap embedding model (`all-mpnet-base-v2`) the
guard would use at runtime, not BERTScore -- three things:

1. Does generated data show the same depth-growing excess in *this* cheap
   embedding space, independent of the BERTScore finding? (An independent
   replication with a different model would materially strengthen G3/G16.)
2. What does a same-thread max-similarity-to-anything-else check, binned by
   the comment's own depth, actually look like on generated vs real? This is
   a conservative *upper bound* on what a true prior-only runtime check would
   see (checking against every other comment, including ones generated
   later, can only match or exceed what a prior-only pool would show).
3. At what ceiling (per depth bin, derived from real's own distribution) does
   a depth-conditioned guard trip a non-trivial, non-saturating share of
   generated comments -- avoiding both the v105 probe-shape failure (fires on
   ~nothing) and a guard so strict it fires on everything.

No API call. Reuses `CommentEmbedder` (the `semantic_mean_cosine`/
`PlanSemanticIndex` model) and the seed-pool/thread loaders already used by
`cross_domain_reply_diversity.py` and `bertscore_pair_diagnosis.py`.

Superseded before any guard was built on top of it -- see
`docs/DECISIONS.md` G20: the natural next mechanism this diagnosis was
calibrating for would have violated `docs/ORIENTATION.md` §4's
non-negotiable "distribution diagnostics never select a Writer candidate"
rule. Kept because the replication finding (G19) stands on its own.

    python3 generalized_card/analysis/reply_diversity_guard_diagnosis.py
    python3 generalized_card/analysis/reply_diversity_guard_diagnosis.py --run <other run tag>
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
    load_generated_comments,
    load_real_comments,
)

DEFAULT_RUN = (
    REPO
    / "artifacts/generalized_card/runs"
    / "generalized_card_camera_gpt54_v103_stance_opening_n10_20260821_v1"
)
SEED_POOL = REPO / "artifacts/generalized_card/seed_pools/camera_product_150_seed42.json"
REAL_DIR = REPO / "data/raw/discussions/camera_product"
DEPTH_BINS = ((0, 1), (1, 2), (2, 4), (4, 7), (7, 999))


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def mean(values: list[float]) -> float:
    return statistics.mean(values) if values else float("nan")


def stdev(values: list[float]) -> float:
    return statistics.pstdev(values) if len(values) > 1 else 0.0


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    rank = (len(ordered) - 1) * pct / 100.0
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def load_matched_real(seed_pool_path: Path, real_dir: Path) -> dict[str, list[ThreadComment]]:
    """The 10 real threads matched to this run's seeds (not the excluded pool)."""

    pool = _load_json(seed_pool_path)
    seed_ids = {str(row["source_raw_post_id"]) for row in pool["seed_posts"]}
    out: dict[str, list[ThreadComment]] = {}
    for product_dir in sorted(p for p in real_dir.iterdir() if p.is_dir()):
        try:
            comments_by_thread, _ = load_real_comments(product_dir)
        except FileNotFoundError:
            continue
        for thread_id, comments in comments_by_thread.items():
            if thread_id in seed_ids:
                out[thread_id] = comments
    return out


def embed_by_thread(
    embedder: CommentEmbedder, comments_by_thread: dict[str, list[ThreadComment]], batch_size: int
) -> dict[str, tuple[list[ThreadComment], Any]]:
    import numpy as np

    thread_ids = list(comments_by_thread)
    all_texts, owner = [], []
    for tid in thread_ids:
        for c in comments_by_thread[tid]:
            all_texts.append(c.text)
            owner.append(tid)
    if not all_texts:
        return {}
    vectors = embedder.encode(all_texts, batch_size=batch_size)
    by_thread_vecs: dict[str, list[np.ndarray]] = defaultdict(list)
    by_thread_comments: dict[str, list[ThreadComment]] = defaultdict(list)
    flat_comments = [c for tid in thread_ids for c in comments_by_thread[tid]]
    for vec, tid, c in zip(vectors, owner, flat_comments):
        by_thread_vecs[tid].append(vec)
        by_thread_comments[tid].append(c)
    return {tid: (by_thread_comments[tid], by_thread_vecs[tid]) for tid in thread_ids}


def depth_binned_pair_means(embedded: dict[str, Any]) -> dict[tuple[int, int], list[float]]:
    import numpy as np

    out: dict[tuple[int, int], list[float]] = defaultdict(list)
    for _tid, (comments, vecs) in embedded.items():
        n = len(comments)
        for i in range(n):
            for j in range(i + 1, n):
                deepest = max(comments[i].depth, comments[j].depth)
                sim = float(np.dot(vecs[i], vecs[j]))
                for low, high in DEPTH_BINS:
                    if low <= deepest < high:
                        out[(low, high)].append(sim)
                        break
    return out


def comment_level_max_similarity(embedded: dict[str, Any]) -> dict[tuple[int, int], list[float]]:
    """For every comment, its max cosine to *any other* comment in the same
    thread (not just prior ones -- an upper bound, see module docstring),
    binned by that comment's own depth."""

    import numpy as np

    out: dict[tuple[int, int], list[float]] = defaultdict(list)
    for _tid, (comments, vecs) in embedded.items():
        n = len(comments)
        for i in range(n):
            best = 0.0
            for j in range(n):
                if i == j:
                    continue
                sim = float(np.dot(vecs[i], vecs[j]))
                if sim > best:
                    best = sim
            depth = comments[i].depth
            for low, high in DEPTH_BINS:
                if low <= depth < high:
                    out[(low, high)].append(best)
                    break
    return out


def report_pair_depth(label: str, bins: dict[tuple[int, int], list[float]]) -> None:
    print(f"\n-- {label}: pair mean cosine by max(depth) --")
    for low, high in DEPTH_BINS:
        vals = bins.get((low, high), [])
        lbl = f"[{low},{high if high < 999 else '+'})"
        print(f"  {lbl:10s} n={len(vals):6d}  mean={mean(vals):.4f}" if vals else f"  {lbl:10s} n=0")


def report_comment_level(label: str, bins: dict[tuple[int, int], list[float]]) -> None:
    print(f"\n-- {label}: per-comment max-similarity-to-anything-else, by own depth --")
    for low, high in DEPTH_BINS:
        vals = bins.get((low, high), [])
        lbl = f"[{low},{high if high < 999 else '+'})"
        if not vals:
            print(f"  {lbl:10s} n=0")
            continue
        print(
            f"  {lbl:10s} n={len(vals):5d}  mean={mean(vals):.4f}  "
            f"p50={percentile(vals, 50):.4f}  p90={percentile(vals, 90):.4f}"
        )


def trip_rate(
    gen_bins: dict[tuple[int, int], list[float]], ceilings: dict[tuple[int, int], float]
) -> None:
    print("\n-- candidate guard trip rate on generated (ceiling = real mean + 1 real stdev) --")
    total_n = 0
    total_trips = 0
    for low, high in DEPTH_BINS:
        lbl = f"[{low},{high if high < 999 else '+'})"
        vals = gen_bins.get((low, high), [])
        ceiling = ceilings.get((low, high), float("inf"))
        trips = sum(1 for v in vals if v > ceiling)
        total_n += len(vals)
        total_trips += trips
        if vals:
            print(
                f"  {lbl:10s} ceiling={ceiling:.4f}  trips={trips}/{len(vals)} "
                f"({trips / len(vals):.3f})"
            )
        else:
            print(f"  {lbl:10s} n=0")
    if total_n:
        print(f"  {'TOTAL':10s} trips={total_trips}/{total_n} ({total_trips / total_n:.3f})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", default=str(DEFAULT_RUN))
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    run_path = Path(args.run)
    gen_comments: dict[str, list[ThreadComment]] = {}
    for sim_dir in sorted(run_path.glob("cleaned/run_*_sampled_reddit")):
        by_thread, _ = load_generated_comments(sim_dir)
        gen_comments.update(by_thread)
    if not gen_comments:
        for sim_dir in sorted(run_path.glob("generated/run_*_sampled_reddit")):
            by_thread, _ = load_generated_comments(sim_dir)
            gen_comments.update(by_thread)
    print(f"generated threads loaded: {len(gen_comments)} ({sum(len(v) for v in gen_comments.values())} comments)")

    real_comments = load_matched_real(SEED_POOL, REAL_DIR)
    print(f"matched real threads loaded: {len(real_comments)} ({sum(len(v) for v in real_comments.values())} comments)")

    embedder = CommentEmbedder(model_name=DEFAULT_MODEL, device="cpu", max_length=256)
    print(f"embedder: {embedder.backend_name} / {embedder.model_name}")

    gen_embedded = embed_by_thread(embedder, gen_comments, args.batch_size)
    real_embedded = embed_by_thread(embedder, real_comments, args.batch_size)

    print("\n" + "=" * 78)
    print("1. Independent replication in embedding-cosine space (not BERTScore)")
    print("=" * 78)
    gen_pair_bins = depth_binned_pair_means(gen_embedded)
    real_pair_bins = depth_binned_pair_means(real_embedded)
    report_pair_depth("generated", gen_pair_bins)
    report_pair_depth("real (matched)", real_pair_bins)
    print("\n-- excess (generated - real), embedding-cosine space --")
    for low, high in DEPTH_BINS:
        lbl = f"[{low},{high if high < 999 else '+'})"
        g = gen_pair_bins.get((low, high), [])
        r = real_pair_bins.get((low, high), [])
        if g and r:
            print(f"  {lbl:10s} {mean(g) - mean(r):+.4f}")

    print("\n" + "=" * 78)
    print("2. Per-comment max-similarity-to-anything-else, by own depth")
    print("   (upper bound on a true prior-only runtime pool -- see docstring)")
    print("=" * 78)
    gen_comment_bins = comment_level_max_similarity(gen_embedded)
    real_comment_bins = comment_level_max_similarity(real_embedded)
    report_comment_level("generated", gen_comment_bins)
    report_comment_level("real (matched)", real_comment_bins)

    print("\n" + "=" * 78)
    print("3. Candidate guard: ceiling = real mean + 1 real stdev per depth bin")
    print("=" * 78)
    ceilings = {}
    for low, high in DEPTH_BINS:
        vals = real_comment_bins.get((low, high), [])
        ceilings[(low, high)] = mean(vals) + stdev(vals) if vals else float("inf")
    trip_rate(gen_comment_bins, ceilings)


if __name__ == "__main__":
    main()
