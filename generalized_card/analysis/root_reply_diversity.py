#!/usr/bin/env python3
"""Is "replies are more diverse than root comments" a property of real Reddit
writing, or an artifact of the ten v103-matched threads?

`bertscore_pair_diagnosis.py` found a sign inversion on the v103 N=10 artifact:
real `reply_reply` pairs are *less* similar than real `root_root` pairs (0.4905
vs 0.4955), while generated `reply_reply` pairs are *more* similar than
generated `root_root` pairs (0.5136 vs 0.5089) -- see `docs/DECISIONS.md` G3.
Ten threads is not enough to know whether the real-side direction is a general
property of Reddit writing or noise from which ten threads got matched.

This checks it on the full evaluation-excluded camera corpus (seed pool
excluded, deduplicated by thread_id -- one Reddit post can sit under two
product folders, `tasks/lessons.md` E7) using the cheap sentence-embedding
scorer (`all-mpnet-base-v2`, the same model `semantic_mean_cosine` uses) rather
than the expensive BERTScore model -- this is a test of the *direction* of a
text property, not a reproduction of `self_bertscore_mean_f1` itself, so the
cheap proxy is the right tool: seconds, not tens of minutes, over ~400 threads.

No API call, no seed-pool leakage, nothing fitted.

    python3 generalized_card/analysis/root_reply_diversity.py
"""

from __future__ import annotations

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

SEED_POOL = REPO / "artifacts/generalized_card/seed_pools/camera_product_150_seed42.json"
REAL_DIR = REPO / "data/raw/discussions/camera_product"
MIN_PAIRS_PER_BUCKET = 2  # need at least 2 pairs (3 comments) for a stable mean


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def mean(values: list[float]) -> float:
    return statistics.mean(values) if values else float("nan")


def load_excluded_threads() -> dict[str, list[ThreadComment]]:
    """Every real camera thread not in the 150-thread evaluation seed pool,
    deduplicated by thread_id (one post can sit under two product folders)."""

    pool = _load_json(SEED_POOL)
    seed_ids = {str(row["source_raw_post_id"]) for row in pool["seed_posts"]}

    out: dict[str, list[ThreadComment]] = {}
    for product_dir in sorted(p for p in REAL_DIR.iterdir() if p.is_dir()):
        try:
            comments_by_thread, _ = load_real_comments(product_dir)
        except FileNotFoundError:
            continue  # product folders with only a scrape log, no data
        for thread_id, comments in comments_by_thread.items():
            if thread_id in seed_ids or thread_id in out:
                continue
            out[thread_id] = comments
    return out


def within_thread_cosine(
    embedder: CommentEmbedder, comments_by_thread: dict[str, list[ThreadComment]], batch_size: int
) -> list[dict[str, Any]]:
    """Per-thread root_root / reply_reply mean cosine, root and reply comments
    embedded together (one pass) so the embedding space is identical for both."""

    import numpy as np

    thread_ids = list(comments_by_thread)
    all_texts, owner = [], []
    for thread_id in thread_ids:
        for comment in comments_by_thread[thread_id]:
            all_texts.append(comment.text)
            owner.append(thread_id)
    if not all_texts:
        return []
    print(f"embedding {len(all_texts)} comments across {len(thread_ids)} threads ...")
    vectors = embedder.encode(all_texts, batch_size=batch_size)

    by_thread_vectors: dict[str, list[np.ndarray]] = defaultdict(list)
    by_thread_comments: dict[str, list[ThreadComment]] = defaultdict(list)
    for vec, thread_id, comment in zip(vectors, owner, [c for cs in comments_by_thread.values() for c in cs]):
        by_thread_vectors[thread_id].append(vec)
        by_thread_comments[thread_id].append(comment)

    rows = []
    for thread_id in thread_ids:
        comments = by_thread_comments[thread_id]
        vecs = by_thread_vectors[thread_id]
        parent = _parent_map(comments)
        root_idx = [i for i, c in enumerate(comments) if _is_root(c.comment_id, parent)]
        reply_idx = [i for i, c in enumerate(comments) if not _is_root(c.comment_id, parent)]

        def bucket_mean(idx: list[int]) -> tuple[float, int]:
            sims = [
                float(np.dot(vecs[i], vecs[j]))
                for a, i in enumerate(idx)
                for j in idx[a + 1 :]
            ]
            return (mean(sims), len(sims))

        root_mean, root_n = bucket_mean(root_idx)
        reply_mean, reply_n = bucket_mean(reply_idx)
        rows.append(
            {
                "thread_id": thread_id,
                "comment_count": len(comments),
                "root_n_comments": len(root_idx),
                "reply_n_comments": len(reply_idx),
                "root_root_pairs": root_n,
                "reply_reply_pairs": reply_n,
                "root_root_mean_cos": root_mean,
                "reply_reply_mean_cos": reply_mean,
            }
        )
    return rows


def main() -> None:
    comments_by_thread = load_excluded_threads()
    print(f"excluded real threads loaded: {len(comments_by_thread)}")

    embedder = CommentEmbedder(model_name=DEFAULT_MODEL, device="cpu", max_length=256)
    print(f"embedder backend: {embedder.backend_name}, model: {embedder.model_name}")

    rows = within_thread_cosine(embedder, comments_by_thread, batch_size=64)
    usable = [
        row
        for row in rows
        if row["root_root_pairs"] >= MIN_PAIRS_PER_BUCKET and row["reply_reply_pairs"] >= MIN_PAIRS_PER_BUCKET
    ]
    print(
        f"\nthreads with >= {MIN_PAIRS_PER_BUCKET} pairs in both buckets: "
        f"{len(usable)} of {len(rows)}\n"
    )

    diffs = [row["reply_reply_mean_cos"] - row["root_root_mean_cos"] for row in usable]
    lower = sum(1 for d in diffs if d < 0)
    print(f"{'metric':32s} value")
    print(f"{'mean root_root cosine':32s} {mean([r['root_root_mean_cos'] for r in usable]):.4f}")
    print(f"{'mean reply_reply cosine':32s} {mean([r['reply_reply_mean_cos'] for r in usable]):.4f}")
    print(f"{'mean (reply_reply - root_root)':32s} {mean(diffs):+.4f}")
    print(f"{'threads where reply_reply < root_root':32s} {lower}/{len(usable)} ({lower/len(usable):.3f})")

    try:
        from scipy import stats as ss

        w = ss.wilcoxon(diffs)
        print(f"{'Wilcoxon p (reply_reply != root_root)':32s} {w.pvalue:.6f}")
    except ImportError:
        pass

    print("\n== distribution of (reply_reply - root_root) per thread ==\n")
    for pct in (5, 25, 50, 75, 95):
        sorted_diffs = sorted(diffs)
        rank = min(len(sorted_diffs) - 1, int(round(pct / 100 * (len(sorted_diffs) - 1))))
        print(f"  p{pct:<3d} {sorted_diffs[rank]:+.4f}")

    out_path = ANALYSIS_DIR / "root_reply_diversity_results.json"
    out_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nwrote {out_path} ({len(rows)} threads, all of them, not just `usable`)")


if __name__ == "__main__":
    main()
