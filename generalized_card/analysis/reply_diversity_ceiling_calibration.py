#!/usr/bin/env python3
"""Calibrate the reply-diversity guard's per-depth-bin ceiling, per domain.

`reply_diversity_guard_diagnosis.py` falsified the mechanism on camera (the
only domain with a generated artifact -- D3). This script produces the
actual numbers a runtime guard would ship: for each registered domain,
from real, evaluation-excluded data only, the per-comment
"max cosine similarity to any other comment in the same thread" distribution,
binned by that comment's own depth -- mean, population stdev, p90, and pair
count, so a sparse domain's thin bins are visible rather than silently
averaged over.

No API call, no generated data needed (three of four domains have never had
one -- D3). Reuses the same loaders and embedder as
`cross_domain_reply_diversity.py`/`reply_diversity_guard_diagnosis.py`.

    python3 generalized_card/analysis/reply_diversity_ceiling_calibration.py
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
MIN_BIN_N = 30  # below this, a per-domain bin is too thin to trust on its own


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


def comment_level_bins(
    embedder: CommentEmbedder, comments_by_thread: dict[str, list[ThreadComment]], batch_size: int
) -> dict[tuple[int, int], list[float]]:
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
    flat = [c for tid in thread_ids for c in comments_by_thread[tid]]
    for vec, tid, c in zip(vectors, owner, flat):
        by_thread_vecs[tid].append(vec)
        by_thread_comments[tid].append(c)

    out: dict[tuple[int, int], list[float]] = defaultdict(list)
    for tid in thread_ids:
        comments = by_thread_comments[tid]
        vecs = by_thread_vecs[tid]
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


def main() -> None:
    embedder = CommentEmbedder(model_name=DEFAULT_MODEL, device="cpu", max_length=256)
    print(f"embedder: {embedder.backend_name} / {embedder.model_name}\n")

    profile: dict[str, dict[str, Any]] = {}
    pooled_bins: dict[tuple[int, int], list[float]] = defaultdict(list)

    for name in sorted(DOMAINS):
        cfg = DOMAINS[name]
        real_dir, seed_pool = cfg["real_dir"], cfg["seed_pool"]
        print(f"{'=' * 78}\n{name}\n{'=' * 78}")
        if not real_dir.exists() or not seed_pool.exists():
            print("  missing real dir or seed pool, skipping")
            continue
        comments_by_thread = load_excluded_threads(real_dir, seed_pool)
        print(f"excluded real threads: {len(comments_by_thread)}")
        bins = comment_level_bins(embedder, comments_by_thread, batch_size=64)
        profile[name] = {}
        for low, high in DEPTH_BINS:
            vals = bins.get((low, high), [])
            pooled_bins[(low, high)].extend(vals)
            key = f"{low}-{high if high < 999 else 'plus'}"
            if len(vals) >= MIN_BIN_N:
                profile[name][key] = {
                    "n": len(vals),
                    "mean": round(mean(vals), 4),
                    "stdev": round(stdev(vals), 4),
                    "p90": round(percentile(vals, 90), 4),
                    "source": "domain",
                }
            else:
                profile[name][key] = {"n": len(vals), "source": "pooled_fallback"}
            print(
                f"  [{low},{high if high < 999 else '+'}) n={len(vals):5d} "
                f"mean={mean(vals):.4f} stdev={stdev(vals):.4f} p90={percentile(vals, 90):.4f}"
                + ("" if len(vals) >= MIN_BIN_N else "  <-- thin, pooled fallback")
            )

    print(f"\n{'=' * 78}\npooled (all four domains, for thin-domain fallback)\n{'=' * 78}")
    pooled: dict[str, Any] = {}
    for low, high in DEPTH_BINS:
        vals = pooled_bins[(low, high)]
        key = f"{low}-{high if high < 999 else 'plus'}"
        pooled[key] = {"n": len(vals), "mean": round(mean(vals), 4), "stdev": round(stdev(vals), 4), "p90": round(percentile(vals, 90), 4)}
        print(f"  [{low},{high if high < 999 else '+'}) n={len(vals):6d} mean={mean(vals):.4f} stdev={stdev(vals):.4f} p90={percentile(vals, 90):.4f}")

    for name in profile:
        for key, row in profile[name].items():
            if row.get("source") == "pooled_fallback":
                row.update({k: pooled[key][k] for k in ("mean", "stdev", "p90")})

    out_path = ANALYSIS_DIR / "reply_diversity_ceiling_calibration_results.json"
    out_path.write_text(json.dumps({"per_domain": profile, "pooled": pooled}, indent=2), encoding="utf-8")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
