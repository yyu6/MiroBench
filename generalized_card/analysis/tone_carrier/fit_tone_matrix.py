"""Refit `tone_realization.REALIZATION_MATRIX` from named run artifacts (E8).

Every number in the shipped module must be reproducible from a committed script.
This is that script. It prints the matrix in the exact literal form the module
carries, so a refit is a copy-paste, and it re-solves the cap sweep so the
claimed closure figures are checkable at the same time.

Usage:
    python3 generalized_card/analysis/tone_carrier/fit_tone_matrix.py [run_tag ...]

Default runs are the two the shipped matrix was fitted on. Pass calibration-run
tags instead to refit off the evaluation seeds -- the recommended move before the
paper run, since the shipped matrix was measured on runs over seeds 2-11 and
those ten are therefore in-sample for the calibration at N=150.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

REPO = Path("/Users/yaoningyu/Desktop/UIUC/GEO")
sys.path.insert(0, str(REPO / "scripts" / "evaluation"))
sys.path.insert(0, str(REPO / "generalized_card"))
from score_thread_semantic_uniformity import load_real_comments, load_generated_comments  # noqa: E402
from score_thread_politeness import PolitenessScorer  # noqa: E402
from generalized_card.tone_realization import (  # noqa: E402
    POLITE_ASSIGNMENT_CAP, TONE_ORDER, REALIZATION_MATRIX,
    invert_tone_rates, set_tone_quota_mode,
)

DEFAULT_RUNS = ("v110_length_transfer_n10_20260824_v1", "v113_v112_gate_n10_20260826_v1")
TARGET_SEEDS = range(2, 12)
runs = tuple(sys.argv[1:]) or DEFAULT_RUNS
pool = json.load(open(REPO / "artifacts/generalized_card/seed_pools/camera_product_150_seed42.json"))["seed_posts"]
by_seed = {int(x["seed_index"]): x for x in pool}
scorer = PolitenessScorer("Intel/polite-guard", "auto", 256)


class T:
    def __init__(self, t):
        self.text = t
        self.thread_id = self.thread_title = self.comment_id = self.parent_id = self.author = ""
        self.depth = 0


def score(ts):
    return scorer.score_comments([T(t) for t in ts], batch_size=64, include_text=False)


rows = []
for tag in runs:
    root = REPO / "artifacts/generalized_card/runs" / tag
    meta = {}
    for d in sorted((root / "cleaned").glob("run_*_sampled_reddit")):
        for post in json.load(open(d / "discussion.json"))["posts"]:
            for rec in post.get("generation_records") or []:
                cid = str((rec.get("comment") or {}).get("comment_id", ""))
                if cid:
                    meta[cid] = str((rec.get("task") or {}).get("tone_target") or "")
    batch = []
    for d in sorted((root / "cleaned").glob("run_*_sampled_reddit")):
        cbt, _ = load_generated_comments(d)
        for cs in cbt.values():
            batch += [(c.text, meta[str(c.comment_id)]) for c in cs if str(c.comment_id) in meta]
    sc = score([t for t, _ in batch])
    rows += [(a, r["pred_label"]) for (_, a), r in zip(batch, sc)]
    print(f"{tag}: {len(batch)} slots")

C = np.zeros((4, 4))
counts = {}
for i, lb in enumerate(TONE_ORDER):
    g = [r for a, r in rows if a == lb]
    counts[lb] = len(g)
    for j, l2 in enumerate(TONE_ORDER):
        C[i, j] = sum(1 for r in g if r == l2) / len(g) if g else 0.0

print(f"\n# paste into tone_realization.REALIZATION_MATRIX  (n={len(rows)})")
print("REALIZATION_MATRIX: tuple[tuple[float, ...], ...] = (")
for i, lb in enumerate(TONE_ORDER):
    print("    (" + ", ".join(f"{C[i,j]:.4f}" for j in range(4))
          + f"),   # assigned {lb}, n={counts[lb]}")
print(")")

shipped = np.array(REALIZATION_MATRIX)
print(f"\nmax |refit - shipped| = {np.abs(C - shipped).max():.4f}")

real, cache = [], {}
for s in TARGET_SEEDS:
    p = by_seed[s]
    d = REPO / "data/raw/discussions/camera_product" / p["source_product_dir"]
    if d not in cache:
        cache[d] = load_real_comments(d)[0]
    real += [c.text for c in (cache[d].get(p["source_raw_post_id"]) or [])]
rr = score(real)
target = {lb: sum(1 for r in rr if r["pred_label"] == lb) / len(rr) for lb in TONE_ORDER}
a_now = np.array([sum(1 for a, _ in rows if a == lb) / len(rows) for lb in TONE_ORDER])
r_now = np.array([sum(1 for _, r in rows if r == lb) / len(rows) for lb in TONE_ORDER])
tvec = np.array([target[lb] for lb in TONE_ORDER])
l2_now = float(np.linalg.norm(r_now - tvec))

print(f"\n{'cap':<10}" + "".join(f"{lb[:9]:>10}" for lb in TONE_ORDER) + "  |"
      + "".join(f"{lb[:9]:>10}" for lb in TONE_ORDER) + f"{'L2':>9}{'closed':>9}")
print(f"{'today':<10}" + "".join(f"{v:>10.4f}" for v in a_now) + "  |"
      + "".join(f"{v:>10.4f}" for v in r_now) + f"{l2_now:>9.4f}{0:>8.0f}%")
set_tone_quota_mode("inverted")
import generalized_card.tone_realization as tr  # noqa: E402
for cap in (0.30, POLITE_ASSIGNMENT_CAP, 0.40, 0.50, 1.00):
    tr._CACHE.clear()
    a = invert_tone_rates(target, cap=cap)
    av = np.array([a[lb] for lb in TONE_ORDER])
    r = C.T @ av
    l2 = float(np.linalg.norm(r - tvec))
    mark = "  <- shipped" if abs(cap - POLITE_ASSIGNMENT_CAP) < 1e-9 else ""
    print(f"{cap:<10.4f}" + "".join(f"{v:>10.4f}" for v in av) + "  |"
          + "".join(f"{v:>10.4f}" for v in r) + f"{l2:>9.4f}{100*(1-l2/l2_now):>8.0f}%{mark}")
print(f"{'target':<10}" + "".join(f"{'':>10}" for _ in TONE_ORDER) + "  |"
      + "".join(f"{tvec[i]:>10.4f}" for i in range(4)))
