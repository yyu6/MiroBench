"""Fix a realization failure from the ASSIGNMENT side, by inverting the confusion matrix.

Everything tried so far attacks realization: more register cues (dead), the omitted
conjunction (dead), hedging (dead), length repair (dead -- occupancy explains 0% of
the polite gap), the bare-assertion frame (9% of the r gap), and the polite lexicon
(dead: generated already carries real's polite-discriminative vocabulary at 1.14x).

But the Writer's failure is CONSISTENT, not random. It realizes `impolite` at 0.87
and `polite` at 0.40, and it does so stably enough to be a measured transfer matrix.
If C[i][j] = P(realize j | assign i) is known and invertible, then the assignment
vector that lands the realized mix on the target is a = solve(C^T, target). Nothing
about the Writer changes; the Planner just asks for the mix that comes out right.

This is calibration on the reference corpus, not tuning: C is a property of the
generator measured on its own output, and the target is real's label mix. It is the
same inverse-calibration `length_calibration` already does for word counts. What it
must not become is fitting to a test p-value (ORIENTATION s4), so the feasibility
question -- does a valid probability vector exist at all -- is settled here, and any
shipped C would be refit on evaluation-excluded threads.

Reports the gate's C, the exact solution, and whether it is a legal distribution.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

REPO = Path("/Users/yaoningyu/Desktop/UIUC/GEO")
sys.path.insert(0, str(REPO / "scripts" / "evaluation"))
from score_thread_semantic_uniformity import load_real_comments, load_generated_comments  # noqa: E402
from score_thread_politeness import PolitenessScorer  # noqa: E402

RUNS = {"v110 (arms off)": "v110_length_transfer_n10_20260824_v1",
        "gate (v112+v113)": "v113_v112_gate_n10_20260826_v1"}
SEEDS = range(2, 12)
LABELS = ("polite", "somewhat_polite", "neutral", "impolite")
pool = json.load(open(REPO / "artifacts/generalized_card/seed_pools/camera_product_150_seed42.json"))["seed_posts"]
by_seed = {int(x["seed_index"]): x for x in pool}
scorer = PolitenessScorer("Intel/polite-guard", "auto", 256)


class T:
    def __init__(self, t):
        self.text = t
        self.thread_id = self.thread_title = self.comment_id = self.parent_id = self.author = ""
        self.depth = 0


def score(texts):
    return scorer.score_comments([T(t) for t in texts], batch_size=64, include_text=False)


def load_run(tag):
    root = REPO / "artifacts/generalized_card/runs" / tag
    assigned = {}
    for d in sorted((root / "cleaned").glob("run_*_sampled_reddit")):
        for post in json.load(open(d / "discussion.json"))["posts"]:
            for rec in post.get("generation_records") or []:
                cid = str((rec.get("comment") or {}).get("comment_id", ""))
                if cid:
                    assigned[cid] = str((rec.get("task") or {}).get("tone_target") or "")
    out = []
    for d in sorted((root / "cleaned").glob("run_*_sampled_reddit")):
        cbt, _ = load_generated_comments(d)
        for cs in cbt.values():
            out.extend((c.text, assigned.get(str(c.comment_id), "")) for c in cs)
    return out


real, cache = [], {}
for s in SEEDS:
    p = by_seed[s]
    d = REPO / "data/raw/discussions/camera_product" / p["source_product_dir"]
    if d not in cache:
        cache[d] = load_real_comments(d)[0]
    real.extend(c.text for c in (cache[d].get(p["source_raw_post_id"]) or []))
rr = score(real)
target = np.array([sum(1 for r in rr if r["pred_label"] == lb) / len(rr) for lb in LABELS])
print("real comment-level label mix (the target):")
print("   " + "  ".join(f"{lb}={v:.4f}" for lb, v in zip(LABELS, target)))

for name, tag in RUNS.items():
    rows = load_run(tag)
    sc = score([t for t, _ in rows])
    pairs = [(a, r["pred_label"]) for (_, a), r in zip(rows, sc) if a in LABELS]
    print(f"\n===== {name}  ({len(pairs)} slots with a tone assignment of {len(rows)}) =====")
    a_now = np.array([sum(1 for a, _ in pairs if a == lb) / len(pairs) for lb in LABELS])
    r_now = np.array([sum(1 for _, r in pairs if r == lb) / len(pairs) for lb in LABELS])
    C = np.zeros((4, 4))
    for i, lb in enumerate(LABELS):
        grp = [r for a, r in pairs if a == lb]
        for j, lb2 in enumerate(LABELS):
            C[i, j] = sum(1 for r in grp if r == lb2) / len(grp) if grp else 0.0
    print(f"{'assigned vs realized':<22}" + "".join(f"{lb:>17}" for lb in LABELS) + f"{'n':>6}")
    for i, lb in enumerate(LABELS):
        n = sum(1 for a, _ in pairs if a == lb)
        print(f"{lb:<22}" + "".join(f"{C[i,j]:>17.3f}" for j in range(4)) + f"{n:>6}")
    print(f"\n{'assignment today':<22}" + "".join(f"{v:>17.4f}" for v in a_now))
    print(f"{'realized today':<22}" + "".join(f"{v:>17.4f}" for v in r_now))
    print(f"{'real (target)':<22}" + "".join(f"{v:>17.4f}" for v in target))

    det = np.linalg.det(C.T)
    print(f"\ndet(C^T) = {det:.6f}   cond = {np.linalg.cond(C.T):.1f}")
    try:
        a_star = np.linalg.solve(C.T, target)
    except np.linalg.LinAlgError:
        print("singular -- no exact inverse"); continue
    print(f"{'exact solution a*':<22}" + "".join(f"{v:>17.4f}" for v in a_star))
    legal = bool((a_star >= -1e-9).all() and abs(a_star.sum() - 1) < 1e-6)
    print(f"legal probability vector? {'YES' if legal else 'NO'}   "
          f"(min {a_star.min():+.4f}, sum {a_star.sum():.4f})")

    # constrained best: nearest legal assignment by projected gradient on the simplex
    def loss(a):
        return float(((C.T @ a - target) ** 2).sum())
    a = a_now.copy()
    for _ in range(60000):
        g = 2 * C @ (C.T @ a - target)
        a = a - 0.30 * g
        a = np.clip(a, 0.0, None)
        a = a / a.sum() if a.sum() > 0 else a_now.copy()
    print(f"{'best legal a':<22}" + "".join(f"{v:>17.4f}" for v in a))
    print(f"{'-> realized':<22}" + "".join(f"{v:>17.4f}" for v in (C.T @ a)))
    print(f"{'residual vs target':<22}" + "".join(f"{v:>+17.4f}" for v in (C.T @ a - target)))
    print(f"   L2 today {np.sqrt(loss(a_now)):.4f}  ->  best legal {np.sqrt(loss(a)):.4f}"
          f"   ({100*(1-np.sqrt(loss(a))/np.sqrt(loss(a_now))):.0f}% of the gap removed)")
