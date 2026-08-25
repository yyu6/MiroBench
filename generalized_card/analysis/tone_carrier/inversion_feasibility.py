"""The Lucas critique against my own result: is C invariant to the assignment vector?

assignment_inversion.py solved C^T a = target and removed 89% of the four-way tone
gap, with C reproducing across v110 and the gate. That solution needs polite on
55.4% of slots against today's 27.4%. Two ways it can be wrong, and neither is
visible in the inversion itself:

  1. C is measured AT today's mix. The Planner assigns polite where polite fits.
     The 28pp of new polite slots are the ones it declined, so their realization
     rate is not the 0.400 the matrix assumes. This is the Lucas critique: the
     transfer matrix is not a structural parameter unless it is stable across the
     slot types the new assignment would reach.
  2. `prompts.py:951` is a hard Planner rule -- "Never pair polite with
     correction_caveat or a disagreeing stance". If fewer than 55.4% of slots
     carry an agreeing stance, the solution is not merely optimistic, it is
     infeasible, and the constrained optimum has to be recomputed on the subset
     that can legally take the label.

Stratifies C by the Planner fields that would gate the new assignment, and
recomputes the best legal assignment under the s951 constraint.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

REPO = Path("/Users/yaoningyu/Desktop/UIUC/GEO")
sys.path.insert(0, str(REPO / "scripts" / "evaluation"))
from score_thread_semantic_uniformity import load_real_comments, load_generated_comments  # noqa: E402
from score_thread_politeness import PolitenessScorer  # noqa: E402

RUNS = {"v110": "v110_length_transfer_n10_20260824_v1", "gate": "v113_v112_gate_n10_20260826_v1"}
SEEDS = range(2, 12)
LABELS = ("polite", "somewhat_polite", "neutral", "impolite")
FIELDS = ("stance", "comment_function", "payload_type", "evidence_mode", "speaker_role")
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
    meta = {}
    for d in sorted((root / "cleaned").glob("run_*_sampled_reddit")):
        for post in json.load(open(d / "discussion.json"))["posts"]:
            for rec in post.get("generation_records") or []:
                cid = str((rec.get("comment") or {}).get("comment_id", ""))
                t = rec.get("task") or {}
                if cid:
                    meta[cid] = {"tone": str(t.get("tone_target") or ""),
                                 **{f: str(t.get(f) or "") for f in FIELDS}}
    rows = []
    for d in sorted((root / "cleaned").glob("run_*_sampled_reddit")):
        cbt, _ = load_generated_comments(d)
        for cs in cbt.values():
            for c in cs:
                m = meta.get(str(c.comment_id))
                if m:
                    rows.append((c.text, m))
    sc = score([t for t, _ in rows])
    return [(m, r["pred_label"]) for (_, m), r in zip(rows, sc)]


real, cache = [], {}
for s in SEEDS:
    p = by_seed[s]
    d = REPO / "data/raw/discussions/camera_product" / p["source_product_dir"]
    if d not in cache:
        cache[d] = load_real_comments(d)[0]
    real.extend(c.text for c in (cache[d].get(p["source_raw_post_id"]) or []))
rr = score(real)
target = np.array([sum(1 for r in rr if r["pred_label"] == lb) / len(rr) for lb in LABELS])

pairs = load_run(RUNS["gate"]) + load_run(RUNS["v110"])
print(f"pooled slots across both runs: {len(pairs)}\n")

print("== 1. is P(realize polite | assign polite) stable across slot types? ==")
pol = [(m, r) for m, r in pairs if m["tone"] == "polite"]
print(f"   base: n={len(pol)}  realized polite {sum(1 for _, r in pol if r=='polite')/len(pol):.3f}")
for f in FIELDS:
    vals = sorted({m[f] for m, _ in pol})
    print(f"   -- {f}")
    for v in vals:
        g = [r for m, r in pol if m[f] == v]
        if len(g) < 12:
            continue
        print(f"      {v:<28} n={len(g):>4}  realized polite {sum(1 for r in g if r=='polite')/len(g):>6.3f}")

print("\n== 2. what fraction of slots can legally take `polite` under prompts.py:951? ==")
stances = {}
for m, _ in pairs:
    stances[m["stance"]] = stances.get(m["stance"], 0) + 1
print("   stance distribution over all slots:")
for k, v in sorted(stances.items(), key=lambda kv: -kv[1]):
    print(f"      {k:<28} {v:>4}  {v/len(pairs):.4f}")
DISAGREE = {"disagree", "hard_disagree", "partial_disagree", "challenge", "correct"}
ok = sum(1 for m, _ in pairs if m["stance"] not in DISAGREE and m["comment_function"] != "correction_caveat")
print(f"\n   slots with a non-disagreeing stance and function != correction_caveat: "
      f"{ok}/{len(pairs)} = {ok/len(pairs):.4f}")
print(f"   the inversion asks for polite on 0.5543 of slots -> "
      f"{'FEASIBLE' if ok/len(pairs) >= 0.5543 else 'INFEASIBLE at that share'}")

print("\n== 3. re-solve with polite capped at the legally assignable share ==")
C = np.zeros((4, 4))
for i, lb in enumerate(LABELS):
    g = [r for m, r in pairs if m["tone"] == lb]
    for j, lb2 in enumerate(LABELS):
        C[i, j] = sum(1 for r in g if r == lb2) / len(g) if g else 0.0
a_now = np.array([sum(1 for m, _ in pairs if m["tone"] == lb) / len(pairs) for lb in LABELS])
r_now = np.array([sum(1 for _, r in pairs if r == lb) / len(pairs) for lb in LABELS])


def best(cap):
    a = a_now.copy()
    for _ in range(80000):
        a = a - 0.30 * (2 * C @ (C.T @ a - target))
        a = np.clip(a, 0.0, None)
        a[0] = min(a[0], cap * max(a.sum(), 1e-9))
        a = a / a.sum() if a.sum() > 0 else a_now.copy()
    return a


print(f"{'polite cap':<14}" + "".join(f"{lb[:9]:>11}" for lb in LABELS)
      + f"{'|  realized':<12}" + "".join(f"{lb[:9]:>11}" for lb in LABELS) + f"{'L2':>9}{'closed':>9}")
l2_now = float(np.linalg.norm(C.T @ a_now - target))
for cap in (0.30, 0.40, 0.50, 0.5543, 0.70, 1.0):
    a = best(cap)
    r = C.T @ a
    l2 = float(np.linalg.norm(r - target))
    print(f"{cap:<14.4f}" + "".join(f"{v:>11.4f}" for v in a) + f"{'|':<12}"
          + "".join(f"{v:>11.4f}" for v in r) + f"{l2:>9.4f}{100*(1-l2/l2_now):>8.0f}%")
print(f"\n{'today':<14}" + "".join(f"{v:>11.4f}" for v in a_now) + f"{'|':<12}"
      + "".join(f"{v:>11.4f}" for v in r_now) + f"{l2_now:>9.4f}")
print(f"{'real target':<14}" + "".join(f"{'':>11}" for _ in LABELS) + f"{'|':<12}"
      + "".join(f"{v:>11.4f}" for v in target))
