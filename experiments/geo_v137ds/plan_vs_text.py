#!/usr/bin/env python3
"""Is the thread too tightly related because the Planner planned it that way,
or because the Writer wrote it that way?

  python3 experiments/geo_v137ds/plan_vs_text.py real_20260902

`semantic_mean_cosine` did not move even with the matched real thread in front
of the Planner, so the constraint is downstream of the Planner's input. Two
candidates remain and they call for opposite fixes: the Planner may be unable to
plan scatter (the twelve frozen decision lenses, the anti-repetition budgets),
or the Writer may be collapsing scatter the Planner did plan.

The decisive quantity is per PAIR of slots, not per thread. For every pair in a
thread we measure how far apart the two PLANS are and how far apart the two
realized TEXTS are, then read the realized distance within each plan-distance
quartile:

  * plans far apart but texts still close  -> the Writer is flattening
  * no pairs whose plans are far apart     -> the Planner cannot plan scatter

Plan and text live on different distributions, so absolute cosines are not
comparable between the two columns. What is comparable is the text column
against the real corpus's own within-thread cosine, which is printed with it.
"""
import argparse, collections, glob, json
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
MODEL = "sentence-transformers/all-mpnet-base-v2"
PLAN_FIELDS = ("semantic_move", "local_topic", "detail_focus", "local_anchor")

ap = argparse.ArgumentParser()
ap.add_argument("prefix")
ap.add_argument("--min-slots", type=int, default=6)
a = ap.parse_args()

threads = []
for f in sorted(glob.glob(str(REPO / "artifacts/generalized_card/runs/*/generated/run_00_sampled_reddit/discussion.json"))):
    if not f.split("/")[-4].startswith(a.prefix):
        continue
    for post in json.load(open(f)).get("posts") or []:
        plans, texts = [], []
        for rec in post.get("generation_records") or []:
            task = rec.get("task") or {}
            comment = rec.get("comment")
            if not isinstance(comment, dict):
                continue
            body = str(comment.get("content") or "").strip()
            plan = " ".join(str(task.get(k) or "") for k in PLAN_FIELDS).strip()
            if body and plan:
                plans.append(plan)
                texts.append(body)
        if len(plans) >= a.min_slots:
            threads.append((plans, texts))

if not threads:
    raise SystemExit(f"{a.prefix}: 没有足够大的 thread")

from sentence_transformers import SentenceTransformer
m = SentenceTransformer(MODEL, device="cpu")

flat_p = [t for th in threads for t in th[0]]
flat_t = [t for th in threads for t in th[1]]
ep = m.encode(flat_p, normalize_embeddings=True, batch_size=128, show_progress_bar=False, convert_to_numpy=True)
et = m.encode(flat_t, normalize_embeddings=True, batch_size=128, show_progress_bar=False, convert_to_numpy=True)

pc, tc, off = [], [], 0
for plans, _ in threads:
    k = len(plans)
    P, T = ep[off:off + k], et[off:off + k]
    off += k
    sp, st = P @ P.T, T @ T.T
    iu = np.triu_indices(k, 1)
    pc.extend(sp[iu].tolist())
    tc.extend(st[iu].tolist())

pc, tc = np.array(pc), np.array(tc)
print(f"{a.prefix}: {len(threads)} 个 thread（>= {a.min_slots} 槽），{len(pc)} 个槽位对\n")
qs = np.quantile(pc, [0, 0.25, 0.5, 0.75, 1.0])
print(f"  {'plan 距离分档':<22}{'对数':>7}{'plan 余弦':>11}{'实际文本余弦':>14}")
labels = ["最远 25%（plan 最散）", "次远 25%", "次近 25%", "最近 25%（plan 最像）"]
for i, lab in enumerate(labels):
    lo, hi = qs[i], qs[i + 1]
    sel = (pc >= lo) & (pc <= hi) if i == 3 else (pc >= lo) & (pc < hi)
    if not sel.any():
        continue
    print(f"  {lab:<22}{sel.sum():>7}{pc[sel].mean():>11.4f}{tc[sel].mean():>14.4f}")
from scipy.stats import spearmanr
rho, p = spearmanr(pc, tc)
print(f"\n  plan 距离 与 文本距离 的相关: rho={rho:+.3f}  p={p:.2e}")
print(f"  全部槽位对的文本余弦均值: {tc.mean():.4f}")
print("  （对照：真实 celebrity thread 内余弦 0.1797，我们整体 0.2332）")
