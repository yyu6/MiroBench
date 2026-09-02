#!/usr/bin/env python3
"""Do the twelve decision lenses actually buy semantic separation?

  python3 experiments/geo_v137ds/perspective_spread.py real_20260902

G186 put the defect in the plan distribution rather than the Writer. Two very
different readings of that remain, and they call for opposite fixes:

  * the slot contract is too tight -- the Planner is boxed in by budgets and
    caps and cannot use the freedom it has; loosen the constraints
  * the menu is too narrow -- every one of the twelve lenses frames a comment
    as one evaluative angle on the post's subject, so even a free and
    well-spread choice among them lands in the same semantic neighbourhood;
    replace the menu

The test distinguishes them. If pairs of slots assigned DIFFERENT perspectives
are barely further apart than pairs sharing ONE perspective, then spreading
usage across the twelve buys almost nothing, and the constraint is not what is
binding -- the menu is. `max_perspective_share 0.34` already forces that spread,
so a narrow menu would mean the pipeline is paying a real cost for it and
getting nothing back.
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
        plans, texts, pids = [], [], []
        for rec in post.get("generation_records") or []:
            task = rec.get("task") or {}
            comment = rec.get("comment")
            if not isinstance(comment, dict):
                continue
            body = str(comment.get("content") or "").strip()
            plan = " ".join(str(task.get(k) or "") for k in PLAN_FIELDS).strip()
            pid = str(task.get("perspective_id") or "")
            if body and plan and pid:
                plans.append(plan); texts.append(body); pids.append(pid)
        if len(plans) >= a.min_slots:
            threads.append((plans, texts, pids))

if not threads:
    raise SystemExit(f"{a.prefix}: 没有足够大的 thread")

from sentence_transformers import SentenceTransformer
m = SentenceTransformer(MODEL, device="cpu")
flat_p = [t for th in threads for t in th[0]]
flat_t = [t for th in threads for t in th[1]]
ep = m.encode(flat_p, normalize_embeddings=True, batch_size=128, show_progress_bar=False, convert_to_numpy=True)
et = m.encode(flat_t, normalize_embeddings=True, batch_size=128, show_progress_bar=False, convert_to_numpy=True)

same_p, diff_p, same_t, diff_t = [], [], [], []
use = collections.Counter()
off = 0
for plans, texts, pids in threads:
    k = len(plans)
    P, T = ep[off:off + k], et[off:off + k]
    off += k
    sp, st = P @ P.T, T @ T.T
    use.update(pids)
    for i in range(k):
        for j in range(i + 1, k):
            if pids[i] == pids[j]:
                same_p.append(sp[i, j]); same_t.append(st[i, j])
            else:
                diff_p.append(sp[i, j]); diff_t.append(st[i, j])

same_p, diff_p = np.array(same_p), np.array(diff_p)
same_t, diff_t = np.array(same_t), np.array(diff_t)
print(f"{a.prefix}: {len(threads)} 个 thread，{len(same_p) + len(diff_p)} 个槽位对\n")
print(f"  {'':<26}{'对数':>8}{'plan 余弦':>11}{'文本余弦':>11}")
print(f"  {'同一个 perspective':<24}{len(same_p):>8}{same_p.mean():>11.4f}{same_t.mean():>11.4f}")
print(f"  {'不同 perspective':<25}{len(diff_p):>8}{diff_p.mean():>11.4f}{diff_t.mean():>11.4f}")
print(f"  {'换 perspective 带来的分散':<20}{'':>8}{same_p.mean() - diff_p.mean():>+11.4f}"
      f"{same_t.mean() - diff_t.mean():>+11.4f}")
print(f"\n  对照：真实 celebrity thread 内文本余弦 0.1797")
print(f"        我们全部槽位对 {np.concatenate([same_t, diff_t]).mean():.4f}")
print(f"\n  12 个 perspective 的使用分布：")
tot = sum(use.values())
for pid, n in sorted(use.items()):
    print(f"    {pid}  {n:>5}  {n / tot * 100:>5.1f}%")
