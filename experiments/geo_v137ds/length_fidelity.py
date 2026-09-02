#!/usr/bin/env python3
"""Does the Writer produce the length the matched real slot asked for?

  python3 experiments/geo_v137ds/length_fidelity.py mprof_ iso2_ win_

Each slot carries `real_word_count`, taken from the real comment it was matched
to, so the asked distribution is the real one by construction. If the generated
threads are short of real's short comments, the loss is either the Planner
refusing to plan a micro slot or the Writer overshooting it, and those call for
different fixes.
"""
import glob, json, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
prefixes = sys.argv[1:] or ["mprof_"]

for pref in prefixes:
    asked, got = [], []
    for f in glob.glob(str(REPO / "artifacts/generalized_card/runs/*/generated/run_00_sampled_reddit/discussion.json")):
        if not f.split("/")[-4].startswith(pref):
            continue
        for post in json.load(open(f)).get("posts") or []:
            for rec in post.get("generation_records") or []:
                task = rec.get("task") or {}
                comment = rec.get("comment")
                if not isinstance(comment, dict):
                    continue
                a = task.get("real_word_count")
                text = str(comment.get("content") or "").strip()
                if a is None or not text:
                    continue
                asked.append(int(a))
                got.append(len(text.split()))
    if not asked:
        print(f"{pref:<12} (没有数据)")
        continue
    a, g = np.array(asked), np.array(got)
    small = a <= 10
    print(f"{pref:<12} {len(a):>5} slot   "
          f"要求中位 {np.median(a):>3.0f} / 实际中位 {np.median(g):>3.0f}   "
          f"要求≤10词 {small.mean()*100:>5.1f}% / 其中实际也≤10词 "
          f"{(g[small] <= 10).mean()*100 if small.any() else float('nan'):>5.1f}%   "
          f"这些槽实际中位 {np.median(g[small]) if small.any() else float('nan'):>3.0f} 词")
