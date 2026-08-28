#!/usr/bin/env python3
"""Is the ONE channel that complies at ~1.0 (E4: a named concrete token) being
used to give slots DISJOINT lexical territory, or is it handing the same few
tokens to everybody?"""
import json, glob, re, statistics
from collections import Counter
from pathlib import Path
REPO = Path("/Users/yaoningyu/Desktop/UIUC/GEO")
TAG = "v128_interaction_n10_20260828_v1"
TOK = re.compile(r"[a-z0-9']+")

by_thread = {}
for p in sorted(glob.glob(str(REPO/"artifacts/generalized_card/runs"/TAG/"generated/run_*/generation_records.json"))):
    for r in json.load(open(p)):
        t = r.get("task") or {}
        a = t.get("concrete_anchors") or []
        by_thread.setdefault(r.get("post_id"), []).append(
            (a, r.get("comment") or {}, t.get("local_task_id")))

print(f"{'thread':30}{'slots':>7}{'anchors/slot':>14}{'distinct':>10}{'Jaccard':>9}{'top anchor share':>18}")
print("-"*90)
jac_all, share_all, distinct_all = [], [], []
for tid, rows in sorted(by_thread.items()):
    sets = [set(x.lower() for x in a) for a, _, _ in rows if a]
    if len(sets) < 4: continue
    pj = []
    for i in range(len(sets)):
        for j in range(i+1, len(sets)):
            u = sets[i] | sets[j]
            if u: pj.append(len(sets[i] & sets[j]) / len(u))
    c = Counter(x for s in sets for x in s)
    top = c.most_common(1)[0]
    jac_all.append(statistics.mean(pj)); distinct_all.append(len(c))
    share_all.append(top[1]/len(sets))
    print(f"{tid[-28:]:30}{len(sets):>7}{statistics.mean(len(s) for s in sets):>14.2f}"
          f"{len(c):>10}{statistics.mean(pj):>9.3f}{top[0][:14]+' '+str(round(100*top[1]/len(sets)))+'%':>18}")
print("-"*90)
print(f"{'MEAN':30}{'':>7}{'':>14}{statistics.mean(distinct_all):>10.1f}"
      f"{statistics.mean(jac_all):>9.3f}{100*statistics.mean(share_all):>17.0f}%")
print(f"\npairwise anchor-set Jaccard = {statistics.mean(jac_all):.3f}  "
      f"(0.000 would mean every slot owns a disjoint token set)")

# do the anchors actually reach the text, and do they separate the comments?
hit = tot = 0
for tid, rows in by_thread.items():
    for a, c, _ in rows:
        txt = set(TOK.findall((c.get("content") or "").lower()))
        for x in a:
            core = TOK.findall(re.sub(r"\(.*?\)", "", x).lower())
            if not core: continue
            tot += 1
            if all(w in txt for w in core): hit += 1
print(f"anchor compliance: {hit}/{tot} = {100*hit/max(tot,1):.1f}% of assigned anchors appear verbatim in the comment")
