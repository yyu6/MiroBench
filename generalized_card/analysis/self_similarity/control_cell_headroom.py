"""Thread-wide: are the scheduled control cells exhausted, or is there headroom?"""
import json, itertools, statistics
from collections import defaultdict, Counter
import numpy as np
from sentence_transformers import SentenceTransformer
P="/Users/yaoningyu/Desktop/UIUC/GEO/artifacts/generalized_card/runs/v128_interaction_n10_20260828_v1/logs/planning_quality.jsonl"
SEM=("semantic_move","local_topic","detail_focus","domain_intent","decision_boundary",
     "reply_delta","reply_novelty_anchor","development_plan")
CTRL=("perspective_id","content_angle","claim_family","comment_function","reply_relation","stance","evidence_mode")

threads=defaultdict(list)
for bi,line in enumerate(open(P)):
    if not line.strip(): continue
    x=json.loads(line)
    ip=x.get("initial_plans")
    if not isinstance(ip,dict): continue
    for sid,plan in ip.items():
        if isinstance(plan,dict): threads[str(x.get("seed_key"))].append((bi,plan))

m=SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
def text(p): return " | ".join(str(p.get(f) or "").strip() for f in SEM if str(p.get(f) or "").strip())
def cell(p): return tuple(str(p.get(f) or "") for f in CTRL)

print(f"{'thread':<10} {'slots':>6} {'cells':>6} {'cells/slot':>11} {'top cell n':>11}  per-field distinct values")
allsame=[]; alldiff=[]
tot_slots=0; tot_cells=0
for sk,items in sorted(threads.items(), key=lambda kv:-len(kv[1])):
    plans=[p for _b,p in items]
    if len(plans)<10: continue
    cells=[cell(p) for p in plans]
    c=Counter(cells)
    fieldn={f:len({str(p.get(f) or "") for p in plans}) for f in CTRL}
    tot_slots+=len(plans); tot_cells+=len(c)
    print(f"{sk:<10} {len(plans):>6} {len(c):>6} {len(c)/len(plans):>11.3f} {c.most_common(1)[0][1]:>11}  "
          + " ".join(f"{f.split('_')[0][:5]}={n}" for f,n in fieldn.items()))
    E=m.encode([text(p) for p in plans],convert_to_numpy=True,normalize_embeddings=True,show_progress_bar=False)
    S=E@E.T
    for i,j in itertools.combinations(range(len(plans)),2):
        (allsame if cells[i]==cells[j] else alldiff).append(float(S[i,j]))

print(f"\nTHREAD-WIDE totals: {tot_slots} slots, {tot_cells} distinct cells -> {tot_cells/tot_slots:.3f} cells/slot")
print(f"pairs sharing the SAME cell : n={len(allsame):6d} ({100*len(allsame)/(len(allsame)+len(alldiff)):.1f}%)  "
      f"mean sim {statistics.mean(allsame):.4f}  >=0.70 {100*sum(s>=.7 for s in allsame)/len(allsame):.2f}%")
print(f"pairs in DIFFERENT cells    : n={len(alldiff):6d} ({100*len(alldiff)/(len(allsame)+len(alldiff)):.1f}%)  "
      f"mean sim {statistics.mean(alldiff):.4f}  >=0.70 {100*sum(s>=.7 for s in alldiff)/len(alldiff):.2f}%")
hs=sum(s>=.7 for s in allsame); hd=sum(s>=.7 for s in alldiff)
print(f"\ncollisions in same cell: {hs}  in different cells: {hd}  -> {100*hd/(hs+hd):.1f}% of collisions cross cells")
