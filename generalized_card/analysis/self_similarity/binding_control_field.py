"""Which single control field, when shared, binds plan similarity most?"""
import json, itertools, statistics
from collections import defaultdict, Counter
import numpy as np
from sentence_transformers import SentenceTransformer
P="/Users/yaoningyu/Desktop/UIUC/GEO/artifacts/generalized_card/runs/v128_interaction_n10_20260828_v1/logs/planning_quality.jsonl"
SEM=("semantic_move","local_topic","detail_focus","domain_intent","decision_boundary",
     "reply_delta","reply_novelty_anchor","development_plan")
CTRL=("perspective_id","content_angle","claim_family","comment_function","reply_relation","stance","evidence_mode")
threads=defaultdict(list)
for line in open(P):
    if not line.strip(): continue
    x=json.loads(line); ip=x.get("initial_plans")
    if isinstance(ip,dict):
        for _s,p in ip.items():
            if isinstance(p,dict): threads[str(x.get("seed_key"))].append(p)
m=SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
def text(p): return " | ".join(str(p.get(f) or "").strip() for f in SEM if str(p.get(f) or "").strip())

same=defaultdict(list); diff=defaultdict(list); base=[]
for sk,plans in threads.items():
    if len(plans)<10: continue
    E=m.encode([text(p) for p in plans],convert_to_numpy=True,normalize_embeddings=True,show_progress_bar=False)
    S=E@E.T
    for i,j in itertools.combinations(range(len(plans)),2):
        s=float(S[i,j]); base.append(s)
        for f in CTRL:
            (same if str(plans[i].get(f) or "")==str(plans[j].get(f) or "") else diff)[f].append(s)
b=statistics.mean(base)
print(f"baseline mean plan similarity (all {len(base)} within-thread pairs): {b:.4f}")
print(f"  real OUTPUT within-thread cosine for reference (G94): 0.2892\n")
print(f"{'control field':<22} {'share pairs':>11} {'sim|same':>9} {'sim|diff':>9} {'lift':>7}")
rows=[]
for f in CTRL:
    ss,dd=same[f],diff[f]
    if not ss or not dd: continue
    rows.append((statistics.mean(ss)-statistics.mean(dd), f, len(ss)/len(base), statistics.mean(ss), statistics.mean(dd)))
for lift,f,sh,a,d in sorted(rows,reverse=True):
    print(f"{f:<22} {100*sh:10.1f}% {a:9.4f} {d:9.4f} {lift:+7.4f}")

# also: non-control structural fields
EXTRA=("branch_id","opener_type","affect_role","context_aperture","_slot_surface_label","claim_key","owned_decision_subject","_required_branch_id")
print()
same2=defaultdict(list); diff2=defaultdict(list)
for sk,plans in threads.items():
    if len(plans)<10: continue
    E=m.encode([text(p) for p in plans],convert_to_numpy=True,normalize_embeddings=True,show_progress_bar=False)
    S=E@E.T
    for i,j in itertools.combinations(range(len(plans)),2):
        s=float(S[i,j])
        for f in EXTRA:
            a,bb=plans[i].get(f),plans[j].get(f)
            if a is None and bb is None: continue
            (same2 if str(a)==str(bb) else diff2)[f].append(s)
rows=[]
for f in EXTRA:
    ss,dd=same2[f],diff2[f]
    if len(ss)<50 or len(dd)<50: continue
    rows.append((statistics.mean(ss)-statistics.mean(dd), f, len(ss)/(len(ss)+len(dd)), statistics.mean(ss), statistics.mean(dd)))
print(f"{'other plan field':<22} {'share pairs':>11} {'sim|same':>9} {'sim|diff':>9} {'lift':>7}")
for lift,f,sh,a,d in sorted(rows,reverse=True):
    print(f"{f:<22} {100*sh:10.1f}% {a:9.4f} {d:9.4f} {lift:+7.4f}")
