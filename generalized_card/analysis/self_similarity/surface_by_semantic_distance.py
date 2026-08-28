"""Is the surface overlap elevated even between semantically DISTANT comment pairs?"""
from __future__ import annotations
import json, sys, itertools, statistics, re
from pathlib import Path
import numpy as np
REPO=Path("/Users/yaoningyu/Desktop/UIUC/GEO")
sys.path.insert(0,str(REPO/"scripts"/"evaluation"))
from score_thread_semantic_uniformity import load_generated_comments, load_real_comments
from sentence_transformers import SentenceTransformer
TOK=re.compile(r"[a-z0-9']+")
pool=json.load(open(REPO/"artifacts/generalized_card/seed_pools/camera_product_150_seed42.json"))["seed_posts"]
by_seed={int(p["seed_index"]):p for p in pool}
m=SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

def grams(t,n):
    w=TOK.findall(t.lower())
    return set(zip(*[w[i:] for i in range(n)])) if len(w)>=n else set()

def collect(texts):
    E=m.encode(texts,convert_to_numpy=True,normalize_embeddings=True,show_progress_bar=False)
    S=E@E.T
    g1=[grams(t,1) for t in texts]; g2=[grams(t,2) for t in texts]
    out=[]
    for i,j in itertools.combinations(range(len(texts)),2):
        u1=g1[i]|g1[j]; u2=g2[i]|g2[j]
        if not u1: continue
        out.append((float(S[i,j]),
                    len(g1[i]&g1[j])/len(u1),
                    len(g2[i]&g2[j])/len(u2) if u2 else 0.0))
    return out

RUN=REPO/"artifacts/generalized_card/runs/generalized_card_camera_gpt54_paper_20260825_v1/cleaned"
cache={}; GP=[]; RP=[]; n=0
for x in sorted(RUN.glob("run_*_sampled_reddit")):
    cbt,_=load_generated_comments(x)
    for tid,cs in cbt.items():
        seed=int(tid.split("seed")[-1]); p=by_seed.get(seed)
        if not p: continue
        dd=REPO/"data/raw/discussions/camera_product"/p["source_product_dir"]
        if dd not in cache:
            try: cache[dd]=load_real_comments(dd)[0]
            except Exception: cache[dd]={}
        rcs=cache[dd].get(p["source_raw_post_id"]) or []
        if len(rcs)<12 or len(cs)<12: continue
        n+=1
        GP+=collect([c.text for c in cs]); RP+=collect([c.text for c in rcs])
        if n>=30: break
    if n>=30: break

print(f"threads: {n}   gen pairs {len(GP)}   real pairs {len(RP)}\n")
BINS=[(-1,0.10),(0.10,0.20),(0.20,0.30),(0.30,0.40),(0.40,0.55),(0.55,1.01)]
print(f"{'cosine bin':<14} {'real n':>7} {'gen n':>7} | {'uni real':>9} {'uni gen':>8} {'ratio':>6} | {'bi real':>8} {'bi gen':>8} {'ratio':>6}")
for lo,hi in BINS:
    r=[x for x in RP if lo<=x[0]<hi]; g=[x for x in GP if lo<=x[0]<hi]
    if len(r)<50 or len(g)<50: continue
    r1,g1=statistics.mean([x[1] for x in r]),statistics.mean([x[1] for x in g])
    r2,g2=statistics.mean([x[2] for x in r]),statistics.mean([x[2] for x in g])
    print(f"[{lo:5.2f},{hi:4.2f}) {len(r):>7} {len(g):>7} | {r1:9.4f} {g1:8.4f} {g1/r1:6.3f} | {r2:8.5f} {g2:8.5f} {g2/r2 if r2 else 0:6.3f}")
print()
print(f"{'POOLED':<14} {len(RP):>7} {len(GP):>7} | "
      f"{statistics.mean([x[1] for x in RP]):9.4f} {statistics.mean([x[1] for x in GP]):8.4f} "
      f"{statistics.mean([x[1] for x in GP])/statistics.mean([x[1] for x in RP]):6.3f} | "
      f"{statistics.mean([x[2] for x in RP]):8.5f} {statistics.mean([x[2] for x in GP]):8.5f} "
      f"{statistics.mean([x[2] for x in GP])/statistics.mean([x[2] for x in RP]):6.3f}")
print(f"\ncosine distribution: real mean {statistics.mean([x[0] for x in RP]):.4f}   gen mean {statistics.mean([x[0] for x in GP]):.4f}")
