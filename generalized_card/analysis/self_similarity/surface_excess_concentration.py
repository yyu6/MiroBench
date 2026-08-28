"""Is the surface excess concentrated in a few carriers, or spread across the distribution?"""
from __future__ import annotations
import json, sys, re, statistics
from collections import Counter
from pathlib import Path
REPO=Path("/Users/yaoningyu/Desktop/UIUC/GEO")
sys.path.insert(0,str(REPO/"scripts"/"evaluation"))
from score_thread_semantic_uniformity import load_generated_comments, load_real_comments
TOK=re.compile(r"[a-z0-9']+")
pool=json.load(open(REPO/"artifacts/generalized_card/seed_pools/camera_product_150_seed42.json"))["seed_posts"]
by_seed={int(p["seed_index"]):p for p in pool}

def mass(texts,n):
    """expected shared-ngram MASS per pair, per ngram type (matches the p_n numerator)."""
    c=Counter(); tot=0
    per=[]
    for t in texts:
        w=TOK.findall(t.lower())
        g=list(zip(*[w[i:] for i in range(n)])) if len(w)>=n else []
        per.append(Counter(g)); tot+=len(g)
    if tot==0: return {},0
    # expected clipped overlap contribution per ngram across all ordered pairs
    acc=Counter(); npair=0
    for i in range(len(per)):
        for j in range(len(per)):
            if i==j: continue
            npair+=1
            for k,v in per[i].items():
                m=min(v,per[j].get(k,0))
                if m: acc[k]+=m
    return {k:v/npair for k,v in acc.items()}, npair

RUN=REPO/"artifacts/generalized_card/runs/generalized_card_camera_gpt54_paper_20260825_v1/cleaned"
cache={}
for N in (1,2):
    G=Counter(); R=Counter(); n=0
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
            a,_=mass([c.text for c in cs],N); b,_=mass([c.text for c in rcs],N)
            for k,v in a.items(): G[k]+=v
            for k,v in b.items(): R[k]+=v
            n+=1
            if n>=25: break
        if n>=25: break
    for k in G: G[k]/=n
    for k in R: R[k]/=n
    exc=sorted(((G.get(k,0)-R.get(k,0)),k) for k in set(G)|set(R))
    pos=[e for e in exc if e[0]>0]; pos.sort(reverse=True)
    tot_pos=sum(e[0] for e in pos)
    print(f"=== {N}-gram, {n} threads ===")
    print(f"  total POSITIVE excess mass per pair: {tot_pos:.4f}   (gen total {sum(G.values()):.3f}, real total {sum(R.values()):.3f})")
    cum=0
    for K in (1,5,10,25,50,100,250,500,1000):
        if K>len(pos): break
        cum=sum(e[0] for e in pos[:K])
        print(f"    top {K:>4} types carry {100*cum/tot_pos:5.1f}% of the positive excess")
    print(f"  distinct types with positive excess: {len(pos)}")
    print(f"  top 10: " + ", ".join(f'"{" ".join(k) if isinstance(k,tuple) else k}"' for _v,k in pos[:10]))
    print()
