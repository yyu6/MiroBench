"""WHICH bigrams do semantically-unrelated comments share? Ours vs real."""
from __future__ import annotations
import json, sys, itertools, re
from collections import Counter
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
def bg(t):
    w=TOK.findall(t.lower())
    return set(zip(w,w[1:]))

def shared(texts, thresh=0.15):
    E=m.encode(texts,convert_to_numpy=True,normalize_embeddings=True,show_progress_bar=False)
    S=E@E.T; B=[bg(t) for t in texts]
    c=Counter(); npair=0
    for i,j in itertools.combinations(range(len(texts)),2):
        if float(S[i,j])>=thresh: continue      # only UNRELATED pairs
        npair+=1
        for g in B[i]&B[j]: c[g]+=1
    return c,npair

RUN=REPO/"artifacts/generalized_card/runs/generalized_card_camera_gpt54_paper_20260825_v1/cleaned"
cache={}; GC=Counter(); RC=Counter(); gn=rn=0; n=0
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
        a,an=shared([c.text for c in cs]); GC+=a; gn+=an
        b,bn=shared([c.text for c in rcs]); RC+=b; rn+=bn
        if n>=30: break
    if n>=30: break
print(f"threads {n}  unrelated gen pairs {gn}  unrelated real pairs {rn}\n")
rows=[]
for g,cnt in GC.items():
    gr=cnt/gn; rr=RC.get(g,0)/rn
    rows.append((gr-rr, gr, rr, g))
rows.sort(reverse=True)
print("Bigrams MOST over-shared between UNRELATED comment pairs (gen rate vs real rate):")
print(f"{'bigram':<26} {'gen %':>8} {'real %':>8} {'ratio':>8}")
for d,gr,rr,g in rows[:30]:
    print(f"{' '.join(g):<26} {100*gr:8.3f} {100*rr:8.3f} {(gr/rr if rr else float('inf')):8.1f}")
print("\nTotal excess bigram-pair mass carried by the top 30:", f"{100*sum(r[0] for r in rows[:30]):.2f} pp of pairs")
print("Total positive excess across all bigrams:", f"{100*sum(r[0] for r in rows if r[0]>0):.2f} pp")
