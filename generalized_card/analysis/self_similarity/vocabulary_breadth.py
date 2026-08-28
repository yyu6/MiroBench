#!/usr/bin/env python3
"""Type count at an EQUAL per-thread token budget (min of the two sides)."""
import json,sys,re,statistics,random
from collections import Counter
from pathlib import Path
REPO=Path("/Users/yaoningyu/Desktop/UIUC/GEO")
sys.path.insert(0,str(REPO/"scripts"/"evaluation"))
from score_thread_semantic_uniformity import load_generated_comments, load_real_comments
TOK=re.compile(r"[a-z0-9']+"); rng=random.Random(0)
pool=json.loads((REPO/"artifacts/generalized_card/seed_pools/camera_product_150_seed42.json").read_text())["seed_posts"]
by={int(p["seed_index"]):p for p in pool}
def load(tag):
    o={}
    for x in sorted((REPO/"artifacts/generalized_card/runs"/tag/"cleaned").glob("run_*_sampled_reddit")):
        cbt,_=load_generated_comments(x)
        for tid,cs in cbt.items(): o[int(tid.split("seed")[-1])]=[TOK.findall(c.text.lower()) for c in cs]
    return o
G=load("v128_interaction_n10_20260828_v1"); H=load("v134_phraseledger_n10_20260828_v1")
cache={}; rows=[]
for s in sorted(set(G)&set(H)):
    p=by[s]; d=REPO/"data/raw/discussions/camera_product"/p["source_product_dir"]
    if d not in cache:
        try: cache[d]=load_real_comments(d)[0]
        except Exception: cache[d]={}
    r=cache[d].get(p["source_raw_post_id"]) or []
    if len(r)<12: continue
    rows.append((s,[TOK.findall(c.text.lower()) for c in r],G[s],H[s]))

print(f"seeds: {[r[0] for r in rows]}")
print(f"\n{'':44}{'real':>9}{'v128':>9}{'v134':>9}")
print("-"*71)
res={0:[],1:[],2:[]}
for s,r,a,b in rows:
    flats=[[w for c in x for w in c] for x in (r,a,b)]
    bud=int(min(len(f) for f in flats)*0.9)
    for k,f in enumerate(flats):
        res[k].append(statistics.mean(len(set(rng.sample(f,bud))) for _ in range(10)))
v=[statistics.mean(res[k]) for k in (0,1,2)]
print(f"{'distinct word types @ equal token budget':44}{v[0]:>9.1f}{v[1]:>9.1f}{v[2]:>9.1f}")
print(f"{'   ratio to real':44}{1.0:>9.2f}{v[1]/v[0]:>9.2f}{v[2]/v[0]:>9.2f}")
for lbl,f in (("hapax share (word used once in thread)", lambda wl:(lambda c: sum(1 for x in c.values() if x==1)/len(c))(Counter(w for cm in wl for w in cm))),
              ("rare-word token share (df<=2 comments)", lambda wl:(lambda df,tot: sum(1 for cm in wl for w in cm if df[w]<=2)/tot)(Counter(w for cm in wl for w in set(cm)), sum(len(cm) for cm in wl)))):
    q=[statistics.mean([f(x[k]) for x in rows]) for k in (1,2,3)]
    print(f"{lbl:44}{q[0]:>9.4f}{q[1]:>9.4f}{q[2]:>9.4f}")
