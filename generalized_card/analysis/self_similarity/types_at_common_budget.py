#!/usr/bin/env python3
"""Type count at exactly 2610 sampled tokens, so real / ours / the constructed
threads are all on one scale."""
import json,sys,re,statistics,random
from pathlib import Path
REPO=Path("/Users/yaoningyu/Desktop/UIUC/GEO")
sys.path.insert(0,str(REPO/"scripts"/"evaluation"))
from score_thread_semantic_uniformity import load_generated_comments, load_real_comments
TOK=re.compile(r"[a-z0-9']+"); rng=random.Random(0)
B=2610
pool=json.loads((REPO/"artifacts/generalized_card/seed_pools/camera_product_150_seed42.json").read_text())["seed_posts"]
by={int(p["seed_index"]):p for p in pool}
def load(tag):
    o={}
    for x in sorted((REPO/"artifacts/generalized_card/runs"/tag/"cleaned").glob("run_*_sampled_reddit")):
        cbt,_=load_generated_comments(x)
        for tid,cs in cbt.items(): o[int(tid.split("seed")[-1])]=[c.text for c in cs]
    return o
G=load("v128_interaction_n10_20260828_v1"); H=load("v134_phraseledger_n10_20260828_v1")
cache={}; res={0:[],1:[],2:[]}
for s in sorted(set(G)&set(H)):
    p=by[s]; d=REPO/"data/raw/discussions/camera_product"/p["source_product_dir"]
    if d not in cache:
        try: cache[d]=load_real_comments(d)[0]
        except Exception: cache[d]={}
    r=cache[d].get(p["source_raw_post_id"]) or []
    if len(r)<12: continue
    for k,ts in enumerate(([c.text for c in r],G[s],H[s])):
        flat=[w for t in ts for w in TOK.findall(t.lower())]
        if len(flat)<B: continue
        res[k].append(statistics.mean(len(set(rng.sample(flat,B))) for _ in range(20)))
for k,l in (("real",0),("v128",1),("v134",2)):
    print(f"{k:6} distinct types @ {B} tokens: {statistics.mean(res[l]):.1f}   (n={len(res[l])} threads)")
print(f"\ndeficit v128 vs real: {statistics.mean(res[1])-statistics.mean(res[0]):+.1f} types "
      f"({100*(statistics.mean(res[1])/statistics.mean(res[0])-1):+.1f}%)")
