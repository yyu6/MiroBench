"""Model-free: assertion architecture vs speaking architecture, real against ours."""
from __future__ import annotations
import json, sys, re, statistics
from pathlib import Path
REPO=Path("/Users/yaoningyu/Desktop/UIUC/GEO")
sys.path.insert(0,str(REPO/"scripts"/"evaluation"))
from score_thread_semantic_uniformity import load_generated_comments, load_real_comments
pool=json.load(open(REPO/"artifacts/generalized_card/seed_pools/camera_product_150_seed42.json"))["seed_posts"]
by_seed={int(p["seed_index"]):p for p in pool}

FIRST=re.compile(r"\b(i|i'm|im|i've|i'd|i'll|my|me|mine|myself)\b",re.I)
ASSERT=re.compile(r"\b(is|are|isn't|aren't)\s+the\b|\bthat's\s+the\b|\bthe\s+(real|only|actual|whole|key|main|whole)\b",re.I)
HEDGEQ=re.compile(r"\?")
SHORT=lambda t: len(t.split())<=12

def prof(texts):
    n=len(texts)
    return {
        "n":n,
        "first_person":sum(bool(FIRST.search(t)) for t in texts)/n,
        "assertion_frame":sum(bool(ASSERT.search(t)) for t in texts)/n,
        "question":sum(bool(HEDGEQ.search(t)) for t in texts)/n,
        "short<=12w":sum(SHORT(t) for t in texts)/n,
        "starts_with_I":sum(bool(re.match(r"\s*(i|i'm|im|i've|i'd)\b",t,re.I)) for t in texts)/n,
        "mean_words":statistics.mean(len(t.split()) for t in texts),
    }

for RUN,label in [("generalized_card_camera_gpt54_paper_20260825_v1","paper50"),
                  ("v128_interaction_n10_20260828_v1","v128")]:
    base=REPO/"artifacts/generalized_card/runs"/RUN/"cleaned"
    cache={}; G=[]; R=[]
    for x in sorted(base.glob("run_*_sampled_reddit")):
        cbt,_=load_generated_comments(x)
        for tid,cs in cbt.items():
            seed=int(tid.split("seed")[-1]); p=by_seed.get(seed)
            if not p: continue
            dd=REPO/"data/raw/discussions/camera_product"/p["source_product_dir"]
            if dd not in cache:
                try: cache[dd]=load_real_comments(dd)[0]
                except Exception: cache[dd]={}
            rcs=cache[dd].get(p["source_raw_post_id"]) or []
            if len(rcs)<10 or len(cs)<10: continue
            G+= [c.text for c in cs]; R+=[c.text for c in rcs]
    if not G: continue
    g,r=prof(G),prof(R)
    print(f"=== {label}  gen {g['n']} / real {r['n']} comments")
    print(f"  {'':<20} {'real':>8} {'gen':>8} {'gen/real':>9}")
    for k in ("first_person","starts_with_I","assertion_frame","question","short<=12w"):
        print(f"  {k:<20} {100*r[k]:7.2f}% {100*g[k]:7.2f}% {g[k]/r[k] if r[k] else 0:9.2f}")
    print(f"  {'mean_words':<20} {r['mean_words']:8.1f} {g['mean_words']:8.1f} {g['mean_words']/r['mean_words']:9.2f}")
    print()
