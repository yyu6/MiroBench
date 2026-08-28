#!/usr/bin/env python3
"""Which 17% of the vocabulary are we missing?"""
import json,sys,re
from collections import Counter
from pathlib import Path
REPO=Path("/Users/yaoningyu/Desktop/UIUC/GEO")
sys.path.insert(0,str(REPO/"scripts"/"evaluation"))
from score_thread_semantic_uniformity import load_generated_comments, load_real_comments
TOK=re.compile(r"[a-z0-9']+")
pool=json.loads((REPO/"artifacts/generalized_card/seed_pools/camera_product_150_seed42.json").read_text())["seed_posts"]
by={int(p["seed_index"]):p for p in pool}
o={}
for x in sorted((REPO/"artifacts/generalized_card/runs/v128_interaction_n10_20260828_v1/cleaned").glob("run_*_sampled_reddit")):
    cbt,_=load_generated_comments(x)
    for tid,cs in cbt.items(): o[int(tid.split("seed")[-1])]=[TOK.findall(c.text.lower()) for c in cs]
cache={}; R=Counter(); O=Counter()
for s in sorted(o):
    p=by[s]; d=REPO/"data/raw/discussions/camera_product"/p["source_product_dir"]
    if d not in cache:
        try: cache[d]=load_real_comments(d)[0]
        except Exception: cache[d]={}
    r=cache[d].get(p["source_raw_post_id"]) or []
    if len(r)<12: continue
    for c in r: R.update(TOK.findall(c.text.lower()))
    for c in o[s]: O.update(c)
print(f"real types {len(R):,}   ours {len(O):,}   we use {100*len(O)/len(R):.0f}% as many")
print(f"types real has that we never use: {len(set(R)-set(O)):,}")
print(f"types we have that real never uses: {len(set(O)-set(R)):,}\n")
miss=[(w,n) for w,n in R.most_common() if w not in O]
print("most frequent real words we NEVER produce (top 60):")
print("  "+", ".join(f"{w}({n})" for w,n in miss[:60]))
digit=sum(n for w,n in miss if any(ch.isdigit() for ch in w))
print(f"\n  of the missing real tokens, {digit:,} occurrences contain a digit "
      f"({100*digit/sum(n for _,n in miss):.0f}% of the missing mass)")
ours=[(w,n) for w,n in O.most_common() if w not in R]
print("\nwords WE produce that real never does (top 40):")
print("  "+", ".join(f"{w}({n})" for w,n in ours[:40]))
