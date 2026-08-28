#!/usr/bin/env python3
"""Before proposing 'give each slot disjoint anchors', check the direction on
real. v134's lesson: an intervention that looks obviously right can move the
generator AWAY from real. So: how concentrated is real's entity vocabulary
across the comments of one thread, versus ours?"""
import json, sys, re, statistics
from collections import Counter
from pathlib import Path
REPO = Path("/Users/yaoningyu/Desktop/UIUC/GEO")
sys.path.insert(0, str(REPO/"scripts"/"evaluation"))
from score_thread_semantic_uniformity import load_generated_comments, load_real_comments

# entity-ish tokens: brand/model names and alphanumeric model codes
WORD = re.compile(r"[A-Za-z][A-Za-z0-9'\-]*|\d+[A-Za-z]+|[A-Za-z]+\d+")
BRANDS = set("""sony canon nikon fuji fujifilm ricoh olympus panasonic pentax leica sigma tamron
zeiss samyang gr rx100 a7 a7r a7s a6000 a6300 a6500 xt x100f x100 x100v eos rebel d850 5d 6d 7d
gh5 om em1 em5 z6 z7 r5 r6 gx a9 nex""".split())
MODEL = re.compile(r"^(?:[a-z]{1,3}\d{1,4}[a-z]{0,3}|\d{1,3}[-/]\d{1,3}(?:\.\d)?|\d{1,3}mm|f/?\d(?:\.\d)?)$")

def entities(text):
    out=set()
    for w in WORD.findall(text):
        lw=w.lower()
        if lw in BRANDS or MODEL.match(lw): out.add(lw)
    return out

def spread(comments):
    sets=[entities(t) for t in comments]
    sets=[s for s in sets if s]
    if len(sets)<4: return None
    pj=[]
    for i in range(len(sets)):
        for j in range(i+1,len(sets)):
            u=sets[i]|sets[j]
            if u: pj.append(len(sets[i]&sets[j])/len(u))
    c=Counter(x for s in sets for x in s)
    return (statistics.mean(pj), len(c), c.most_common(1)[0][1]/len(sets),
            len(sets)/len(comments), statistics.mean(len(s) for s in sets))

pool=json.loads((REPO/"artifacts/generalized_card/seed_pools/camera_product_150_seed42.json").read_text())["seed_posts"]
by={int(p["seed_index"]):p for p in pool}
G={}
for x in sorted((REPO/"artifacts/generalized_card/runs/v128_interaction_n10_20260828_v1/cleaned").glob("run_*_sampled_reddit")):
    cbt,_=load_generated_comments(x)
    for tid,cs in cbt.items(): G[int(tid.split("seed")[-1])]=[c.text for c in cs]
cache={}; R=[]; O=[]
for s in sorted(G):
    p=by[s]; d=REPO/"data/raw/discussions/camera_product"/p["source_product_dir"]
    if d not in cache:
        try: cache[d]=load_real_comments(d)[0]
        except Exception: cache[d]={}
    r=cache[d].get(p["source_raw_post_id"]) or []
    if len(r)<12: continue
    a=spread([c.text for c in r]); b=spread(G[s])
    if a and b: R.append(a); O.append(b)
print(f"{len(R)} matched threads\n")
names=("pairwise entity Jaccard","distinct entities in thread","top entity's comment share",
       "share of comments naming any","entities per naming comment")
print(f"{'':32}{'real':>10}{'ours':>10}{'ratio':>9}")
print("-"*63)
for i,n in enumerate(names):
    a=statistics.mean(x[i] for x in R); b=statistics.mean(x[i] for x in O)
    print(f"{n:32}{a:>10.3f}{b:>10.3f}{b/a if a else float('nan'):>9.2f}")
print("\nper-thread pairwise entity Jaccard (real -> ours):")
for s,a,b in zip([s for s in sorted(G) if True][:len(R)], R, O):
    print(f"   {a[0]:.3f} -> {b[0]:.3f}   {'ours MORE concentrated' if b[0]>a[0] else 'ours more spread'}")
print(f"\nours more concentrated in {sum(1 for a,b in zip(R,O) if b[0]>a[0])}/{len(R)} threads")
