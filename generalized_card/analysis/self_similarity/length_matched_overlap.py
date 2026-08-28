#!/usr/bin/env python3
"""Is v134's overlap drop real, or is it just shorter comments?

G133: a length-matched ratio is the only honest comparison. Shorter comments
share fewer bigrams mechanically, so an arm that merely shortens the output
looks like an arm that diversified it.
"""
from __future__ import annotations
import json, re, statistics, sys, random
from pathlib import Path
REPO = Path("/Users/yaoningyu/Desktop/UIUC/GEO")
sys.path.insert(0, str(REPO / "scripts" / "evaluation"))
from score_thread_semantic_uniformity import load_generated_comments, load_real_comments
TOK = re.compile(r"[a-z0-9']+")
rng = random.Random(0)

def bg(w): return set(zip(w, w[1:]))
def overlap(wl):
    s=[bg(w) for w in wl]; v=[]
    for i in range(len(s)):
        for j in range(i+1,len(s)):
            u=s[i]|s[j]
            if u: v.append(len(s[i]&s[j])/len(u))
    return statistics.mean(v) if v else float("nan")

def truncate_to(wl, target):
    """Cut every comment to `target` words (contiguous random window)."""
    out=[]
    for w in wl:
        if len(w) <= target: out.append(list(w)); continue
        st = rng.randrange(0, len(w)-target+1)
        out.append(list(w[st:st+target]))
    return out

def load(tag, sub):
    root = REPO/"artifacts/generalized_card/runs"/tag/sub; out={}
    for x in sorted(root.glob("run_*_sampled_reddit")):
        cbt,_ = load_generated_comments(x)
        for tid,cs in cbt.items(): out[int(tid.split("seed")[-1])]=[TOK.findall(c.text.lower()) for c in cs]
    return out

pool=json.loads((REPO/"artifacts/generalized_card/seed_pools/camera_product_150_seed42.json").read_text())["seed_posts"]
by={int(p["seed_index"]):p for p in pool}
a=load("v134_phraseledger_n10_20260828_v1","cleaned")
b=load("v128_interaction_n10_20260828_v1","cleaned")
cache={}; rows=[]
for sidx in sorted(set(a)&set(b)):
    p=by[sidx]; d=REPO/"data/raw/discussions/camera_product"/p["source_product_dir"]
    if d not in cache:
        try: cache[d]=load_real_comments(d)[0]
        except Exception: cache[d]={}
    rcs=cache[d].get(p["source_raw_post_id"]) or []
    if len(rcs)<12: continue
    rows.append((sidx,[TOK.findall(c.text.lower()) for c in rcs],b[sidx],a[sidx]))

print(f"{'':26}{'real':>10}{'v128':>10}{'v134':>10}")
print("-"*56)
for lbl,f in (("comments / thread", lambda w: float(len(w))),
              ("mean words / comment", lambda w: statistics.mean([len(x) for x in w]) if w else 0.0),
              ("median words", lambda w: statistics.median([len(x) for x in w]) if w else 0.0)):
    v=[statistics.mean([f(x[k]) for x in rows]) for k in (1,2,3)]
    print(f"{lbl:26}"+"".join(f"{q:>10.2f}" for q in v))

print("\nlength-matched: every comment in all three cut to the same word count")
print(f"{'cut to':>8}{'real':>10}{'v128':>10}{'v134':>10}{'v128/real':>11}{'v134/real':>11}")
for T in (20,30,40,60):
    vals=[]
    for k in (1,2,3):
        vals.append(statistics.mean([overlap(truncate_to(x[k],T)) for x in rows]))
    r,c128,c134=vals
    print(f"{T:>8}{r:>10.5f}{c128:>10.5f}{c134:>10.5f}{c128/r:>11.2f}{c134/r:>11.2f}")

print("\npaired per-thread (length-matched at 30 words): does v134 beat v128?")
w=0
for sidx,r,g128,g134 in rows:
    o128=overlap(truncate_to(g128,30)); o134=overlap(truncate_to(g134,30)); rr=overlap(truncate_to(r,30))
    hit = o134 < o128
    w += hit
    print(f"  seed{sidx:<4} real {rr:.5f}   v128 {o128:.5f}   v134 {o134:.5f}   {'v134 better' if hit else 'v128 better'}")
print(f"\n  v134 closer to real in {w}/{len(rows)} threads")
