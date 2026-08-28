#!/usr/bin/env python3
"""Where in the PAIR DISTRIBUTION does the excess live?

The suite's own numbers hint at the answer: self_bertscore mean/median FAIL
hard (d +0.92/+0.96) while self_bertscore_TOP_K passes (d -0.30). If our most
similar pairs are fine and our typical pair is too similar, the defect is a
raised FLOOR -- we never produce two genuinely unrelated comments -- not a
few near-duplicates. This checks that on the cheap surface proxy.
"""
import json,sys,re,statistics
from pathlib import Path
REPO=Path("/Users/yaoningyu/Desktop/UIUC/GEO")
sys.path.insert(0,str(REPO/"scripts"/"evaluation"))
from score_thread_semantic_uniformity import load_generated_comments, load_real_comments
TOK=re.compile(r"[a-z0-9']+")

def pairs(wl, n):
    s=[set(zip(*[w[i:] for i in range(n)])) if n>1 else set(w) for w in wl]
    out=[]
    for i in range(len(s)):
        for j in range(i+1,len(s)):
            u=s[i]|s[j]
            if u: out.append(len(s[i]&s[j])/len(u))
    return sorted(out)

def pct(v,q): 
    if not v: return float("nan")
    k=(len(v)-1)*q; f=int(k); c=min(f+1,len(v)-1)
    return v[f]+(v[c]-v[f])*(k-f)

pool=json.loads((REPO/"artifacts/generalized_card/seed_pools/camera_product_150_seed42.json").read_text())["seed_posts"]
by={int(p["seed_index"]):p for p in pool}
def load(tag):
    out={}
    for x in sorted((REPO/"artifacts/generalized_card/runs"/tag/"cleaned").glob("run_*_sampled_reddit")):
        cbt,_=load_generated_comments(x)
        for tid,cs in cbt.items(): out[int(tid.split("seed")[-1])]=[TOK.findall(c.text.lower()) for c in cs]
    return out
g=load("v128_interaction_n10_20260828_v1")
cache={}; rows=[]
for s in sorted(g):
    p=by[s]; d=REPO/"data/raw/discussions/camera_product"/p["source_product_dir"]
    if d not in cache:
        try: cache[d]=load_real_comments(d)[0]
        except Exception: cache[d]={}
    rcs=cache[d].get(p["source_raw_post_id"]) or []
    if len(rcs)<12: continue
    rows.append((s,[TOK.findall(c.text.lower()) for c in rcs],g[s]))
print(f"{len(rows)} threads, v128 vs its matched real\n")
for n,name in ((1,"UNIGRAM"),(2,"BIGRAM")):
    print(f"{name} Jaccard between comment pairs -- percentile of the pair distribution")
    print(f"{'pctile':>8}{'real':>10}{'ours':>10}{'ratio':>8}")
    for q in (0.05,0.10,0.25,0.50,0.75,0.90,0.95,0.99):
        r=statistics.mean([pct(pairs(x[1],n),q) for x in rows])
        o=statistics.mean([pct(pairs(x[2],n),q) for x in rows])
        print(f"{int(q*100):>7}%{r:>10.4f}{o:>10.4f}{o/r if r else float('nan'):>8.2f}")
    # how many pairs are essentially unrelated
    thr=0.02 if n==2 else 0.10
    rz=statistics.mean([sum(1 for v in pairs(x[1],n) if v<thr)/max(len(pairs(x[1],n)),1) for x in rows])
    oz=statistics.mean([sum(1 for v in pairs(x[2],n) if v<thr)/max(len(pairs(x[2],n)),1) for x in rows])
    print(f"  share of pairs below {thr}:  real {rz:.3f}   ours {oz:.3f}\n")
