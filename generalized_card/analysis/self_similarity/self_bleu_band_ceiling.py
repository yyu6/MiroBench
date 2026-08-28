#!/usr/bin/env python3
"""Redo the v134-motivating ablation on the TARGET metric, not on my proxy.

The 80.5% figure that justified v134 was measured on 2-gram Jaccard overlap.
self_bleu_4 is a different quantity (4-grams, brevity penalty, clipped counts).
So: if we deleted the high-DF function band PERFECTLY, what would self_bleu_4
actually do? That is the ceiling of this entire direction.
"""
import json,sys,re,statistics,random
from collections import Counter
from pathlib import Path
REPO=Path("/Users/yaoningyu/Desktop/UIUC/GEO")
sys.path.insert(0,str(REPO/"scripts"/"evaluation"))
from score_thread_semantic_uniformity import load_generated_comments, load_real_comments
from score_thread_self_bleu import score_thread, ThreadComment
rng=random.Random(0)
TOK=re.compile(r"[a-z0-9']+")
FUNCTION=set("""a an the and or but if so then than that this these those there here
is are was were be been being am s re ve d ll t of to in on at for with from by about
into over after before as like just only very really pretty quite kind sort bit lot much
many more most less least i you he she it we they me him her us them my your his its our
their what which who whom whose when where why how all any both each few other some such
no nor not too own same can could would should may might must will shall do does did have
has had get got go goes went one two 1 2 up out off down again still even also""".split())
def fn(g): return g[0] in FUNCTION and g[1] in FUNCTION

def mk(texts):
    return [ThreadComment(thread_id="t",comment_id=str(i),parent_id=None,author="a",depth=0,
                          text=t,thread_title="") for i,t in enumerate(texts)]

def strip_band(texts, frac=0.10):
    wl=[TOK.findall(t.lower()) for t in texts]; n=len(wl); df=Counter()
    for w in wl:
        for g in set(zip(w,w[1:])): df[g]+=1
    hot={g for g,v in df.items() if v/n>frac and fn(g)}
    out=[];rm=[]
    for w in wl:
        keep=[];d=0;i=0
        while i<len(w):
            if i+1<len(w) and (w[i],w[i+1]) in hot: keep.append(w[i]); i+=2; d+=1
            else: keep.append(w[i]); i+=1
        out.append(" ".join(keep)); rm.append(d)
    return out,rm

def strip_rand(texts,rm):
    out=[]
    for t,k in zip(texts,rm):
        w=TOK.findall(t.lower())
        if k<=0 or len(w)<=k: out.append(" ".join(w)); continue
        drop=set(rng.sample(range(len(w)),k))
        out.append(" ".join(x for i,x in enumerate(w) if i not in drop))
    return out

pool=json.loads((REPO/"artifacts/generalized_card/seed_pools/camera_product_150_seed42.json").read_text())["seed_posts"]
by={int(p["seed_index"]):p for p in pool}
cache={}; base=[];abl=[];ctl=[];real=[]
for x in sorted((REPO/"artifacts/generalized_card/runs/v128_interaction_n10_20260828_v1/cleaned").glob("run_*_sampled_reddit")):
    cbt,_=load_generated_comments(x)
    for tid,cs in cbt.items():
        p=by.get(int(tid.split("seed")[-1]))
        if not p: continue
        d=REPO/"data/raw/discussions/camera_product"/p["source_product_dir"]
        if d not in cache:
            try: cache[d]=load_real_comments(d)[0]
            except Exception: cache[d]={}
        rcs=cache[d].get(p["source_raw_post_id"]) or []
        if len(rcs)<12 or len(cs)<12: continue
        g=[c.text for c in cs]
        s,rm=strip_band(g)
        K="self_bleu_4"
        base.append(score_thread("t",mk(g),False)[K])
        abl.append(score_thread("t",mk(s),False)[K])
        ctl.append(score_thread("t",mk(strip_rand(g,rm)),False)[K])
        real.append(score_thread("t",rcs,False)[K])
mb,ma,mc,mr=(statistics.mean(v) for v in (base,abl,ctl,real))
print(f"self_bleu_4, {len(base)} v128 threads -- PERFECT deletion of the high-DF function band")
print(f"  REAL                 {mr:.5f}")
print(f"  ours, as generated   {mb:.5f}   excess {mb-mr:+.5f}  ({100*(mb-mr)/mr:+.1f}%)")
print(f"  ours, band removed   {ma:.5f}   excess {ma-mr:+.5f}  ({100*(ma-mr)/mr:+.1f}%)")
print(f"  ours, random control {mc:.5f}   excess {mc-mr:+.5f}  ({100*(mc-mr)/mr:+.1f}%)")
net=(mb-ma)-(mb-mc)
print(f"\n  NET band-specific effect on self_bleu_4: {net:+.5f} = {100*net/(mb-mr):.1f}% of the excess")
print(f"  band beats random control in {sum(1 for a,c in zip(abl,ctl) if a<c)}/{len(abl)} threads")
