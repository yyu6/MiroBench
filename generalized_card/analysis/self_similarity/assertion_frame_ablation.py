"""Controlled ablation: remove the assertion-frame bigrams, with a matched random control."""
from __future__ import annotations
import json, sys, re, itertools, random, statistics
from pathlib import Path
REPO=Path("/Users/yaoningyu/Desktop/UIUC/GEO")
sys.path.insert(0,str(REPO/"scripts"/"evaluation"))
from score_thread_semantic_uniformity import load_generated_comments, load_real_comments
TOK=re.compile(r"[a-z0-9']+")
rng=random.Random(0)
pool=json.load(open(REPO/"artifacts/generalized_card/seed_pools/camera_product_150_seed42.json"))["seed_posts"]
by_seed={int(p["seed_index"]):p for p in pool}
FRAME=re.compile(r"\b(?:is|are|isn't|aren't)\s+the\b|\bthat's\s+the\b|\bthe\s+(?:real|only|actual|whole|key|main)\b",re.I)

def self_bleu2(texts):
    """mean over ordered pairs of clipped bigram precision (the p2 term)."""
    G=[]
    for t in texts:
        w=TOK.findall(t.lower()); G.append(list(zip(w,w[1:])))
    G=[g for g in G if g]
    if len(G)<2: return None
    from collections import Counter
    idx=[(i,j) for i in range(len(G)) for j in range(len(G)) if i!=j]
    if len(idx)>4000: idx=rng.sample(idx,4000)
    vals=[]
    for i,j in idx:
        c=Counter(G[i]); r=Counter(G[j])
        clip=sum(min(v,r[k]) for k,v in c.items())
        vals.append((clip+1)/(len(G[i])+1))
    return statistics.mean(vals)

def strip_frames(t):
    return FRAME.sub(" ", t)

def strip_random_matched(t, n_removed_words):
    w=t.split()
    if n_removed_words<=0 or len(w)<=n_removed_words: return t
    drop=set(rng.sample(range(len(w)),n_removed_words))
    return " ".join(x for i,x in enumerate(w) if i not in drop)

RUN=REPO/"artifacts/generalized_card/runs/generalized_card_camera_gpt54_paper_20260825_v1/cleaned"
cache={}; base=[];abl=[];ctrl=[];real=[]
n=0
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
        gt=[c.text for c in cs]; rt=[c.text for c in rcs]
        gs=[strip_frames(t) for t in gt]
        removed=[max(0,len(a.split())-len(b.split())) for a,b in zip(gt,gs)]
        gc=[strip_random_matched(t,k) for t,k in zip(gt,removed)]
        b=self_bleu2(gt); a=self_bleu2(gs); c=self_bleu2(gc); r=self_bleu2(rt)
        if None in (b,a,c,r): continue
        base.append(b); abl.append(a); ctrl.append(c); real.append(r)
        n+=1
        if n>=30: break
    if n>=30: break
mb,ma,mc,mr=map(statistics.mean,(base,abl,ctrl,real))
print(f"threads: {n}")
print(f"  REAL   self-bleu2-ish  {mr:.5f}")
print(f"  OURS   base            {mb:.5f}   excess over real {mb-mr:+.5f}  ({100*(mb-mr)/mr:+.1f}%)")
print(f"  OURS   frames removed  {ma:.5f}   excess {ma-mr:+.5f}  ({100*(ma-mr)/mr:+.1f}%)")
print(f"  OURS   random control  {mc:.5f}   excess {mc-mr:+.5f}  ({100*(mc-mr)/mr:+.1f}%)")
print()
frame_effect=(mb-ma)-(mb-mc)
print(f"  raw drop from removing frames : {mb-ma:+.5f}")
print(f"  drop from removing SAME MASS randomly : {mb-mc:+.5f}")
print(f"  NET frame-specific effect     : {frame_effect:+.5f}   = {100*frame_effect/(mb-mr):.1f}% of the excess")
w=sum(1 for a_,c_ in zip(abl,ctrl) if a_<c_)
print(f"  frames beat the random control in {w}/{n} threads")
