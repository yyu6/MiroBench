#!/usr/bin/env python3
"""Price the register defect on self_bertscore.

The discriminator (AUC 0.981 on ONE comment) names the tells. Controlled
ablation on OUR OWN text: delete the tell tokens, delete the same token mass
at random, compare both against the matched real thread. This is the project's
standard instrument (G63, G127). It bounds the token-presence channel; per G80
it is NOT a prediction of what an arm that stops generating them would do.
"""
import json, sys, re, itertools, statistics, random
from pathlib import Path
REPO = Path("/Users/yaoningyu/Desktop/UIUC/GEO")
sys.path.insert(0, str(REPO/"scripts"/"evaluation"))
sys.path.insert(0, str(REPO/"bert_score-master"))
from score_thread_semantic_uniformity import load_generated_comments, load_real_comments
rng = random.Random(17)
TOK = re.compile(r"[A-Za-z0-9']+")

# top generated-ish unigrams from the held-out discriminator (disc.py)
TELLS = set("""check here actually body order still real feels feel whole review email
subject normal setup clip makes thanks matters inbox grip lot kind mine changes
pretty fast that""".split())

def strip_tells(text):
    out=[]
    for w in text.split():
        core=TOK.sub(lambda m:m.group(0), w)
        if TOK.findall(w) and TOK.findall(w)[0].lower() in TELLS: continue
        out.append(w)
    return " ".join(out)

def strip_random(text, k):
    ws=text.split()
    if k<=0 or len(ws)<=k: return text
    drop=set(rng.sample(range(len(ws)), k))
    return " ".join(w for i,w in enumerate(ws) if i not in drop)

from bert_score import BERTScorer
sc=BERTScorer(model_type="microsoft/deberta-xlarge-mnli", num_layers=40, batch_size=32,
              idf=False, device="cpu", lang="en", rescale_with_baseline=False)
def mf1(t):
    pr=list(itertools.combinations(range(len(t)),2))
    _,_,F=sc.score([t[i] for i,_ in pr],[t[j] for _,j in pr],batch_size=64)
    return statistics.mean(F.tolist())

pool=json.loads((REPO/"artifacts/generalized_card/seed_pools/camera_product_150_seed42.json").read_text())["seed_posts"]
by={int(p["seed_index"]):p for p in pool}
G={}
for x in sorted((REPO/"artifacts/generalized_card/runs/v128_interaction_n10_20260828_v1/cleaned").glob("run_*_sampled_reddit")):
    cbt,_=load_generated_comments(x)
    for tid,cs in cbt.items(): G[int(tid.split("seed")[-1])]=[" ".join(c.text.split()) for c in cs]
rows=[]
for S in sorted(G):
    p=by[S]
    real=[" ".join(c.text.split()) for c in (load_real_comments(REPO/"data/raw/discussions/camera_product"/p["source_product_dir"])[0].get(p["source_raw_post_id"]) or [])][:40]
    ours=G[S][:40]
    if len(real)<12 or len(ours)<12: continue
    abl=[strip_tells(t) for t in ours]
    ks=[len(o.split())-len(a.split()) for o,a in zip(ours,abl)]
    ctl=[strip_random(o,k) for o,k in zip(ours,ks)]
    b,a,c,r=mf1(ours),mf1(abl),mf1(ctl),mf1(real)
    rows.append((S,sum(ks),b,a,c,r))
    print(f"seed{S:<3} removed {sum(ks):>4} words  ours {b:.4f} -> minus-tells {a:.4f} "
          f"(random ctl {c:.4f})   real {r:.4f}", flush=True)
B,A,C,R=[statistics.mean(x[i] for x in rows) for i in (2,3,4,5)]
net=(B-A)-(B-C)
print(f"\n{len(rows)} threads, {statistics.mean(x[1] for x in rows):.0f} words removed/thread")
print(f"  real                 {R:.4f}")
print(f"  ours as generated    {B:.4f}   excess {B-R:+.4f}")
print(f"  ours minus tells     {A:.4f}   excess {A-R:+.4f}")
print(f"  ours minus random    {C:.4f}   excess {C-R:+.4f}   <- control")
print(f"\n  NET tell-specific effect: {net:+.5f} = {100*net/(B-R):.0f}% of the excess")
print(f"  beats control in {sum(1 for x in rows if x[3]<x[4])}/{len(rows)} threads")
