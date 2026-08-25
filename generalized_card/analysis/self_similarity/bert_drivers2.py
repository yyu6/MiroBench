"""Do the surface-register features predict self_bertscore inside real threads?"""
from __future__ import annotations
import csv, json, math, random, re, sys
from pathlib import Path
import numpy as np
REPO=Path("/Users/yaoningyu/Desktop/UIUC/GEO")
sys.path.insert(0,str(REPO/"scripts"/"evaluation"))
from score_thread_self_bleu import tokenize
from score_thread_semantic_uniformity import load_real_comments, load_generated_comments
MARKUP=re.compile(r"https?://|\bwww\.|\bu/\w|\br/\w|[*_`~]|&gt;|^\s*>|\\|\[[^\]]*\]\(")
rng=random.Random(0)
KEYS=["log_tokens","log_n","comma_free","sent_per_comment","markup","punct_per_word",
      "sd_comma_free","jaccard_all","ttr"]

def feats(texts):
    toks=[tokenize(t) for t in texts]; keep=[i for i,t in enumerate(toks) if t]
    texts=[texts[i] for i in keep]; toks=[toks[i] for i in keep]
    n=len(toks)
    if n<2: return None
    sets=[set(t) for t in toks]
    idx=[(i,j) for i in range(n) for j in range(i+1,n)]
    if len(idx)>4000: idx=rng.sample(idx,4000)
    jac=sum(len(sets[i]&sets[j])/max(1,len(sets[i]|sets[j])) for i,j in idx)/len(idx)
    pooled=[w for t in toks for w in t]
    cf=[1.0 if t.count(",")==0 else 0.0 for t in texts]
    sents=[max(1,len(re.findall(r"[.!?]+",t))) for t in texts]
    return {
        "log_tokens": math.log(sum(len(t) for t in toks)/n),
        "log_n": math.log(n),
        "comma_free": float(np.mean(cf)),
        "sent_per_comment": float(np.mean(sents)),
        "markup": float(np.mean([1.0 if MARKUP.search(t) else 0.0 for t in texts])),
        "punct_per_word": float(np.mean([sum(1 for c in t if c in ",.;:!?()\"'-—…")/max(1,len(t.split())) for t in texts])),
        "sd_comma_free": float(np.std(cf)),
        "jaccard_all": jac,
        "ttr": len(set(pooled))/len(pooled),
    }

rows={r["thread_id"]:r for r in csv.DictReader(open(REPO/"artifacts/baselines/camera_product_gpt5mini/real/thread_scores.csv"))}
real_f={}
for d in sorted((REPO/"data/raw/discussions/camera_product").iterdir()):
    if not d.is_dir(): continue
    try: cbt,_=load_real_comments(d)
    except Exception: continue
    for tid,cs in cbt.items():
        f=feats([c.text for c in cs])
        if f: real_f[tid]=f
ids=[t for t in real_f if t in rows and float(rows[t]["comment_count"])>=2]
X=np.array([[real_f[t][k] for k in KEYS] for t in ids])
Z=(X-X.mean(0))/X.std(0)
A=np.column_stack([np.ones(len(Z)),Z])
print(f"real threads: {len(ids)}\n")
for name,y in (("self_bertscore_mean_f1",np.array([float(rows[t]["self_bertscore_mean_f1"]) for t in ids])),
               ("self_bleu_4",np.array([float(rows[t]["self_bleu_4"]) for t in ids]))):
    beta,*_=np.linalg.lstsq(A,y,rcond=None); pred=A@beta
    r2=1-((y-pred)**2).sum()/((y-y.mean())**2).sum()
    print(f"{name}   R^2={r2:.3f}")
    for i in sorted(range(len(KEYS)),key=lambda i:-abs(beta[i+1])):
        print(f"   {KEYS[i]:<18} beta={beta[i+1]:+.5f}  simple r={float(np.corrcoef(X[:,i],y)[0,1]):+.3f}")
    print()

RUN=REPO/"artifacts/generalized_card/runs/generalized_card_camera_gpt54_paper_20260825_v1"
pool=json.load(open(REPO/"artifacts/generalized_card/seed_pools/camera_product_150_seed42.json"))["seed_posts"]
by_seed={int(x["seed_index"]):x for x in pool}
gp=[]
cache={}
for dd in sorted((RUN/"cleaned").glob("run_*_sampled_reddit")):
    cbt,_=load_generated_comments(dd)
    for tid,cs in cbt.items():
        p=by_seed[int(tid.split("seed")[-1])]
        rid=p["source_raw_post_id"]
        if rid not in real_f: continue
        f=feats([c.text for c in cs])
        if f: gp.append((f,real_f[rid]))
print(f"matched pairs: {len(gp)}")
print(f"{'feature':<20}{'real':>10}{'gen':>10}{'gap in real SD':>16}{'predicted bertscore effect':>28}")
ybs=np.array([float(rows[t]["self_bertscore_mean_f1"]) for t in ids])
beta,*_=np.linalg.lstsq(A,ybs,rcond=None)
tot=0.0
for i,k in enumerate(KEYS):
    rv=np.mean([p[1][k] for p in gp]); gv=np.mean([p[0][k] for p in gp]); sd=X[:,i].std()
    eff=beta[i+1]*(gv-rv)/sd; tot+=eff
    print(f"{k:<20}{rv:>10.4f}{gv:>10.4f}{(gv-rv)/sd:>+16.2f}{eff:>+28.5f}")
print(f"{'TOTAL predicted':<20}{'':>10}{'':>10}{'':>16}{tot:>+28.5f}   observed gap +0.0124")
