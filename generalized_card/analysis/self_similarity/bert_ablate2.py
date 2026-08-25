"""Which component of Reddit markup carries the Self-BERTScore effect."""
from __future__ import annotations
import csv, json, re, sys, time
from pathlib import Path
import numpy as np
REPO=Path("/Users/yaoningyu/Desktop/UIUC/GEO")
sys.path.insert(0,str(REPO/"scripts"/"evaluation"))
from score_thread_semantic_uniformity import load_real_comments, load_generated_comments
from score_thread_self_bertscore import load_bert_scorer
RUN=REPO/"artifacts/generalized_card/runs/generalized_card_camera_gpt54_paper_20260825_v1"
pool=json.load(open(REPO/"artifacts/generalized_card/seed_pools/camera_product_150_seed42.json"))["seed_posts"]
by_seed={int(x["seed_index"]):x for x in pool}

URL=re.compile(r"https?://\S+|\bwww\.\S+")
USER=re.compile(r"\b[ur]/\w+")
EMPH=re.compile(r"[*_`~]")
QUOTE=re.compile(r"&gt;|^\s*>+\s*|\\")
def sub(p): return lambda t: re.sub(r"\s+"," ",p.sub("",t)).strip()

cache={}; threads=[]
for seed,p in sorted(by_seed.items()):
    if seed>=50: break
    d=REPO/"data/raw/discussions/camera_product"/p["source_product_dir"]
    if d not in cache: cache[d]=load_real_comments(d)[0]
    cs=cache[d].get(p["source_raw_post_id"]) or []
    if 8<=len(cs)<=26: threads.append((p["source_raw_post_id"],[c.text for c in cs]))
threads=threads[:20]

# prevalence, real vs generated, over all 50 matched threads
GA,RA=[],[]
for dd in sorted((RUN/"cleaned").glob("run_*_sampled_reddit")):
    cbt,_=load_generated_comments(dd)
    for tid,cs in cbt.items():
        p=by_seed[int(tid.split("seed")[-1])]
        d=REPO/"data/raw/discussions/camera_product"/p["source_product_dir"]
        if d not in cache: cache[d]=load_real_comments(d)[0]
        GA+=[c.text for c in cs]; RA+=[c.text for c in (cache[d].get(p["source_raw_post_id"]) or [])]
print(f"prevalence over all matched comments (real {len(RA)}, generated {len(GA)})")
print(f"{'channel':<26}{'real':>9}{'gen':>9}{'ratio':>9}")
for name,p in (("URL / www",URL),("u/ or r/ mention",USER),("* _ ` ~ emphasis",EMPH),("quote marker / escape",QUOTE)):
    r=np.mean([1.0 if p.search(t) else 0.0 for t in RA]); g=np.mean([1.0 if p.search(t) else 0.0 for t in GA])
    print(f"{name:<26}{r:>9.4f}{g:>9.4f}{(g/r if r else float('nan')):>9.2f}")

scorer,_,dev,mt,nl,_=load_bert_scorer(bert_score_path=REPO/"bert_score-master",
    model_type="microsoft/deberta-xlarge-mnli", num_layers=None, batch_size=8,
    device="auto", idf=False, idf_sents=[], rescale_with_baseline=False, local_files_only=True)
def tf1(texts):
    c,r=[],[]
    for i in range(len(texts)):
        for j in range(i+1,len(texts)): c.append(texts[i]); r.append(texts[j])
    _,_,f1=scorer.score(c,r,batch_size=8)
    return float(np.mean([float(x) for x in f1]))
print(f"\nablation on {len(threads)} real threads (device={dev})")
base=[tf1(t) for _,t in threads]; print(f"{'real (baseline)':<30}{np.mean(base):.4f}")
for name,fn in (("- URL / www",sub(URL)),("- u/ or r/ mention",sub(USER)),
                ("- emphasis chars",sub(EMPH)),("- quote marker / escape",sub(QUOTE))):
    v=[tf1([fn(x) for x in t]) for _,t in threads]
    d=np.mean(v)-np.mean(base)
    print(f"{name:<30}{np.mean(v):.4f}   {d:+.4f}  = {100*d/0.0124:+.0f}% of the gap")
