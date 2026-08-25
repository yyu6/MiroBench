"""Exact per-thread Self-BERTScore effect of URLs, on all 50 matched real threads.

Only pairs touching a URL-bearing comment change, so only those are rescored --
the thread delta is exact, not sampled.
"""
from __future__ import annotations
import csv, json, re, sys, time
from pathlib import Path
import numpy as np
REPO=Path("/Users/yaoningyu/Desktop/UIUC/GEO")
sys.path.insert(0,str(REPO/"scripts"/"evaluation"))
from score_thread_semantic_uniformity import load_real_comments
from score_thread_self_bertscore import load_bert_scorer
SP=(Path(__file__).resolve().parent / "_cache")
pool=json.load(open(REPO/"artifacts/generalized_card/seed_pools/camera_product_150_seed42.json"))["seed_posts"]
URL=re.compile(r"https?://\S+|\bwww\.\S+")
MEDIA=("preview.redd.it","i.redd.it","i.imgur.com","imgur.com","v.redd.it")
def strip_all(t): return re.sub(r"\s+"," ",URL.sub("",t)).strip()
def strip_ref(t): return re.sub(r"\s+"," ",URL.sub(lambda m:"" if not any(h in m.group().lower() for h in MEDIA) else m.group(),t)).strip()

scorer,_,dev,_,_,_=load_bert_scorer(bert_score_path=REPO/"bert_score-master",
    model_type="microsoft/deberta-xlarge-mnli", num_layers=None, batch_size=8,
    device="auto", idf=False, idf_sents=[], rescale_with_baseline=False, local_files_only=True)
print("device", dev, flush=True)

cache={}; out={}
t0=time.time()
for p in pool[:50]:
    d=REPO/"data/raw/discussions/camera_product"/p["source_product_dir"]
    if d not in cache: cache[d]=load_real_comments(d)[0]
    texts=[c.text for c in (cache[d].get(p["source_raw_post_id"]) or [])]
    n=len(texts); total=n*(n-1)//2
    seed=int(p["seed_index"])
    if total==0: out[seed]={"delta_all":0.0,"delta_ref":0.0,"n":n}; continue
    has=[bool(URL.search(t)) for t in texts]
    idx=[(i,j) for i in range(n) for j in range(i+1,n) if has[i] or has[j]]
    if not idx: out[seed]={"delta_all":0.0,"delta_ref":0.0,"n":n}; continue
    o_c=[texts[i] for i,j in idx]; o_r=[texts[j] for i,j in idx]
    a_c=[strip_all(x) for x in o_c]; a_r=[strip_all(x) for x in o_r]
    f_c=[strip_ref(x) for x in o_c]; f_r=[strip_ref(x) for x in o_r]
    _,_,f_o=scorer.score(o_c,o_r,batch_size=8)
    _,_,f_a=scorer.score(a_c,a_r,batch_size=8)
    _,_,f_f=scorer.score(f_c,f_r,batch_size=8)
    da=float(sum(float(a)-float(o) for a,o in zip(f_a,f_o)))/total
    df=float(sum(float(a)-float(o) for a,o in zip(f_f,f_o)))/total
    out[seed]={"delta_all":da,"delta_ref":df,"n":n,"affected_pairs":len(idx),"total_pairs":total}
    print(f"  seed {seed:>3} n={n:>3} affected {len(idx):>5}/{total:>6}  delta_all {da:+.5f}  delta_ref {df:+.5f}  [{time.time()-t0:.0f}s]", flush=True)
json.dump(out, open(SP/"url_delta.json","w"), indent=1)
print(f"done in {time.time()-t0:.0f}s")
