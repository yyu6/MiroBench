"""Causal ablation: move REAL text toward the generator's surface register and
see whether real's Self-BERTScore rises to meet the generator's.

Real text only -- no LLM writing -- so nothing here can be an artifact of my
own prose style.  Reproduces the shipped per-thread number first (E6).
"""
from __future__ import annotations
import csv, json, math, random, re, sys, time
from pathlib import Path
import numpy as np
REPO=Path("/Users/yaoningyu/Desktop/UIUC/GEO")
sys.path.insert(0,str(REPO/"scripts"/"evaluation"))
from score_thread_semantic_uniformity import load_real_comments
from score_thread_self_bertscore import load_bert_scorer

SEED_POOL=json.load(open(REPO/"artifacts/generalized_card/seed_pools/camera_product_150_seed42.json"))["seed_posts"]
by_seed={int(x["seed_index"]):x for x in SEED_POOL}
real_csv={r["thread_id"]:r for r in csv.DictReader(open(REPO/"artifacts/baselines/camera_product_gpt5mini/real/thread_scores.csv"))}

URL=re.compile(r"https?://\S+|\bwww\.\S+")
USER=re.compile(r"\b[ur]/\w+")
FMT=re.compile(r"[*_`~]|&gt;|\\")
SENT=re.compile(r"(?<=[.!?])\s+")

def strip_markup(t: str) -> str:
    t=URL.sub("", t); t=USER.sub("", t); t=FMT.sub("", t)
    t=re.sub(r"^\s*>+\s*", "", t)
    return re.sub(r"\s+"," ",t).strip()

def join_sentences(t: str, rng) -> str:
    parts=[p for p in SENT.split(t) if p.strip()]
    if len(parts)<2: return t
    out=parts[0].rstrip()
    for p in parts[1:]:
        if rng.random() < 0.72 and out.endswith((".",)):
            out=out[:-1]+", "+(p[0].lower()+p[1:] if p[:1].isalpha() else p)
        else:
            out=out+" "+p
    return re.sub(r"\s+"," ",out).strip()

# pick matched real threads of workable size
cache={}
threads=[]
for seed,p in sorted(by_seed.items()):
    if seed>=50: break
    d=REPO/"data/raw/discussions/camera_product"/p["source_product_dir"]
    if d not in cache: cache[d]=load_real_comments(d)[0]
    cs=cache[d].get(p["source_raw_post_id"]) or []
    if 8<=len(cs)<=26:
        threads.append((p["source_raw_post_id"], [c.text for c in cs]))
threads=threads[:20]
pairs_total=sum(len(t)*(len(t)-1)//2 for _,t in threads)
print(f"threads {len(threads)}  comments {sum(len(t) for _,t in threads)}  pairs {pairs_total}")

scorer,h,dev,mt,nl,fb = load_bert_scorer(
    bert_score_path=REPO/"bert_score-master", model_type="microsoft/deberta-xlarge-mnli",
    num_layers=None, batch_size=8, device="auto", idf=False, idf_sents=[],
    rescale_with_baseline=False, local_files_only=True)
print(f"model={mt} layers={nl} device={dev} fallback={fb}")

def thread_f1(texts):
    cands,refs=[],[]
    for i in range(len(texts)):
        for j in range(i+1,len(texts)):
            cands.append(texts[i]); refs.append(texts[j])
    _,_,f1=scorer.score(cands,refs,batch_size=8)
    return float(np.mean([float(x) for x in f1]))

def surface(texts):
    cf=np.mean([1.0 if t.count(",")==0 else 0.0 for t in texts])
    se=np.mean([max(1,len(re.findall(r"[.!?]+",t))) for t in texts])
    mk=np.mean([1.0 if (URL.search(t) or USER.search(t) or FMT.search(t)) else 0.0 for t in texts])
    return cf,se,mk

conds={}
rng=random.Random(7)
for name, fn in (("real (baseline)", lambda ts,_r: ts),
                 ("real, sentences comma-joined", lambda ts,r: [join_sentences(t,r) for t in ts]),
                 ("real, markup stripped", lambda ts,_r: [strip_markup(t) for t in ts]),
                 ("real, both", lambda ts,r: [join_sentences(strip_markup(t),r) for t in ts])):
    t0=time.time(); vals=[]; surf=[]
    r=random.Random(7)
    for tid,texts in threads:
        tt=fn(texts,r)
        vals.append(thread_f1(tt)); surf.append(surface(tt))
    conds[name]=(vals,surf)
    cf=np.mean([s[0] for s in surf]); se=np.mean([s[1] for s in surf]); mk=np.mean([s[2] for s in surf])
    print(f"{name:<32} mean self_bertscore {np.mean(vals):.4f}   comma-free {cf:.3f}  sent/comment {se:.2f}  markup {mk:.3f}   [{time.time()-t0:.0f}s]")

base=conds["real (baseline)"][0]
ship=[float(real_csv[tid]["self_bertscore_mean_f1"]) for tid,_ in threads]
print(f"\n[E6] baseline vs shipped real thread_scores.csv: max |diff| = {max(abs(a-b) for a,b in zip(base,ship)):.5f}")
print(f"     baseline mean {np.mean(base):.4f}   shipped mean {np.mean(ship):.4f}")
print(f"\ngenerated on the full 50 matched threads: 0.5076   real: 0.4952   gap +0.0124")
for name,(vals,_) in conds.items():
    if name=="real (baseline)": continue
    d=np.mean(vals)-np.mean(base)
    print(f"  {name:<32} moves real by {d:+.4f}  = {100*d/0.0124:+.0f}% of the observed generated-vs-real gap")
