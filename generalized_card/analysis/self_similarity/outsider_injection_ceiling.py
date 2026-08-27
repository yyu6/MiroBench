"""Ceiling test: if 12% of slots really were topical outsiders, is 0.13 reachable?

Two donor sources, because "inject real text" and "inject off-topic text" are
different claims and only the second is a mechanism we could build:
  real  -- real comments from OTHER products (upper bound on realism)
  self  -- our OWN generated comments from other threads (off-topic, our prose)
If `self` closes the gap, a first-pass outsider schedule is worth paying for.
"""
import json, sys, re, csv, math, random, collections, statistics as st
from pathlib import Path
import numpy as np
REPO = Path("/Users/yaoningyu/Desktop/UIUC/GEO")
sys.path.insert(0, str(REPO / "scripts" / "evaluation"))
from score_thread_self_bertscore import (DEFAULT_BERT_SCORE_PATH, DEFAULT_MODEL,
    load_bert_scorer, score_pairs_with_device_fallback)
from score_thread_semantic_uniformity import load_generated_comments, load_real_comments
from score_thread_self_bleu import symmetric_pair_bleu, tokenize

RUNS = REPO/"artifacts/generalized_card/runs"
BASE = "v122_writer_retries_n10_20260828_v1"
POOL = json.load(open(REPO/"artifacts/generalized_card/seed_pools/camera_product_150_seed42.json"))
BY = {int(r["seed_index"]):r for r in POOL["seed_posts"]}
SEED = re.compile(r"seed(\d+)$")
def norm(t): return " ".join(str(t or "").split())[:120]
OUT_SHARE, OUT_LONG_SHARE, LONG_WORDS = 0.12, 0.30, 40

def harvest(pats):
    lab={}
    for pat, root in pats:
        for p in sorted(root.glob(pat)):
            for t in json.load(open(p)).get("threads", []):
                for c in t.get("comments", []):
                    k=norm(c.get("text"))
                    if not k: continue
                    d=lab.setdefault(k,{})
                    nm=p.name
                    if nm.startswith("politeness"): d["tone"]=c.get("pred_label")
                    elif nm.startswith("storyseeker"):
                        v=c.get("story_probability",c.get("probability"))
                        if v is not None: d["story"]=float(v)
                    elif nm.startswith("go_emotions"):
                        e=c.get("dominant_emotion") or c.get("label")
                        if e: d["emo"]=e
    return lab
F=["politeness_results.json","storyseeker_results.json","go_emotions_results.json"]
GEN=harvest([(f"cleaned/run_*_sampled_reddit/{f}",RUNS/BASE) for f in F])
REALL=harvest([(f"*/{f}",REPO/"data/raw/discussions/camera_product") for f in F])

REALROW={}
with open(RUNS/BASE/"matched_evaluation/matched_real_thread_scores.csv") as fh:
    for r in csv.DictReader(fh): REALROW[r["thread_id"]]=r

gen={}
for sd in sorted((RUNS/BASE).glob("cleaned/run_*_sampled_reddit")):
    c,_=load_generated_comments(sd); gen.update(c)

class C:
    __slots__=("text","comment_id","author","thread_id","parent_id","depth")
    def __init__(s,t,i): s.text=t;s.comment_id=str(i);s.author="";s.thread_id="t";s.parent_id="";s.depth=0
def pair_means(texts):
    n=len(texts); nodes=[C(t,i) for i,t in enumerate(texts)]
    specs=[{"thread_id":"t","left":nodes[i],"right":nodes[j]} for i in range(n) for j in range(i+1,n)]
    idf=[s["left"].text for s in specs]+[s["right"].text for s in specs]
    kw=dict(bert_score_path=DEFAULT_BERT_SCORE_PATH,model_type=DEFAULT_MODEL,num_layers=None,
            idf=False,idf_sents=idf,rescale_with_baseline=False,local_files_only=False)
    sc,*_r,fb=load_bert_scorer(batch_size=32,device="cpu",**kw)
    if fb: raise SystemExit("wrong bertscore model")
    pr,*_=score_pairs_with_device_fallback(scorer=sc,pair_specs=specs,batch_size=32,
        requested_device="cpu",fallback_used=False,**kw)
    b=st.mean([p["bert_f1"] for p in pr])
    tok=[tokenize(t) for t in texts]
    l4=st.mean([symmetric_pair_bleu(tok[i],tok[j],4) for i in range(n) for j in range(i+1,n)])
    return b,l4
def lcv(texts):
    L=[len((t or "").split()) for t in texts]
    a=sum(L)/len(L)
    return math.sqrt(sum((v-a)**2 for v in L)/len(L))/a if a>0 else 0.0
def labels(texts,src):
    tn=[src.get(norm(t),{}).get("tone") for t in texts]; tn=[x for x in tn if x]
    so=[src.get(norm(t),{}).get("story") for t in texts]; so=[x for x in so if x is not None]
    em=[src.get(norm(t),{}).get("emo") for t in texts]; em=[x for x in em if x]
    c=collections.Counter(tn); n=max(1,len(tn)); ec=collections.Counter(em); en=max(1,len(em))
    H=-sum((v/en)*math.log(v/en) for v in ec.values()) if ec else 0.0
    return dict(polite=c["polite"]/n,impolite=c["impolite"]/n,neutral=c["neutral"]/n,
                story=st.mean(so) if so else float("nan"),emoH=H)

# donor pools
real_by_product={}
for pdir in sorted((REPO/"data/raw/discussions/camera_product").glob("*")):
    if not pdir.is_dir(): continue
    try: th,_=load_real_comments(pdir)
    except Exception: continue
    real_by_product[pdir.name]=[c.text for cl in th.values() for c in cl]

rows=[]
rng=random.Random(20260827)
real_cache={}
for tid in sorted(gen):
    seed=BY[int(SEED.search(tid).group(1))]
    pdname=str(seed["source_product_dir"]); rid=str(seed["source_raw_post_id"])
    if rid not in REALROW: continue
    base=[c.text for c in gen[tid]]
    n=len(base)
    if n<6 or n>70: continue
    tgt=REALROW[rid]
    k=max(1,round(n*OUT_SHARE)); klong=max(1,round(k*OUT_LONG_SHARE))
    idx=list(range(n)); rng.shuffle(idx)
    long_slots=[i for i in idx if len(base[i].split())>=LONG_WORDS][:klong]
    rest=[i for i in idx if i not in long_slots][:k-len(long_slots)]
    picks=long_slots+rest
    variants={}
    for src in ("real","self"):
        if src=="real":
            donors=[t for name,ts in real_by_product.items() if name!=pdname for t in ts]
        else:
            donors=[c.text for o,cl in gen.items() if o!=tid for c in cl]
        longd=[t for t in donors if len(t.split())>=LONG_WORDS]
        shortd=[t for t in donors if len(t.split())<LONG_WORDS]
        rr=random.Random(hash((tid,src))%10**6)
        new=list(base)
        for i in picks:
            pooled = longd if i in long_slots else shortd
            new[i]=rr.choice(pooled)
        variants[src]=new
    out={"tid":tid,"n":n,"k":k,"klong":len(long_slots)}
    bb,bl=pair_means(base)
    out["base"]={"bert":bb,"bleu4":bl,"lcv":lcv(base),**labels(base,GEN)}
    for src,texts in variants.items():
        vb,vl=pair_means(texts)
        src_lab = {**GEN, **REALL} if src=="real" else GEN
        out[src]={"bert":vb,"bleu4":vl,"lcv":lcv(texts),**labels(texts,src_lab)}
    out["target"]={"bert":float(tgt["self_bertscore_mean_f1"]),"bleu4":float(tgt["self_bleu_4"]),
                   "story":float(tgt["mean_story_probability"]),"emoH":float(tgt["emotion_entropy"]),
                   "polite":float(tgt["polite_rate"]),"impolite":float(tgt["impolite_rate"]),
                   "neutral":float(tgt["neutral_rate"])}
    rows.append(out)
    print(f"{tid:<34} n={n:>3} k={k:>2}(long {len(long_slots)})  "
          f"bert real={out['target']['bert']:.4f} base={bb:.4f} "
          f"+real={out['real']['bert']:.4f} +self={out['self']['bert']:.4f}",flush=True)
json.dump(rows,open(Path(__file__).with_name("o1.json"),"w"))
print("\n%-16s %10s %10s %10s"%("metric","base","+real","+self"))
for key in ("bert","bleu4","story","emoH","polite","impolite","neutral"):
    def dev(src):
        v=[100*(r[src][key]-r["target"][key])/r["target"][key] for r in rows
           if r["target"].get(key) and r[src].get(key)==r[src].get(key)]
        return st.mean(v) if v else float("nan")
    print("%-16s %+9.2f%% %+9.2f%% %+9.2f%%"%(key,dev("base"),dev("real"),dev("self")))
