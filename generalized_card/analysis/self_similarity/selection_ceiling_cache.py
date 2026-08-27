"""Cache everything selection needs, so objective variants cost nothing.

Every metric that selection can move is either a pairwise mean (self-BERTScore,
Self-BLEU-4, semantic cosine) or a per-comment scalar/label (words, tone, story,
emotion). Score each candidate ONCE into matrices + feature vectors and the
whole search becomes arithmetic -- a new objective is then a re-run of seconds,
not of the BERTScore model.
"""
import json, sys, re, csv, math, collections
from pathlib import Path
import numpy as np
REPO = Path("/Users/yaoningyu/Desktop/UIUC/GEO")
sys.path.insert(0, str(REPO / "scripts" / "evaluation"))
from score_thread_self_bertscore import (DEFAULT_BERT_SCORE_PATH, DEFAULT_MODEL,
    load_bert_scorer, score_pairs_with_device_fallback)
from score_thread_semantic_uniformity import load_generated_comments, load_real_comments
from score_thread_self_bleu import symmetric_pair_bleu, tokenize

RUNS = REPO/"artifacts/generalized_card/runs"
ARMS = {"v122":"v122_writer_retries_n10_20260828_v1","v124":"v124_planledger_n10_20260828_v1",
        "v125":"v125_outsider_n10_20260828_v1"}
POOL = json.load(open(REPO/"artifacts/generalized_card/seed_pools/camera_product_150_seed42.json"))
BY = {int(r["seed_index"]):r for r in POOL["seed_posts"]}
SEED = re.compile(r"seed(\d+)$")
def norm(t): return " ".join(str(t or "").split())[:120]

# ---- per-comment labels + embeddings, joined by text (no model re-run) -------
def harvest(globs):
    lab = {}
    for pat, root in globs:
        for p in sorted(root.glob(pat)):
            kind = p.name
            for t in json.load(open(p)).get("threads", []):
                for c in t.get("comments", []):
                    k = norm(c.get("text"))
                    if not k: continue
                    d = lab.setdefault(k, {})
                    if kind.startswith("politeness"): d["tone"] = c.get("pred_label")
                    elif kind.startswith("storyseeker"):
                        v = c.get("story_probability", c.get("probability"))
                        if v is not None: d["story"] = float(v)
                    elif kind.startswith("go_emotions"):
                        e = c.get("dominant_emotion") or c.get("label")
                        if e: d["emo"] = e
                    elif kind.startswith("semantic"):
                        e = c.get("embedding")
                        if e: d["emb"] = e
    return lab
FILES = ["politeness_results.json","storyseeker_results.json","go_emotions_results.json",
         "semantic_uniformity_results.json"]
GEN = harvest([(f"cleaned/run_*_sampled_reddit/{f}", RUNS/tag) for tag in ARMS.values() for f in FILES])
REAL = harvest([(f"*/{f}", REPO/"data/raw/discussions/camera_product") for f in FILES])
print(f"labels: gen={len(GEN)} real={len(REAL)}", flush=True)

# ---- real thread targets straight from the matched score table --------------
REALROW = {}
with open(RUNS/ARMS["v122"]/"matched_evaluation/matched_real_thread_scores.csv") as fh:
    for r in csv.DictReader(fh): REALROW[r["thread_id"]] = r

def load(tag):
    o = {}
    for sd in sorted((RUNS/tag).glob("cleaned/run_*_sampled_reddit")):
        c, _ = load_generated_comments(sd); o.update(c)
    return o
pools = {k: load(t) for k, t in ARMS.items()}

class C:
    __slots__=("text","comment_id","author","thread_id","parent_id","depth")
    def __init__(s,t,i): s.text=t;s.comment_id=str(i);s.author="";s.thread_id="t";s.parent_id="";s.depth=0

def bert_matrix(texts):
    n=len(texts); nodes=[C(t,i) for i,t in enumerate(texts)]
    specs=[{"thread_id":"t","left":nodes[i],"right":nodes[j]} for i in range(n) for j in range(i+1,n)]
    idf=[s["left"].text for s in specs]+[s["right"].text for s in specs]
    kw=dict(bert_score_path=DEFAULT_BERT_SCORE_PATH,model_type=DEFAULT_MODEL,num_layers=None,
            idf=False,idf_sents=idf,rescale_with_baseline=False,local_files_only=False)
    sc,*_r,fb=load_bert_scorer(batch_size=32,device="cpu",**kw)
    if fb: raise SystemExit("wrong bertscore model")
    pr,*_=score_pairs_with_device_fallback(scorer=sc,pair_specs=specs,batch_size=32,
        requested_device="cpu",fallback_used=False,**kw)
    M=np.zeros((n,n))
    for s,p in zip(specs,pr):
        i,j=int(s["left"].comment_id),int(s["right"].comment_id); M[i,j]=M[j,i]=p["bert_f1"]
    return M

def bleu_matrix(texts):
    tok=[tokenize(t) for t in texts]; n=len(tok); M=np.zeros((n,n))
    for i in range(n):
        for j in range(i+1,n): M[i,j]=M[j,i]=symmetric_pair_bleu(tok[i],tok[j],4)
    return M

def cos_matrix(texts, src):
    E=[]
    miss=0
    for t in texts:
        e=src.get(norm(t),{}).get("emb")
        if e is None: miss+=1; e=[0.0]*768
        E.append(e)
    A=np.array(E,dtype=float); nn=np.linalg.norm(A,axis=1,keepdims=True); nn[nn==0]=1
    A=A/nn; M=A@A.T; np.fill_diagonal(M,0.0)
    return M, miss

def length_cv(texts):
    L=[len((t or "").split()) for t in texts]
    if not L: return 0.0
    a=sum(L)/len(L)
    if a<=0: return 0.0
    return math.sqrt(sum((v-a)**2 for v in L)/len(L))/a

out={"threads":[]}
real_cache={}
for tid in sorted(pools["v122"]):
    seed=BY[int(SEED.search(tid).group(1))]
    pd=REPO/"data/raw/discussions/camera_product"/str(seed["source_product_dir"])
    if pd not in real_cache: real_cache[pd]=load_real_comments(pd)[0]
    rid=str(seed["source_raw_post_id"])
    if rid not in real_cache[pd] or rid not in REALROW: continue
    byid={}
    for arm in ARMS:
        for c in pools[arm].get(tid,[]): byid.setdefault(c.comment_id,{})[arm]=c.text
    slots=[k for k,v in byid.items() if len(v)==3]
    if len(slots)<6 or len(slots)>70: continue
    cand=[];owner=[];armof=[]
    for k in slots:
        for a in ARMS: cand.append(byid[k][a]); owner.append(k); armof.append(a)
    B=bert_matrix(cand); L4=bleu_matrix(cand); Cs,miss=cos_matrix(cand,GEN)
    feat=[]
    for t in cand:
        d=GEN.get(norm(t),{})
        feat.append({"w":len((t or "").split()),"tone":d.get("tone"),
                     "story":d.get("story"),"emo":d.get("emo")})
    cover=sum(1 for f in feat if f["tone"]) / len(feat)
    r=REALROW[rid]
    rt=[c.text for c in real_cache[pd][rid]]
    target={"bert":float(r["self_bertscore_mean_f1"]),"bleu4":float(r["self_bleu_4"]),
            "cos":float(r["semantic_mean_cosine"]),"story":float(r["mean_story_probability"]),
            "emoH":float(r["emotion_entropy"]),"polite":float(r["polite_rate"]),
            "impolite":float(r["impolite_rate"]),"neutral":float(r["neutral_rate"]),
            "lcv":length_cv(rt)}
    out["threads"].append({"tid":tid,"rid":rid,"slots":slots,"owner":owner,"arm":armof,
        "B":B.tolist(),"L4":L4.tolist(),"C":Cs.tolist(),"feat":feat,"target":target,
        "label_cover":cover,"emb_missing":miss,"texts":cand})
    print(f"{tid:<34} slots={len(slots):>3} cand={len(cand):>4} cover={cover:.2f} embmiss={miss}",flush=True)
cache = Path(__file__).with_name("_cache")
cache.mkdir(exist_ok=True)
json.dump(out, open(cache / "selection_ceiling_cache.json", "w"))
print("cached", len(out["threads"]), "threads")
