"""What carries the within-thread excess? Mask candidates and re-measure.

G102: within-thread pairwise BERTScore sits +0.031 above our own cross-thread
mean while real's sits -0.005 below its own. Mask one surface channel at a time
on BOTH corpora and see which one collapses our excess. Whatever does is the
channel a first-pass control has to own.
"""
import sys,re,random,statistics as st
from pathlib import Path
REPO=Path("/Users/yaoningyu/Desktop/UIUC/GEO")
sys.path.insert(0,str(REPO/"scripts"/"evaluation"))
from score_thread_self_bertscore import (DEFAULT_BERT_SCORE_PATH, DEFAULT_MODEL,
    load_bert_scorer, score_pairs_with_device_fallback)
from score_thread_semantic_uniformity import load_generated_comments, load_real_comments
RUNS=REPO/"artifacts/generalized_card/runs"
class C:
    __slots__=("text","comment_id","author","thread_id","parent_id","depth")
    def __init__(s,t,i): s.text=t;s.comment_id=str(i);s.author="";s.thread_id="t";s.parent_id="";s.depth=0
MODEL_RE=re.compile(r"\b([A-Z][A-Za-z]*\s?)?[A-Za-z]{1,4}[- ]?\d{1,4}\s?(mark\s?)?([IVXivx]{1,4}|[A-Za-z]{1,3})?\b")
BRAND_RE=re.compile(r"\b(canon|nikon|sony|fuji(film)?|ricoh|olympus|panasonic|leica|pentax|sigma|tamron|gopro|dji)\b",re.I)
CAPS_RE=re.compile(r"\b[A-Z]{2,}\d*\b")
HEDGE_RE=re.compile(r"\b(if you|if your|only if|unless|as long as|depends on|matters more than|more than the|rather than|instead of)\b",re.I)
def mask(t,kind):
    s=str(t or "")
    if kind=="none": return s
    if kind in ("entity","entity+hedge"):
        s=BRAND_RE.sub("BRAND",s); s=CAPS_RE.sub("MODEL",s); s=MODEL_RE.sub("MODEL",s)
    if kind in ("hedge","entity+hedge"):
        s=HEDGE_RE.sub("HEDGE",s)
    return s
def score(pairs_texts):
    nodes=[C(t,i) for i,t in enumerate(pairs_texts[0])]
    specs=[{"thread_id":"t","left":nodes[i],"right":nodes[j]} for i,j in pairs_texts[1]]
    idf=[s["left"].text for s in specs]+[s["right"].text for s in specs]
    kw=dict(bert_score_path=DEFAULT_BERT_SCORE_PATH,model_type=DEFAULT_MODEL,num_layers=None,
            idf=False,idf_sents=idf,rescale_with_baseline=False,local_files_only=False)
    sc,*_r,fb=load_bert_scorer(batch_size=32,device="cpu",**kw)
    if fb: raise SystemExit("wrong model")
    pr,*_=score_pairs_with_device_fallback(scorer=sc,pair_specs=specs,batch_size=32,
        requested_device="cpu",fallback_used=False,**kw)
    return st.mean([p["bert_f1"] for p in pr])
rng=random.Random(20260827)
gen={}
for sd in sorted((RUNS/"v122_writer_retries_n10_20260828_v1").glob("cleaned/run_*_sampled_reddit")):
    c,_=load_generated_comments(sd); gen.update(c)
gen_threads=[[c.text for c in v] for v in gen.values() if len(v)>=8]
real_threads=[]
for pdir in sorted((REPO/"data/raw/discussions/camera_product").glob("*")):
    if not pdir.is_dir(): continue
    try: th,_=load_real_comments(pdir)
    except Exception: continue
    for cl in th.values():
        if len(cl)>=8: real_threads.append([c.text for c in cl])
rng.shuffle(real_threads)
def build(threads,per=6,k=10):
    ts=threads[:k]; picked=[];origin=[]
    for i,t in enumerate(ts):
        for x in rng.sample(t,min(per,len(t))): picked.append(x); origin.append(i)
    within=[(i,j) for i in range(len(picked)) for j in range(i+1,len(picked)) if origin[i]==origin[j]]
    cross =[(i,j) for i in range(len(picked)) for j in range(i+1,len(picked)) if origin[i]!=origin[j]]
    rng.shuffle(cross); cross=cross[:600]
    return picked,within,cross
sets={"real":build(real_threads),"ours":build(gen_threads)}
print(f"{'mask':<14}{'corpus':<7}{'within':>9}{'cross':>9}{'excess':>9}")
for kind in ("none","entity","hedge","entity+hedge"):
    for name,(picked,within,cross) in sets.items():
        m=[mask(t,kind) for t in picked]
        w=score((m,within)); x=score((m,cross))
        print(f"{kind:<14}{name:<7}{w:>9.4f}{x:>9.4f}{w-x:>+9.4f}",flush=True)
