"""Does the shaping pipeline cause the uniform level effect?

G103's corpus-level term (+0.0183) is the same "every pair uniformly ~+0.02 too
similar" residual the v118/v119 session found and could not name. One candidate:
every comment passes through eight measured-profile shaping layers (surface
skeleton, tone overlay, opener, closing move, rhythm, register, length
calibration, evaluation tier), and each pulls its output toward that profile's
centre. If that is the cause, CROSS-thread similarity should rise monotonically
as the layers were added across versions.

Cross-thread only: it is the channel free of any topic confound.
"""
import sys,re,random,statistics as st,json
from pathlib import Path
REPO=Path("/Users/yaoningyu/Desktop/UIUC/GEO")
sys.path.insert(0,str(REPO/"scripts"/"evaluation"))
from score_thread_self_bertscore import (DEFAULT_BERT_SCORE_PATH, DEFAULT_MODEL,
    load_bert_scorer, score_pairs_with_device_fallback)
from score_thread_semantic_uniformity import load_generated_comments, load_real_comments
RUNS=REPO/"artifacts/generalized_card/runs"
ORDER=[
 ("v92","generalized_card_camera_gpt54_v92_named_n10_20260818_v1"),
 ("v97","generalized_card_camera_gpt54_v97_keyboard_n10_20260819_v1"),
 ("v101","generalized_card_camera_gpt54_v101_register_n10_20260820_v1"),
 ("v103","generalized_card_camera_gpt54_v103_stance_opening_n10_20260821_v1"),
 ("v108","generalized_card_camera_gpt54_v108_coverage_nonrepeat_n10_20260823_v1"),
 ("v113","v113_v112_gate_n10_20260826_v1"),
 ("v122","v122_writer_retries_n10_20260828_v1"),
 ("v125b","v125b_outsider_n10_20260828_v1"),
]
class C:
    __slots__=("text","comment_id","author","thread_id","parent_id","depth")
    def __init__(s,t,i): s.text=t;s.comment_id=str(i);s.author="";s.thread_id="t";s.parent_id="";s.depth=0
def crossmean(threads, rng, per=6, k=10, cap=600):
    ts=[t for t in threads if len(t)>=8][:k]
    if len(ts)<4: return None,0
    picked=[];origin=[]
    for i,t in enumerate(ts):
        for x in rng.sample(t,min(per,len(t))): picked.append(x); origin.append(i)
    pairs=[(i,j) for i in range(len(picked)) for j in range(i+1,len(picked)) if origin[i]!=origin[j]]
    rng.shuffle(pairs); pairs=pairs[:cap]
    nodes=[C(t,i) for i,t in enumerate(picked)]
    specs=[{"thread_id":"t","left":nodes[i],"right":nodes[j]} for i,j in pairs]
    idf=[s["left"].text for s in specs]+[s["right"].text for s in specs]
    kw=dict(bert_score_path=DEFAULT_BERT_SCORE_PATH,model_type=DEFAULT_MODEL,num_layers=None,
            idf=False,idf_sents=idf,rescale_with_baseline=False,local_files_only=False)
    sc,*_r,fb=load_bert_scorer(batch_size=32,device="cpu",**kw)
    if fb: raise SystemExit("wrong model")
    pr,*_=score_pairs_with_device_fallback(scorer=sc,pair_specs=specs,batch_size=32,
        requested_device="cpu",fallback_used=False,**kw)
    return st.mean([p["bert_f1"] for p in pr]), len(pr)
real_threads=[]
for pdir in sorted((REPO/"data/raw/discussions/camera_product").glob("*")):
    if not pdir.is_dir(): continue
    try: th,_=load_real_comments(pdir)
    except Exception: continue
    for cl in th.values():
        if len(cl)>=8: real_threads.append([c.text for c in cl])
rng=random.Random(20260827); rng.shuffle(real_threads)
rm,rn=crossmean(real_threads, random.Random(20260827))
print(f"{'version':<8}{'cross-thread mean':>20}{'vs real':>10}{'pairs':>8}")
print(f"{'REAL':<8}{rm:>20.4f}{'--':>10}{rn:>8}")
for name,tag in ORDER:
    d=RUNS/tag
    if not d.exists(): print(f"{name:<8}{'(missing)':>20}"); continue
    gen={}
    for sd in sorted(d.glob("cleaned/run_*_sampled_reddit")):
        c,_=load_generated_comments(sd); gen.update(c)
    if not gen:
        for sd in sorted(d.glob("generated/run_*_sampled_reddit")):
            try:
                c,_=load_generated_comments(sd); gen.update(c)
            except Exception: pass
    threads=[[c.text for c in v] for v in gen.values()]
    m,n=crossmean(threads, random.Random(20260827))
    if m is None: print(f"{name:<8}{'(too few threads)':>20}"); continue
    print(f"{name:<8}{m:>20.4f}{m-rm:>+10.4f}{n:>8}",flush=True)
