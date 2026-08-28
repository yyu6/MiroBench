#!/usr/bin/env python3
"""THE DECISIVE TEST for whether any within-thread arm can ever work.

G107 measured a corpus-level offset: cross-thread pairwise BERTScore sat
+0.0205 to +0.0402 above real across seven versions and never moved. If that
holds for v128, then comments from DIFFERENT threads -- sharing no topic, no
plan, no author, no conversation -- are already too similar, and no amount of
rearranging within a thread can fix it. Everything priced at zero today would
then be zero for one reason.

Verifying it independently on the current release, with a like-for-like real
control drawn the same way.
"""
import json, sys, re, itertools, statistics, random
from pathlib import Path
REPO = Path("/Users/yaoningyu/Desktop/UIUC/GEO")
sys.path.insert(0, str(REPO/"scripts"/"evaluation"))
sys.path.insert(0, str(REPO/"bert_score-master"))
from score_thread_semantic_uniformity import load_generated_comments, load_real_comments
rng = random.Random(101)
TOK = re.compile(r"[a-z0-9']+")

def pick(by_thread, per_thread=6, lo=25, hi=140):
    """One sample of comments, at most `per_thread` from any single thread."""
    out=[]
    for tid, cs in by_thread.items():
        ok=[c for c in cs if lo <= len(TOK.findall(c.lower())) <= hi]
        rng.shuffle(ok)
        out += [(tid, t) for t in ok[:per_thread]]
    rng.shuffle(out)
    return out

def cross_pairs(items, n=700):
    """Pairs whose two comments come from DIFFERENT threads."""
    idx=[(i,j) for i in range(len(items)) for j in range(i+1,len(items))
         if items[i][0]!=items[j][0]]
    rng.shuffle(idx)
    return idx[:n]

def within_pairs(items, n=700):
    idx=[(i,j) for i in range(len(items)) for j in range(i+1,len(items))
         if items[i][0]==items[j][0]]
    rng.shuffle(idx)
    return idx[:n]

G={}
for x in sorted((REPO/"artifacts/generalized_card/runs/v128_interaction_n10_20260828_v1/cleaned").glob("run_*_sampled_reddit")):
    cbt,_=load_generated_comments(x)
    for tid,cs in cbt.items(): G[tid]=[" ".join(c.text.split()) for c in cs]

pool=json.loads((REPO/"artifacts/generalized_card/seed_pools/camera_product_150_seed42.json").read_text())["seed_posts"]
used={p["source_raw_post_id"] for p in pool if 2<=int(p["seed_index"])<=11}
R={}
for d in sorted((REPO/"data/raw/discussions/camera_product").iterdir()):
    if not d.is_dir(): continue
    try: bt,_=load_real_comments(d)
    except Exception: continue
    for pid,cs in bt.items():
        if pid in used or len(cs)<12: continue
        R[pid]=[" ".join(c.text.split()) for c in cs]
    if len(R)>=40: break
print(f"generated threads {len(G)}   held-out real threads {len(R)}", flush=True)

from bert_score import BERTScorer
sc=BERTScorer(model_type="microsoft/deberta-xlarge-mnli", num_layers=40, batch_size=32,
              idf=False, device="cpu", lang="en", rescale_with_baseline=False)
def score(items, idx):
    _,_,F=sc.score([items[i][1] for i,_ in idx],[items[j][1] for _,j in idx],batch_size=64)
    return statistics.mean(F.tolist()), statistics.median(F.tolist())

gi=pick(G); ri=pick(R)
print(f"generated comments {len(gi)}   real comments {len(ri)}", flush=True)
for label, items in (("REAL (held out)", ri), ("v128", gi)):
    cp=cross_pairs(items); wp=within_pairs(items)
    cm,_=score(items,cp); wm,_=score(items,wp)
    print(f"{label:18} cross-thread {cm:.4f} (n={len(cp)})   within-thread {wm:.4f} (n={len(wp)})", flush=True)
