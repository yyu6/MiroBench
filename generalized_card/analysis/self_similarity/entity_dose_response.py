#!/usr/bin/env python3
"""Price 'spread the entity vocabulary' on self_bertscore (G136's rule).

Same construction design as G139: threads built from GENUINE real comments,
nothing synthesised, tokens held constant, hill-climbed to a target pairwise
ENTITY Jaccard spanning real's 0.192 and our 0.454. Then read off what moving
0.454 -> 0.192 is worth on self_bertscore.

The v109 A/B cannot answer this -- it is a single-seed run and G76 shows one
thread's run-to-run noise (sd 2.94%) exceeds the gap.
"""
from __future__ import annotations
import json, re, sys, random, statistics, argparse
from pathlib import Path
REPO = Path("/Users/yaoningyu/Desktop/UIUC/GEO")
sys.path.insert(0, str(REPO/"scripts"/"evaluation"))
from score_thread_semantic_uniformity import load_real_comments
TOK = re.compile(r"[a-z0-9']+")
WORD = re.compile(r"[A-Za-z][A-Za-z0-9'\-]*|\d+[A-Za-z]+|[A-Za-z]+\d+")
BRANDS = set("""sony canon nikon fuji fujifilm ricoh olympus panasonic pentax leica sigma tamron
zeiss samyang gr rx100 a7 a7r a7s a6000 a6300 a6500 xt x100f x100 x100v eos rebel d850 5d 6d 7d
gh5 om em1 em5 z6 z7 r5 r6 gx a9 nex""".split())
MODEL = re.compile(r"^(?:[a-z]{1,3}\d{1,4}[a-z]{0,3}|\d{1,3}[-/]\d{1,3}(?:\.\d)?|\d{1,3}mm|f/?\d(?:\.\d)?)$")
rng = random.Random(23)

def ents(t):
    return {w.lower() for w in WORD.findall(t) if w.lower() in BRANDS or MODEL.match(w.lower())}
def toks(ts): return sum(len(TOK.findall(t.lower())) for t in ts)
def ent_jac(ts):
    S=[e for e in (ents(t) for t in ts) if e]
    if len(S)<4: return None
    v=[]
    for i in range(len(S)):
        for j in range(i+1,len(S)):
            u=S[i]|S[j]
            if u: v.append(len(S[i]&S[j])/len(u))
    return statistics.mean(v) if v else None

def build(pool, target, n=40, lo=2100, hi=2700, iters=6000):
    cur=None
    for _ in range(400):
        c=rng.sample(pool,n)
        if lo<=toks(c)<=hi and ent_jac(c) is not None: cur=c; break
    if cur is None: return None
    best=abs(ent_jac(cur)-target)
    for _ in range(iters):
        i=rng.randrange(n); cand=rng.choice(pool)
        if cand in cur: continue
        tr=list(cur); tr[i]=cand
        if not (lo<=toks(tr)<=hi): continue
        j=ent_jac(tr)
        if j is None: continue
        d=abs(j-target)
        if d<=best: best,cur=d,tr
        if best<0.004: break
    return cur

ap=argparse.ArgumentParser(); ap.add_argument("--targets",default="0.15,0.25,0.35,0.45"); ap.add_argument("--reps",type=int,default=4)
a=ap.parse_args()
pool=[]
for d in sorted((REPO/"data/raw/discussions/camera_product").iterdir()):
    if not d.is_dir(): continue
    try: bt,_=load_real_comments(d)
    except Exception: continue
    for cs in bt.values():
        for c in cs:
            n=len(TOK.findall(c.text.lower()))
            if 20<=n<=150: pool.append(" ".join(c.text.split()))
pool=sorted(set(pool))
print(f"pool {len(pool):,} real comments", flush=True)
specs=[]
for t in [float(x) for x in a.targets.split(",")]:
    for r in range(a.reps):
        th=build(pool,t)
        if not th: print(f"  target={t} rep={r}: FAILED"); continue
        specs.append({"target":t,"rep":r,"texts":th,"ent_jac":ent_jac(th),"tokens":toks(th),
                      "distinct_ents":len({e for x in th for e in ents(x)})})
        print(f"  target={t} rep={r}: ent_jac={specs[-1]['ent_jac']:.3f} "
              f"distinct={specs[-1]['distinct_ents']} tok={specs[-1]['tokens']}", flush=True)
out="/private/tmp/claude-501/-Users-yaoningyu-Desktop-UIUC-GEO/d8816651-1679-43a5-8d4b-21a1a35e5936/scratchpad/ent_dose_threads.json"
Path(out).write_text(json.dumps(specs)); print(f"\nwrote {len(specs)} -> {out}")
