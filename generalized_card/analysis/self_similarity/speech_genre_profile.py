#!/usr/bin/env python3
"""Speech-genre profile. The floor read suggests our comments are all the SAME
KIND of utterance even when the topic differs. This counts kinds."""
import json,sys,re,statistics
from pathlib import Path
REPO=Path("/Users/yaoningyu/Desktop/UIUC/GEO")
sys.path.insert(0,str(REPO/"scripts"/"evaluation"))
from score_thread_semantic_uniformity import load_generated_comments, load_real_comments
TOK=re.compile(r"[a-z0-9']+")
pool=json.loads((REPO/"artifacts/generalized_card/seed_pools/camera_product_150_seed42.json").read_text())["seed_posts"]
by={int(p["seed_index"]):p for p in pool}
def load(tag):
    o={}
    for x in sorted((REPO/"artifacts/generalized_card/runs"/tag/"cleaned").glob("run_*_sampled_reddit")):
        cbt,_=load_generated_comments(x)
        for tid,cs in cbt.items(): o[int(tid.split("seed")[-1])]=[c.text for c in cs]
    return o
G=load("v128_interaction_n10_20260828_v1"); H=load("v134_phraseledger_n10_20260828_v1")
cache={}; rows=[]
for s in sorted(set(G)&set(H)):
    p=by[s]; d=REPO/"data/raw/discussions/camera_product"/p["source_product_dir"]
    if d not in cache:
        try: cache[d]=load_real_comments(d)[0]
        except Exception: cache[d]={}
    r=cache[d].get(p["source_raw_post_id"]) or []
    if len(r)<12: continue
    rows.append(([c.text for c in r],G[s],H[s]))

URL=re.compile(r"https?://|www\.")
NUM=re.compile(r"\b\d")
CAPS=re.compile(r"\b[A-Z]{3,}\b")
GENRES={
 "quote-and-react (>)"   : lambda t: t.strip().startswith(">") or "\n>" in t,
 "has a link"            : lambda t: bool(URL.search(t)),
 "one-liner (<12 words)" : lambda t: len(TOK.findall(t))<12,
 "long (>=120 words)"    : lambda t: len(TOK.findall(t))>=120,
 "pure question"         : lambda t: t.strip().endswith("?") and len(TOK.findall(t))<25,
 "conditional 'if'"      : lambda t: bool(re.search(r"\bif\b", t.lower())),
 "has a number/spec"     : lambda t: bool(NUM.search(t)),
 "SHOUTING caps"         : lambda t: bool(CAPS.search(t)),
 "1st-person anecdote"   : lambda t: bool(re.search(r"\bi (have|had|own|owned|shot|shoot|use|used|bought|got)\b", t.lower())),
}
print(f"{'genre':26}{'real':>9}{'v128':>9}{'v134':>9}{'v128/real':>11}")
print("-"*64)
for name,f in GENRES.items():
    v=[statistics.mean([sum(1 for t in x[k] if f(t))/len(x[k]) for x in rows]) for k in (0,1,2)]
    print(f"{name:26}{v[0]:>9.3f}{v[1]:>9.3f}{v[2]:>9.3f}{(v[1]/v[0] if v[0] else float('nan')):>11.2f}")
# how many DISTINCT genres does a thread use, and how evenly
def prof(ts):
    return [sum(1 for t in ts if f(t))/len(ts) for f in GENRES.values()]
print()
for lbl,k in (("real",0),("v128",1),("v134",2)):
    ps=[prof(x[k]) for x in rows]
    ent=[]
    for p_ in ps:
        s=sum(p_) or 1
        q=[v/s for v in p_ if v>0]
        ent.append(-sum(v*__import__("math").log(v) for v in q))
    print(f"{lbl:6} genre-mix entropy {statistics.mean(ent):.3f}")
