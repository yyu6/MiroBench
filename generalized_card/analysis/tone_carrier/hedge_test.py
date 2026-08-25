"""Is generated appreciation landing one class down because it is hedged?"""
from __future__ import annotations
import json, re, sys
from collections import Counter
from pathlib import Path
import numpy as np
REPO=Path("/Users/yaoningyu/Desktop/UIUC/GEO")
sys.path.insert(0,str(REPO/"generalized_card"))
sys.path.insert(0,str(REPO/"scripts"/"evaluation"))
from score_thread_semantic_uniformity import load_real_comments, load_generated_comments
from score_thread_politeness import PolitenessScorer
RUN=REPO/"artifacts/generalized_card/runs/generalized_card_camera_gpt54_paper_20260825_v1"
pool=json.load(open(REPO/"artifacts/generalized_card/seed_pools/camera_product_150_seed42.json"))["seed_posts"]
scorer=PolitenessScorer("Intel/polite-guard","auto",256)
class T:
    def __init__(self,t):
        self.text=t; self.thread_id=""; self.thread_title=""; self.comment_id=""
        self.parent_id=""; self.author=""; self.depth=0
lab=lambda ts: [x["pred_label"] for x in scorer.score_comments([T(t) for t in ts],batch_size=32,include_text=False)]
HEDGE=re.compile(r"\b(?:kind of|kinda|sort of|sorta|i'd say|i think|i guess|probably|"
                 r"more or less|somewhat|a bit|a little|fairly|arguably|to be fair|"
                 r"i suppose|maybe|might be|can be|tends to|generally)\b", re.I)
strip=lambda t: re.sub(r"\s{2,}", " ", HEDGE.sub("", t)).strip()
cache={}; real=[]
for p in pool[:50]:
    d=REPO/"data/raw/discussions/camera_product"/p["source_product_dir"]
    if d not in cache: cache[d]=load_real_comments(d)[0]
    real += [c.text for c in (cache[d].get(p["source_raw_post_id"]) or [])]
gen=[]
for d in sorted((RUN/"cleaned").glob("run_*_sampled_reddit")):
    cbt,_=load_generated_comments(d)
    for tid,cs in cbt.items(): gen += [c.text for c in cs]
print(f"hedge prevalence: real {np.mean([bool(HEDGE.search(t)) for t in real]):.3f}  "
      f"generated {np.mean([bool(HEDGE.search(t)) for t in gen]):.3f}")
print(f"hedges per 100 words: real {100*sum(len(HEDGE.findall(t)) for t in real)/sum(len(t.split()) for t in real):.2f}  "
      f"generated {100*sum(len(HEDGE.findall(t)) for t in gen)/sum(len(t.split()) for t in gen):.2f}")
g0=lab(gen); g1=lab([strip(t) for t in gen]); r0=lab(real); r1=lab([strip(t) for t in real])
def rate(ls,l="polite"): return sum(1 for x in ls if x==l)/len(ls)
print(f"\n{'':<26}{'polite':>9}{'somewhat':>10}{'neutral':>9}{'impolite':>10}")
for name,ls in (("real, untouched",r0),("real, hedges stripped",r1),
                ("generated, untouched",g0),("generated, hedges stripped",g1)):
    print(f"{name:<26}" + "".join(f"{rate(ls,k):>9.3f}" if k!="somewhat" else f"{rate(ls,'somewhat_polite'):>10.3f}"
                                   for k in ("polite","somewhat","neutral","impolite")))
moved=sum(1 for a,b in zip(g0,g1) if a!="polite" and b=="polite")
print(f"\ngenerated comments that flip TO polite when hedges are removed: {moved} "
      f"({100*moved/len(gen):.2f}% of all comments; the gap needs 9.1%)")
