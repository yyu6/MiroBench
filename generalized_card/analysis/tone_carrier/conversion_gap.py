"""Where does the polite label go, when the positive move is present?"""
from __future__ import annotations
import json, re, sys, statistics as st
from collections import Counter
from pathlib import Path
import numpy as np
REPO=Path("/Users/yaoningyu/Desktop/UIUC/GEO")
sys.path.insert(0,str(REPO/"generalized_card"))
sys.path.insert(0,str(REPO/"scripts"/"evaluation"))
from generalized_card.register_realization import REGISTER_MOVES
from score_thread_semantic_uniformity import load_real_comments, load_generated_comments
from score_thread_politeness import PolitenessScorer
RUN=REPO/"artifacts/generalized_card/runs/generalized_card_camera_gpt54_paper_20260825_v1"
pool=json.load(open(REPO/"artifacts/generalized_card/seed_pools/camera_product_150_seed42.json"))["seed_posts"]
P={m["name"]: re.compile(m["pattern"], re.I) for m in REGISTER_MOVES}
SENT=re.compile(r"(?<=[.!?])\s+")
scorer=PolitenessScorer("Intel/polite-guard","auto",256)
class T:
    def __init__(self,t):
        self.text=t; self.thread_id=""; self.thread_title=""; self.comment_id=""
        self.parent_id=""; self.author=""; self.depth=0
def rows(texts):
    return scorer.score_comments([T(t) for t in texts],batch_size=32,include_text=False)
def conj(t):
    return any(P["any_intensifier"].search(s) and P["plain_verdict"].search(s) for s in SENT.split(t))
cache={}; real=[]
for p in pool[:50]:
    d=REPO/"data/raw/discussions/camera_product"/p["source_product_dir"]
    if d not in cache: cache[d]=load_real_comments(d)[0]
    real += [c.text for c in (cache[d].get(p["source_raw_post_id"]) or [])]
gen=[]
for d in sorted((RUN/"cleaned").glob("run_*_sampled_reddit")):
    cbt,_=load_generated_comments(d)
    for tid,cs in cbt.items(): gen += [c.text for c in cs]
R=rows(real); G=rows(gen)
for name, texts, rr in (("real", real, R), ("generated", gen, G)):
    idx=[i for i,t in enumerate(texts) if conj(t)]
    print(f"\n{name}: conjunction comments {len(idx)}")
    print("  labels:", dict(Counter(rr[i]['pred_label'] for i in idx).most_common()))
    for lab in ("polite","impolite"):
        sel=[i for i in idx if rr[i]['pred_label']==lab]
        if not sel: continue
        pp=[rr[i]['polite_probability'] for i in sel]
        wl=[len(texts[i].split()) for i in sel]
        print(f"  {lab:<10} n={len(sel):>3}  mean P(polite)={np.mean(pp):.3f}  mean words={np.mean(wl):.0f}")
    ni=[i for i in idx if rr[i]['pred_label']!='polite']
    print(f"  non-polite: mean P(polite)={np.mean([rr[i]['polite_probability'] for i in ni]):.3f}, "
          f"mean P(impolite)={np.mean([rr[i]['impolite_probability'] for i in ni]):.3f}, "
          f"mean words={np.mean([len(texts[i].split()) for i in ni]):.0f}")
# length is the obvious confound: check conversion by comment length
print(f"\n{'words':<12}{'real n':>8}{'real P(pol)':>13}{'gen n':>8}{'gen P(pol)':>12}")
for lo,hi in ((0,14),(15,29),(30,59),(60,119),(120,10**9)):
    ri=[i for i,t in enumerate(real) if lo<=len(t.split())<=hi]
    gi=[i for i,t in enumerate(gen) if lo<=len(t.split())<=hi]
    rp=sum(1 for i in ri if R[i]['pred_label']=='polite')/len(ri) if ri else float('nan')
    gp=sum(1 for i in gi if G[i]['pred_label']=='polite')/len(gi) if gi else float('nan')
    print(f"{f'{lo}-{hi}':<12}{len(ri):>8}{rp:>13.3f}{len(gi):>8}{gp:>12.3f}")
