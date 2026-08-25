"""Is the omitted conjunction the move that actually buys the polite label?

`register_realization` cues intensifier and plain_verdict separately and
explicitly omits `intensified_positive` as "the same construction twice". The
measured carrier sentences are exactly that conjunction, so this tests whether
the whole is worth more than either part -- on real comments, with the shipped
classifier.
"""
from __future__ import annotations
import json, re, sys
from pathlib import Path
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
def label(texts):
    return [x["pred_label"] for x in scorer.score_comments([T(t) for t in texts],batch_size=32,include_text=False)]

cache={}
real=[]
for p in pool[:50]:
    d=REPO/"data/raw/discussions/camera_product"/p["source_product_dir"]
    if d not in cache: cache[d]=load_real_comments(d)[0]
    real += [c.text for c in (cache[d].get(p["source_raw_post_id"]) or [])]
gen=[]
for d in sorted((RUN/"cleaned").glob("run_*_sampled_reddit")):
    cbt,_=load_generated_comments(d)
    for tid,cs in cbt.items(): gen += [c.text for c in cs]
print(f"real {len(real)}  generated {len(gen)}")
RL=label(real); GL=label(gen)

def same_sentence_conj(text):
    """Intensifier AND a positive verdict inside ONE sentence."""
    for s in SENT.split(text):
        if P["any_intensifier"].search(s) and P["plain_verdict"].search(s):
            return True
    return False

def bucket(text):
    i=bool(P["any_intensifier"].search(text)); v=bool(P["plain_verdict"].search(text))
    if same_sentence_conj(text): return "conjunction (same sentence)"
    if i and v: return "both, different sentences"
    if v: return "verdict only"
    if i: return "intensifier only"
    return "neither"
ORDER=["conjunction (same sentence)","both, different sentences","verdict only","intensifier only","neither"]
print(f"\n{'bucket':<30}{'real n':>8}{'P(polite)':>11}{'gen n':>8}{'P(polite)':>11}{'prevalence real/gen':>22}")
for b in ORDER:
    ri=[i for i,t in enumerate(real) if bucket(t)==b]
    gi=[i for i,t in enumerate(gen) if bucket(t)==b]
    rp=sum(1 for i in ri if RL[i]=="polite")/len(ri) if ri else float('nan')
    gp=sum(1 for i in gi if GL[i]=="polite")/len(gi) if gi else float('nan')
    print(f"{b:<30}{len(ri):>8}{rp:>11.3f}{len(gi):>8}{gp:>11.3f}"
          f"{len(ri)/len(real):>11.3f}{len(gi)/len(gen):>11.3f}")
r_conj=sum(1 for t in real if same_sentence_conj(t))/len(real)
g_conj=sum(1 for t in gen if same_sentence_conj(t))/len(gen)
print(f"\nsame-sentence conjunction prevalence: real {r_conj:.4f}   generated {g_conj:.4f}   ratio {g_conj/r_conj:.2f}x")
