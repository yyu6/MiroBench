"""Punctuation / clause-shape uniformity, controlled for comment length."""
import json, re, sys, statistics as st
from collections import defaultdict
from pathlib import Path
REPO=Path("/Users/yaoningyu/Desktop/UIUC/GEO")
sys.path.insert(0,str(REPO/"scripts"/"evaluation"))
from score_thread_self_bleu import tokenize
from score_thread_semantic_uniformity import load_real_comments, load_generated_comments
RUN=REPO/"artifacts/generalized_card/runs/generalized_card_camera_gpt54_paper_20260825_v1"
pool=json.load(open(REPO/"artifacts/generalized_card/seed_pools/camera_product_150_seed42.json"))["seed_posts"]
by_seed={int(x["seed_index"]):x for x in pool}
MARKUP=re.compile(r"https?://|\bwww\.|\bu/\w|\br/\w|[*_`~]|&gt;|^\s*>|\\|\[[^\]]*\]\(")
G,R=[],[]
cache={}
for dd in sorted((RUN/"cleaned").glob("run_*_sampled_reddit")):
    cbt,_=load_generated_comments(dd)
    for tid,comments in cbt.items():
        p=by_seed[int(tid.split("seed")[-1])]
        rd=REPO/"data/raw/discussions/camera_product"/p["source_product_dir"]
        if rd not in cache: cache[rd]=load_real_comments(rd)[0]
        G+=[c.text for c in comments]
        R+=[c.text for c in (cache[rd].get(p["source_raw_post_id"]) or [])]
print(f"comments: generated {len(G)}  real {len(R)}\n")

def band(w):
    for lo,hi,lab in ((0,9,"1-9"),(10,24,"10-24"),(25,49,"25-49"),(50,99,"50-99"),(100,10**9,"100+")):
        if lo<=w<=hi: return lab

def stats(texts):
    d=defaultdict(list)
    for t in texts:
        w=len(t.split()); b=band(w)
        d[b].append({
            "comma": t.count(","),
            "no_comma": 1.0 if t.count(",")==0 else 0.0,
            "sent": max(1,len(re.findall(r"[.!?]+", t))),
            "markup": 1.0 if MARKUP.search(t) else 0.0,
            "punct_per_word": sum(1 for ch in t if ch in ",.;:!?()\"'-—…") / max(1,w),
            "n": 1,
        })
    return d

sg, sr = stats(G), stats(R)
print(f"{'length band':<12}{'n real':>8}{'n gen':>7}{'  no-comma share':>18}{'  commas/comment':>18}{'  sentences':>14}{'  markup share':>16}")
for b in ("1-9","10-24","25-49","50-99","100+"):
    if b not in sg or b not in sr: continue
    def m(d,k): return st.mean([x[k] for x in d[b]])
    print(f"{b:<12}{len(sr[b]):>8}{len(sg[b]):>7}"
          f"   real {m(sr,'no_comma'):.2f} gen {m(sg,'no_comma'):.2f}"
          f"   real {m(sr,'comma'):.2f} gen {m(sg,'comma'):.2f}"
          f"   {m(sr,'sent'):.2f}/{m(sg,'sent'):.2f}"
          f"      {m(sr,'markup'):.3f}/{m(sg,'markup'):.3f}")
print()
for lab,d in (("real",sr),("generated",sg)):
    allc=[x for b in d for x in d[b]]
    print(f"{lab:<10} no-comma share {st.mean([x['no_comma'] for x in allc]):.3f}   "
          f"commas/comment {st.mean([x['comma'] for x in allc]):.2f}   "
          f"punct/word {st.mean([x['punct_per_word'] for x in allc]):.3f}   "
          f"markup share {st.mean([x['markup'] for x in allc]):.3f}")
