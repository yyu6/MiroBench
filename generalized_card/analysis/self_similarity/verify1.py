"""Check 1: does shortening real text actually raise real's self_bleu_4?

This is the causal test of the length claim, on real text, with the shipped
scorer -- not a decomposition identity.
"""
import json, sys, statistics as st
from pathlib import Path
from scipy.stats import mannwhitneyu, ks_2samp
REPO=Path("/Users/yaoningyu/Desktop/UIUC/GEO")
sys.path.insert(0,str(REPO/"scripts"/"evaluation"))
from score_thread_self_bleu import tokenize, pairwise_self_bleu_for_order
from score_thread_semantic_uniformity import load_real_comments
SP=(Path(__file__).resolve().parent / "_cache")
pool=json.load(open(REPO/"artifacts/generalized_card/seed_pools/camera_product_150_seed42.json"))["seed_posts"]
gen={g["seed"]:g for g in json.load(open(SP/"gen_sb4.json"))}
cache={}
def truncate(words, frac):
    keep=max(1,int(round(len(words)*frac)))
    return words[:keep]
rows=[]
for p in pool[:50]:
    d=REPO/"data/raw/discussions/camera_product"/p["source_product_dir"]
    if d not in cache: cache[d]=load_real_comments(d)[0]
    texts=[c.text for c in (cache[d].get(p["source_raw_post_id"]) or [])]
    if len(texts)<2: continue
    rows.append((int(p["seed_index"]), texts))
G=[gen[s]["sb4"] for s,_ in rows]
def score(frac_fn):
    out=[]
    for s,texts in rows:
        toks=[]
        for t in texts:
            w=t.split()
            toks.append(tokenize(" ".join(truncate(w, frac_fn(len(w))))))
        out.append(pairwise_self_bleu_for_order(toks,4))
    return out
base=score(lambda n: 1.0)
print(f"real, untouched                          {st.mean(base):.5f}   (generated {st.mean(G):.5f})")
for name, fn in (
    ("real, every comment cut to 0.891 (the run's global ratio)", lambda n: 0.891),
    ("real, cut by the run's per-band ratio", lambda n: (1.224 if n<10 else 1.001 if n<20 else 0.992 if n<35
        else 0.907 if n<50 else 0.881 if n<70 else 0.877 if n<=100 else 0.910 if n<=150 else 0.802 if n<=300 else 0.699)),
):
    v=score(fn)
    print(f"{name:<58}{st.mean(v):.5f}   gap to generated {st.mean(G)-st.mean(v):+.5f}  "
          f"MWU {mannwhitneyu(v,G,alternative='two-sided').pvalue:.4f}  KS {ks_2samp(v,G).pvalue:.4f}")
print(f"\nreference: untouched real vs generated   gap {st.mean(G)-st.mean(base):+.5f}  "
      f"MWU {mannwhitneyu(base,G,alternative='two-sided').pvalue:.4f}  KS {ks_2samp(base,G).pvalue:.4f}")
