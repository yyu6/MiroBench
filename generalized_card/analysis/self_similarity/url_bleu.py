import json, re, sys, statistics as st
from pathlib import Path
from scipy.stats import mannwhitneyu, ks_2samp
REPO=Path("/Users/yaoningyu/Desktop/UIUC/GEO")
sys.path.insert(0,str(REPO/"scripts"/"evaluation"))
from score_thread_self_bleu import tokenize, pairwise_self_bleu_for_order
from score_thread_semantic_uniformity import load_real_comments
SP=(Path(__file__).resolve().parent / "_cache")
pool=json.load(open(REPO/"artifacts/generalized_card/seed_pools/camera_product_150_seed42.json"))["seed_posts"]
URL=re.compile(r"https?://\S+|\bwww\.\S+")
strip=lambda t: re.sub(r"\s+"," ",URL.sub("",t)).strip()
gen={g["seed"]:g for g in json.load(open(SP/"gen_sb4.json"))}
cache={}; base=[]; stripped=[]; g=[]
tok_r=tok_rs=w_r=0
for p in pool[:50]:
    d=REPO/"data/raw/discussions/camera_product"/p["source_product_dir"]
    if d not in cache: cache[d]=load_real_comments(d)[0]
    texts=[c.text for c in (cache[d].get(p["source_raw_post_id"]) or [])]
    if len(texts)<2: continue
    a=[tokenize(t) for t in texts]; b=[tokenize(strip(t)) for t in texts]
    tok_r+=sum(len(x) for x in a); tok_rs+=sum(len(x) for x in b); w_r+=sum(len(t.split()) for t in texts)
    base.append(pairwise_self_bleu_for_order(a,4)); stripped.append(pairwise_self_bleu_for_order(b,4))
    g.append(gen[int(p["seed_index"])]["sb4"])
print(f"real self_bleu_4            {st.mean(base):.5f}")
print(f"real, URLs stripped         {st.mean(stripped):.5f}   ({st.mean(stripped)-st.mean(base):+.5f})")
print(f"generated self_bleu_4       {st.mean(g):.5f}")
print(f"\ngap real-vs-generated       {st.mean(g)-st.mean(base):+.5f}  MWU {mannwhitneyu(base,g,alternative='two-sided').pvalue:.4f}  KS {ks_2samp(base,g).pvalue:.4f}")
print(f"gap if real had no URLs     {st.mean(g)-st.mean(stripped):+.5f}  MWU {mannwhitneyu(stripped,g,alternative='two-sided').pvalue:.4f}  KS {ks_2samp(stripped,g).pvalue:.4f}")
print(f"\nURLs are {100*(tok_r-tok_rs)/tok_r:.2f}% of real's tokens; real tokens/word {tok_r/w_r:.4f} -> {tok_rs/w_r:.4f} without them (generated 1.1450)")
