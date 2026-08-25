import csv, json, statistics as st
from pathlib import Path
from scipy.stats import mannwhitneyu, ks_2samp
REPO = Path("/Users/yaoningyu/Desktop/UIUC/GEO")
RUN = REPO/"artifacts/generalized_card/runs/generalized_card_camera_gpt54_paper_20260825_v1"
pool = json.load(open(REPO/"artifacts/generalized_card/seed_pools/camera_product_150_seed42.json"))["seed_posts"]
by_seed = {int(p["seed_index"]): p for p in pool}
real = {r["thread_id"]: r for r in csv.DictReader(open(REPO/"artifacts/baselines/camera_product_gpt5mini/real/thread_scores.csv"))}
gen = list(csv.DictReader(open(RUN/"evaluation/revised_generated_thread_scores.csv")))
pairs = []
for g in gen:
    rid = by_seed[int(float(g["seed_index"]))]["source_raw_post_id"]
    if rid in real:
        pairs.append((real[rid], g))
print(f"matched: {len(pairs)}\n")
cols = ["self_bleu_2","self_bleu_3","self_bleu_4",
        "self_bertscore_mean_f1","self_bertscore_median_f1","self_bertscore_top_k_mean_f1",
        "self_bertscore_mean_precision","self_bertscore_mean_recall",
        "semantic_mean_cosine","semantic_median_cosine","semantic_top_k_mean_cosine","semantic_p90_cosine"]
print(f"{'column':<32}{'real':>9}{'gen':>9}{'gap':>10}{'rel':>8}{'MWU':>9}{'KS':>9}")
for c in cols:
    a = [float(r[c]) for r, _ in pairs if c in r]
    b = [float(g[c]) for r, g in pairs if c in r]
    if not a: 
        print(f"{c:<32}  (absent from real baseline)"); continue
    gap = st.mean(b)-st.mean(a)
    print(f"{c:<32}{st.mean(a):>9.4f}{st.mean(b):>9.4f}{gap:>+10.4f}{100*gap/st.mean(a):>7.1f}%"
          f"{mannwhitneyu(a,b,alternative='two-sided').pvalue:>9.4f}{ks_2samp(a,b).pvalue:>9.4f}")
