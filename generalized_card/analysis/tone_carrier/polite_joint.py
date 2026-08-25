"""Does one appreciative-sentence mechanism move all three tone metrics?

Also checks the side effects on the two self-similarity metrics, since the
carrier is drawn from a finite pool and repeated text inflates both.
"""
from __future__ import annotations
import csv, json, random, re, statistics as st, sys
from pathlib import Path
import numpy as np
from scipy.stats import mannwhitneyu, ks_2samp
REPO = Path("/Users/yaoningyu/Desktop/UIUC/GEO")
sys.path.insert(0, str(REPO / "scripts" / "evaluation"))
from score_thread_semantic_uniformity import load_real_comments, load_generated_comments
from score_thread_politeness import PolitenessScorer
from score_thread_self_bleu import tokenize, pairwise_self_bleu_for_order
RUN = REPO / "artifacts/generalized_card/runs/generalized_card_camera_gpt54_paper_20260825_v1"
pool = json.load(open(REPO / "artifacts/generalized_card/seed_pools/camera_product_150_seed42.json"))["seed_posts"]
by_seed = {int(x["seed_index"]): x for x in pool}
matched = {by_seed[s]["source_raw_post_id"] for s in range(50)}
real_csv = {r["thread_id"]: r for r in csv.DictReader(open(REPO/"artifacts/baselines/camera_product_gpt5mini/real/thread_scores.csv"))}
scorer = PolitenessScorer("Intel/polite-guard", "auto", 256)
SENT = re.compile(r"(?<=[.!?])\s+")
class T:
    def __init__(self, t):
        self.text=t; self.thread_id=""; self.thread_title=""; self.comment_id=""
        self.parent_id=""; self.author=""; self.depth=0
score = lambda ts: scorer.score_comments([T(t) for t in ts], batch_size=32, include_text=False)
share_of = lambda rows, lab: sum(1 for x in rows if x["pred_label"]==lab)/len(rows) if rows else 0.0

gen = {}
for d in sorted((RUN/"cleaned").glob("run_*_sampled_reddit")):
    cbt,_ = load_generated_comments(d)
    for tid, cs in cbt.items():
        gen[int(tid.split("seed")[-1])] = [c.text for c in cs]

carriers=[]
for d in sorted((REPO/"data/raw/discussions/camera_product").iterdir()):
    if not d.is_dir() or len(carriers)>4000: continue
    try: cbt,_=load_real_comments(d)
    except Exception: continue
    for tid,cs in cbt.items():
        if tid in matched: continue
        for c in cs:
            for s_ in SENT.split(c.text):
                if 3<=len(s_.split())<=18: carriers.append(s_.strip())
rows=score(carriers[:4000])
POOL=[carriers[i] for i,x in enumerate(rows) if x["pred_label"]=="polite" and x["polite_probability"]>0.90]

R = {s: real_csv[by_seed[s]["source_raw_post_id"]] for s in gen if by_seed[s]["source_raw_post_id"] in real_csv}
def realv(k): return [float(R[s][k]) for s in sorted(R)]

def evaluate(share, seed=11):
    rng=random.Random(seed)
    pol,imp,neu,sb4 = [],[],[],[]
    for s in sorted(R):
        texts=[(rng.choice(POOL)+" "+t) if rng.random()<share else t for t in gen[s]]
        rr=score(texts)
        pol.append(share_of(rr,"polite")); imp.append(share_of(rr,"impolite")); neu.append(share_of(rr,"neutral"))
        sb4.append(pairwise_self_bleu_for_order([tokenize(t) for t in texts],4))
    return pol,imp,neu,sb4

print(f"{'metric':<16}{'real':>9}{'today':>9}{'+10%':>9}{'+22%':>9}   {'MWU today':>10}{'MWU +22%':>10}{'KS +22%':>9}")
res={}
for share in (0.0,0.10,0.22):
    res[share]=evaluate(share)
for i,(name,key) in enumerate((("polite_rate","polite_rate"),("impolite_rate","impolite_rate"),
                               ("neutral_rate","neutral_rate"),("self_bleu_4","self_bleu_4"))):
    rv=realv(key)
    a,b,c = res[0.0][i], res[0.10][i], res[0.22][i]
    print(f"{name:<16}{st.mean(rv):>9.4f}{st.mean(a):>9.4f}{st.mean(b):>9.4f}{st.mean(c):>9.4f}   "
          f"{mannwhitneyu(rv,a,alternative='two-sided').pvalue:>10.4f}"
          f"{mannwhitneyu(rv,c,alternative='two-sided').pvalue:>10.4f}"
          f"{ks_2samp(rv,c).pvalue:>9.4f}")
