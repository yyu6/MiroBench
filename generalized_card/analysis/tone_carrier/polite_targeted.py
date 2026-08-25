"""Target the carrier at the over-full bucket instead of at random.

Generated has too many `impolite` comments (0.5349 vs real 0.4220) and too few
`neutral` (0.1143 vs 0.1577), so a random carrier steals from the bucket that is
already short. This routes it to `impolite` comments only.
"""
from __future__ import annotations
import csv, json, random, re, statistics as st, sys
from pathlib import Path
from scipy.stats import mannwhitneyu, ks_2samp
REPO = Path("/Users/yaoningyu/Desktop/UIUC/GEO")
sys.path.insert(0, str(REPO / "scripts" / "evaluation"))
from score_thread_semantic_uniformity import load_real_comments, load_generated_comments
from score_thread_politeness import PolitenessScorer
from score_thread_self_bleu import tokenize, pairwise_self_bleu_for_order
RUN = REPO / "artifacts/generalized_card/runs/generalized_card_camera_gpt54_paper_20260825_v1"
pool = json.load(open(REPO/"artifacts/generalized_card/seed_pools/camera_product_150_seed42.json"))["seed_posts"]
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
sh = lambda rows,l: sum(1 for x in rows if x["pred_label"]==l)/len(rows) if rows else 0.0

gen={}
for d in sorted((RUN/"cleaned").glob("run_*_sampled_reddit")):
    cbt,_=load_generated_comments(d)
    for tid,cs in cbt.items(): gen[int(tid.split("seed")[-1])]=[c.text for c in cs]
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
R={s: real_csv[by_seed[s]["source_raw_post_id"]] for s in gen if by_seed[s]["source_raw_post_id"] in real_csv}
seeds=sorted(R)
base={s:score(gen[s]) for s in seeds}

def run(share, only_impolite, seed=11):
    rng=random.Random(seed); out={"polite_rate":[],"impolite_rate":[],"neutral_rate":[],"self_bleu_4":[]}
    for s in seeds:
        texts=[]
        for t,b in zip(gen[s], base[s]):
            eligible = (b["pred_label"]=="impolite") if only_impolite else True
            texts.append((rng.choice(POOL)+" "+t) if (eligible and rng.random()<share) else t)
        rr=score(texts)
        out["polite_rate"].append(sh(rr,"polite")); out["impolite_rate"].append(sh(rr,"impolite"))
        out["neutral_rate"].append(sh(rr,"neutral"))
        out["self_bleu_4"].append(pairwise_self_bleu_for_order([tokenize(t) for t in texts],4))
    return out

print(f"{'':>34}{'real':>9}{'today':>9}" + "".join(f"{f'imp@{int(100*s)}%':>10}" for s in (0.25,0.40,0.55)))
res={s: run(s, True) for s in (0.25,0.40,0.55)}
today={k:[float(R[s][k]) for s in seeds] for k in ("polite_rate","impolite_rate","neutral_rate","self_bleu_4")}
b0=run(0.0, True)
for k in ("polite_rate","impolite_rate","neutral_rate","self_bleu_4"):
    rv=today[k]
    line=f"{k:<34}{st.mean(rv):>9.4f}{st.mean(b0[k]):>9.4f}"
    for s in (0.25,0.40,0.55): line+=f"{st.mean(res[s][k]):>10.4f}"
    print(line)
print()
print(f"{'':>34}{'MWU today':>9}" + "".join(f"{f'MWU@{int(100*s)}':>10}" for s in (0.25,0.40,0.55)))
for k in ("polite_rate","impolite_rate","neutral_rate","self_bleu_4"):
    rv=today[k]
    line=f"{k:<34}{mannwhitneyu(rv,b0[k],alternative='two-sided').pvalue:>9.4f}"
    for s in (0.25,0.40,0.55):
        line+=f"{mannwhitneyu(rv,res[s][k],alternative='two-sided').pvalue:>10.4f}"
    print(line)
print()
print(f"{'':>34}{'KS today':>9}" + "".join(f"{f'KS@{int(100*s)}':>10}" for s in (0.25,0.40,0.55)))
for k in ("polite_rate","impolite_rate","neutral_rate","self_bleu_4"):
    rv=today[k]
    line=f"{k:<34}{ks_2samp(rv,b0[k]).pvalue:>9.4f}"
    for s in (0.25,0.40,0.55):
        line+=f"{ks_2samp(rv,res[s][k]).pvalue:>10.4f}"
    print(line)
