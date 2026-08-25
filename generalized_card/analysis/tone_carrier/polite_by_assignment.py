"""The buildable routing: cue the carrier on slots the PLANNER assigned polite.

Routing by the evaluation classifier's own label is forbidden (ORIENTATION s4:
distribution diagnostics never select a Writer candidate). The Planner already
assigns tone_target=polite to 588 of 1974 slots -- 29.8%, against real's 0.3020
-- so the assignment is right and only the realization fails. This measures what
raising that realization rate would do.
"""
from __future__ import annotations
import csv, json, random, re, statistics as st, sys
from pathlib import Path
from scipy.stats import mannwhitneyu, ks_2samp
REPO=Path("/Users/yaoningyu/Desktop/UIUC/GEO")
sys.path.insert(0,str(REPO/"scripts"/"evaluation"))
from score_thread_semantic_uniformity import load_real_comments, load_generated_comments
from score_thread_politeness import PolitenessScorer
from score_thread_self_bleu import tokenize, pairwise_self_bleu_for_order
RUN=REPO/"artifacts/generalized_card/runs/generalized_card_camera_gpt54_paper_20260825_v1"
pool=json.load(open(REPO/"artifacts/generalized_card/seed_pools/camera_product_150_seed42.json"))["seed_posts"]
by_seed={int(x["seed_index"]):x for x in pool}
matched={by_seed[s]["source_raw_post_id"] for s in range(50)}
real_csv={r["thread_id"]:r for r in csv.DictReader(open(REPO/"artifacts/baselines/camera_product_gpt5mini/real/thread_scores.csv"))}
scorer=PolitenessScorer("Intel/polite-guard","auto",256)
SENT=re.compile(r"(?<=[.!?])\s+")
class T:
    def __init__(self,t):
        self.text=t; self.thread_id=""; self.thread_title=""; self.comment_id=""
        self.parent_id=""; self.author=""; self.depth=0
score=lambda ts: scorer.score_comments([T(t) for t in ts],batch_size=32,include_text=False)
sh=lambda rows,l: sum(1 for x in rows if x["pred_label"]==l)/len(rows) if rows else 0.0

# join scored comments to their assigned tone
assigned={}
for d in sorted((RUN/"cleaned").glob("run_*_sampled_reddit")):
    disc=json.load(open(d/"discussion.json"))
    for post in disc["posts"]:
        for rec in post.get("generation_records") or []:
            cid=str((rec.get("comment") or {}).get("comment_id",""))
            if cid: assigned[cid]=(rec.get("task") or {}).get("tone_target")
gen={}
for d in sorted((RUN/"cleaned").glob("run_*_sampled_reddit")):
    cbt,_=load_generated_comments(d)
    for tid,cs in cbt.items():
        gen[int(tid.split("seed")[-1])]=[(c.text, assigned.get(str(c.comment_id))) for c in cs]

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

R={s:real_csv[by_seed[s]["source_raw_post_id"]] for s in gen if by_seed[s]["source_raw_post_id"] in real_csv}
seeds=sorted(R)
base={s:score([t for t,_ in gen[s]]) for s in seeds}
pa=[(t,a,b) for s in seeds for (t,a),b in zip(gen[s],base[s]) if a=="polite"]
print(f"slots assigned tone_target=polite: {len(pa)}")
print(f"of those, realized as polite by the scorer: {sum(1 for _,_,b in pa if b['pred_label']=='polite')} "
      f"({100*sum(1 for _,_,b in pa if b['pred_label']=='polite')/len(pa):.1f}%)")
print(f"their current label mix: {dict((l, sum(1 for _,_,b in pa if b['pred_label']==l)) for l in ('polite','somewhat_polite','neutral','impolite'))}")

def run(share, seed=11):
    rng=random.Random(seed); out={"polite_rate":[],"impolite_rate":[],"neutral_rate":[],"self_bleu_4":[]}
    for s in seeds:
        texts=[]
        for (t,a),b in zip(gen[s],base[s]):
            fix = a=="polite" and b["pred_label"]!="polite" and rng.random()<share
            texts.append((rng.choice(POOL)+" "+t) if fix else t)
        rr=score(texts)
        out["polite_rate"].append(sh(rr,"polite")); out["impolite_rate"].append(sh(rr,"impolite"))
        out["neutral_rate"].append(sh(rr,"neutral"))
        out["self_bleu_4"].append(pairwise_self_bleu_for_order([tokenize(t) for t in texts],4))
    return out

shares=(0.0,0.40,0.70,1.00)
res={s:run(s) for s in shares}
today={k:[float(R[s][k]) for s in seeds] for k in ("polite_rate","impolite_rate","neutral_rate","self_bleu_4")}
print(f"\n{'metric':<18}{'real':>9}" + "".join(f"{f'fix {int(100*s)}%':>11}" for s in shares))
for k in today:
    print(f"{k:<18}{st.mean(today[k]):>9.4f}" + "".join(f"{st.mean(res[s][k]):>11.4f}" for s in shares))
print(f"\n{'metric':<18}{'':>9}" + "".join(f"{f'MWU {int(100*s)}%':>11}" for s in shares))
for k in today:
    print(f"{k:<18}{'':>9}" + "".join(f"{mannwhitneyu(today[k],res[s][k],alternative='two-sided').pvalue:>11.4f}" for s in shares))
print(f"\n{'metric':<18}{'':>9}" + "".join(f"{f'KS {int(100*s)}%':>11}" for s in shares))
for k in today:
    print(f"{k:<18}{'':>9}" + "".join(f"{ks_2samp(today[k],res[s][k]).pvalue:>11.4f}" for s in shares))
