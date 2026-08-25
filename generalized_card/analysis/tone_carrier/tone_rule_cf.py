"""The buildable rule, measured: polite-assigned first-person experience slots."""
from __future__ import annotations
import csv, json, random, re, statistics as st, sys
from collections import Counter
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

task_by_cid={}
for d in sorted((RUN/"cleaned").glob("run_*_sampled_reddit")):
    disc=json.load(open(d/"discussion.json"))
    for post in disc["posts"]:
        for rec in post.get("generation_records") or []:
            cid=str((rec.get("comment") or {}).get("comment_id",""))
            if cid: task_by_cid[cid]=rec.get("task") or {}

# Every value below sits under the polite-assigned base neutral rate of 0.055;
# the four story/experience ones are exactly 0.000. neutral_rate is already the
# shortest bucket, so the rule is built to avoid draining it.
NARROW = {
    ("payload_type", "personal_story"),
    ("evidence_mode", "firsthand_experience"),
    ("story_mode", "specific_personal_story"),
    ("story_mode", "messy_multi_step_story"),
}
BROAD = NARROW | {
    ("affect_role", "admiration"),
    ("comment_function", "personal_datapoint"),
    ("speaker_role", "datapoint_only"),
    ("surface_texture", "abbrev_shorthand"),
}
RULE = BROAD

def routed(t):
    if t.get("tone_target") != "polite":
        return False
    return any(t.get(f) == v for f, v in RULE)

gen={}
for d in sorted((RUN/"cleaned").glob("run_*_sampled_reddit")):
    cbt,_=load_generated_comments(d)
    for tid,cs in cbt.items():
        gen[int(tid.split("seed")[-1])]=[(c.text, task_by_cid.get(str(c.comment_id)) or {}) for c in cs]
allr=[(t,x) for s in gen for x,t in ((a,b) for a,b in gen[s])]
n_routed=sum(1 for s in gen for _,t in gen[s] if routed(t))
print(f"routed slots: {n_routed} of {sum(len(v) for v in gen.values())} = {100*n_routed/sum(len(v) for v in gen.values()):.2f}%")
base={s:score([t for t,_ in gen[s]]) for s in gen}
lab=[(b["pred_label"]) for s in gen for b in base[s]]
rl=[b["pred_label"] for s in gen for (txt,t),b in zip(gen[s],base[s]) if routed(t)]
print(f"  their current label mix: {dict(Counter(rl).most_common())}")

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
def run(share, seed=11):
    rng=random.Random(seed); out={"polite_rate":[],"impolite_rate":[],"neutral_rate":[],"self_bleu_4":[]}
    for s in seeds:
        texts=[(rng.choice(POOL)+" "+t) if (routed(tk) and rng.random()<share) else t for t,tk in gen[s]]
        rr=score(texts)
        out["polite_rate"].append(sh(rr,"polite")); out["impolite_rate"].append(sh(rr,"impolite"))
        out["neutral_rate"].append(sh(rr,"neutral"))
        out["self_bleu_4"].append(pairwise_self_bleu_for_order([tokenize(t) for t in texts],4))
    return out
shares=(0.0,0.5,0.8,1.0)
res={x:run(x) for x in shares}
today={k:[float(R[s][k]) for s in seeds] for k in ("polite_rate","impolite_rate","neutral_rate","self_bleu_4")}
print(f"\n{'metric':<16}{'real':>9}" + "".join(f"{f'@{int(100*x)}%':>10}" for x in shares))
for k in today: print(f"{k:<16}{st.mean(today[k]):>9.4f}" + "".join(f"{st.mean(res[x][k]):>10.4f}" for x in shares))
print(f"\n{'metric':<16}{'':>9}" + "".join(f"{f'MWU@{int(100*x)}':>10}" for x in shares))
for k in today: print(f"{k:<16}{'':>9}" + "".join(f"{mannwhitneyu(today[k],res[x][k],alternative='two-sided').pvalue:>10.4f}" for x in shares))
print(f"\n{'metric':<16}{'':>9}" + "".join(f"{f'KS@{int(100*x)}':>10}" for x in shares))
for k in today: print(f"{k:<16}{'':>9}" + "".join(f"{ks_2samp(today[k],res[x][k]).pvalue:>10.4f}" for x in shares))
