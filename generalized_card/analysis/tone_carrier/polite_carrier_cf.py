"""What does it take to move the generator's polite_rate?

Two measurements, both with the shipped classifier, reproducing the shipped
per-thread rate first (E6):

  1. how far the closest non-polite generated comments are from flipping;
  2. the exact effect of prepending ONE real appreciative sentence, drawn from
     real threads OUTSIDE the 50 matched seeds, to a share of generated
     comments -- so the required rate is a measured number.
"""
from __future__ import annotations
import csv, json, random, re, statistics as st, sys
from pathlib import Path
import numpy as np
REPO = Path("/Users/yaoningyu/Desktop/UIUC/GEO")
sys.path.insert(0, str(REPO / "scripts" / "evaluation"))
from score_thread_semantic_uniformity import load_real_comments, load_generated_comments
from score_thread_politeness import PolitenessScorer
RUN = REPO / "artifacts/generalized_card/runs/generalized_card_camera_gpt54_paper_20260825_v1"
pool = json.load(open(REPO / "artifacts/generalized_card/seed_pools/camera_product_150_seed42.json"))["seed_posts"]
by_seed = {int(x["seed_index"]): x for x in pool}
matched_rids = {by_seed[s]["source_raw_post_id"] for s in range(50)}
gen_csv = {int(float(r["seed_index"])): r for r in csv.DictReader(open(RUN / "evaluation/revised_generated_thread_scores.csv"))}

scorer = PolitenessScorer("Intel/polite-guard", "auto", 256)
SENT = re.compile(r"(?<=[.!?])\s+")

gen_threads = {}
for d in sorted((RUN / "cleaned").glob("run_*_sampled_reddit")):
    cbt, _ = load_generated_comments(d)
    for tid, cs in cbt.items():
        gen_threads[int(tid.split("seed")[-1])] = cs

class T:
    def __init__(self, text): 
        self.text=text; self.thread_id=""; self.thread_title=""; self.comment_id=""
        self.parent_id=""; self.author=""; self.depth=0

def score_texts(texts):
    return scorer.score_comments([T(t) for t in texts], batch_size=32, include_text=False)

base = {s: score_texts([c.text for c in cs]) for s, cs in gen_threads.items()}
rate = lambda rows: sum(1 for x in rows if x["pred_label"]=="polite")/len(rows) if rows else 0.0
e6 = max(abs(rate(base[s]) - float(gen_csv[s]["polite_rate"])) for s in base)
print(f"[E6] generated polite_rate max |reproduced - shipped| = {e6:.5f}")
print(f"     thread-mean polite_rate {st.mean([rate(base[s]) for s in base]):.4f}  (real 0.3020)")

flat = [(s, i, x) for s in base for i, x in enumerate(base[s])]
nonp = sorted([t for t in flat if t[2]["pred_label"] != "polite"],
              key=lambda t: -t[2]["polite_probability"])
need = int(round(0.3020*len(flat))) - sum(1 for t in flat if t[2]["pred_label"]=="polite")
print(f"\ncomments that must flip: {need} of {len(flat)}")
gaps = [max(t[2]["class_probabilities"].values()) - t[2]["polite_probability"] for t in nonp[:need]]
print(f"the {need} closest need P(polite) to rise by a median of {st.median(gaps):.3f} "
      f"(their current P(polite) runs {nonp[0][2]['polite_probability']:.3f} down to {nonp[need-1][2]['polite_probability']:.3f})")

# real appreciative carrier sentences from threads OUTSIDE the matched 50
carriers = []
for d in sorted((REPO/"data/raw/discussions/camera_product").iterdir()):
    if not d.is_dir() or len(carriers) > 4000: continue
    try: cbt,_ = load_real_comments(d)
    except Exception: continue
    for tid, cs in cbt.items():
        if tid in matched_rids: continue
        for c in cs:
            for sent in SENT.split(c.text):
                w = sent.split()
                if 3 <= len(w) <= 18: carriers.append(sent.strip())
print(f"\ncandidate sentences from non-matched real threads: {len(carriers)}")
cs_rows = score_texts(carriers[:4000])
polite_carriers = [carriers[i] for i,x in enumerate(cs_rows) if x["pred_label"]=="polite" and x["polite_probability"]>0.90]
print(f"of those, {len(polite_carriers)} score polite at P>0.90 -- the carrier pool")
print("examples:", [c[:60] for c in polite_carriers[:4]])

rng = random.Random(11)
print(f"\n{'share of generated comments given ONE real carrier sentence':<58}{'polite_rate':>12}")
for share in (0.00, 0.05, 0.10, 0.15, 0.20):
    vals=[]
    for s, cs in gen_threads.items():
        texts=[]
        for c in cs:
            if rng.random() < share:
                texts.append(rng.choice(polite_carriers) + " " + c.text)
            else:
                texts.append(c.text)
        vals.append(rate(score_texts(texts)))
    print(f"  {share:>4.0%}{'':<52}{st.mean(vals):>12.4f}")
print(f"  real                                                      {0.3020:>12.4f}")
