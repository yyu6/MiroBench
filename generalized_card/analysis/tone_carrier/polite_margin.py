"""How far is the generator from real's polite_rate, in decision-margin terms?

Reproduces the shipped per-thread polite_rate on both sides first (E6), then
reports the full P(polite) distribution, so the size of the required nudge is a
measured quantity rather than an assumption.
"""
from __future__ import annotations
import csv, json, statistics as st, sys
from pathlib import Path
import numpy as np
REPO = Path("/Users/yaoningyu/Desktop/UIUC/GEO")
sys.path.insert(0, str(REPO / "scripts" / "evaluation"))
from score_thread_semantic_uniformity import load_real_comments, load_generated_comments
from score_thread_politeness import PolitenessScorer
OUT = Path("/private/tmp/claude-501/-Users-yaoningyu-Desktop-UIUC-GEO/1f41c5a0-3c0b-415c-9b23-9fb381e5c727/scratchpad")
RUN = REPO / "artifacts/generalized_card/runs/generalized_card_camera_gpt54_paper_20260825_v1"
pool = json.load(open(REPO / "artifacts/generalized_card/seed_pools/camera_product_150_seed42.json"))["seed_posts"]
by_seed = {int(x["seed_index"]): x for x in pool}
real_csv = {r["thread_id"]: r for r in csv.DictReader(open(REPO / "artifacts/baselines/camera_product_gpt5mini/real/thread_scores.csv"))}
gen_csv = {}
for r in csv.DictReader(open(RUN / "evaluation/revised_generated_thread_scores.csv")):
    gen_csv[int(float(r["seed_index"]))] = r

scorer = PolitenessScorer("Intel/polite-guard", "auto", 256)
print("device:", scorer.device, flush=True)

cache = {}
gen_threads, real_threads = {}, {}
for d in sorted((RUN / "cleaned").glob("run_*_sampled_reddit")):
    cbt, _ = load_generated_comments(d)
    for tid, cs in cbt.items():
        seed = int(tid.split("seed")[-1])
        gen_threads[seed] = cs
        p = by_seed[seed]
        rd = REPO / "data/raw/discussions/camera_product" / p["source_product_dir"]
        if rd not in cache:
            cache[rd] = load_real_comments(rd)[0]
        real_threads[seed] = cache[rd].get(p["source_raw_post_id"]) or []

def score(threads):
    out = {}
    for seed, cs in threads.items():
        rows = scorer.score_comments(cs, batch_size=32, include_text=True)
        out[seed] = rows
    return out

print("scoring generated...", flush=True)
G = score(gen_threads)
print("scoring real...", flush=True)
R = score(real_threads)

def rate(rows, label="polite"):
    return sum(1 for x in rows if x["pred_label"] == label) / len(rows) if rows else 0.0

gd = max(abs(rate(G[s]) - float(gen_csv[s]["polite_rate"])) for s in G)
rd = max(abs(rate(R[s]) - float(real_csv[by_seed[s]["source_raw_post_id"]]["polite_rate"]))
         for s in R if by_seed[s]["source_raw_post_id"] in real_csv)
print(f"\n[E6] polite_rate max |reproduced - shipped|: generated {gd:.5f}   real {rd:.5f}")
print(f"     means: real {st.mean([rate(R[s]) for s in R]):.4f}   generated {st.mean([rate(G[s]) for s in G]):.4f}")

gp = np.array([x["polite_probability"] for s in G for x in G[s]])
rp = np.array([x["polite_probability"] for s in R for x in R[s]])
gl = [x["pred_label"] for s in G for x in G[s]]
rl = [x["pred_label"] for s in R for x in R[s]]
from collections import Counter
print(f"\ncomment label mix   real {dict(Counter(rl))}")
print(f"                    gen  {dict(Counter(gl))}")
print(f"\nP(polite) distribution over comments")
print(f"{'quantile':<10}{'real':>9}{'generated':>11}")
for q in (50, 75, 90, 95, 99):
    print(f"  p{q:<7}{np.percentile(rp,q):>9.4f}{np.percentile(gp,q):>11.4f}")
print(f"  mean   {rp.mean():>11.4f}{gp.mean():>11.4f}")

need = int(round((rp_rate := (np.array(rl)=='polite').mean()) * len(gl))) - sum(1 for x in gl if x=='polite')
print(f"\ncomments that must flip to `polite` to close the gap: {need} of {len(gl)} ({100*need/len(gl):.1f}%)")
nonpolite = sorted((x["polite_probability"], x["class_probabilities"]) for s in G for x in G[s] if x["pred_label"] != "polite")
top = nonpolite[-need:] if need>0 else []
if top:
    margins = [max(cp.values()) - p for p, cp in top]
    print(f"the {need} closest non-polite generated comments need P(polite) to rise by a median of "
          f"{st.median(margins):.4f} (mean {st.mean(margins):.4f}) to become argmax")
    print(f"their current P(polite) runs {top[0][0]:.3f} to {top[-1][0]:.3f}")
json.dump({"gen":{str(s):[{k:v for k,v in x.items() if k in ('comment_id','pred_label','polite_probability','text')} for x in G[s]] for s in G},
           "real":{str(s):[{k:v for k,v in x.items() if k in ('comment_id','pred_label','polite_probability','text')} for x in R[s]] for s in R}},
          open(OUT/"polite_rows.json","w"))
print("\nwrote polite_rows.json")
