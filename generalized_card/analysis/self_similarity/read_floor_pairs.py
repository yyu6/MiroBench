#!/usr/bin/env python3
"""Stop aggregating. Compute every pairwise BERTScore for a matched real thread
and our thread, then PRINT the actual comment pairs at the same percentile of
each distribution. The floor is at the bottom, so the bottom is what to read."""
import json, sys, re, itertools, statistics
from pathlib import Path
REPO = Path("/Users/yaoningyu/Desktop/UIUC/GEO")
sys.path.insert(0, str(REPO / "scripts" / "evaluation"))
sys.path.insert(0, str(REPO / "bert_score-master"))
from score_thread_semantic_uniformity import load_generated_comments, load_real_comments

SEEDS = [int(x) for x in (sys.argv[1:] or ["4", "7"])]
pool = json.loads((REPO/"artifacts/generalized_card/seed_pools/camera_product_150_seed42.json").read_text())["seed_posts"]
by = {int(p["seed_index"]): p for p in pool}
G = {}
for x in sorted((REPO/"artifacts/generalized_card/runs/v128_interaction_n10_20260828_v1/cleaned").glob("run_*_sampled_reddit")):
    cbt, _ = load_generated_comments(x)
    for tid, cs in cbt.items(): G[int(tid.split("seed")[-1])] = [" ".join(c.text.split()) for c in cs]

from bert_score import BERTScorer
sc = BERTScorer(model_type="microsoft/deberta-xlarge-mnli", num_layers=40,
                batch_size=32, idf=False, device="cpu", lang="en", rescale_with_baseline=False)

def scored(texts):
    pr = list(itertools.combinations(range(len(texts)), 2))
    P, R, F = sc.score([texts[i] for i,_ in pr], [texts[j] for _,j in pr], batch_size=64)
    return sorted(zip(F.tolist(), pr))

def show(label, texts, ranked, qs=(0.02, 0.10, 0.50)):
    print("\n" + "="*84); print(label); print("="*84)
    for q in qs:
        f, (i, j) = ranked[min(int(q*len(ranked)), len(ranked)-1)]
        print(f"\n--- {int(q*100)}th percentile pair   BERTScore F1 = {f:.4f} ---")
        print("  A:", texts[i][:320] + ("..." if len(texts[i])>320 else ""))
        print("  B:", texts[j][:320] + ("..." if len(texts[j])>320 else ""))

for S in SEEDS:
    p = by[S]
    real = [" ".join(c.text.split()) for c in (load_real_comments(REPO/"data/raw/discussions/camera_product"/p["source_product_dir"])[0].get(p["source_raw_post_id"]) or [])]
    ours = G.get(S) or []
    if len(real) < 12 or len(ours) < 12: continue
    real = real[:44]; ours = ours[:44]
    rr, ro = scored(real), scored(ours)
    print("\n\n" + "#"*84)
    print(f"# SEED {S}: {p.get('title','')[:64]}")
    print(f"# real  {len(real)} comments, {len(rr)} pairs, mean F1 {statistics.mean(f for f,_ in rr):.4f}")
    print(f"# ours  {len(ours)} comments, {len(ro)} pairs, mean F1 {statistics.mean(f for f,_ in ro):.4f}")
    print("#"*84)
    show("REAL", real, rr)
    show("OURS (v128)", ours, ro)
