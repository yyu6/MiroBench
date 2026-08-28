#!/usr/bin/env python3
"""Paired per-seed test of an arm against its base run.

The shipped eval compares generated against real as two INDEPENDENT samples
(Cliff d, MWU). For deciding whether an arm changed anything, that throws away
the pairing: the arm and its base ran the same seeds against the same matched
real threads, so the per-seed difference is the right statistic and it has far
more power at N=10. sd of Cliff d at N=10 is 0.265; a paired Wilcoxon on ten
seeds resolves effects several times smaller.
"""
from __future__ import annotations
import argparse, csv, itertools, math, statistics
from pathlib import Path
RUNS = Path("/Users/yaoningyu/Desktop/UIUC/GEO/artifacts/generalized_card/runs")
METRICS = ["self_bertscore_mean_f1", "self_bleu_4", "semantic_mean_cosine",
           "polite_rate", "impolite_rate", "neutral_rate", "emotion_entropy",
           "hard_disagree_rate", "length_cv", "mean_story_probability"]

def load(tag, which):
    p = RUNS/tag/"matched_evaluation"/f"matched_{which}_thread_scores.csv"
    if not p.exists(): return None
    out = {}
    for x in csv.DictReader(open(p)):
        k = x.get("matched_seed_idx") or x.get("seed_index")
        if k is None: continue
        out[int(k)] = x
    return out

def wilcoxon_p(diffs):
    """Exact two-sided signed-rank p for small n (n<=12), ties dropped."""
    d = [x for x in diffs if x != 0]
    n = len(d)
    if n < 3: return float("nan")
    order = sorted(range(n), key=lambda i: abs(d[i]))
    ranks = [0.0]*n
    i = 0
    while i < n:
        j = i
        while j+1 < n and abs(d[order[j+1]]) == abs(d[order[i]]): j += 1
        avg = (i+j)/2 + 1
        for k in range(i, j+1): ranks[order[k]] = avg
        i = j+1
    w = sum(ranks[i] for i in range(n) if d[i] > 0)
    if n > 12:
        mu = n*(n+1)/4; sd = math.sqrt(n*(n+1)*(2*n+1)/24)
        z = (w-mu)/sd
        return 2*(1-0.5*(1+math.erf(abs(z)/math.sqrt(2))))
    total = 0; extreme = 0
    tgt = min(w, n*(n+1)/2 - w)
    for signs in itertools.product([0,1], repeat=n):
        s = sum(ranks[i] for i in range(n) if signs[i])
        total += 1
        if min(s, n*(n+1)/2 - s) <= tgt: extreme += 1
    return extreme/total

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="v128_interaction_n10_20260828_v1")
    ap.add_argument("--arm", required=True)
    a = ap.parse_args()
    B, A = load(a.base, "generated"), load(a.arm, "generated")
    R = load(a.arm, "real") or load(a.base, "real")
    if not A: raise SystemExit(f"no matched_evaluation for {a.arm}")
    seeds = sorted(set(A) & set(B) & set(R))
    print(f"paired on {len(seeds)} seeds: {seeds}\n")
    print(f"{'metric':<26} {'base gap':>9} {'arm gap':>9} {'change':>9} {'better':>7} {'wilcoxon p':>11}")
    print("-"*76)
    for m in METRICS:
        try:
            bg = [float(B[s][m]) - float(R[s][m]) for s in seeds]
            ag = [float(A[s][m]) - float(R[s][m]) for s in seeds]
        except (KeyError, ValueError):
            continue
        # closer to real = smaller |gap|
        d = [abs(x) - abs(y) for x, y in zip(bg, ag)]   # >0 means arm is closer
        wins = sum(1 for x in d if x > 0)
        print(f"{m:<26} {statistics.mean(bg):+9.4f} {statistics.mean(ag):+9.4f} "
              f"{statistics.mean(ag)-statistics.mean(bg):+9.4f} {wins:>4}/{len(seeds)} {wilcoxon_p(d):11.4f}")
    print("\n'better' counts seeds where the arm is closer to its own matched real.")
    print("wilcoxon p is on the paired change in |gap|; small p = the arm moved it consistently.")

if __name__ == "__main__":
    main()
