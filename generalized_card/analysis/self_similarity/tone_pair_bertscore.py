#!/usr/bin/env python3
"""Would rebalancing the tone mix lower self_bertscore for free?

Half the DeepSeek corpus lands in one Polite-Guard bucket (impolite 50.6% vs a
real 38.1%). If two comments that share a tone are more alike than two that do
not, then the tone quota fix priced separately for `impolite_rate` also buys
`self_bertscore` -- and the size of that second payment is algebraic from the
same three pair populations the cross-model test uses.

Reported as observed pair means, not as a claim about a thread that was never
generated: a real rebalance changes which comments exist, not just their labels.
"""
from __future__ import annotations
import argparse, glob, json, random, statistics, sys, itertools
from pathlib import Path
from collections import defaultdict

REPO = Path("/Users/yaoningyu/Desktop/UIUC/GEO")
sys.path.insert(0, str(REPO / "scripts" / "evaluation"))
LAB = ["polite", "somewhat_polite", "neutral", "impolite"]
DS = ["v137ds_10_20260829_v1", "v137ds_40more_20260829_v1",
      "v137ds_s21_20260830_v2", "v137ds_s36_20260830_v2"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs-per-cell", type=int, default=260)
    ap.add_argument("--batch-size", type=int, default=32)
    a = ap.parse_args()
    rng = random.Random(0)

    by_thread = defaultdict(lambda: defaultdict(list))
    for t in DS:
        for run in sorted(glob.glob(str(REPO / "artifacts/generalized_card/runs" / t / "cleaned/run_*_sampled_reddit"))):
            pf = Path(run) / "politeness_results.json"
            if not pf.exists():
                continue
            for th in json.load(open(pf))["threads"]:
                for c in th["comments"]:
                    if c["text"].strip():
                        by_thread[th["thread_id"]][c["pred_label"]].append(c["text"])
    print(f"threads {len(by_thread)}")

    cells = {}
    for x, y in itertools.combinations_with_replacement(LAB, 2):
        prs = []
        tries = 0
        while len(prs) < a.pairs_per_cell and tries < 60000:
            tries += 1
            th = rng.choice(list(by_thread))
            L, R = by_thread[th].get(x, []), by_thread[th].get(y, [])
            if x == y:
                if len(L) < 2:
                    continue
                i, j = rng.sample(range(len(L)), 2)
                prs.append((L[i], L[j]))
            else:
                if not L or not R:
                    continue
                prs.append((rng.choice(L), rng.choice(R)))
        if len(prs) >= 40:
            cells[(x, y)] = prs

    sys.path.insert(0, str(REPO / "bert_score-master"))
    from bert_score import BERTScorer
    sc = BERTScorer(model_type="microsoft/deberta-xlarge-mnli", lang="en", idf=False,
                    rescale_with_baseline=False, device="cpu", batch_size=a.batch_size)
    if getattr(sc._tokenizer, "model_max_length", 0) > 100000:
        sc._tokenizer.model_max_length = 512

    F = {}
    for (x, y), prs in cells.items():
        _, _, f1 = sc.score([p for p, _ in prs], [q for _, q in prs], batch_size=a.batch_size)
        F[(x, y)] = F[(y, x)] = statistics.mean(float(v) for v in f1)
        print(f"  {x:16} x {y:16} n={len(prs):>4}  F1 = {F[(x, y)]:.5f}")

    def expected(mix):
        num = den = 0.0
        for x in LAB:
            for y in LAB:
                if (x, y) not in F:
                    continue
                w = mix[LAB.index(x)] * mix[LAB.index(y)]
                num += w * F[(x, y)]; den += w
        return num / den if den else float("nan")

    now = [0.287, 0.062, 0.146, 0.506]
    fixed = [0.355, 0.089, 0.175, 0.381]
    real = [0.355, 0.089, 0.175, 0.381]
    print(f"\n  expected pair F1 at TODAY's mix   {now}  = {expected(now):.5f}")
    print(f"  expected pair F1 at the FIXED mix {fixed} = {expected(fixed):.5f}")
    print(f"  free change in self_bertscore from the tone fix alone : {expected(fixed)-expected(now):+.5f}")
    print(f"  (the gap to close is +0.0060; d +0.17)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
