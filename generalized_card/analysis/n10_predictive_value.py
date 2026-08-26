#!/usr/bin/env python3
"""Does a LARGE p-value at N=10 predict passing at N=150?

`ORIENTATION.md` s2 trap 1 warns that "a large p-value at N=10 is not evidence of
a match" and prices it at |Cliff|=0.25: ~87% pass at N=10 against ~4% at N=150.
That is a statement about a metric that merely *passes* at N=10. The working
assumption this measures is narrower and more defensible:

    "N=10 is the cheap measurement. If the N=10 p-value is LARGE -- 0.7, 0.8 --
     then N=150 will probably also clear 0.05."

Whether that holds is not a matter of opinion. With the true bias FIXED and the
two runs drawn independently, the N=10 p-value carries **zero** information about
N=150. It is informative only because the bias is *unknown*: a large p at N=10 is
evidence the underlying bias is small, and it is that inference the heuristic
rests on. So the honest form of the question is a posterior:

    P(passes at N=150 | the N=10 reading was p)

which this computes by simulating over a prior on the true bias, drawing an N=10
sample and an independent N=150 sample from the same biased population, and
binning the N=150 outcome by the N=10 reading. Everything is resampled from this
domain's own 763 real threads, so the marginal shape is the real one.

Usage:  python3 generalized_card/analysis/n10_predictive_value.py [--reps 4000]
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.stats import ks_2samp, mannwhitneyu

REPO = Path(__file__).resolve().parents[2]
REAL = REPO / "artifacts/baselines/camera_product_gpt5mini/real/thread_scores.csv"
METRICS = ("self_bertscore_mean_f1", "self_bleu_4", "emotion_entropy", "polite_rate")
# Bias magnitudes the generator has actually shown on these metrics across
# v110/v113/v117: self_bertscore 1.6-2.6%, self_bleu_4 8.7-18.8%,
# emotion_entropy -10.0 to +5.5%, polite_rate -47 to -56%. The prior spans zero to
# somewhat past the worst of them, in both directions.
PRIOR = {
    "self_bertscore_mean_f1": 0.06,
    "self_bleu_4": 0.30,
    "emotion_entropy": 0.20,
    "polite_rate": 0.70,
}
BANDS = ((0.05, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01))


def column(name: str) -> np.ndarray:
    rows = list(csv.DictReader(open(REAL)))
    return np.array(
        [float(r[name]) for r in rows if r.get(name) not in (None, "", "nan")]
    )


def read(values, bias, n, rng):
    idx = rng.permutation(len(values))
    generated = values[idx[:n]] * (1.0 + bias)
    real = values[idx[n:2 * n]]
    return (
        mannwhitneyu(generated, real, alternative="two-sided").pvalue,
        ks_2samp(generated, real).pvalue,
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--reps", type=int, default=4000)
    args = ap.parse_args()
    rng = np.random.default_rng(20260827)

    print(f"\nP(passes at N=150 | the N=10 reading), {args.reps} draws per metric")
    print("  'the N=10 reading' = min(MWU p, KS p), the binding one under the "
          "shipped rule")
    print("  prior on the true relative bias: uniform +-the range each metric has "
          "actually shown\n")
    for metric in METRICS:
        values = column(metric)
        span = PRIOR[metric]
        rows = []
        for _ in range(args.reps):
            bias = rng.uniform(-span, span)
            m10, k10 = read(values, bias, 10, rng)
            m150, k150 = read(values, bias, 150, rng)
            rows.append((min(m10, k10), (m150 > 0.05) and (k150 > 0.05)))
        arr = np.array(rows, dtype=float)
        overall = arr[:, 1].mean()
        passed10 = arr[arr[:, 0] > 0.05]
        print(f"  {metric}   (prior +-{span:.0%})")
        print(f"    {'N=10 reading':<22}{'n':>7}{'P(pass at N=150)':>20}")
        print(f"    {'(unconditional)':<22}{len(arr):>7}{overall:>20.2f}")
        print(f"    {'passed at N=10':<22}{len(passed10):>7}"
              f"{passed10[:, 1].mean() if len(passed10) else float('nan'):>20.2f}")
        for lo, hi in BANDS:
            sel = arr[(arr[:, 0] >= lo) & (arr[:, 0] < hi)]
            if len(sel) < 25:
                print(f"    {f'p in [{lo:.2f},{hi:.2f})':<22}{len(sel):>7}{'(thin)':>20}")
                continue
            print(f"    {f'p in [{lo:.2f},{hi:.2f})':<22}{len(sel):>7}{sel[:, 1].mean():>20.2f}")
        print()


if __name__ == "__main__":
    main()
