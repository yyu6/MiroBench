#!/usr/bin/env python3
"""How much gap closure each acceptance standard actually requires.

`docs/DECISIONS.md` G42 opens "The user's target is not `p > 0.05` but
`p ~ 0.5-0.6`" and derives from it that ~90% closure is needed at N=150. That
premise is **uncited**: no user statement of a 0.5-0.6 target exists anywhere in
this repository. The only user-quoted criterion is `docs/ORIENTATION.md` §1's
verbatim "只要是肉眼无法识别出 generated 和 real 并且 p value 大于 0.05 就可以".
The likely origin of 0.5 is §2 trap 5 / J2, where 0.50 is the *joint* rate at
which a perfect generator passes all 24 raw tests at once -- a different quantity
from a per-metric p-value.

G42's simulation is sound and is reproduced here. What it never did is apply
**J2**, this project's own VERIFIED recommendation: report under Holm-Bonferroni
over the 24 tests rather than raw p > 0.05. That changes the requirement, because
Holm makes rejection harder and therefore passing easier.

The Holm column here is computed with the Bonferroni bound (every test compared
to 0.05/24), which is uniformly more conservative than step-down Holm, so the
figures are a **lower bound** on the Holm pass rate.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.stats import ks_2samp, mannwhitneyu

REPO = Path(__file__).resolve().parents[2]
REAL = REPO / "artifacts/baselines/camera_product_gpt5mini/real/thread_scores.csv"
N_TESTS = 24

# Relative bias of generated against matched real. Recomputed 2026-08-27 from the
# newest artifact that carries each metric honestly:
#
#   self_bleu_4, self_bertscore, emotion_entropy -- `v117_calibration_20260826_v1`,
#     full coverage with all three v115/v116/v117 arms on. Its pool is NOT the
#     evaluation pool, so these are the best available reading and not a paired
#     comparison with v110/v113.
#   polite_rate / impolite_rate / neutral_rate -- `v113_v112_gate_n10_20260826_v1`.
#     The calibration run's tone numbers are meaningless by construction (its quota
#     was deliberately flat), so the last honest tone reading is the v113 gate.
#
# Trend on the shared v110/v113 evaluation pool, for scale:
#   self_bertscore +2.60% -> +2.41%;  self_bleu_4 +18.84% -> +12.96%.
BIAS = {
    "self_bleu_4": 0.0867,
    "self_bertscore_mean_f1": 0.0159,
    "emotion_entropy": -0.0999,
    "polite_rate": -0.4719,
    "impolite_rate": 0.4969,
    "neutral_rate": -0.3378,
}


def column(name: str) -> np.ndarray:
    rows = list(csv.DictReader(open(REAL)))
    return np.array([float(r[name]) for r in rows if r.get(name) not in (None, "", "nan")])


def simulate(values, bias, closure, n, reps, rng):
    effective = bias * (1.0 - closure)
    mwu, ks = [], []
    for _ in range(reps):
        idx = rng.permutation(len(values))
        generated = values[idx[:n]] * (1.0 + effective)
        real = values[idx[n:2 * n]]
        mwu.append(mannwhitneyu(generated, real, alternative="two-sided").pvalue)
        ks.append(ks_2samp(generated, real).pvalue)
    return np.array(mwu), np.array(ks)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reps", type=int, default=400)
    ap.add_argument("--n", type=int, default=150)
    args = ap.parse_args()
    rng = np.random.default_rng(20260825)

    print(f"\nP(metric passes) at N={args.n}, {args.reps} replications over "
          f"{len(column('self_bleu_4'))} real camera threads\n")
    print("  raw  = the shipped rule (run_evaluate `_metric_status`): MWU p>0.05 AND KS p>0.05")
    print("  Holm = J2's VERIFIED recommendation, Bonferroni-bounded: both p > 0.05/24\n")
    print(f"  {'metric':<26} {'closure':>8} {'raw':>7} {'Holm>=':>8}")
    for metric, bias in BIAS.items():
        values = column(metric)
        for closure in (0.0, 0.10, 0.25, 0.50, 0.75, 0.90, 1.0):
            mwu, ks = simulate(values, bias, closure, args.n, args.reps, rng)
            raw = float(np.mean((mwu > 0.05) & (ks > 0.05)))
            holm = float(np.mean((mwu > 0.05 / N_TESTS) & (ks > 0.05 / N_TESTS)))
            tag = "  <- perfect generator" if closure == 1.0 else ""
            print(f"  {metric:<26} {closure:>7.0%} {raw:>7.2f} {holm:>8.2f}{tag}")
        print()
    print("  Reading: under the shipped raw rule the operational bar is ~90% closure,")
    print("  which is G42's figure. Under J2 the same pass probability arrives at")
    print("  ~50-75% closure. Which standard is reported is the user's open decision")
    print("  (docs/ORIENTATION.md §2 trap 5); it decides what is worth building.")


if __name__ == "__main__":
    main()
