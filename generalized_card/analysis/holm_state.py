#!/usr/bin/env python3
"""Where the generator actually stands under the reporting standard now in force.

The user selected Holm-Bonferroni over the 24 tests on 2026-08-25 (`J2`, `G51`).
That changes two things this script reports and nothing else changes:

  1. what the CURRENT paid N=10 artifact scores, applying Holm to its own 24
     saved p-values rather than the raw per-test rule; and
  2. what each metric would score at N=150, which is the paper's scale and has
     never been run -- simulated from that metric's own relative bias against
     matched real, over the 763-thread real baseline.

(1) is the honest present state. (2) is what decides where the remaining work
goes, because an N=10 pass is optimistic by construction (`ORIENTATION.md` §2
trap 1: the tests are unpaired while the data is paired by seed).
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import ks_2samp, mannwhitneyu

REPO = Path(__file__).resolve().parents[2]
REAL = REPO / "artifacts/baselines/camera_product_gpt5mini/real/thread_scores.csv"
DEFAULT_TAG = "v110_length_transfer_n10_20260824_v1"
STRUCTURAL = {"avg_depth", "structural_virality"}


def holm(pvalues: dict[str, float], alpha: float = 0.05) -> dict[str, bool]:
    """Step-down Holm. Returns {name: rejected}. Survival = not rejected = PASS."""
    order = sorted(pvalues, key=lambda k: pvalues[k])
    rejected: dict[str, bool] = {}
    stopped = False
    for rank, name in enumerate(order):
        threshold = alpha / (len(order) - rank)
        if stopped or pvalues[name] > threshold:
            stopped = True
            rejected[name] = False
        else:
            rejected[name] = True
    return rejected


def load_eval(tag: str):
    path = REPO / "artifacts/generalized_card/runs" / tag / "matched_evaluation/matched_seed_group_eval.json"
    return json.loads(path.read_text(encoding="utf-8"))


def column(name: str) -> np.ndarray:
    rows = list(csv.DictReader(open(REAL)))
    return np.array([float(r[name]) for r in rows if r.get(name) not in (None, "", "nan")])


def simulate(values, bias, closure, n, reps, rng) -> float:
    effective = bias * (1.0 - closure)
    survive = 0
    for _ in range(reps):
        idx = rng.permutation(len(values))
        generated = values[idx[:n]] * (1.0 + effective)
        real = values[idx[n:2 * n]]
        mwu = mannwhitneyu(generated, real, alternative="two-sided").pvalue
        ks = ks_2samp(generated, real).pvalue
        # Bonferroni bound on Holm: conservative, so this under-reports passing.
        survive += int(mwu > 0.05 / 24 and ks > 0.05 / 24)
    return survive / reps


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default=DEFAULT_TAG)
    ap.add_argument("--reps", type=int, default=300)
    ap.add_argument("--target", type=float, default=0.80, help="Holm pass probability to solve closure for")
    ap.add_argument("--n", type=int, default=150, help="sample size to simulate (the paper's scale is 150)")
    args = ap.parse_args()
    data = load_eval(args.tag)

    pvalues, meta = {}, {}
    for metric, row in data.items():
        mwu = float(row["mwu_p_value"])
        ks = float(row["ks_p_value"])
        pvalues[f"{metric}::mwu"] = mwu
        pvalues[f"{metric}::ks"] = ks
        gen = float(row["generated_mean"])
        real = float(row["real_mean"])
        meta[metric] = (mwu, ks, gen, real, float(row.get("cliffs_delta", 0.0)))

    rejected = holm(pvalues)
    print(f"\n=== {args.tag}, N=10: the shipped raw rule vs Holm over the same 24 tests ===\n")
    print(f"  {'metric':<26} {'MWU':>9} {'KS':>9} {'Cliff':>7}  {'raw':<8} {'Holm':<6}")
    raw_pass = holm_pass = 0
    for metric, (mwu, ks, gen, real, cliff) in meta.items():
        raw = "PASS" if (mwu > 0.05 and ks > 0.05) else ("PARTIAL" if (mwu > 0.05) != (ks > 0.05) else "FAIL")
        surv = not (rejected[f"{metric}::mwu"] or rejected[f"{metric}::ks"])
        raw_pass += raw == "PASS"
        holm_pass += surv
        print(f"  {metric:<26} {mwu:>9.4f} {ks:>9.4f} {cliff:>+7.2f}  {raw:<8} {'PASS' if surv else 'FAIL':<6}")
    print(f"\n  raw: {raw_pass}/12 PASS     Holm: {holm_pass}/12 PASS")
    print("  N=10 p-values are optimistic (unpaired tests on seed-paired data), so this is")
    print("  a gate reading, not the paper's claim. The paper's scale is N=150.")

    print("\n=== the same generator at N=150, simulated, Holm (Bonferroni-bounded) ===\n")
    rng = np.random.default_rng(20260825)
    print(f"  {'metric':<26} {'rel. bias':>10} {'P(pass)':>9}  {'closure for P>=' + format(args.target, '.2f'):>22}")
    rows = []
    for metric, (mwu, ks, gen, real, cliff) in meta.items():
        if metric in STRUCTURAL:
            print(f"  {metric:<26} {'--':>10} {'--':>9}  {'copied from the real tree':>22}")
            continue
        try:
            values = column(metric)
        except KeyError:
            print(f"  {metric:<26} {'--':>10} {'--':>9}  {'not in the baseline':>22}")
            continue
        bias = (gen - real) / real if real else 0.0
        now = simulate(values, bias, 0.0, args.n, args.reps, rng)
        need = None
        if now < args.target:
            for closure in (0.25, 0.50, 0.75, 0.90, 1.00):
                if simulate(values, bias, closure, args.n, args.reps, rng) >= args.target:
                    need = closure
                    break
        label = "already there" if now >= args.target else (f"~{need:.0%}" if need else "not reachable by closure alone")
        rows.append((metric, bias, now, label))
        print(f"  {metric:<26} {bias:>+9.1%} {now:>9.2f}  {label:>22}")
    print("\n  Ranked by distance from passing at the paper's scale:")
    for metric, bias, now, label in sorted(rows, key=lambda r: r[2]):
        if now < args.target:
            print(f"    {metric:<26} P(pass)={now:.2f}  needs {label}")


if __name__ == "__main__":
    main()
