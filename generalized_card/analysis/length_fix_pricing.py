#!/usr/bin/env python3
"""Price the beat-budget length fix on `self_bleu_4`, exactly, over all 10 threads.

`length_instrument_rdd.py` establishes that the enumerated per-slot beat plan is
the causal instrument on realized length and that it never reaches the two bands
carrying 88% of the word deficit. This script asks the next question: if those
bands realized at the compliance the enumerated band already achieves
(realized/assigned = 0.959, measured at assigned [101,150)), how much of
`self_bleu_4`'s gap closes?

Two estimators, because they disagree by ~3x and the honest answer is the range:

  cells       the Oaxaca-style pair reweighting `composition_decomposition.py`
              uses for G43 -- generated's within-cell means held fixed, pair
              cells from real's own quantiles. Reported at 5/10/20/40 bins
              because the estimator is granularity-sensitive.
  continuous  within-thread OLS of each comment's mean pairwise BLEU on log
              tokens, applied to each comment's own log-length change. Sees
              within-cell lengthening, which the cell estimator cannot.

Unit discipline: `real_word_count` is a WHITESPACE word count while the scorer's
`tokenize` splits punctuation (1.15x inflation). The counterfactual is computed
as a ratio entirely in whitespace words and then applied to the token length.
Getting this wrong understates the fix by ~3x.

`self_bleu_4` is free to compute exactly, so nothing here is approximated, and
every thread's recomputed metric is checked against the shipped value first
(rule E6).
"""
from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "evaluation"))
from score_thread_self_bleu import symmetric_pair_bleu, tokenize  # noqa: E402
from score_thread_semantic_uniformity import (  # noqa: E402
    load_generated_comments,
    load_real_comments,
)

RUNS = REPO / "artifacts/generalized_card/runs"
SEED_POOL = REPO / "artifacts/generalized_card/seed_pools/camera_product_150_seed42.json"
REAL_DIR = REPO / "data/raw/discussions/camera_product"
SEED_SUFFIX = re.compile(r"seed(\d+)$")
TARGET_RATIO = 0.959
BANDS_NARROW = [(35, 100)]
BANDS_WIDE = [(35, 100), (252, 10 ** 6)]

mean = lambda values: statistics.fmean(values) if values else float("nan")  # noqa: E731


def load(tag: str) -> list[dict]:
    run = RUNS / tag
    assigned: dict[tuple[str, str], int] = {}
    for path in sorted(run.glob("generated/**/generation_records.json")):
        for rec in json.load(open(path)):
            key = (str(rec.get("post_id") or ""), str((rec.get("comment") or {}).get("comment_id", "")))
            assigned[key] = int((rec.get("task") or {}).get("real_word_count") or 0)
    pool = {int(r["seed_index"]): r for r in json.load(open(SEED_POOL))["seed_posts"]}
    threads = []
    for sim_dir in sorted(run.glob("cleaned/run_*_sampled_reddit")):
        generated_by_thread, _ = load_generated_comments(sim_dir)
        shipped = {r["thread_id"]: r["self_bleu_4"]
                   for r in json.load(open(sim_dir / "self_bleu_results.json"))["threads"]}
        for thread_id, comments in generated_by_thread.items():
            seed = pool[int(SEED_SUFFIX.search(thread_id).group(1))]
            real_by_thread, _ = load_real_comments(REAL_DIR / str(seed["source_product_dir"]))
            real = real_by_thread[str(seed["source_raw_post_id"])]
            gen_tokens = [tokenize(c.text) for c in comments]
            real_tokens = [tokenize(c.text) for c in real]
            gen_pairs = [(symmetric_pair_bleu(gen_tokens[i], gen_tokens[j], 4), i, j)
                         for i in range(len(gen_tokens)) for j in range(i + 1, len(gen_tokens))]
            recomputed = mean([v for v, _, _ in gen_pairs])
            if abs(recomputed - shipped[thread_id]) > 1e-9:
                raise SystemExit(f"FIDELITY FAIL {thread_id}: {recomputed} != {shipped[thread_id]}")
            real_pairs = [(symmetric_pair_bleu(real_tokens[i], real_tokens[j], 4),
                           float(len(real_tokens[i])), float(len(real_tokens[j])))
                          for i in range(len(real_tokens)) for j in range(i + 1, len(real_tokens))]
            threads.append({
                "thread_id": thread_id, "gen_pairs": gen_pairs, "real_pairs": real_pairs,
                "tokens": [float(len(t)) for t in gen_tokens],
                "words": [float(len(c.text.split())) for c in comments],
                "assigned": [assigned.get((thread_id, c.comment_id), 0) for c in comments],
                "shipped": shipped[thread_id],
            })
    print(f"fidelity: recomputed self_bleu_4 reproduces the shipped value on all "
          f"{len(threads)} threads (max delta < 1e-9)")
    return threads


def counterfactual_tokens(thread: dict, bands, full: bool) -> list[float]:
    out = []
    for token_len, word_len, assigned in zip(thread["tokens"], thread["words"], thread["assigned"]):
        if assigned <= 0 or word_len <= 0:
            out.append(token_len)
            continue
        hit = full or any(low <= assigned <= high for low, high in bands)
        target = assigned if full else TARGET_RATIO * assigned
        out.append(token_len * (target / word_len) if (hit and target > word_len) else token_len)
    return out


def bucket(value, cuts):
    for index, cut in enumerate(cuts):
        if value <= cut:
            return index
    return len(cuts)


def price_cells(threads, bins, bands, full=False):
    quantiles = [i / bins for i in range(1, bins)]
    got, fixed, real_mean = [], [], []
    for thread in threads:
        lengths = [v for _, a, b in thread["real_pairs"] for v in (a, b)]
        percentile = statistics.quantiles(sorted(lengths), n=1000)
        cuts = [percentile[max(0, int(q * 1000) - 1)] for q in quantiles]
        cell = defaultdict(list)
        for value, i, j in thread["gen_pairs"]:
            cell[tuple(sorted((bucket(thread["tokens"][i], cuts), bucket(thread["tokens"][j], cuts))))].append(value)
        cell_mean = {k: mean(v) for k, v in cell.items()}
        counter = counterfactual_tokens(thread, bands, full)
        weights = defaultdict(int)
        for _, i, j in thread["gen_pairs"]:
            weights[tuple(sorted((bucket(counter[i], cuts), bucket(counter[j], cuts))))] += 1
        num = den = 0.0
        for key, count in weights.items():
            if key in cell_mean:
                num += count * cell_mean[key]
                den += count
        got.append(thread["shipped"])
        fixed.append(num / den if den else float("nan"))
        real_mean.append(mean([v for v, _, _ in thread["real_pairs"]]))
    g, f, r = mean(got), mean(fixed), mean(real_mean)
    return (g - f) / (g - r)


def price_continuous(threads, bands, full=False):
    got, fixed, real_mean = [], [], []
    for thread in threads:
        per = defaultdict(list)
        for value, i, j in thread["gen_pairs"]:
            per[i].append(value)
            per[j].append(value)
        index = sorted(per)
        x = np.log([max(1.0, thread["tokens"][i]) for i in index])
        y = np.array([mean(per[i]) for i in index])
        design = np.column_stack([np.ones(len(x)), x])
        slope = np.linalg.lstsq(design, y, rcond=None)[0][1]
        counter = counterfactual_tokens(thread, bands, full)
        n_pairs = len(thread["gen_pairs"])
        delta = sum((len(per[i]) / n_pairs) * slope
                    * (math.log(max(1.0, counter[i])) - math.log(max(1.0, thread["tokens"][i]))) * 0.5
                    for i in index)
        got.append(thread["shipped"])
        fixed.append(thread["shipped"] + delta)
        real_mean.append(mean([v for v, _, _ in thread["real_pairs"]]))
    g, f, r = mean(got), mean(fixed), mean(real_mean)
    return (g - f) / (g - r)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default="v110_length_transfer_n10_20260824_v1")
    args = ap.parse_args()
    threads = load(args.tag)
    cases = [("extend the beat plan to 35-100", BANDS_NARROW, False),
             ("also lift the 12-beat cap (252+)", BANDS_WIDE, False),
             ("FULL compliance (realized = assigned)", [], True)]
    print("\nshare of self_bleu_4's gap closed\n")
    print(f"  {'counterfactual':<38} {'cells@5':>8} {'@10':>6} {'@20':>6} {'@40':>6} {'continuous':>11}")
    for label, bands, full in cases:
        cells = [price_cells(threads, b, bands, full) for b in (5, 10, 20, 40)]
        cont = price_continuous(threads, bands, full)
        print(f"  {label:<38} " + " ".join(f"{c:>7.1%}" for c in cells) + f" {cont:>10.1%}")
    print("\n  The two estimator families bracket the answer; quote the range, not a point.")


if __name__ == "__main__":
    main()
