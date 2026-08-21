#!/usr/bin/env python3
"""What "statistically indistinguishable" can actually mean here.

A perfect generator produces a second sample of real threads. So the ceiling of
any acceptance test is what that test does when **both** samples are real. This
measures it, on this domain's own per-thread metric tables, and it settles three
things the project had been assuming:

1. The current standard -- all 12 required metrics passing both Mann-Whitney and
   KS at alpha 0.05 -- fails a perfect generator **half the time at N=150**.
2. Holm-Bonferroni over the same 24 tests restores it to ~0.95, which is what a
   correct standard has to read.
3. The standing `|Cliff| <= 0.10` steering target sits **below the noise floor**.
   At N=10 the null 95th percentile of `|Cliff|` is ~0.52 for every metric, so
   any N=10 reading under about 0.5 is indistinguishable from two real samples.

Domain-portable by construction: it reads
`data/raw/discussions/<domain>/*/thread_metrics_summary.csv`, so the same three
numbers can be produced for laptop, cell_phone or headphone without changing a
line. It makes no API call and loads no model.

    python3 generalized_card/analysis/acceptance_standard.py
    python3 generalized_card/analysis/acceptance_standard.py --domain headphone_product
"""

from __future__ import annotations

import argparse
import csv
import glob
import random
import statistics
from pathlib import Path

import numpy as np
from scipy import stats as ss

REPO = Path(__file__).resolve().parents[2]
# The 12 the matched evaluation requires; see `run_evaluate.REQUIRED_THREAD_METRICS`.
METRICS = (
    "self_bleu_4", "self_bertscore_mean_f1", "semantic_mean_cosine",
    "hard_disagree_rate", "polite_rate", "impolite_rate", "neutral_rate",
    "length_cv", "avg_depth", "structural_virality", "mean_story_probability",
    "emotion_entropy",
)
ALPHA = 0.05


def load(domain: str, min_comments: int) -> list[dict[str, float]]:
    """Per-thread metric rows, deduplicated -- one Reddit post can sit under two
    product folders, which double-counts it (see `tasks/lessons.md`)."""

    rows: list[dict[str, float]] = []
    seen: set[str] = set()
    pattern = str(REPO / "data/raw/discussions" / domain / "*/thread_metrics_summary.csv")
    for path in sorted(glob.glob(pattern)):
        with open(path) as handle:
            for row in csv.DictReader(handle):
                thread_id = str(row.get("thread_id") or "").strip()
                if not thread_id or thread_id in seen:
                    continue
                try:
                    if int(float(row.get("comment_count") or 0)) < min_comments:
                        continue
                    values = {metric: float(row[metric]) for metric in METRICS}
                except (KeyError, TypeError, ValueError):
                    continue
                seen.add(thread_id)
                rows.append(values)
    return rows


def _tests(a: list[dict], b: list[dict]) -> tuple[list[float], list[float]]:
    """Return the 24 p-values and the 12 |Cliff| values for one real/real split."""

    ps: list[float] = []
    cliffs: list[float] = []
    for metric in METRICS:
        x = np.array([row[metric] for row in a])
        y = np.array([row[metric] for row in b])
        try:
            u, mwu = ss.mannwhitneyu(x, y, alternative="two-sided")
        except ValueError:
            u, mwu = len(x) * len(y) / 2.0, 1.0
        _, ks = ss.ks_2samp(x, y)
        ps += [float(mwu), float(ks)]
        cliffs.append(abs(2.0 * float(u) / (len(x) * len(y)) - 1.0))
    return ps, cliffs


def holm_accepts(ps: list[float], alpha: float = ALPHA) -> bool:
    """Holm-Bonferroni: True when no hypothesis of difference is rejected."""

    order = sorted(range(len(ps)), key=lambda i: ps[i])
    total = len(ps)
    for rank, index in enumerate(order):
        if ps[index] <= alpha / (total - rank):
            return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", default="camera_product")
    parser.add_argument("--min-comments", type=int, default=5)
    parser.add_argument("--repeats-small", type=int, default=3000)
    parser.add_argument("--repeats-large", type=int, default=700)
    parser.add_argument("--seed", type=int, default=20260821)
    args = parser.parse_args()

    rows = load(args.domain, args.min_comments)
    print(f"domain {args.domain}: {len(rows)} distinct real threads with "
          f">= {args.min_comments} comments and all 12 metrics")

    plans = [(10, args.repeats_small), (150, args.repeats_large)]
    summary: dict[int, dict[str, object]] = {}
    for n, repeats in plans:
        if 2 * n > len(rows):
            print(f"\nN={n}: not enough threads for two disjoint samples")
            continue
        rng = random.Random(args.seed + n)
        raw = holm = tight = 0
        counts: list[int] = []
        per_metric = {metric: 0 for metric in METRICS}
        cliff_draws = {metric: [] for metric in METRICS}
        for _ in range(repeats):
            pick = rng.sample(rows, 2 * n)
            ps, cliffs = _tests(pick[:n], pick[n:])
            passed = [ps[2 * i] > ALPHA and ps[2 * i + 1] > ALPHA for i in range(len(METRICS))]
            counts.append(sum(passed))
            raw += all(passed)
            holm += holm_accepts(ps)
            tight += all(c <= 0.10 for c in cliffs)
            for metric, ok, cliff in zip(METRICS, passed, cliffs):
                per_metric[metric] += ok
                cliff_draws[metric].append(cliff)
        summary[n] = {
            "raw": raw / repeats,
            "holm": holm / repeats,
            "tight": tight / repeats,
            "counts": counts,
            "per_metric": {m: v / repeats for m, v in per_metric.items()},
            "floor": {m: float(np.percentile(v, 95)) for m, v in cliff_draws.items()},
        }
        ordered = sorted(counts)
        print(f"\n### N={n}, {repeats} real-vs-real repeats")
        print(f"  metrics passing: mean {statistics.mean(counts):.2f}  "
              f"p10 {ordered[repeats // 10]}  min {min(counts)}  max {max(counts)}")

    if not summary:
        return 0
    ns = sorted(summary)
    print(f"\n{'standard':<46}" + "".join(f"{'N=' + str(n):>10}" for n in ns))
    for label, key in (("current: all 24 raw p > 0.05", "raw"),
                       ("Holm-Bonferroni over the 24 tests", "holm"),
                       ("effect size only: every |Cliff| <= 0.10", "tight")):
        print(f"{label:<46}" + "".join(f"{summary[n][key]:>10.3f}" for n in ns))
    print("  A perfect generator is a second sample of real threads, so the row")
    print("  reading ~0.95 is the standard that does not fail correct work.")

    print(f"\n{'metric':<28}" + "".join(f"{'P(pass) N=' + str(n):>16}" for n in ns)
          + "".join(f"{'|Cliff| p95 N=' + str(n):>20}" for n in ns))
    for metric in METRICS:
        print(f"{metric:<28}"
              + "".join(f"{summary[n]['per_metric'][metric]:>16.3f}" for n in ns)
              + "".join(f"{summary[n]['floor'][metric]:>20.3f}" for n in ns))
    print("  A |Cliff| inside its own p95 column is indistinguishable from two")
    print("  real samples. Steer by the distance to that floor, not to zero.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
