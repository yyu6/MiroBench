#!/usr/bin/env python3
"""Price a subject re-mention cap on `self_bleu_4` before building it (rule J7).

The v109 gate closed the naming-*shape* gap: mentions per distinct designator
went 4.286 (v108) -> 2.333 (v109) against a matched real 2.432, and distinct
designators 21 -> 69 against 118. What did not close is the volume of the one
name everybody uses. On the same seed-8 thread, generated names the seed
product's designator **81** times in 186 comments; matched real names its top
designator **40** times. Every repeated 4-gram in the v109 audit is a form of
that phrase (`the canon eos r5` in 18 comments, `on the canon eos r5` in 6).

`self_bleu_decomposition` also shows the excess has moved: in v109 the pooled
3- and 4-gram precisions are now *below* real (0.95x, 0.93x) and the entire
remaining log excess is p1 (56.3%), p2 (54.9%) and the brevity penalty. p1 is
plain vocabulary overlap, and the thread's vocabulary breadth is the measured
deficit: types/sqrt(tokens) 14.73 generated against 18.68 real.

So the next mechanism is not another offer, it is a **cap**: stop the Writer
restating the thread's subject by name when a pronoun would do. This script
prices that cap exactly, by editing the shipped artifact's text and re-running
the project's own scorer:

  - fidelity first: recomputed `self_bleu_4` must reproduce the shipped value.
  - ablation: reduce the top designator's mentions to the matched real thread's
    own count, replacing the excess noun phrase with `it`, then rescore.
  - the reverse direction on real, so the relationship is shown to be
    asymmetric rather than assumed.

Per J7 the number this prints is an **upper bound**, not a budget: it edits text
that a prompt rule can only influence.

    python3 generalized_card/analysis/subject_rename_ablation.py
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCORER_DIR = REPO / "scripts" / "evaluation"
PACKAGE_ROOT = REPO / "generalized_card"
for extra in (str(SCORER_DIR), str(PACKAGE_ROOT)):
    if extra not in sys.path:
        sys.path.insert(0, extra)

from generalized_card.content_profile_analysis import DESIGNATOR  # noqa: E402
from score_thread_self_bleu import pairwise_self_bleu_for_order, tokenize  # noqa: E402
from score_thread_semantic_uniformity import (  # noqa: E402
    load_generated_comments,
    load_real_comments,
)

RUNS = REPO / "artifacts/generalized_card/runs"
TREATED = "v109_entity_spread_seed8_20260824_v1"
SEED_POOL = REPO / "artifacts/generalized_card/seed_pools/camera_product_150_seed42.json"
REAL_DIR = REPO / "data/raw/discussions/camera_product"
SEED_SUFFIX = re.compile(r"seed(\d+)$")

# The designator plus the optional brand/line words and determiner in front of
# it, so the excess mention is removed as a noun phrase rather than leaving
# "on the canon eos it".
SUBJECT_NP = re.compile(
    r"\b(?:the|my|a|an|this|that)?\s*(?:canon\s+)?(?:eos\s+)?(r5c?|r6)\b",
    re.IGNORECASE,
)


def mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else float("nan")


def self_bleu_4(texts: list[str]) -> float:
    """The shipped metric: mean symmetric pairwise BLEU-4 over the thread."""

    return pairwise_self_bleu_for_order([tokenize(t) for t in texts], 4)


def designator_counts(texts: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for text in texts:
        for match in DESIGNATOR.finditer(text):
            key = match.group().casefold()
            counts[key] = counts.get(key, 0) + 1
    return counts


def cap_subject(texts: list[str], budget: int) -> tuple[list[str], int]:
    """Replace the subject noun phrase with `it` past a per-thread budget.

    Deterministic and order-preserving: the first `budget` mentions in reading
    order are kept, every later one becomes `it`. Reading order is the closest
    available stand-in for "the Writer had already named it, so a pronoun is
    enough now", which is what the prompt rule would say.
    """

    kept = 0
    replaced = 0
    out: list[str] = []
    for text in texts:
        pieces: list[str] = []
        cursor = 0
        for match in SUBJECT_NP.finditer(text):
            pieces.append(text[cursor : match.start()])
            if kept < budget:
                pieces.append(match.group())
                kept += 1
            else:
                lead = match.group()[: len(match.group()) - len(match.group().lstrip())]
                pieces.append(f"{lead}it")
                replaced += 1
            cursor = match.end()
        pieces.append(text[cursor:])
        out.append("".join(pieces))
    return out, replaced


def load() -> tuple[list[str], list[str], float]:
    run = RUNS / TREATED
    sim_dir = next(iter(sorted(run.glob("cleaned/run_*_sampled_reddit"))))
    generated_by_thread, _ = load_generated_comments(sim_dir)
    thread_id, comments = next(iter(generated_by_thread.items()))
    shipped = json.loads((sim_dir / "self_bleu_results.json").read_text(encoding="utf-8"))
    shipped_value = next(
        row["self_bleu_4"] for row in shipped["threads"] if row["thread_id"] == thread_id
    )

    pool = json.loads(SEED_POOL.read_text(encoding="utf-8"))
    seed_index = int(SEED_SUFFIX.search(thread_id).group(1))
    seed = next(row for row in pool["seed_posts"] if int(row["seed_index"]) == seed_index)
    real_by_thread, _ = load_real_comments(REAL_DIR / str(seed["source_product_dir"]))
    real = real_by_thread[str(seed["source_raw_post_id"])]

    return [c.text for c in comments], [c.text for c in real], float(shipped_value)


def main() -> None:
    argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    ).parse_args()

    generated, real, shipped = load()
    recomputed = self_bleu_4(generated)
    print("\n== fidelity: recomputed self_bleu_4 must reproduce the shipped value ==\n")
    print(f"  shipped={shipped:.12f} recomputed={recomputed:.12f} delta={abs(shipped - recomputed):.2e}")
    if abs(shipped - recomputed) > 1e-9:
        raise SystemExit("recomputation does not reproduce the shipped metric")

    gen_counts = designator_counts(generated)
    real_counts = designator_counts(real)
    gen_top_name, gen_top = max(gen_counts.items(), key=lambda kv: kv[1])
    real_top_name, real_top = max(real_counts.items(), key=lambda kv: kv[1])
    real_bleu = self_bleu_4(real)

    print("\n== the residual being priced ==\n")
    print(f"  generated top designator   {gen_top_name!r} x{gen_top} of {sum(gen_counts.values())} mentions")
    print(f"  matched real top designator {real_top_name!r} x{real_top} of {sum(real_counts.values())} mentions")
    print(f"  generated self_bleu_4 {recomputed:.6f}   real {real_bleu:.6f}   gap {recomputed - real_bleu:+.6f}")

    print("\n== ablation: cap the generated subject's mentions ==\n")
    print(f"  {'budget':>8s} {'replaced':>9s} {'self_bleu_4':>12s} {'gap vs real':>12s} {'gap closed':>11s}")
    base_gap = recomputed - real_bleu
    for budget in (real_top, 60, 30, 20, 10, 0):
        capped, replaced = cap_subject(generated, budget)
        value = self_bleu_4(capped)
        gap = value - real_bleu
        print(
            f"  {budget:8d} {replaced:9d} {value:12.6f} {gap:+12.6f} "
            f"{(base_gap - gap) / base_gap:10.1%}"
        )

    print("\n== reverse direction on real, to show the relationship is asymmetric ==\n")
    for budget in (real_top, 10, 0):
        capped, replaced = cap_subject(real, budget)
        value = self_bleu_4(capped)
        print(f"  real capped at {budget:3d} (replaced {replaced:3d}) -> self_bleu_4 {value:.6f}")

    print("\n  J7: these are upper bounds. A prompt rule influences the Writer's")
    print("  input, not its output, so the shipped effect will be a fraction of this.")


if __name__ == "__main__":
    main()
