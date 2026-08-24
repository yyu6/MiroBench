#!/usr/bin/env python3
"""How much of each priority metric's gap is length/depth *composition*?

The user's target is not `p > 0.05`, it is `p ~ 0.5-0.6`: the two distributions
have to overlap, which needs Cliff's delta near zero rather than a shrunken mean
gap. `self_bleu_4` currently has Cliff **+0.42 with 10/10 threads on the same
side** and `self_bertscore_mean_f1` is worse, so a 5-9% mechanism cannot get
there. That makes it worth knowing, before building anything, which part of each
gap is **composition** -- the mix of comment lengths and depths the thread is
made of -- and which part is **content**, the similarity of two comments that are
already the same length and depth.

The distinction matters because the two are not equally movable. Composition is
mechanical, already under Planner control (every slot carries `real_word_count`
copied from its matched real comment), domain-neutral, and verifiable offline.
Content is the thing four consecutive mechanisms failed to move.

Method: an exact pair-level reweighting (Oaxaca-style). Both metrics are means
over unordered within-thread comment pairs, so for any pair-level covariate the
thread mean can be re-expressed as a weighted average of per-cell means, and
generated's metric can be evaluated **at real's cell distribution**:

    actual_gen      = sum_cell  w_gen(cell)  * mean_gen(cell)
    reweighted_gen  = sum_cell  w_real(cell) * mean_gen(cell)
    real            = sum_cell  w_real(cell) * mean_real(cell)

`actual_gen - reweighted_gen` is the composition component; `reweighted_gen -
real` is the within-cell content component. Cells are built from real's own
quantiles so the binning is not tuned to generated.

`self_bleu_4` pairs are computed here with the project's scorer functions and
checked against the shipped thread value first (rule E6). `self_bertscore` pairs
have to be supplied, because the scorer does not save them by default:

    python3 scripts/evaluation/score_thread_self_bertscore.py \\
      artifacts/generalized_card/runs/<tag>/cleaned/run_00_sampled_reddit \\
      --target-kind generated --device cpu --batch-size 32 --include-pairs \\
      --output-file <gen_pairs.json>

    python3 generalized_card/analysis/composition_decomposition.py bleu
    python3 generalized_card/analysis/composition_decomposition.py bertscore \\
      --generated-pairs <gen_pairs.json> --real-pairs <real_pairs.json>
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
SCORER_DIR = REPO / "scripts" / "evaluation"
if str(SCORER_DIR) not in sys.path:
    sys.path.insert(0, str(SCORER_DIR))

from score_thread_self_bleu import symmetric_pair_bleu, tokenize  # noqa: E402
from score_thread_semantic_uniformity import (  # noqa: E402
    load_generated_comments,
    load_real_comments,
)

RUNS = REPO / "artifacts/generalized_card/runs"
TREATED = "v109_entity_spread_seed8_20260824_v1"
SEED_POOL = REPO / "artifacts/generalized_card/seed_pools/camera_product_150_seed42.json"
REAL_DIR = REPO / "data/raw/discussions/camera_product"
SEED_SUFFIX = re.compile(r"seed(\d+)$")
QUANTILES = (0.2, 0.4, 0.6, 0.8)


def mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else float("nan")


def cut_points(values: list[float]) -> list[float]:
    ordered = sorted(values)
    return [statistics.quantiles(ordered, n=100)[int(q * 100) - 1] for q in QUANTILES]


def bucket(value: float, cuts: list[float]) -> int:
    for index, cut in enumerate(cuts):
        if value <= cut:
            return index
    return len(cuts)


def cell_means(
    pairs: list[tuple[float, float, float]], cuts: list[float]
) -> tuple[dict[tuple[int, int], float], dict[tuple[int, int], int]]:
    """Per-cell mean metric and pair count, cells keyed by sorted bucket pair."""

    grouped: dict[tuple[int, int], list[float]] = defaultdict(list)
    for value, left, right in pairs:
        key = tuple(sorted((bucket(left, cuts), bucket(right, cuts))))
        grouped[key].append(value)
    return (
        {key: mean(vals) for key, vals in grouped.items()},
        {key: len(vals) for key, vals in grouped.items()},
    )


def decompose(
    generated: list[tuple[float, float, float]],
    real: list[tuple[float, float, float]],
    *,
    label: str,
    covariate: str,
) -> None:
    """Print the composition/content split for one metric and one covariate."""

    cuts = cut_points([v for _, a, b in real for v in (a, b)])
    gen_mean, gen_n = cell_means(generated, cuts)
    real_mean, real_n = cell_means(real, cuts)

    actual_gen = mean([v for v, _, _ in generated])
    actual_real = mean([v for v, _, _ in real])
    total_real = sum(real_n.values())

    shared = [key for key in real_n if key in gen_mean]
    missing = {key: real_n[key] for key in real_n if key not in gen_mean}
    covered = sum(real_n[key] for key in shared)
    reweighted = sum(real_n[key] * gen_mean[key] for key in shared) / covered

    print(f"\n== {label}: composition vs content, binned by {covariate} ==\n")
    print(f"  real {covariate} quintile cuts: " + ", ".join(f"{c:.0f}" for c in cuts))
    print(f"  real cells covered by generated: {covered}/{total_real} pairs "
          f"({covered / total_real:.1%})")
    if missing:
        print(f"  real cells generated never produces: {missing}")
    print()
    print(f"  generated, as run                {actual_gen:.6f}")
    print(f"  generated at real's {covariate:<12s} {reweighted:.6f}")
    print(f"  matched real                     {actual_real:.6f}")
    gap = actual_gen - actual_real
    content = reweighted - actual_real
    print()
    print(f"  total gap                        {gap:+.6f}")
    print(f"  composition component            {actual_gen - reweighted:+.6f}  "
          f"({(actual_gen - reweighted) / gap:6.1%} of the gap)")
    print(f"  within-cell content component    {content:+.6f}  "
          f"({content / gap:6.1%} of the gap)")

    print(f"\n  per-cell detail ({covariate} bucket pair -> mean metric):\n")
    print(f"  {'cell':>8s} {'real n':>7s} {'gen n':>7s} {'real':>9s} {'gen':>9s} {'gen-real':>9s}")
    for key in sorted(set(real_mean) | set(gen_mean)):
        print(
            f"  {str(key):>8s} {real_n.get(key, 0):7d} {gen_n.get(key, 0):7d} "
            f"{real_mean.get(key, float('nan')):9.4f} {gen_mean.get(key, float('nan')):9.4f} "
            f"{gen_mean.get(key, float('nan')) - real_mean.get(key, float('nan')):+9.4f}"
        )


def load_texts() -> tuple[list[str], list[str], float]:
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


def bleu_pairs(texts: list[str]) -> list[tuple[float, float, float]]:
    toks = [tokenize(t) for t in texts]
    out = []
    for i in range(len(toks)):
        for j in range(i + 1, len(toks)):
            out.append(
                (symmetric_pair_bleu(toks[i], toks[j], 4), float(len(toks[i])), float(len(toks[j])))
            )
    return out


def cmd_bleu(_: Any) -> None:
    generated, real, shipped = load_texts()
    gen_pairs = bleu_pairs(generated)
    recomputed = mean([v for v, _, _ in gen_pairs])
    print("\n== fidelity: recomputed self_bleu_4 must reproduce the shipped value ==\n")
    print(f"  shipped={shipped:.12f} recomputed={recomputed:.12f} delta={abs(shipped - recomputed):.2e}")
    if abs(shipped - recomputed) > 1e-9:
        raise SystemExit("recomputation does not reproduce the shipped metric")
    decompose(gen_pairs, bleu_pairs(real), label="self_bleu_4", covariate="token count")


def _pair_rows(path: Path, key: str) -> tuple[list[tuple[float, float, float]], dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    thread = data["threads"][0] if "threads" in data else data
    words = {}
    for row in thread.get("comments") or []:
        words[str(row["comment_id"])] = float(
            row.get("words") if row.get("words") is not None else len(str(row.get("text", "")).split())
        )
    depth = {str(row["comment_id"]): float(row.get("depth", 0)) for row in thread.get("comments") or []}
    out = []
    for pair in thread["pairs"]:
        left, right = str(pair["left_comment_id"]), str(pair["right_comment_id"])
        if key == "words":
            out.append((float(pair["bert_f1"]), words[left], words[right]))
        else:
            out.append((float(pair["bert_f1"]), depth[left], depth[right]))
    return out, thread


def cmd_bertscore(args: Any) -> None:
    if not args.generated_pairs or not args.real_pairs:
        raise SystemExit("bertscore needs --generated-pairs and --real-pairs")
    shipped = json.loads(
        (RUNS / TREATED / "cleaned/run_00_sampled_reddit/self_bertscore_results.json").read_text(
            encoding="utf-8"
        )
    )["threads"][0]["mean_bert_f1"]
    for covariate, key in (("word count", "words"), ("depth", "depth")):
        gen_rows, gen_thread = _pair_rows(Path(args.generated_pairs), key)
        real_rows, _ = _pair_rows(Path(args.real_pairs), key)
        if key == "words":
            recomputed = mean([v for v, _, _ in gen_rows])
            print("\n== fidelity: supplied generated pairs must reproduce the shipped metric ==\n")
            print(f"  shipped={shipped:.12f} recomputed={recomputed:.12f} "
                  f"delta={abs(shipped - recomputed):.2e}")
            if abs(shipped - recomputed) > 1e-9:
                raise SystemExit("supplied pairs do not reproduce the shipped metric")
        decompose(gen_rows, real_rows, label="self_bertscore_mean_f1", covariate=covariate)


COMMANDS = {"bleu": cmd_bleu, "bertscore": cmd_bertscore}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("command", choices=[*COMMANDS, "all"])
    parser.add_argument("--generated-pairs", default="")
    parser.add_argument("--real-pairs", default="")
    args = parser.parse_args()
    for name in (list(COMMANDS) if args.command == "all" else [args.command]):
        COMMANDS[name](args)


if __name__ == "__main__":
    main()
