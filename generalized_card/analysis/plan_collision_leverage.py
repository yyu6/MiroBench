#!/usr/bin/env python3
"""The generator flags its own duplicate plans, fails to repair them, and ships.

Found by reading the v109 gate's own `logs/planning_quality.jsonl` rather than by
hypothesising. In the accepted plan batches for the seed-8 thread:

  - **38 of 60 batches were still `healthy: false` when accepted.**
  - **63 unresolved collision issues** remain (58 `semantic_collision`,
    3 `duplicate_claim`, 2 `duplicate_reference`), with reported plan-embedding
    similarity **mean 0.790, max 1.000**, and **55 of 63 at or above the
    configured 0.72 threshold**.
  - **87 of 186 slots (46.8%)** are named in an unresolved collision.

The repair loop (`plan_quality.repair_rounds = 3`) emits exactly the right
instruction -- "Assign a materially different local move rather than renaming the
claim" -- and when it fails, the batch is accepted anyway. So the duplication is
detected, described, and then shipped. Read by eye in the highest-similarity
realized pairs, this is unmistakable: slots 123 and 150 both realize as "Put the
same speedlight on the Canon EOS R5, set the sync point you care about, and fire
it in-camera", and slots 104 and 158 both realize as "on a motionless clip the
Canon's stabilization has nothing to work on".

This script prices the defect for both priority metrics, exactly and for free.

Both metrics are means over unordered within-thread comment pairs, so a pair
whose two slots were flagged as colliding can be re-valued at what a
*non-colliding* pair of the same length cell actually scores. That imputation is
the counterfactual "the repair loop had succeeded": the pair still exists, the
thread keeps its matched structural slots (`docs/ORIENTATION.md` §4 forbids
dropping them), only its similarity falls to the non-duplicate level.

    python3 generalized_card/analysis/plan_collision_leverage.py bleu
    python3 generalized_card/analysis/plan_collision_leverage.py bertscore \\
      --generated-pairs <gen_pairs.json>
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
from score_thread_semantic_uniformity import load_generated_comments  # noqa: E402

RUNS = REPO / "artifacts/generalized_card/runs"
TREATED = "v109_entity_spread_seed8_20260824_v1"
COLLISION_CODES = {"semantic_collision", "duplicate_claim", "duplicate_reference"}
SIMILARITY = re.compile(r"similarity=([0-9]*\.?[0-9]+)")
QUANTILES = (0.25, 0.5, 0.75)


def mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else float("nan")


def collisions(tag: str) -> tuple[set[tuple[int, int]], list[float]]:
    """Unresolved (sample_id, other_sample_id) pairs from the accepted batches."""

    path = RUNS / tag / "logs/planning_quality.jsonl"
    flagged: set[tuple[int, int]] = set()
    sims: list[float] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if not row.get("attempts"):
            continue
        for issue in row["attempts"][-1].get("issues") or []:
            if issue.get("code") not in COLLISION_CODES:
                continue
            left, right = issue.get("sample_id"), issue.get("other_sample_id")
            if left is None or right is None:
                continue
            flagged.add((min(int(left), int(right)), max(int(left), int(right))))
            found = SIMILARITY.search(issue.get("message", ""))
            if found:
                sims.append(float(found.group(1)))
    return flagged, sims


def slot_index(tag: str) -> dict[str, int]:
    """comment_id -> the Planner sample id the collision log refers to."""

    out: dict[str, int] = {}
    for path in sorted((RUNS / tag).glob("generated/run_*/generation_records.json")):
        for rec in json.loads(path.read_text(encoding="utf-8")):
            comment = rec.get("comment") or {}
            task = rec.get("task") or {}
            if not (comment.get("content") or "").strip():
                continue
            sample = task.get("real_sample_id")
            if sample is None:
                continue
            out[str(comment.get("comment_id"))] = int(sample)
    return out


def cuts_from(values: list[float]) -> list[float]:
    ordered = sorted(values)
    return [statistics.quantiles(ordered, n=100)[int(q * 100) - 1] for q in QUANTILES]


def bucket(value: float, cuts: list[float]) -> int:
    for index, cut in enumerate(cuts):
        if value <= cut:
            return index
    return len(cuts)


def price(
    label: str,
    pairs: list[tuple[float, str, str, float, float]],
    tag: str,
    real_mean: float,
) -> None:
    flagged, sims = collisions(tag)
    samples = slot_index(tag)
    cuts = cuts_from([v for _, _, _, a, b in pairs for v in (a, b)])

    is_flagged: list[bool] = []
    for _, left, right, _, _ in pairs:
        a, b = samples.get(left), samples.get(right)
        key = (min(a, b), max(a, b)) if a is not None and b is not None else None
        is_flagged.append(key in flagged)

    clean_cell: dict[tuple[int, int], list[float]] = defaultdict(list)
    for (value, _, _, la, lb), bad in zip(pairs, is_flagged):
        if not bad:
            clean_cell[tuple(sorted((bucket(la, cuts), bucket(lb, cuts))))].append(value)
    clean_mean = {key: mean(vals) for key, vals in clean_cell.items()}

    actual = mean([p[0] for p in pairs])
    flagged_vals = [p[0] for p, bad in zip(pairs, is_flagged) if bad]
    clean_vals = [p[0] for p, bad in zip(pairs, is_flagged) if not bad]

    imputed = []
    for (value, _, _, la, lb), bad in zip(pairs, is_flagged):
        cell = tuple(sorted((bucket(la, cuts), bucket(lb, cuts))))
        imputed.append(clean_mean.get(cell, value) if bad else value)
    repaired = mean(imputed)

    print(f"\n== {label}: price of the unresolved plan collisions ==\n")
    print(f"  unresolved collision pairs logged     {len(flagged)}")
    print(f"  slots named in one                    {len({s for pair in flagged for s in pair})}")
    if sims:
        print(f"  reported plan similarity              mean {mean(sims):.3f}  max {max(sims):.3f}")
    print(f"  comment pairs whose two slots collide {len(flagged_vals)} of {len(pairs)}")
    print()
    print(f"  flagged pairs, observed               {mean(flagged_vals):.6f}")
    print(f"  non-flagged pairs, observed           {mean(clean_vals):.6f}")
    print(f"  flagged minus non-flagged             {mean(flagged_vals) - mean(clean_vals):+.6f}")
    print()
    print(f"  generated as run                      {actual:.6f}")
    print(f"  generated with collisions repaired    {repaired:.6f}")
    print(f"  matched real                          {real_mean:.6f}")
    gap = actual - real_mean
    print()
    print(f"  total gap                             {gap:+.6f}")
    print(f"  closed by repairing the collisions    {actual - repaired:+.6f}  "
          f"({(actual - repaired) / gap:.1%} of the gap)")
    print("\n  J7: an upper bound. It assumes every flagged collision becomes an")
    print("  ordinary pair of the same lengths; a Planner-side fix will land under it.")


def load_generated() -> tuple[list[str], list[str], float]:
    sim_dir = next(iter(sorted((RUNS / TREATED).glob("cleaned/run_*_sampled_reddit"))))
    by_thread, _ = load_generated_comments(sim_dir)
    thread_id, comments = next(iter(by_thread.items()))
    shipped = json.loads((sim_dir / "self_bleu_results.json").read_text(encoding="utf-8"))
    value = next(row["self_bleu_4"] for row in shipped["threads"] if row["thread_id"] == thread_id)
    return [c.comment_id for c in comments], [c.text for c in comments], float(value)


def cmd_bleu(_: Any) -> None:
    ids, texts, shipped = load_generated()
    toks = [tokenize(t) for t in texts]
    pairs = [
        (
            symmetric_pair_bleu(toks[i], toks[j], 4),
            str(ids[i]),
            str(ids[j]),
            float(len(toks[i])),
            float(len(toks[j])),
        )
        for i in range(len(toks))
        for j in range(i + 1, len(toks))
    ]
    recomputed = mean([p[0] for p in pairs])
    print("\n== fidelity ==\n")
    print(f"  self_bleu_4 shipped={shipped:.12f} recomputed={recomputed:.12f} "
          f"delta={abs(shipped - recomputed):.2e}")
    if abs(shipped - recomputed) > 1e-9:
        raise SystemExit("recomputation does not reproduce the shipped metric")
    price("self_bleu_4", pairs, TREATED, real_mean=0.0282702737326884)


def cmd_bertscore(args: Any) -> None:
    if not args.generated_pairs:
        raise SystemExit("bertscore needs --generated-pairs")
    data = json.loads(Path(args.generated_pairs).read_text(encoding="utf-8"))
    thread = data["threads"][0] if "threads" in data else data
    words = {
        str(row["comment_id"]): float(len(str(row.get("text", "")).split()))
        for row in thread["comments"]
    }
    pairs = [
        (
            float(p["bert_f1"]),
            str(p["left_comment_id"]),
            str(p["right_comment_id"]),
            words[str(p["left_comment_id"])],
            words[str(p["right_comment_id"])],
        )
        for p in thread["pairs"]
    ]
    shipped = json.loads(
        (RUNS / TREATED / "cleaned/run_00_sampled_reddit/self_bertscore_results.json").read_text(
            encoding="utf-8"
        )
    )["threads"][0]["mean_bert_f1"]
    recomputed = mean([p[0] for p in pairs])
    print("\n== fidelity ==\n")
    print(f"  self_bertscore shipped={shipped:.12f} recomputed={recomputed:.12f} "
          f"delta={abs(shipped - recomputed):.2e}")
    if abs(shipped - recomputed) > 1e-9:
        raise SystemExit("supplied pairs do not reproduce the shipped metric")
    price("self_bertscore_mean_f1", pairs, TREATED, real_mean=0.48873049542745606)


COMMANDS = {"bleu": cmd_bleu, "bertscore": cmd_bertscore}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("command", choices=[*COMMANDS, "all"])
    parser.add_argument("--generated-pairs", default="")
    args = parser.parse_args()
    for name in (list(COMMANDS) if args.command == "all" else [args.command]):
        COMMANDS[name](args)


if __name__ == "__main__":
    main()
