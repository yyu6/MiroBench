#!/usr/bin/env python3
"""Why do cross-branch mid-depth replies converge in realization? (G26 follow-up)

`docs/DECISIONS.md` G26 established where `self_bertscore_mean_f1`'s excess
actually lives: depth bins `[2,4)` and `[4,7)` carry **82.7%** of it, while
`[7,+)` -- the only population either shipped mechanism (v105, v108) acts on --
carries 11.9%. G3 additionally showed the mid-depth excess is **not**
within-branch (`same_branch` +0.0056, p=0.32), and G21 showed the Planner's own
plan fields already separate with depth (0.4516 -> 0.2918 for non-ancestor
pairs). So the open question is a Writer-realization one: two mid-depth replies
in *different* branches receive well-separated plans and still produce text that
is too similar.

This script tests the most direct available explanation, using the run's own
saved Writer prompts (`generation_records.json`), at zero cost:

  **Is the Writer's prompt itself converging with depth?**

If an increasing share of a slot's prompt is text it shares with every other
slot in the thread -- accumulated ledgers, the branch table, the forbidden-
subject list -- then the slot's own distinguishing assignment becomes a smaller
fraction of what the Writer reads, and outputs would be expected to converge
regardless of how well the Planner separated the plans. This is the same
"prompt dilution" mechanism v67 diagnosed and fixed once (at slot 140 the slot's
own assignment had fallen to 19% of its prompt); this asks whether it is back,
and specifically whether it tracks the depth profile G26 localised.

Three measurements, all on the shipped artifact:

1. `shared` -- per-slot share of prompt lines that are boilerplate (a line
   appearing in >=80% of that thread's prompts) vs slot-unique, binned by depth.
2. `pairs` -- per-pair prompt similarity (Jaccard over line sets and over token
   sets) binned by depth, and the same for same-branch vs cross-branch pairs.
3. `link` -- correlation between a pair's prompt similarity and its realized
   text similarity, to establish whether prompt convergence plausibly transmits.

RESULT (2026-08-23, v108 N=10) -- the hypothesis is REJECTED, and the negative
result is more informative than a confirmation would have been:

1. Prompt dilution does NOT happen. Slot-unique share of the prompt is flat at
   85.5 / 86.3 / 86.9 / 87.0 / 85.2% across the five depth bins. v67's ledger
   caps are holding; the slot's own assignment is not being crowded out.
2. Prompt similarity between two slots **decreases** with depth -- line Jaccard
   0.3516 -> 0.2840 -> 0.2439 -> 0.2302 -> 0.2481. The Writer's inputs separate
   with depth, exactly as G21 found for the Planner's plan fields alone. This
   now holds for the *entire rendered prompt*, not just the plan.
3. Realized text similarity stays **flat** while the prompts diverge: token
   Jaccard 0.0971 / 0.0865 / 0.0917 / 0.0969 / 0.0818. r(prompt, text) = 0.320
   pooled over 26,520 pairs, and essentially constant per bin (0.33-0.37).
4. Cross-branch pairs receive *more* distinct prompts than same-branch pairs at
   every depth (e.g. 0.2212 vs 0.2642 at [4,7)) -- yet per G3 they are the
   population carrying the excess.

**Consequence for the mechanism search.** The Writer's whole input -- plan
fields (G21) and rendered prompt (here) -- already separates with depth, and the
output does not follow. So the convergence is produced inside the Writer,
downstream of every lever this project is permitted to pull. G26 correctly
showed the two shipped mechanisms were aimed at 11.9% of the defect; this shows
that re-aiming an input-side mechanism at the remaining 82.7% has a poor prior,
because input separation demonstrably fails to transmit at every depth. That is
a mechanism-level reason for G22's reading, replacing the weaker
"four mechanisms failed" argument.

No model, no API, seconds to run.

    python3 generalized_card/analysis/prompt_convergence_diagnosis.py shared
    python3 generalized_card/analysis/prompt_convergence_diagnosis.py all
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
SCORER_DIR = REPO / "scripts" / "evaluation"
if str(SCORER_DIR) not in sys.path:
    sys.path.insert(0, str(SCORER_DIR))

from score_thread_self_bleu import tokenize  # noqa: E402

DEFAULT_RUN = (
    REPO
    / "artifacts/generalized_card/runs"
    / "generalized_card_camera_gpt54_v108_coverage_nonrepeat_n10_20260823_v1"
)
BINS = ((0, 1), (1, 2), (2, 4), (4, 7), (7, 999))
BOILERPLATE_THRESHOLD = 0.80


def mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else float("nan")


def bin_label(low: int, high: int) -> str:
    return f"[{low},{high})" if high < 999 else f"[{low},+)"


def depth_bin(depth: int) -> tuple[int, int] | None:
    for low, high in BINS:
        if low <= depth < high:
            return (low, high)
    return None


def load_records(run: Path) -> dict[str, list[dict[str, Any]]]:
    """Per thread: one row per generated slot with prompt, text, depth, branch."""

    by_thread: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path in sorted(run.glob("generated/run_*/generation_records.json")):
        for rec in json.loads(path.read_text(encoding="utf-8")):
            comment = rec.get("comment") or {}
            task = rec.get("task") or {}
            text = (comment.get("content") or "").strip()
            if not text:
                continue
            by_thread[str(rec["post_id"])].append(
                {
                    "prompt": rec.get("prompt") or "",
                    "text": text,
                    "depth": int(comment.get("depth", task.get("depth", 0)) or 0),
                    "branch": task.get("branch_id"),
                }
            )
    return dict(by_thread)


def cmd_shared(threads: dict[str, list[dict[str, Any]]]) -> None:
    """How much of a slot's prompt is text every other slot also sees?"""

    print("\n== per-slot prompt composition, by the slot's own depth ==\n")
    print("A 'boilerplate' line appears in >=80% of that thread's prompts.\n")
    acc: dict[tuple[int, int], list[tuple[float, int, int]]] = defaultdict(list)
    for rows in threads.values():
        line_sets = [set(r["prompt"].splitlines()) for r in rows]
        counts: Counter[str] = Counter()
        for lines in line_sets:
            counts.update(lines)
        n = len(rows)
        boiler = {ln for ln, c in counts.items() if c >= BOILERPLATE_THRESHOLD * n}
        for row, lines in zip(rows, line_sets):
            key = depth_bin(row["depth"])
            if key is None:
                continue
            total_chars = len(row["prompt"])
            uniq_chars = sum(len(ln) for ln in lines - boiler)
            acc[key].append(
                (uniq_chars / total_chars if total_chars else 0.0, total_chars, uniq_chars)
            )

    print(f"{'depth':>8s} {'slots':>6s} {'prompt chars':>13s} {'slot-unique':>12s} {'unique share':>13s}")
    for low, high in BINS:
        rows = acc.get((low, high), [])
        if not rows:
            continue
        print(
            f"{bin_label(low, high):>8s} {len(rows):6d} {mean([r[1] for r in rows]):13.0f} "
            f"{mean([r[2] for r in rows]):12.0f} {mean([r[0] for r in rows]):12.1%}"
        )


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def cmd_pairs(threads: dict[str, list[dict[str, Any]]]) -> None:
    """Prompt similarity between two slots, by depth and by same/cross branch."""

    print("\n== pairwise PROMPT similarity, by max(depth) of the pair ==\n")
    by_depth: dict[tuple[int, int], list[float]] = defaultdict(list)
    by_depth_tok: dict[tuple[int, int], list[float]] = defaultdict(list)
    same_branch: dict[tuple[int, int], list[float]] = defaultdict(list)
    cross_branch: dict[tuple[int, int], list[float]] = defaultdict(list)

    for rows in threads.values():
        lines = [set(r["prompt"].splitlines()) for r in rows]
        toks = [set(tokenize(r["prompt"])) for r in rows]
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                key = depth_bin(max(rows[i]["depth"], rows[j]["depth"]))
                if key is None:
                    continue
                jl = _jaccard(lines[i], lines[j])
                by_depth[key].append(jl)
                by_depth_tok[key].append(_jaccard(toks[i], toks[j]))
                bi, bj = rows[i]["branch"], rows[j]["branch"]
                if bi is not None and bj is not None:
                    (same_branch if bi == bj else cross_branch)[key].append(jl)

    print(f"{'depth':>8s} {'pairs':>7s} {'line Jaccard':>13s} {'token Jaccard':>14s} {'same-branch':>12s} {'cross-branch':>13s}")
    for low, high in BINS:
        key = (low, high)
        if not by_depth.get(key):
            continue
        sb, cb = same_branch.get(key, []), cross_branch.get(key, [])
        print(
            f"{bin_label(low, high):>8s} {len(by_depth[key]):7d} {mean(by_depth[key]):13.4f} "
            f"{mean(by_depth_tok[key]):14.4f} {mean(sb):12.4f} {mean(cb):13.4f}"
        )


def cmd_link(threads: dict[str, list[dict[str, Any]]]) -> None:
    """Does prompt similarity track realized text similarity, within a depth bin?"""

    print("\n== prompt similarity vs realized text similarity ==\n")
    rows_by_bin: dict[tuple[int, int], list[tuple[float, float]]] = defaultdict(list)
    for rows in threads.values():
        plines = [set(r["prompt"].splitlines()) for r in rows]
        ttoks = [set(tokenize(r["text"])) for r in rows]
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                key = depth_bin(max(rows[i]["depth"], rows[j]["depth"]))
                if key is None:
                    continue
                rows_by_bin[key].append((_jaccard(plines[i], plines[j]), _jaccard(ttoks[i], ttoks[j])))

    def pearson(xs: list[float], ys: list[float]) -> float:
        mx, my = mean(xs), mean(ys)
        num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        den = (sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys)) ** 0.5
        return num / den if den else float("nan")

    print(f"{'depth':>8s} {'pairs':>7s} {'text Jaccard':>13s} {'r(prompt,text)':>15s}")
    allp, allt = [], []
    for low, high in BINS:
        vals = rows_by_bin.get((low, high), [])
        if len(vals) < 30:
            continue
        ps, ts = [v[0] for v in vals], [v[1] for v in vals]
        allp.extend(ps)
        allt.extend(ts)
        print(f"{bin_label(low, high):>8s} {len(vals):7d} {mean(ts):13.4f} {pearson(ps, ts):15.3f}")
    if allp:
        print(f"\n  pooled r(prompt similarity, text similarity) = {pearson(allp, allt):.3f} over {len(allp)} pairs")


COMMANDS = {"shared": cmd_shared, "pairs": cmd_pairs, "link": cmd_link}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("command", choices=[*COMMANDS, "all"])
    parser.add_argument("--run", default=str(DEFAULT_RUN))
    args = parser.parse_args()

    run = Path(args.run)
    threads = load_records(run)
    total = sum(len(v) for v in threads.values())
    print(f"run: {run.name}\nthreads: {len(threads)}  slots with text: {total}")
    if not total:
        raise SystemExit("no generation records found")
    names = list(COMMANDS) if args.command == "all" else [args.command]
    for name in names:
        COMMANDS[name](threads)


if __name__ == "__main__":
    main()
