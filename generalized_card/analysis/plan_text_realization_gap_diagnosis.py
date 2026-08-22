#!/usr/bin/env python3
"""Is the depth-growing reply convergence a Planner problem, a Writer
realization gap, or both -- before building anything in either category.

G16/G19 established the excess is real, broad (not concentrated on any one
tree relation), and grows with depth, replicated in two embedding models.
G20 ruled out a Writer-candidate-vs-metric-shaped-band mechanism as a
forbidden category (`docs/ORIENTATION.md` SS4). The remaining legitimate
category is a Planner-side plan-structure contract, the same category as
`reply_increment_problem` (v105). Before extending that mechanism's scope
(v105 only compared a reply to its ancestor chain), this asks the load-bearing
question directly:

1. Does the Planner's *own* plan-field similarity (the full `SEMANTIC_FIELDS`
   set `PlanSemanticIndex` already embeds) grow with depth the same way the
   *text* does -- independent of any "real" comparison, since plans have no
   real-side equivalent? If plans are already about as distinct at depth 7
   as at depth 1, no Planner-side fix has anything to catch.
2. Does that plan-level pattern hold on **non-ancestor** pairs (siblings,
   cousins, cross-branch) -- the population v105's ancestor-chain-only scope
   structurally cannot see? If the pattern is ancestor-only, v105 already
   covered it (and the N=10 gate already showed that didn't move the
   metric). If it also holds broadly, widening the scope is still untried.
3. Across the same pairs, how well does plan-field similarity predict actual
   *text* similarity (Pearson r)? A low correlation means the Planner is
   already producing sufficiently distinct plans for many pairs whose *text*
   still converges -- a realization gap, the same pattern already found for
   `polite_rate` (G5-G7) and `abstract_verdict_close` (G14) -- and no
   Planner-side fix, however broad, can close a gap that lives downstream of
   the plan.

No API call. Reuses `PlanSemanticIndex` (`planning_quality.py`) for plan
embeddings and the same cheap `all-mpnet-base-v2` model for text embeddings
(the model already cross-validated against BERTScore's direction in G19).

    python3 generalized_card/analysis/plan_text_realization_gap_diagnosis.py
    python3 generalized_card/analysis/plan_text_realization_gap_diagnosis.py --run <other run tag>
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = REPO / "generalized_card"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from generalized_card.planning_quality import (  # noqa: E402
    PlanSemanticIndex,
    plan_semantic_text,
)

DEFAULT_RUN = (
    REPO
    / "artifacts/generalized_card/runs"
    / "generalized_card_camera_gpt54_v103_stance_opening_n10_20260821_v1"
)
DEFAULT_MODEL = "sentence-transformers/all-mpnet-base-v2"
DEPTH_BINS = ((0, 1), (1, 2), (2, 4), (4, 7), (7, 999))


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def mean(values: list[float]) -> float:
    return statistics.mean(values) if values else float("nan")


def load_threads(run_path: Path) -> dict[str, list[dict[str, Any]]]:
    """Per thread: [{comment_id, parent_id, depth, plan_text, text}]."""

    threads: dict[str, list[dict[str, Any]]] = {}
    search_roots = list(run_path.glob("cleaned/run_*_sampled_reddit")) or list(
        run_path.glob("generated/run_*_sampled_reddit")
    )
    for sim_dir in sorted(search_roots):
        data = _load_json(sim_dir / "discussion.json")
        for post in data.get("posts") or []:
            thread_id = str(post.get("post_id") or "")
            rows = []
            for record in post.get("generation_records") or []:
                task = dict(record.get("task") or {})
                if not task:
                    continue
                comment = dict(record.get("comment") or {})
                text = str(comment.get("content") or record.get("final_text") or "")
                rows.append(
                    {
                        "comment_id": int(task.get("local_task_id") or 0),
                        "parent_id": int(task.get("local_parent_task_id") or 0),
                        "depth": int(task.get("depth") or 0),
                        "plan_text": plan_semantic_text(task),
                        "text": text,
                    }
                )
            rows.sort(key=lambda row: row["comment_id"])
            if rows:
                threads[thread_id] = rows
    return threads


def is_ancestor_descendant(a: dict[str, Any], b: dict[str, Any], by_id: dict[int, dict[str, Any]]) -> bool:
    """True if b is an ancestor of a, or a is an ancestor of b."""

    def ancestors(node: dict[str, Any]) -> set[int]:
        seen: set[int] = set()
        current = node.get("parent_id", 0)
        depth_guard = 0
        while current and current > 0 and current not in seen and depth_guard < 64:
            seen.add(current)
            parent = by_id.get(current)
            if parent is None:
                break
            current = parent.get("parent_id", 0)
            depth_guard += 1
        return seen

    return b["comment_id"] in ancestors(a) or a["comment_id"] in ancestors(b)


def pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2:
        return float("nan")
    mx, my = mean(xs), mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    denx = sum((x - mx) ** 2 for x in xs) ** 0.5
    deny = sum((y - my) ** 2 for y in ys) ** 0.5
    if denx == 0 or deny == 0:
        return float("nan")
    return num / (denx * deny)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", default=str(DEFAULT_RUN))
    args = parser.parse_args()

    threads = load_threads(Path(args.run))
    total_comments = sum(len(rows) for rows in threads.values())
    print(f"threads: {len(threads)}  comments: {total_comments}")

    plan_index = PlanSemanticIndex(model_name=DEFAULT_MODEL, device="cpu")
    all_plan_texts = [row["plan_text"] for rows in threads.values() for row in rows]
    all_texts = [row["text"] for rows in threads.values() for row in rows]
    print("embedding plan texts and comment texts (shared cheap model) ...")
    plan_vectors = {t: v for t, v in zip(all_plan_texts, plan_index.encode_texts(all_plan_texts))}
    text_vectors = {t: v for t, v in zip(all_texts, plan_index.encode_texts(all_texts))}

    import numpy as np

    def sim(vec_map: dict[str, Any], key_field: str, a: dict[str, Any], b: dict[str, Any]) -> float:
        va, vb = vec_map.get(a[key_field]), vec_map.get(b[key_field])
        if va is None or vb is None:
            return float("nan")
        return float(np.dot(va, vb))

    plan_bins: dict[tuple[int, int], list[float]] = defaultdict(list)
    text_bins: dict[tuple[int, int], list[float]] = defaultdict(list)
    plan_bins_nonancestor: dict[tuple[int, int], list[float]] = defaultdict(list)
    text_bins_nonancestor: dict[tuple[int, int], list[float]] = defaultdict(list)
    all_plan_sims: list[float] = []
    all_text_sims: list[float] = []
    ancestor_flags: list[bool] = []

    for thread_id, rows in threads.items():
        by_id = {row["comment_id"]: row for row in rows}
        n = len(rows)
        for i in range(n):
            for j in range(i + 1, n):
                a, b = rows[i], rows[j]
                ps = sim(plan_vectors, "plan_text", a, b)
                ts = sim(text_vectors, "text", a, b)
                if ps != ps or ts != ts:  # NaN guard
                    continue
                deepest = max(a["depth"], b["depth"])
                ancestor = is_ancestor_descendant(a, b, by_id)
                for low, high in DEPTH_BINS:
                    if low <= deepest < high:
                        plan_bins[(low, high)].append(ps)
                        text_bins[(low, high)].append(ts)
                        if not ancestor:
                            plan_bins_nonancestor[(low, high)].append(ps)
                            text_bins_nonancestor[(low, high)].append(ts)
                        break
                all_plan_sims.append(ps)
                all_text_sims.append(ts)
                ancestor_flags.append(ancestor)

    print("\n== 1. plan-field similarity by depth (all pairs) vs text similarity by depth ==")
    print(f"{'depth':10s} {'plan n':>8s} {'plan mean':>10s} {'text n':>8s} {'text mean':>10s}")
    for low, high in DEPTH_BINS:
        lbl = f"[{low},{high if high < 999 else '+'})"
        pv, tv = plan_bins.get((low, high), []), text_bins.get((low, high), [])
        print(f"{lbl:10s} {len(pv):8d} {mean(pv):10.4f} {len(tv):8d} {mean(tv):10.4f}")

    print("\n== 2. same, restricted to NON-ancestor pairs (siblings/cousins/cross-branch) ==")
    print(f"{'depth':10s} {'plan n':>8s} {'plan mean':>10s} {'text n':>8s} {'text mean':>10s}")
    for low, high in DEPTH_BINS:
        lbl = f"[{low},{high if high < 999 else '+'})"
        pv, tv = plan_bins_nonancestor.get((low, high), []), text_bins_nonancestor.get((low, high), [])
        print(f"{lbl:10s} {len(pv):8d} {mean(pv):10.4f} {len(tv):8d} {mean(tv):10.4f}")

    r_all = pearson(all_plan_sims, all_text_sims)
    ancestor_pairs = [(p, t) for p, t, a in zip(all_plan_sims, all_text_sims, ancestor_flags) if a]
    nonancestor_pairs = [(p, t) for p, t, a in zip(all_plan_sims, all_text_sims, ancestor_flags) if not a]
    r_anc = pearson([p for p, _ in ancestor_pairs], [t for _, t in ancestor_pairs])
    r_non = pearson([p for p, _ in nonancestor_pairs], [t for _, t in nonancestor_pairs])

    print("\n== 3. does plan similarity predict text similarity? (Pearson r) ==")
    print(f"  all pairs        n={len(all_plan_sims):6d}  r={r_all:+.4f}")
    print(f"  ancestor pairs   n={len(ancestor_pairs):6d}  r={r_anc:+.4f}")
    print(f"  non-ancestor     n={len(nonancestor_pairs):6d}  r={r_non:+.4f}")


if __name__ == "__main__":
    main()
