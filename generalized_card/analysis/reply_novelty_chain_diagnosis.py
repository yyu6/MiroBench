#!/usr/bin/env python3
"""Falsify the `--reply-novelty-scope chain` threshold before wiring it up.

`docs/DECISIONS.md` G3 traced `self_bertscore_mean_f1`'s excess to reply
chains that restate the same argument as they go deeper (measured: excess
grows from +0.0004 at depth 1-2 to +0.0432 at depth 7+,
`bertscore_pair_diagnosis.py depth`). The codebase already has a reply-novelty
contract (`planning_quality.reply_increment_problem`) that checks a reply's
`reply_novelty_anchor` against its immediate parent only. This script asks the
question that has to be answered before that check is extended to the whole
ancestor chain: **does the existing 0.76 similarity threshold, applied to
every ancestor instead of just the parent, actually flag the diagnosed
chains** -- on the real, already-shipped v103 artifact, with the real
`PlanSemanticIndex` model, not a guess.

No API call. No new generation. Reads `discussion.json` from a run that
already exists on disk.

    python3 generalized_card/analysis/reply_novelty_chain_diagnosis.py
    python3 generalized_card/analysis/reply_novelty_chain_diagnosis.py --run <other run tag>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = REPO / "generalized_card"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from generalized_card.planning_quality import (  # noqa: E402
    PlanSemanticIndex,
    reply_increment_problem,
)

DEFAULT_RUN = (
    REPO
    / "artifacts/generalized_card/runs"
    / "generalized_card_camera_gpt54_v103_stance_opening_n10_20260821_v1"
)
DEFAULT_MODEL = "sentence-transformers/all-mpnet-base-v2"

# The two chains the qualitative reading (docs/ORIENTATION.md SS6.3) named by
# hand. Kept here so the script states, explicitly, whether it reproduces
# what a human found -- not just a pooled trip-rate number.
NAMED_CHAINS = {
    "sampled_run00_post00_seed002": {41, 42, 43, 44, 45},
    "sampled_run01_post01_seed008": None,  # too many candidate ids to hand-pick; report all trips instead
}


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _thread_plans(post: dict[str, Any]) -> list[dict[str, Any]]:
    """Reconstruct the plan ledger `evaluate_plan_batch` would have built.

    `generation_records[i]["task"]` is the real Planner task dict persisted
    for every generated comment. `local_task_id`/`local_parent_task_id` are
    the thread-local ids `_sample_id`/`_parent_sample_id` read (confirmed
    equal to `real_sample_id`/`real_parent_sample_id` on the real artifact,
    checked this session -- both point at the same values here).
    """

    plans = []
    for record in post.get("generation_records") or []:
        task = dict(record.get("task") or {})
        if not task:
            continue
        task["sample_id"] = int(task.get("local_task_id") or 0)
        plans.append(task)
    plans.sort(key=lambda row: row["sample_id"])
    return plans


def run_thread(
    index: PlanSemanticIndex, plans: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Replay `evaluate_plan_batch`'s ledger-building loop for one thread,
    under both scopes, and return the trips each scope raised."""

    index.prepare(plans)
    trips_parent_only: list[dict[str, Any]] = []
    trips_chain: list[dict[str, Any]] = []
    for scope, trips in (("parent_only", trips_parent_only), ("chain", trips_chain)):
        seen: list[dict[str, Any]] = []
        for plan in plans:
            problem = reply_increment_problem(
                plan,
                parent_plans=seen,
                semantic_similarity=index.similarity,
                required=True,
                novelty_scope=scope,
            )
            if problem:
                trips.append(
                    {
                        "sample_id": plan["sample_id"],
                        "parent_sample_id": int(plan.get("local_parent_task_id") or 0),
                        "message": problem,
                    }
                )
            seen.append(plan)
    return trips_parent_only, trips_chain


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default=str(DEFAULT_RUN))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    run = Path(args.run)
    index = PlanSemanticIndex(model_name=args.model, device=args.device)

    total_parent_only = 0
    total_chain = 0
    new_from_chain: list[tuple[str, int, int, str]] = []
    named_chain_hit = {name: False for name in NAMED_CHAINS}

    for sim_dir in sorted(run.glob("cleaned/run_*_sampled_reddit")):
        discussion = _load_json(sim_dir / "discussion.json")
        for post in discussion.get("posts") or []:
            post_id = str(post.get("post_id") or "")
            plans = _thread_plans(post)
            if not plans:
                continue
            trips_parent_only, trips_chain = run_thread(index, plans)
            total_parent_only += len(trips_parent_only)
            total_chain += len(trips_chain)
            parent_only_ids = {row["sample_id"] for row in trips_parent_only}
            for row in trips_chain:
                if row["sample_id"] not in parent_only_ids:
                    new_from_chain.append(
                        (post_id, row["sample_id"], row["parent_sample_id"], row["message"])
                    )
                if post_id in NAMED_CHAINS:
                    target = NAMED_CHAINS[post_id]
                    if target is None or row["sample_id"] in target:
                        named_chain_hit[post_id] = True

            print(
                f"{post_id:36s} plans={len(plans):3d} "
                f"parent_only_trips={len(trips_parent_only):2d} chain_trips={len(trips_chain):2d}"
            )

    print(f"\ntotal: parent_only={total_parent_only} chain={total_chain}")
    print(f"newly caught by chain (missed by parent_only): {len(new_from_chain)}")
    for post_id, sample_id, parent_id, message in new_from_chain:
        print(f"  {post_id} S{sample_id} (parent S{parent_id}): {message}")

    print("\n== the two chains found by reading the actual text ==")
    for post_id, hit in named_chain_hit.items():
        print(f"  {post_id}: {'CAUGHT' if hit else 'NOT caught'} by chain scope")


if __name__ == "__main__":
    main()
