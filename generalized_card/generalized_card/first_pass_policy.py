"""Structural coverage invariants for first-pass generation.

Planner output is a required contract for every anonymous matched slot.  A
missing plan is a schema failure, not permission to silently shorten a thread.
"""

from __future__ import annotations

from typing import Any


def retain_explicitly_planned_tasks(
    tasks: list[Any],
    comment_plans: dict[int, dict[str, Any]],
) -> tuple[list[Any], dict[str, Any]]:
    """Validate full Planner coverage and retain the complete task skeleton."""

    planned_ids = {int(sample_id) for sample_id in comment_plans}
    retained = list(tasks)
    retained_ids = {
        int(getattr(task, "real_sample_id", 0) or 0) for task in retained
    }
    structural_ids = {
        int(getattr(task, "real_sample_id", 0) or 0)
        for task in tasks
        if int(getattr(task, "real_sample_id", 0) or 0) > 0
    }
    missing = sorted(structural_ids - planned_ids)
    if missing:
        raise RuntimeError(
            "Comment Planner omitted required structural slots before Writer "
            f"generation: {missing}"
        )
    return retained, {
        "structural_slots": len(structural_ids),
        "planner_returned_slots": len(planned_ids),
        "writer_task_slots": len(retained),
        "omitted_structural_slot_ids": sorted(structural_ids - retained_ids),
        "unused_plan_slot_ids": sorted(planned_ids - retained_ids),
        "policy": "require_full_planner_coverage_before_writer",
    }


def generation_coverage(
    tasks: list[Any],
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Summarize Writer coverage; incomplete posts are not evaluable."""

    skipped = [row for row in records if bool(row.get("skipped"))]
    generated = [row for row in records if isinstance(row.get("comment"), dict)]
    failed_ids = sorted(
        {
            _safe_int((row.get("task") or {}).get("local_task_id"), 0)
            for row in skipped
            if _safe_int((row.get("task") or {}).get("local_task_id"), 0) > 0
        }
    )
    return {
        "writer_task_slots": len(tasks),
        "writer_records": len(records),
        "generated_comments": len(generated),
        "skipped_comments": len(skipped),
        "failed_task_ids": failed_ids,
        "complete_share": round(len(generated) / max(1, len(tasks)), 6),
        "complete": len(generated) == len(tasks) and not skipped,
        "policy": "require_complete_writer_coverage_for_evaluation",
    }


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default
