"""Preserve healthy Planner state during bounded slot-local repair."""

from __future__ import annotations

from typing import Any


_FIELD_SCOPED_REPAIRS = {
    "long_form_capacity": ("development_plan",),
}


def repair_merge_fields(report: Any, sample_id: int) -> tuple[str, ...]:
    """Return the fields a slot's single remaining diagnostic may replace.

    Whole-plan replacement remains necessary when a slot has multiple repair
    conflicts. Once exactly one field-local conflict remains, accepting unrelated
    model changes can undo an earlier successful repair.
    """

    issue_codes = {
        str(issue.code)
        for issue in report.repair_issues
        if int(issue.sample_id) == int(sample_id)
    }
    if len(issue_codes) != 1:
        return ()
    return _FIELD_SCOPED_REPAIRS.get(next(iter(issue_codes)), ())


def apply_repair_candidate(
    selected: dict[str, Any],
    candidate: dict[str, Any],
    *,
    merge_fields: tuple[str, ...],
) -> dict[str, Any]:
    """Apply a full candidate or merge only the named failing fields."""

    if not merge_fields:
        return dict(candidate)
    merged = dict(selected)
    for field in merge_fields:
        merged[field] = candidate.get(field, "")
    return merged


def render_field_repair_instruction(merge_fields: tuple[str, ...]) -> str:
    if not merge_fields:
        return ""
    rendered = ", ".join(merge_fields)
    return (
        f"Only {rendered} is still failing. Preserve every other field exactly; "
        "changes returned for other fields will be ignored."
    )
