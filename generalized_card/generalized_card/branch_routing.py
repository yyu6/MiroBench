"""Domain-neutral structural routing for first-pass comment planning.

The router sees only anonymous comment identifiers and parent links. It gives
each root discussion chain a distinct Planner branch when one is available;
reply slots inherit their root's branch. It never reads comment text.
"""

from __future__ import annotations

import math
from typing import Any, Iterable


def required_branch_count(
    comments: Iterable[dict[str, Any]], *, maximum: int | None = None
) -> int:
    """Return a stable semantic-axis budget for root discussion chains.

    A top-level comment is not automatically a new topic. Narrow real Reddit
    threads commonly contain many roots that agree, disagree, ask a short
    follow-up, or add a small example around the same decision. Requiring one
    mutually-exclusive branch per root fabricates false topical breadth and
    paradoxically makes the Writer repeat the same seed premise in polished
    wording. The Planner receives a compact number of reusable decision axes;
    separate root instances vary social/discourse function around those axes.
    """

    root_count = len(_root_ids(list(comments)))
    # The capacity grows with the visible number of independent roots but much
    # slower than one-for-one. This is topology-only, not domain-specific.
    capacity = max(3, int(round(2.0 * math.sqrt(max(1, root_count)))))
    branch_count = min(max(3, root_count), capacity)
    if maximum is not None:
        branch_count = min(max(3, int(maximum)), branch_count)
    return max(3, branch_count)


def root_branch_schedule(
    comments: Iterable[dict[str, Any]], *, branch_ids: Iterable[int]
) -> dict[int, int]:
    """Map every one-indexed structural slot to a root-chain branch ID."""

    rows = list(comments)
    available = [int(value) for value in branch_ids if int(value) > 0]
    if not rows or not available:
        return {}
    roots = _root_ids(rows)
    root_to_branch = {
        root: available[index % len(available)]
        for index, root in enumerate(roots)
    }
    parent_slots = _parent_slots(rows)
    return {
        sample_id: root_to_branch[_resolve_root(sample_id, parent_slots)]
        for sample_id in range(1, len(rows) + 1)
    }


def parent_slot_schedule(comments: Iterable[dict[str, Any]]) -> dict[int, int]:
    """Return only anonymous structural parent relationships."""

    return _parent_slots(list(comments))


# v148 arm. Each routed slot normally arrives carrying its direction already
# decided: a branch goal, a required perspective, an exclusion, an owned subject,
# and the list of subjects it may not touch. The Planner is then filling in a
# grid rather than deciding anything, and the measurements say it feels the grid:
# `perspective_id` offers `seed_local` as an escape and it is used in 0.0% of
# 1,878 slots, while `content_angle` -- eight shopping categories -- takes
# `unclear_mixed` for 66.9% of them. The Planner is already reporting that the
# taxonomy does not fit this domain, and is routed into it anyway.
#
# `structural` keeps only what the thread's shape requires -- which branch, which
# parent, which siblings -- and drops every field that decides what the comment
# is about. Paired with the matched real text, the slot's own real comment
# becomes the guide instead.
BRANCH_DICTATION_MODE = "full"


def set_branch_dictation(mode: str) -> bool:
    global BRANCH_DICTATION_MODE
    BRANCH_DICTATION_MODE = str(mode or "full").strip().lower()
    return BRANCH_DICTATION_MODE == "structural"


def render_branch_requirements(
    schedule: dict[int, int],
    *,
    sample_ids: Iterable[int],
    branch_goals: dict[int, str] | None = None,
    branch_perspectives: dict[int, str] | None = None,
    branch_exclusions: dict[int, str] | None = None,
    branch_subjects: dict[int, str] | None = None,
    parent_slots: dict[int, int] | None = None,
) -> str:
    parent_by_child = dict(parent_slots or {})
    children_by_parent: dict[int, list[int]] = {}
    for child, parent in parent_by_child.items():
        children_by_parent.setdefault(int(parent), []).append(int(child))
    for children in children_by_parent.values():
        children.sort()

    root_instances: dict[int, tuple[int, int]] = {}
    roots_by_branch: dict[int, list[int]] = {}
    for sample_id, branch_id in schedule.items():
        if sample_id not in parent_by_child:
            roots_by_branch.setdefault(int(branch_id), []).append(int(sample_id))
    for branch_id, roots in roots_by_branch.items():
        for ordinal, sample_id in enumerate(roots, start=1):
            root_instances[sample_id] = (ordinal, len(roots))

    rows = []
    for sample_id in sample_ids:
        if sample_id not in schedule:
            continue
        branch_id = schedule[sample_id]
        structural = BRANCH_DICTATION_MODE == "structural"
        goal = "" if structural else str((branch_goals or {}).get(branch_id) or "").strip()
        parent = (parent_slots or {}).get(sample_id)
        row = f"- S{sample_id}: required_branch=B{branch_id}"
        if goal:
            row += f"; branch_goal={goal}"
        perspective = "" if structural else str((branch_perspectives or {}).get(branch_id) or "").strip()
        if perspective:
            row += f"; required_perspective={perspective}"
        exclusion = "" if structural else str((branch_exclusions or {}).get(branch_id) or "").strip()
        if exclusion:
            row += f"; branch_exclusion={exclusion}"
        subject = "" if structural else str((branch_subjects or {}).get(branch_id) or "").strip()
        if subject:
            other_subjects = [
                value
                for other_id, value in sorted((branch_subjects or {}).items())
                if other_id != branch_id and str(value or "").strip()
            ]
            row += f"; owned_decision_subject={subject}"
            if other_subjects:
                row += "; forbidden_other_subjects=" + " | ".join(other_subjects)
        if parent:
            row += f"; direct_parent=S{parent}"
            siblings = children_by_parent.get(int(parent), [])
            if len(siblings) > 1:
                ordinal = siblings.index(int(sample_id)) + 1
                sibling_ids = ",".join(f"S{value}" for value in siblings)
                row += (
                    f"; sibling_group={sibling_ids}"
                    f"; sibling_turn={ordinal}/{len(siblings)}"
                )
        elif sample_id in root_instances:
            ordinal, count = root_instances[sample_id]
            if count > 1:
                row += f"; root_branch_instance={ordinal}/{count}"
        rows.append(row)
    return "\n".join(rows) or "- no structural branch route is available"


def _root_ids(rows: list[dict[str, Any]]) -> list[int]:
    parent_slots = _parent_slots(rows)
    roots: list[int] = []
    for sample_id in range(1, len(rows) + 1):
        root = _resolve_root(sample_id, parent_slots)
        if root not in roots:
            roots.append(root)
    return roots


def _parent_slots(rows: list[dict[str, Any]]) -> dict[int, int]:
    identifiers: dict[str, int] = {}
    for sample_id, row in enumerate(rows, start=1):
        for key in _comment_keys(row):
            identifiers[key] = sample_id
    parent_slots: dict[int, int] = {}
    for sample_id, row in enumerate(rows, start=1):
        parent = identifiers.get(str(row.get("parent_id") or "").strip())
        if parent is not None and parent != sample_id:
            parent_slots[sample_id] = parent
    return parent_slots


def _resolve_root(sample_id: int, parent_slots: dict[int, int]) -> int:
    current = sample_id
    seen: set[int] = set()
    while current in parent_slots and current not in seen:
        seen.add(current)
        current = parent_slots[current]
    return current


def _comment_keys(row: dict[str, Any]) -> tuple[str, ...]:
    keys: list[str] = []
    for value in (
        row.get("comment_fullname"),
        row.get("fullname"),
        row.get("name"),
        row.get("comment_id"),
        row.get("id"),
    ):
        text = str(value or "").strip()
        if text:
            keys.append(text)
            if not text.startswith("t1_"):
                keys.append(f"t1_{text}")
    return tuple(dict.fromkeys(keys))
