from __future__ import annotations

from dataclasses import replace
from sampling_generator.engine.model import CommentTask
from sampling_generator.engine.util import safe_int
from sampling_generator.engine.writer_validation import real_slot_requires_substantive_writer
from typing import Any

def nearest_generated_ancestor(
    task: CommentTask,
    *,
    task_by_id: dict[int, CommentTask],
    actual_by_task: dict[int, dict[str, Any]],
) -> int | None:
    parent_id = task.local_parent_task_id
    visited: set[int] = set()
    while parent_id is not None and parent_id not in visited:
        if parent_id in actual_by_task:
            return parent_id
        visited.add(parent_id)
        parent_task = task_by_id.get(parent_id)
        parent_id = parent_task.local_parent_task_id if parent_task else None
    return None

def align_task_to_generated_parent(
    task: CommentTask,
    *,
    task_by_id: dict[int, CommentTask],
    actual_by_task: dict[int, dict[str, Any]],
) -> tuple[CommentTask, dict[str, Any] | None]:
    """Align planned parent/depth metadata with the tree actually generated."""

    parent_id = task.local_parent_task_id
    parent_comment = actual_by_task.get(parent_id) if parent_id is not None else None
    repaired_parent = False
    if parent_id is not None and parent_comment is None:
        parent_id = nearest_generated_ancestor(
            task,
            task_by_id=task_by_id,
            actual_by_task=actual_by_task,
        )
        parent_comment = actual_by_task.get(parent_id) if parent_id is not None else None
        repaired_parent = True

    expected_depth = safe_int(parent_comment.get("depth"), 0) + 1 if parent_comment else 0
    updates: dict[str, Any] = {}
    if task.local_parent_task_id != parent_id:
        updates["local_parent_task_id"] = parent_id
    if task.depth != expected_depth:
        updates["depth"] = expected_depth
    if repaired_parent:
        updates["context_transform"] = "parent_gist" if parent_comment else "minor_detail_focus"
    if updates:
        task = replace(task, **updates)
    return task, parent_comment

def writer_payload_for_token_cap(task: CommentTask) -> str:
    if real_slot_requires_substantive_writer(task):
        return ""
    return task.payload_type
