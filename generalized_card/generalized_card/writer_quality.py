"""Domain-neutral Writer candidate quality policy.

This module controls only generated text from the current thread.  It never
receives matched evaluation comments or raw held-out reference text.
"""

from __future__ import annotations

import re
from dataclasses import replace
from types import ModuleType
from typing import Any

from .generation_diversity import (
    distribution_target_with_slot_progress,
    joint_candidate_diagnostics,
)
from .length_fidelity import length_band_problem, length_ceiling_problem
from .length_policy import is_soft_length_problem


DISTRIBUTION_PROBLEM_MARKERS = (
    "lexical_overlap_high:",
    "semantic_overlap_high:",
    "semantic_overlap_low:",
)


SINGLE_STAGE_DIAGNOSTIC_PROBLEMS = frozenset(
    {
        "opening_reused",
        "opener_family_reused",
        "template_phrase_reused",
        "first_person_frame_unwanted",
        "uncertainty_frame_unwanted",
        "question_mark_unwanted",
        "meta_template_quote_heading",
        "long_helpful_too_generic",
        "missing_concrete_anchor",
    }
)
SINGLE_STAGE_DIAGNOSTIC_PREFIXES = DISTRIBUTION_PROBLEM_MARKERS

HARD_REALIZATION_PROBLEMS = frozenset(
    {
        "empty",
        "exact_duplicate",
        "parent_copy",
        "placeholder_literal",
        "planner_skeleton_residue",
    }
)

_QUOTE_TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?", re.I)


def is_single_stage_diagnostic(problem: str) -> bool:
    """Return whether a non-empty realization may be retained and audited."""

    return (
        is_soft_length_problem(problem)
        or problem in SINGLE_STAGE_DIAGNOSTIC_PROBLEMS
        or problem.startswith(SINGLE_STAGE_DIAGNOSTIC_PREFIXES)
    )


def hard_realization_problems(problems: list[str]) -> list[str]:
    """Return failures that cannot be persisted as a generated comment."""

    return [problem for problem in problems if problem in HARD_REALIZATION_PROBLEMS]


def planned_quote_has_distinct_reply(text: str, parent_text: str) -> bool:
    """Recognize a bounded parent excerpt followed by an independent reply.

    This is a syntax exception for a Planner-assigned quote opener, not a
    semantic copy detector.  The caller must still require the assigned
    ``opener_type=quote`` before waiving ``parent_copy``.
    """

    quote_lines: list[str] = []
    reply_lines: list[str] = []
    for line in str(text or "").splitlines():
        stripped = line.strip()
        if stripped.startswith(">"):
            quote_lines.append(stripped.lstrip("> "))
        elif stripped:
            reply_lines.append(stripped)
    if not quote_lines or not reply_lines:
        return False

    quoted = _QUOTE_TOKEN_RE.findall(" ".join(quote_lines).lower())
    parent = _QUOTE_TOKEN_RE.findall(str(parent_text or "").lower())
    reply = _QUOTE_TOKEN_RE.findall(" ".join(reply_lines).lower())
    if not quoted or len(quoted) > 24 or len(reply) < 6:
        return False
    if len(parent) > 5 and quoted == parent:
        return False
    if not _contains_token_sequence(parent, quoted):
        return False
    # A quote plus a second full copy of the parent is still a parent copy.
    if len(parent) >= 6 and _contains_token_sequence(reply, parent):
        return False
    return True


def _contains_token_sequence(haystack: list[str], needle: list[str]) -> bool:
    if not needle or len(needle) > len(haystack):
        return False
    width = len(needle)
    return any(
        haystack[index : index + width] == needle
        for index in range(len(haystack) - width + 1)
    )


def writer_distribution_problems(
    module: ModuleType,
    *,
    text: str,
    previous_comments: list[dict[str, Any]] | None,
    previous_texts: list[str],
    calibration: dict[str, Any],
    thread_target: dict[str, Any],
    task: Any,
) -> tuple[dict[str, Any], list[str]]:
    """Return evaluator-aligned, current-thread-only candidate failures."""

    if not text:
        return {}, []
    effective_target = distribution_target_with_slot_progress(
        thread_target,
        local_task_id=getattr(task, "local_task_id", 0),
    )
    diagnostics = joint_candidate_diagnostics(
        text=text,
        previous_texts=previous_texts,
        lexical_calibration=calibration,
        thread_target=effective_target,
        semantic_index=getattr(module, "GENERALIZED_PLAN_SEMANTIC_INDEX", None),
    )
    problems: list[str] = []
    checker = getattr(module, "lexical_overlap_problem", None)
    if callable(checker):
        measured = str(
            checker(text=text, previous_comments=previous_comments, task=task) or ""
        )
        problems.extend(parse_distribution_problems(measured))
    floor_problem = substantive_length_floor_problem(text, task)
    if floor_problem:
        problems.append(floor_problem)
    # Registered soft, so it can only trigger a Writer retry and can never make
    # a matched structural slot blocking (`docs/ORIENTATION.md` §4).
    band_problem = length_band_problem(text, task)
    if band_problem:
        problems.append(band_problem)
    # Also soft, and deliberately independent of the band check: a tail
    # overshoot sits inside its own assigned band and raises nothing there.
    ceiling_problem = length_ceiling_problem(text, task)
    if ceiling_problem:
        problems.append(ceiling_problem)
    return diagnostics, deduplicate_problems(problems)


def substantive_length_floor_problem(text: str, task: Any) -> str:
    """Prevent a substantive real-shaped slot from evading overlap checks as a fragment."""

    if task is None:
        return ""
    real_words = safe_int(getattr(task, "real_word_count", 0), 0)
    if real_words < 16:
        return ""
    if str(getattr(task, "real_surface_shape", "")) in {
        "micro_reaction",
        "short_direct_answer",
        "short_question",
    }:
        return ""
    if str(getattr(task, "payload_type", "")) in {
        "low_info_reaction",
        "joke",
        "meta_or_template",
        "narrow_question",
        "fragment_datapoint",
    }:
        return ""
    if str(getattr(task, "comment_function", "")) in {
        "reaction",
        "offtopic_noise",
        "question_followup",
    }:
        return ""
    if str(getattr(task, "utterance_mode", "")) in {
        "fragment_only",
        "question_only",
        "joke_only",
    }:
        return ""
    minimum = 8 if real_words < 22 else max(8, min(32, round(real_words * 0.5)))
    actual = len(text.split())
    if actual < minimum:
        return f"substantive_length_floor:{actual}<{minimum}"
    return ""


def writer_hard_recovery_task(
    task: Any,
    *,
    problems: list[str],
    previous_candidate_text: str = "",
) -> Any:
    """Ask for the same planned move after an unpersistable Writer result."""

    if task is None:
        return task
    instructions = [
        "The previous output could not be stored. Realize the same assigned "
        "discussion move, parent relation, role, factual anchors, and payload."
    ]
    if any(
        item in problems
        for item in (
            "empty",
            "placeholder_literal",
            "planner_skeleton_residue",
            "meta_template_quote_heading",
        )
    ):
        instructions.append(
            "Return one non-empty ordinary comment body. Do not output control labels, "
            "placeholder text, fake resource names, or an invented quote heading."
        )
    if any(item in problems for item in ("exact_duplicate", "parent_copy")):
        instructions.append(
            "Do not copy the parent or previous output; state the same move in "
            "new wording."
        )
    failed_candidate = " ".join(str(previous_candidate_text or "").split())[:260]
    if failed_candidate:
        instructions.append(
            "Do not repeat this failed output: "
            + failed_candidate
        )
    instructions.append("Output one ordinary comment body only.")
    repair_note = " ".join(instructions)
    try:
        return replace(
            task,
            planner_intent=f"{getattr(task, 'planner_intent', '')} {repair_note}".strip(),
            must_not_do=(
                f"{getattr(task, 'must_not_do', '')} Do not reuse the failed output."
            ).strip(),
        )
    except TypeError:
        # Test adapters can use lightweight task objects. Production uses the
        # immutable CommentTask dataclass and takes the branch above.
        return task


def parse_distribution_problems(value: str) -> list[str]:
    """Split only at known diagnostic boundaries, not at field separators."""

    text = str(value or "").strip(" ;")
    if not text:
        return []
    starts = sorted(
        {
            position
            for marker in DISTRIBUTION_PROBLEM_MARKERS
            for position in _marker_positions(text, marker)
        }
    )
    if not starts:
        return [text]
    rows: list[str] = []
    for index, start in enumerate(starts):
        end = starts[index + 1] - 1 if index + 1 < len(starts) else len(text)
        row = text[start:end].strip(" ;")
        if row:
            rows.append(row)
    return rows


def _marker_positions(text: str, marker: str) -> list[int]:
    starts: list[int] = []
    offset = 0
    while True:
        found = text.find(marker, offset)
        if found < 0:
            return starts
        if found == 0 or text[found - 1] == ";":
            starts.append(found)
        offset = found + len(marker)


def last_writer_problems(result: dict[str, Any]) -> list[str]:
    attempts = list(result.get("attempts") or [])
    if attempts:
        return list(attempts[-1].get("problems") or [])
    return [item for item in str(result.get("skip_reason") or "").split(",") if item]


def deduplicate_problems(problems: list[str]) -> list[str]:
    return list(dict.fromkeys(str(item) for item in problems if str(item)))


def annotate_writer_attempts(
    attempts: list[dict[str, Any]],
    *,
    start_at: int,
    repair_round: int,
) -> list[dict[str, Any]]:
    annotated: list[dict[str, Any]] = []
    for index, attempt in enumerate(attempts, start=1):
        row = dict(attempt)
        row["attempt"] = start_at + index
        row["local_repair_round"] = repair_round
        annotated.append(row)
    return annotated


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default
