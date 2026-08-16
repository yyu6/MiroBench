"""Domain-neutral Writer candidate quality policy.

This module controls only generated text from the current thread.  It never
receives matched evaluation comments or raw held-out reference text.
"""

from __future__ import annotations

import math
import re
from dataclasses import replace
from types import ModuleType
from typing import Any

from .generation_diversity import (
    distribution_target_with_slot_progress,
    joint_candidate_diagnostics,
)
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

# The three repetition codes, which the guard can promote out of the advisory
# set. They are the ones with a measured defect behind them and no dependency on
# any other change.
#
# In one generated 186-comment thread the 4-gram "that's the part" occurs in 12
# of 186 comments (6.5%); in its matched real thread the most-shared 4-gram
# occurs in 3 of 200 (1.5%) and there is effectively no shared phrasing at all.
# The same frame family was measured at 20% of comments back at policy v72 and
# 0 times in 39,265 real tokens, and it has survived a rewording (v73), a prompt
# rebuild (v74) and a route lock (v75).
#
# `template_phrase_reused` already fires on it -- 38 times in v76a and 40 in
# v76b, about 21% of slots -- and is then discarded: all 186 slots ran exactly
# one attempt and 85 were accepted through
# `accepted_first_pass_distribution_diagnostics`, on a known-failing candidate.
#
# All three are already in `REPAIRABLE_WRITER_PROBLEMS`, so promoting them
# cannot push a slot down the `skip: True` path that drops comments. They are
# also satisfiable on their own: not reusing a phrase needs no new entity, which
# is why `missing_concrete_anchor` stays advisory here.
REPETITION_DIAGNOSTIC_PROBLEMS = frozenset(
    {
        "opening_reused",
        "opener_family_reused",
        "template_phrase_reused",
    }
)
REPETITION_DIAGNOSTIC_PREFIXES = ("repeated_frame:",)

GUARD_ADVISORY = "off"
GUARD_BLOCKING = "blocking"

#: Set by the adapter from `GENERALIZED_CARD_REPETITION_GUARD`.
REPETITION_GUARD_MODE = GUARD_ADVISORY


def set_repetition_guard(mode: str) -> str:
    """Select whether repeated phrasing may force another Writer attempt."""

    global REPETITION_GUARD_MODE
    value = str(mode or "").strip().lower()
    REPETITION_GUARD_MODE = (
        GUARD_BLOCKING if value == GUARD_BLOCKING else GUARD_ADVISORY
    )
    return REPETITION_GUARD_MODE


def advisory_problems() -> frozenset[str]:
    """Return the problems this run tolerates on an accepted candidate."""

    if REPETITION_GUARD_MODE == GUARD_BLOCKING:
        return SINGLE_STAGE_DIAGNOSTIC_PROBLEMS - REPETITION_DIAGNOSTIC_PROBLEMS
    return SINGLE_STAGE_DIAGNOSTIC_PROBLEMS


def is_repetition_problem(problem: str) -> bool:
    """Return whether a code describes reused phrasing rather than invalidity."""

    return problem in REPETITION_DIAGNOSTIC_PROBLEMS or problem.startswith(
        REPETITION_DIAGNOSTIC_PREFIXES
    )


def only_style_problems(problems: list[str]) -> bool:
    """Return whether every residual failure is phrasing, not validity.

    Run v77 dropped 14 of 186 comments. Promoting the repetition codes to
    blocking made them non-distribution failures, so
    `consider_distribution_candidate` never registered those candidates as a
    fallback and `best_distribution_candidate` stayed None on exhaustion.
    A comment that reuses a phrase is still a comment; losing it also breaks the
    matched thread's structure, which is what `avg_depth` and
    `structural_virality` -- two of the four metrics that currently pass -- are
    measured on.
    """

    if not problems:
        return False
    # Anything the run would have tolerated on a first-pass acceptance counts
    # here too. Run v78 still lost 4 slots whose final residue was
    # `missing_concrete_anchor` or `question_mark_unwanted` -- both advisory, both
    # accepted without comment on attempt 1. Rejecting on attempt 5 what attempt
    # 1 would have kept is not a stricter policy, only an inconsistent one.
    return all(
        is_repetition_problem(problem) or is_single_stage_diagnostic(problem)
        for problem in problems
    )

HARD_REALIZATION_PROBLEMS = frozenset(
    {
        "empty",
        "exact_duplicate",
        "parent_copy",
        "placeholder_literal",
        "planner_skeleton_residue",
    }
)


def is_single_stage_diagnostic(problem: str) -> bool:
    """Return whether a non-empty realization may be retained and audited."""

    return (
        is_soft_length_problem(problem)
        or problem in advisory_problems()
        or problem.startswith(SINGLE_STAGE_DIAGNOSTIC_PREFIXES)
    )


def hard_realization_problems(problems: list[str]) -> list[str]:
    """Return failures that cannot be persisted as a generated comment."""

    return [problem for problem in problems if problem in HARD_REALIZATION_PROBLEMS]


# These are candidate-output failures, not permanent task failures.  A new
# candidate can fix every item without changing the assigned discussion move.
# Unknown problem codes remain non-repairable so programming/configuration
# errors cannot turn into an unbounded model loop.
REPAIRABLE_WRITER_PROBLEMS = frozenset(
    {
        "empty",
        "exact_duplicate",
        "parent_copy",
        "placeholder_literal",
        "planner_skeleton_residue",
        "meta_template_quote_heading",
        "long_helpful_too_generic",
        "missing_concrete_anchor",
        "low_info_too_long",
        "length_too_long",
        "real_slot_too_short",
        "opening_reused",
        "opener_family_reused",
        "template_phrase_reused",
        "first_person_frame_unwanted",
        "uncertainty_frame_unwanted",
        "question_mark_unwanted",
    }
)

REPAIRABLE_WRITER_PREFIXES = (
    # A new candidate can drop a stock frame without changing its assigned move.
    # It has to be listed here or `backend.py:2022` returns skip: True and the
    # slot is lost, which is the trap the v77 drop already demonstrated.
    "repeated_frame:",
    "lexical_overlap_high:",
    "semantic_overlap_high:",
    "semantic_overlap_low:",
    "substantive_length_floor:",
)

DISTRIBUTION_WRITER_PREFIXES = (
    "lexical_overlap_high:",
    "semantic_overlap_high:",
    "semantic_overlap_low:",
)

LOCAL_REPAIR_STRATEGIES = (
    (
        "surface_reconstruction",
        "Rebuild from the local subject or constraint. Use a different opener, "
        "clause order, connective pattern, and cadence; omit acknowledgements "
        "and conversational setup used earlier in the thread.",
    ),
    (
        "direct_local_consequence",
        "State the assigned result, caveat, or action directly, then add only "
        "the local consequence if needed. Do not begin with agreement, reaction, "
        "first-person setup, or a paraphrase of a nearby comment.",
    ),
    (
        "constraint_first",
        "Lead with the unused boundary condition or tradeoff already present in "
        "the plan, then stop after its local implication.",
    ),
    (
        "evidence_first",
        "Lead with one visible anchor, observation, or datapoint already assigned "
        "to this slot. Follow it with at most one local implication.",
    ),
    (
        "parent_relation_first",
        "Realize the assigned reply relation first: answer, correct, qualify, or "
        "react to the parent without restating the parent or the thread consensus.",
    ),
    (
        "cadence_reset",
        "Reset the sentence route completely. Change sentence count, clause "
        "boundaries, and connective rhythm while preserving the same local move.",
    ),
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
    frame_problem = repeated_frame_problem(text, previous_texts)
    if frame_problem:
        problems.append(frame_problem)
    return diagnostics, deduplicate_problems(problems)


# The core's `template_phrase_signature` reads only `tokens[:28]`, so it sees a
# frame only when it opens the comment. Measured on run v76a, the "that's the
# part" family occurs in 15 of 186 comments and the head window catches 4; the
# other 11 sit at token 20, 52, 62, 80 of their comment. That is why promoting
# `template_phrase_reused` to blocking in v77 forced 51 slots to retry and left
# the frame at 7.6% -- the retries were aimed at other families.
#
# This check reads the whole comment. It is kept separate from the core
# signature rather than replacing it, because that signature also decides
# `first_person_frame_unwanted` and `uncertainty_frame_unwanted`, which are
# genuinely about how a comment opens.
#
# The families are LLM discourse tics, not domain vocabulary, so this is
# domain-neutral: it holds for any subject the writer discusses.
REPEATED_FRAME_PATTERNS = {
    "part_frame": r"\b(?:that'?s|that is|thats|was|is)\s+the\s+(?:annoying\s+|real\s+|only\s+)?(?:part|bit|thing)\b|\bthe\s+(?:annoying\s+|real\s+)?(?:part|bit)\s+that\b",
    "basically_frame": r"\bbasically\s+(?:the|how|where|what|it)\b|\bthat'?s\s+basically\b",
    "feels_like_frame": r"\b(?:feels?|felt)\s+like\b",
    "worth_frame": r"\b(?:value\s+proposition|pays\s+for\s+itself|worth\s+it\s+if)\b",
    "matters_frame": r"\b(?:what|that)\s+(?:actually\s+)?matters\b|\bthe\s+real\s+question\b",
}

#: How many earlier comments in this thread may already carry a frame before a
#: new one is a repeat. Real threads share almost no phrasing at all: in the
#: matched thread for seed 8 the most-shared 4-gram reaches 3 of 200 comments
#: (1.5%), so a budget of 2 is still permissive.
REPEATED_FRAME_BUDGET = 2


def frame_families(text: str) -> set[str]:
    """Return every stock frame family present anywhere in the comment."""

    value = str(text or "").replace("’", "'").replace("‘", "'").lower()
    if not value:
        return set()
    return {
        name
        for name, pattern in REPEATED_FRAME_PATTERNS.items()
        if re.search(pattern, value)
    }


def repeated_frame_problem(text: str, previous_texts: list[str] | None) -> str:
    """Flag a stock frame this thread has already leaned on.

    Emitted only under the blocking guard, so `--repetition-guard off` renders
    and validates exactly as every release before it did.
    """

    if REPETITION_GUARD_MODE != GUARD_BLOCKING:
        return ""
    families = frame_families(text)
    if not families:
        return ""
    counts: dict[str, int] = {}
    for earlier in previous_texts or ():
        for name in frame_families(earlier):
            counts[name] = counts.get(name, 0) + 1
    repeated = sorted(
        name for name in families if counts.get(name, 0) >= REPEATED_FRAME_BUDGET
    )
    if not repeated:
        return ""
    return "repeated_frame:" + ",".join(repeated)


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
        "thanks_ack",
        "joke_reaction",
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


def writer_local_repair_task(
    task: Any,
    *,
    problems: list[str],
    repair_round: int,
    previous_candidate_text: str = "",
) -> Any:
    """Keep the planned local move while specifying a repair direction."""

    if task is None:
        return task
    instructions = [
        "This is a local rewrite of the same assigned discussion move. Keep its parent relation, role, factual anchors, and intended payload."
    ]
    lexical = any(item.startswith("lexical_overlap_high:") for item in problems)
    semantic_high = any(
        item.startswith("semantic_overlap_high:") for item in problems
    )
    semantic_low = any(
        item.startswith("semantic_overlap_low:") for item in problems
    )
    surface_reuse = any(
        item in {"opening_reused", "opener_family_reused", "template_phrase_reused"}
        for item in problems
    )
    strategy, strategy_instruction = LOCAL_REPAIR_STRATEGIES[
        (max(1, repair_round) - 1) % len(LOCAL_REPAIR_STRATEGIES)
    ]
    if lexical or surface_reuse:
        instructions.append(strategy_instruction)
    if semantic_high:
        if strategy == "surface_reconstruction":
            instructions.append(
                "Keep the assigned topic but realize its unused evidence role, boundary condition, or decision implication rather than repeating the thread's existing conclusion."
            )
        else:
            instructions.append(
                "Answer the parent through one unresolved constraint or consequence already present in the assigned plan. Do not summarize, endorse, or negate an earlier comment."
            )
    if semantic_low:
        if strategy == "surface_reconstruction":
            instructions.append(
                "Tie the response explicitly to the visible seed or parent issue and then make the assigned local implication. Do not introduce a generic side topic."
            )
        else:
            instructions.append(
                "Use a direct parent-grounded statement: identify the existing issue, then give only the assigned consequence, caveat, or next step."
            )
    if any(item.startswith("substantive_length_floor:") for item in problems):
        instructions.append(
            "This slot represents a substantive local reply. Do not evade the constraint with a fragment, acknowledgement, or bare slogan."
        )
    if "real_slot_too_short" in problems:
        instructions.append(
            "Expand the same local move to the requested substantive length. Add "
            "only assigned context, reasoning, or pacing; do not add another claim."
        )
    if "length_too_long" in problems:
        instructions.append(
            "Compress the same local move by removing setup and secondary clauses. "
            "Do not replace it with a generic acknowledgement."
        )
    if "low_info_too_long" in problems:
        instructions.append(
            "This is a low-information slot. Keep only the assigned reaction, "
            "answer, question, joke, or reference nudge."
        )
    if "question_mark_unwanted" in problems:
        instructions.append(
            "Use a statement or fragment, not a question, while keeping the same local point."
        )
    if "first_person_frame_unwanted" in problems:
        instructions.append(
            "Remove the first-person setup and state the assigned observation directly."
        )
    if "uncertainty_frame_unwanted" in problems:
        instructions.append(
            "Remove uncertainty-preface wording and state the assigned point directly."
        )
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
            "Do not copy the parent or any previous candidate. Reconstruct the "
            "same assigned move from a different sentence route."
        )
    if any(
        item in problems for item in ("missing_concrete_anchor", "long_helpful_too_generic")
    ):
        instructions.append(
            "Use one visible factual anchor already assigned to this slot and keep "
            "the response local. Do not invent a replacement anchor."
        )
    failed_candidate = compact_generated_candidate(previous_candidate_text)
    if failed_candidate:
        instructions.append(
            "Previous failed candidate to avoid reproducing or lightly paraphrasing: "
            + failed_candidate
        )
    forbidden_candidate = (
        f" Never output this failed candidate again: {failed_candidate}."
        if failed_candidate
        else ""
    )
    instructions.append(
        f"Local repair strategy: {strategy}. This is repair round {repair_round}; output one comment body only."
    )
    evidence = bounded_retry_evidence(problems)
    if evidence:
        instructions.append(
            "Measured generated-thread evidence to avoid copying or paraphrasing: "
            + evidence
        )
    repair_note = " ".join(instructions)
    try:
        return replace(
            task,
            planner_intent=f"{getattr(task, 'planner_intent', '')} {repair_note}".strip(),
            must_not_do=(
                f"{getattr(task, 'must_not_do', '')} Do not reuse an earlier "
                f"generated sentence route.{forbidden_candidate}"
            ).strip(),
        )
    except TypeError:
        # Test adapters can use lightweight task objects. Production uses the
        # immutable CommentTask dataclass and takes the branch above.
        return task


def only_repairable_writer_problems(problems: list[str]) -> bool:
    if not problems:
        return True
    return all(
        problem in REPAIRABLE_WRITER_PROBLEMS
        or problem.startswith(REPAIRABLE_WRITER_PREFIXES)
        for problem in problems
    )


def only_distribution_writer_problems(problems: list[str]) -> bool:
    """Return whether a candidate passed every hard content/surface guard."""

    return bool(problems) and all(
        problem.startswith(DISTRIBUTION_WRITER_PREFIXES) for problem in problems
    )


def distribution_candidate_is_reachable(diagnostics: dict[str, Any]) -> bool:
    """Check whether accepting a candidate can still reach the final BLEU band."""

    lexical = dict(diagnostics.get("self_bleu") or {})
    if not lexical.get("available"):
        return True
    final_upper = safe_float(lexical.get("final_upper"), math.inf)
    planned = safe_int(lexical.get("planned_comment_count"), 0)
    processed = safe_int(lexical.get("processed_slot_count"), 0)
    if planned > 0 and processed >= planned:
        return safe_float(lexical.get("proposed_mean"), math.inf) <= final_upper + 1e-12
    return safe_float(lexical.get("minimum_final_mean"), math.inf) <= final_upper + 1e-12


def distribution_candidate_rank(diagnostics: dict[str, Any]) -> tuple[float, ...]:
    """Rank hard-valid candidates by final reachability and target distance."""

    lexical = dict(diagnostics.get("self_bleu") or {})
    semantic = dict(diagnostics.get("semantic_cosine") or {})
    reachable_penalty = 0.0 if distribution_candidate_is_reachable(diagnostics) else 1.0
    return (
        reachable_penalty,
        safe_float(diagnostics.get("joint_target_distance"), math.inf),
        safe_float(lexical.get("target_distance"), math.inf),
        safe_float(semantic.get("target_distance"), math.inf),
    )


def compact_generated_candidate(text: str, *, limit: int = 260) -> str:
    """Bound generated retry evidence without exposing reference text."""

    value = " ".join(str(text or "").split()).replace(";", ",").replace("|", "/")
    if not value:
        return ""
    return value if len(value) <= limit else value[: limit - 3].rstrip() + "..."


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


def bounded_retry_evidence(problems: list[str], *, limit: int = 420) -> str:
    """Expose bounded generated-thread diagnostics to a local retry prompt."""

    rows: list[str] = []
    for problem in problems:
        text = str(problem or "")
        fields: list[str] = []
        if "shared=" in text:
            fields.append("shared=" + text.split("shared=", 1)[1].split(";nearest=", 1)[0])
        if "nearest=" in text:
            fields.append("nearest=" + text.split("nearest=", 1)[1])
        if fields:
            rows.append(";".join(fields))
    value = " || ".join(dict.fromkeys(rows))
    return value if len(value) <= limit else value[: limit - 3].rstrip() + "..."


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


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default
