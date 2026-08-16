"""Planner-owned task contracts for generalized CARD generation."""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
from typing import Any, Callable


# CARD's surface rebalancer is useful for role and answer-shape diversity, but
# its historical demotions may also rewrite the planned claim or matched slot.
# These fields define the generalized boundary that the surface pass cannot cross.
PLANNER_AND_SLOT_INVARIANTS = (
    "local_task_id",
    "local_parent_task_id",
    "depth",
    "branch_id",
    "branch_goal",
    "visible_scope",
    "local_anchor",
    "content_angle",
    "evidence_mode",
    "length_bucket",
    "real_sample_id",
    "real_parent_sample_id",
    "real_word_count",
    # `speaker_id` is a matched-slot fact, derived from the same real comment as
    # `real_sample_id`, so the surface rebalancer must not rewrite it. Listed
    # here for the reason `semantic_move` is: a field the surface pass can reach
    # but does not own is a field that silently disappears. That is exactly how
    # a reply's planned move was lost in 347 of 347 slots.
    "speaker_id",
    "semantic_move",
    "local_topic",
    "reply_relation",
    "stance",
    "detail_focus",
    "avoid_repeating",
    "claim_key",
    "claim_family",
    "perspective_id",
    "domain_intent",
    "decision_boundary",
    "reply_delta",
    "parent_semantic_move",
    "parent_decision_boundary",
    "opening_style",
    "development_plan",
    "context_aperture",
    "real_surface_shape",
    "concrete_anchors",
)


# The shared CARD expander derives a few social labels from the matched raw
# comment after the generalized Planner has already assigned its slot contract.
# In generalized CARD that order is backwards: the raw comment is an anonymous
# structural template, while the Planner is the owner of semantic and social
# controls. Keep the structural fields (tree, word scale, and surface shape),
# but restore the explicit plan before the Writer sees the task.
#
# `semantic_move` is listed for the same reason `decision_boundary` is. Slot
# expansion replaces a reply's move with its novelty anchor, which is often a
# bare noun phrase ("telephoto reach on a compact camera"), and that phrase then
# becomes the reply's entire semantic content: 61 of 61 replies in v70. The
# increment still reaches the Writer -- `reply_delta` and `reply_novelty_anchor`
# are separate rendered fields in every reply prompt -- so overwriting the move
# only discards the Planner's fuller proposition.
PLANNER_OWNED_TASK_FIELDS = (
    "comment_function",
    "content_angle",
    "semantic_move",
    "evidence_mode",
    "story_mode",
    "voice",
    "payload_type",
    "speaker_role",
    "reply_relation",
    "stance",
    "detail_focus",
    "claim_key",
    "claim_family",
    "perspective_id",
    "domain_intent",
    "decision_boundary",
    "opening_style",
    "context_aperture",
)


def restore_planner_task_contract(task: Any, plan: dict[str, Any], *, core: Any) -> Any:
    """Restore Planner-owned controls without changing anonymous slot shape.

    ``task`` has already inherited the matched thread's parent relation,
    word-count target, and surface texture. This function only replaces fields
    explicitly returned by the structured Comment Planner, then recomputes the
    dependent utterance mode from that restored contract and the anonymous word
    count. It neither reads real text nor selects among Writer candidates.
    """

    if not plan:
        return task

    updates: dict[str, Any] = {}
    for field in PLANNER_OWNED_TASK_FIELDS:
        value = plan.get(field)
        if isinstance(value, str) and value.strip():
            updates[field] = value.strip()

    payload = str(updates.get("payload_type") or getattr(task, "payload_type", ""))
    function = str(
        updates.get("comment_function") or getattr(task, "comment_function", "")
    )
    voice = str(updates.get("voice") or getattr(task, "voice", ""))
    role = str(updates.get("speaker_role") or getattr(task, "speaker_role", ""))
    word_count = int(getattr(task, "real_word_count", 0) or 0)

    # Length remains an anonymous structural target. Do not let a semantic
    # label turn a medium or long matched slot into a canned short response.
    updates["length_bucket"] = core.length_bucket_for_payload(
        payload_type=payload,
        word_count=word_count,
    )
    updates["utterance_mode"] = core.infer_utterance_mode(
        payload_type=payload,
        speaker_role=role,
        comment_function=function,
        voice=voice,
        real_word_count=word_count,
    )

    story = str(updates.get("story_mode") or getattr(task, "story_mode", "no_story"))
    evidence = str(
        updates.get("evidence_mode") or getattr(task, "evidence_mode", "")
    )
    stance = str(updates.get("stance") or getattr(task, "stance", ""))
    updates["allow_first_person_frame"] = (
        story != "no_story"
        or payload in {"personal_story", "fragment_datapoint"}
        or evidence == "firsthand_experience"
    )
    updates["allow_uncertainty_frame"] = (
        stance == "uncertain"
        or function == "question_followup"
        or payload == "narrow_question"
    )
    return replace(task, **updates)


def rebalance_card_surfaces(
    tasks: list[Any],
    **kwargs: Any,
) -> tuple[list[Any], dict[str, Any]]:
    """Keep the Planner's coherent slot contract intact.

    CARD's historical surface rebalancer contains fixed finance-era global
    priors.  Applying it after a held-out domain template mutates roles,
    payloads, and voices a second time.  Generalized CARD keeps the same
    Planner/Writer architecture but makes the calibrated Planner the single
    owner of those controls.  This pass therefore only reports; the share
    arguments are accepted so the caller's contract is unchanged and ignored.
    """

    if not tasks:
        return [], _report([], [])

    del kwargs
    result = list(tasks)
    report = _report(tasks, result)
    report["policy"] = "planner_owned_generalized_surface_contract"
    report["restored_substantive_payload_task_ids"] = []
    report["restored_substantive_payload_count"] = 0
    return result, report


def _collapses_substantive_slot(original: Any, candidate: Any) -> bool:
    try:
        real_words = int(getattr(original, "real_word_count", 0) or 0)
    except (TypeError, ValueError):
        real_words = 0
    original_payload = str(getattr(original, "payload_type", "") or "")
    candidate_payload = str(getattr(candidate, "payload_type", "") or "")
    candidate_function = str(getattr(candidate, "comment_function", "") or "")
    low_information = {
        "low_info_reaction",
        "bare_answer",
        "narrow_question",
        "joke",
        "meta_or_template",
        "side_tangent",
    }
    return (
        real_words >= 35
        and original_payload not in low_information
        and (
            candidate_payload in low_information
            or candidate_function in {"reaction", "offtopic_noise"}
        )
    )


def _combined_instruction(original: Any, balanced: Any, *, prefix: str) -> str:
    original_text = " ".join(str(original or "").split())
    balanced_text = " ".join(str(balanced or "").split())
    if not balanced_text or balanced_text == original_text:
        return original_text
    if original_text and original_text in balanced_text:
        return balanced_text
    if not original_text:
        return balanced_text
    return f"{original_text} {prefix}: {balanced_text}"


def _report(before: list[Any], after: list[Any]) -> dict[str, Any]:
    before_by_id = {int(task.local_task_id): task for task in before}
    changed_ids = [
        int(task.local_task_id)
        for task in after
        if int(task.local_task_id) in before_by_id
        and _surface_signature(task)
        != _surface_signature(before_by_id[int(task.local_task_id)])
    ]
    return {
        "task_count": len(after),
        "surface_rebalanced_task_ids": changed_ids,
        "surface_rebalanced_count": len(changed_ids),
        "before": _counts(before),
        "after": _counts(after),
        "preserved_fields": list(PLANNER_AND_SLOT_INVARIANTS),
    }


def _surface_signature(task: Any) -> tuple[str, ...]:
    return tuple(
        str(getattr(task, field, "") or "")
        for field in (
            "speaker_role",
            "payload_type",
            "comment_function",
            "voice",
            "utterance_mode",
            "surface_texture",
            "surface_skeleton",
            "tone_shape",
        )
    )


def _counts(tasks: list[Any]) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for field in (
        "speaker_role",
        "payload_type",
        "comment_function",
        "voice",
        "utterance_mode",
        "surface_texture",
        "tone_shape",
        "length_bucket",
    ):
        counts = Counter(str(getattr(task, field, "") or "unassigned") for task in tasks)
        result[field] = dict(sorted(counts.items()))
    return result
