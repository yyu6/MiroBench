"""Compile fixed slot targets and Planner choices into one realizable contract."""

from __future__ import annotations

from typing import Any


LOW_INFORMATION_PAYLOADS = {
    "low_info_reaction",
    "bare_answer",
    "narrow_question",
    "joke",
    "side_tangent",
    "meta_or_template",
}


def reconcile_planner_contract(plan: dict[str, Any]) -> list[dict[str, Any]]:
    """Reconcile dependent routing labels without changing the semantic move.

    Story/affect targets and anonymous capacity are already fixed before this
    function runs. The model still chooses the local contribution. These
    corrections only make its dependent evidence and surface route agree with
    those two authorities.
    """

    events: list[dict[str, Any]] = []
    words = _as_int(plan.get("_slot_word_count"))
    surface = _value(plan, "_slot_surface_label")
    story = _value(plan, "story_mode") or "no_story"
    affect = _value(plan, "affect_role")
    parent_id = _as_int(plan.get("parent_sample_id"))
    social_route = (
        affect in {"gratitude", "relief"}
        or _value(plan, "speaker_role") == "gratitude_reply"
        or _value(plan, "reply_delta_type") == "social_close"
    )

    if story == "no_story":
        if _value(plan, "evidence_mode") == "firsthand_experience":
            _replace(
                plan,
                events,
                "evidence_mode",
                "small_observation",
                "no_story_uses_non_narrative_evidence",
            )
        if _value(plan, "payload_type") == "personal_story":
            _replace(
                plan,
                events,
                "payload_type",
                "fragment_datapoint" if words > 5 else "bare_answer",
                "no_story_uses_non_narrative_payload",
            )
    else:
        for field, value in (
            ("evidence_mode", "firsthand_experience"),
            ("comment_function", "personal_datapoint"),
            ("payload_type", "personal_story" if words >= 70 else "fragment_datapoint"),
            ("speaker_role", "datapoint_only"),
        ):
            _replace(plan, events, field, value, "scheduled_story_joint_contract")

    if 0 < words <= 5 and surface == "micro":
        for field, value in (
            ("story_mode", "no_story"),
            ("evidence_mode", "none_assertion"),
            ("comment_function", "reaction"),
            ("payload_type", "low_info_reaction"),
            ("speaker_role", "side_observer"),
        ):
            _replace(plan, events, field, value, "micro_slot_capacity")

    if social_route:
        if affect not in {"gratitude", "relief"}:
            _replace(
                plan,
                events,
                "affect_role",
                "gratitude",
                "planner_social_close_preserves_joint_affect",
            )
        for field, value in (
            ("story_mode", "no_story"),
            ("evidence_mode", "none_assertion"),
            ("speaker_role", "gratitude_reply"),
            ("comment_function", "reaction"),
            ("payload_type", "soft_helpful" if words >= 35 else "low_info_reaction"),
        ):
            _replace(plan, events, field, value, "social_close_joint_contract")
        if parent_id > 0:
            _replace(
                plan,
                events,
                "reply_delta_type",
                "social_close",
                "social_close_joint_contract",
            )

    payload = _value(plan, "payload_type")
    if words >= 35 and surface in {"ordinary_turn", "long_turn"} and payload in LOW_INFORMATION_PAYLOADS:
        _replace(
            plan,
            events,
            "payload_type",
            "fragment_datapoint",
            "substantive_slot_preserves_information_density",
        )

    return events


def _replace(
    plan: dict[str, Any],
    events: list[dict[str, Any]],
    field: str,
    value: str,
    reason: str,
) -> None:
    before = str(plan.get(field) or "").strip()
    if before == value:
        return
    plan[field] = value
    events.append({"field": field, "before": before, "after": value, "reason": reason})


def _value(plan: dict[str, Any], field: str) -> str:
    return str(plan.get(field) or "").strip().lower()


def _as_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0
