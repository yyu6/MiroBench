from __future__ import annotations

from types import SimpleNamespace

from generalized_card.plan_repair import (
    apply_repair_candidate,
    render_field_repair_instruction,
    repair_merge_fields,
)


def _report(*codes: str) -> SimpleNamespace:
    return SimpleNamespace(
        repair_issues=tuple(
            SimpleNamespace(sample_id=9, code=code) for code in codes
        )
    )


def test_single_long_form_failure_merges_only_development_plan() -> None:
    fields = repair_merge_fields(_report("long_form_capacity"), 9)
    selected = {
        "development_plan": "",
        "evidence_mode": "small_observation",
        "story_mode": "no_story",
    }
    candidate = {
        "development_plan": "one || two || three || four || five",
        "evidence_mode": "firsthand_experience",
        "story_mode": "no_story",
    }

    merged = apply_repair_candidate(selected, candidate, merge_fields=fields)

    assert fields == ("development_plan",)
    assert merged["development_plan"] == candidate["development_plan"]
    assert merged["evidence_mode"] == "small_observation"
    assert "Only development_plan is still failing" in (
        render_field_repair_instruction(fields)
    )


def test_multiple_blocking_failures_keep_full_plan_replacement() -> None:
    fields = repair_merge_fields(
        _report("long_form_capacity", "social_contract_conflict"),
        9,
    )
    candidate = {"development_plan": "five beats", "story_mode": "no_story"}

    assert fields == ()
    assert apply_repair_candidate(
        {"development_plan": "", "story_mode": "specific_personal_story"},
        candidate,
        merge_fields=fields,
    ) == candidate


def test_nonblocking_companion_issue_keeps_joint_repair_prompt_coherent() -> None:
    assert repair_merge_fields(
        _report("long_form_capacity", "semantic_collision"),
        9,
    ) == ()
