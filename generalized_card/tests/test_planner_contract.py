from __future__ import annotations

import itertools

from generalized_card.planner_contract import reconcile_planner_contract
from generalized_card.planning_quality import evaluate_plan_batch


def _plan(**updates: str) -> dict[str, str]:
    plan = {
        "sample_id": "1",
        "parent_sample_id": "2",
        "_slot_word_count": "40",
        "_slot_surface_label": "ordinary_turn",
        "story_mode": "no_story",
        "affect_role": "neutral",
        "evidence_mode": "firsthand_experience",
        "payload_type": "meta_or_template",
        "comment_function": "reaction",
        "speaker_role": "side_observer",
        "reply_delta_type": "useful_extension",
        "semantic_move": "make one local contribution",
    }
    plan.update(updates)
    return plan


def _blocking_codes(plan: dict[str, str]) -> set[str]:
    report = evaluate_plan_batch(
        {1: plan},
        max_perspective_share=1.0,
        require_reply_novelty=False,
    )
    return {issue.code for issue in report.blocking_issues}


def test_v94_failure_shapes_compile_to_realisable_contracts() -> None:
    cases = (
        _plan(
            parent_sample_id="",
            _slot_word_count="12",
            _slot_surface_label="short_turn",
            payload_type="advice",
            comment_function="recommendation_advice",
            speaker_role="advisor",
        ),
        _plan(
            parent_sample_id="42",
            _slot_word_count="36",
            affect_role="neutral",
            evidence_mode="none_assertion",
            payload_type="soft_helpful",
            speaker_role="gratitude_reply",
            reply_delta_type="social_close",
        ),
        _plan(
            parent_sample_id="",
            _slot_word_count="118",
            _slot_surface_label="long_turn",
            evidence_mode="small_observation",
            payload_type="meta_or_template",
        ),
    )

    for plan in cases:
        reconcile_planner_contract(plan)
        assert not _blocking_codes(plan)


def test_contract_compiler_covers_story_social_and_capacity_cross_product() -> None:
    for story, affect, role, payload, words, surface in itertools.product(
        ("no_story", "specific_personal_story"),
        ("neutral", "gratitude"),
        ("side_observer", "gratitude_reply"),
        ("meta_or_template", "personal_story", "advice"),
        (3, 40, 118),
        ("micro", "ordinary_turn", "long_turn"),
    ):
        plan = _plan(
            story_mode=story,
            affect_role=affect,
            speaker_role=role,
            payload_type=payload,
            _slot_word_count=str(words),
            _slot_surface_label=surface,
        )
        reconcile_planner_contract(plan)
        assert not _blocking_codes(plan), plan
