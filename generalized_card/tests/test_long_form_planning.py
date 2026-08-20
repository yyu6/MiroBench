from __future__ import annotations

import unittest
from dataclasses import dataclass

from generalized_card.long_form_planning import (
    development_beats,
    development_plan_problem,
    expected_development_beats,
    normalize_development_plan,
    reconcile_development_plan_capacity,
    render_development_guidance,
)


@dataclass(frozen=True)
class Task:
    real_word_count: int
    development_plan: str = ""


class LongFormPlanningTest(unittest.TestCase):
    def test_short_slots_do_not_receive_a_synthetic_depth_target(self) -> None:
        self.assertEqual(expected_development_beats(100), 0)
        self.assertEqual(render_development_guidance(Task(80)), "")

    def test_short_slot_drops_copied_development_schema_text(self) -> None:
        plan = {
            "_slot_word_count": "30",
            "development_plan": "none for a short slot || otherwise add beats",
        }
        event = reconcile_development_plan_capacity(plan)
        self.assertEqual(plan["development_plan"], "")
        self.assertEqual(event["reason"], "slot_has_no_long_form_capacity")

    def test_long_slot_keeps_real_development_beats(self) -> None:
        plan = {
            "_slot_word_count": "180",
            "development_plan": "show the setup || report the friction",
        }
        self.assertIsNone(reconcile_development_plan_capacity(plan))
        self.assertIn("show the setup", plan["development_plan"])

    def test_long_tail_capacity_matches_the_observed_realization_rate(self) -> None:
        # The Writer realizes about 21 words per planned beat, measured over the
        # long slots of a 197-comment thread. The budget has to track that rate
        # or long slots are under-specified, come out short, and flatten the
        # thread's length spread.
        self.assertEqual(expected_development_beats(160), 8)

    def test_beat_budget_stops_where_the_planner_saturates(self) -> None:
        # Raising the ceiling to 40 did not raise the reachable length. Measured
        # over the v96 slots that carried a beat plan: asked ~6 the Planner
        # supplied 5.2 and the slot realized 0.95x its matched length; asked
        # 14-40 it supplied 9.5 and realized 0.60x. The largest plan any slot
        # received was 26. Scale above the saturation point is carried by
        # `comment_structure`, not by an unreachable beat request.
        self.assertEqual(expected_development_beats(320), 12)
        self.assertEqual(expected_development_beats(900), 12)

    def test_beat_budget_is_monotonic_and_bounded(self) -> None:
        budgets = [expected_development_beats(words) for words in range(101, 1200, 17)]
        self.assertEqual(budgets, sorted(budgets))
        self.assertLessEqual(max(budgets), 12)
        self.assertGreaterEqual(min(budgets), 3)

    def test_normalizer_accepts_lists_and_removes_duplicate_beats(self) -> None:
        normalized = normalize_development_plan(
            [
                "1. state the visible constraint",
                "2. explain its local consequence",
                "state the visible constraint",
            ]
        )
        self.assertEqual(
            development_beats(normalized),
            ["state the visible constraint", "explain its local consequence"],
        )

    def test_long_plan_problem_reports_missing_content_capacity(self) -> None:
        problem = development_plan_problem(
            {
                "_slot_word_count": "400",
                "development_plan": "state the constraint || explain the consequence",
            }
        )
        self.assertIn("development_plan contains 2", problem)
        self.assertEqual(
            development_plan_problem(
                {
                    "_slot_word_count": "400",
                    "development_plan": " || ".join(
                        f"beat {index}" for index in range(1, 20)
                    ),
                }
            ),
            "",
        )

    def test_writer_guidance_keeps_each_planned_beat_visible_once(self) -> None:
        guidance = render_development_guidance(
            Task(
                300,
                "start from the observed friction || explain the workflow effect || add a boundary || close with the local reaction",
            )
        )
        self.assertIn("1. start from the observed friction", guidance)
        self.assertIn("4. close with the local reaction", guidance)
        self.assertIn("Realize each beat once", guidance)

    def test_direct_reply_planner_requires_beats_for_a_long_slot(self) -> None:
        # Every long slot below the root previously received no development
        # plan at all, because this planner's schema omitted the field.
        from types import SimpleNamespace

        from generalized_card.reply_planning import render_direct_reply_planner_prompt

        backend = SimpleNamespace(compact=lambda value, limit: str(value)[:limit])
        prompt = render_direct_reply_planner_prompt(
            config=SimpleNamespace(community_context="a camera community"),
            backend=backend,
            seed_post=SimpleNamespace(title="t", body="b", content="b"),
            comments=[{"body": "word " * 300, "depth": 2}],
            all_comments=[
                {"comment_id": "c1", "parent_id": None, "depth": 1, "body": "x"},
                {"comment_id": "c2", "parent_id": "t1_c1", "depth": 2, "body": "word " * 300},
            ],
            sample_offset=1,
            prior_plans=[{"sample_id": "1", "semantic_move": "parent move"}],
            slot_distribution="- none",
        )
        self.assertIn('"development_plan"', prompt)
        self.assertIn("development_plan: required, about 12 beats", prompt)

    def test_direct_reply_planner_skips_beats_for_a_short_slot(self) -> None:
        from types import SimpleNamespace

        from generalized_card.reply_planning import render_direct_reply_planner_prompt

        backend = SimpleNamespace(compact=lambda value, limit: str(value)[:limit])
        prompt = render_direct_reply_planner_prompt(
            config=SimpleNamespace(community_context="a camera community"),
            backend=backend,
            seed_post=SimpleNamespace(title="t", body="b", content="b"),
            comments=[{"body": "short reply", "depth": 2}],
            all_comments=[
                {"comment_id": "c1", "parent_id": None, "depth": 1, "body": "x"},
                {"comment_id": "c2", "parent_id": "t1_c1", "depth": 2, "body": "short reply"},
            ],
            sample_offset=1,
            prior_plans=[{"sample_id": "1", "semantic_move": "parent move"}],
            slot_distribution="- none",
            slot_controls={
                2: {
                    "story_mode": "no_story",
                    "tone_class": "polite",
                    "affect_role": "approval",
                    "opener_type": "content_phrase",
                }
            },
        )
        self.assertIn("development_plan: none; this slot is short", prompt)
        self.assertIn("Story contract: no_story", prompt)
        self.assertIn("Affect contract: approval", prompt)
        self.assertIn("Opening grammar: content_phrase", prompt)
        self.assertNotIn(
            "Allowed reply_delta_type: corroborating_datapoint", prompt
        )


if __name__ == "__main__":
    unittest.main()
