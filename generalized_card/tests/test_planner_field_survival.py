from __future__ import annotations

import inspect
import random
import re
import unittest
from types import SimpleNamespace

from generalized_card import backend as backend_module
from generalized_card.domain_claim import enrich_domain_claim_fields
from generalized_card.generation_distribution import enrich_distribution_plan_fields
from generalized_card.long_form_planning import enrich_development_plan_fields


class PlannerFieldSurvivalTest(unittest.TestCase):
    """Every generalized planner field needs an explicit enrich step.

    The shared CARD JSON parser keeps only the fields it declares, so a new
    planner field is dropped in silence. That happened to `domain_claim`: the
    mechanism shipped, ran a full 520-comment generation, and carried a claim on
    0 slots while every log looked healthy. These assertions make the omission
    fail in the suite instead of in a paid run.
    """

    #: Fields the generalized Planner asks for that the shared parser does not keep.
    GENERALIZED_PLANNER_FIELDS = (
        "tone_class",
        "affect_role",
        "decision_boundary",
        "development_plan",
        "domain_claim",
    )

    def test_every_generalized_field_survives_normalization(self) -> None:
        payload = {
            "comment_plans": [
                {
                    "sample_id": "S1",
                    "tone_class": "polite",
                    "affect_role": "approval",
                    "decision_boundary": "whether the adapter holds focus",
                    "development_plan": "beat one || beat two",
                    "domain_claim": "EF glass adapts to RF bodies with the adapter",
                }
            ]
        }
        normalized: dict[int, dict[str, str]] = {1: {}}
        for enrich in (
            enrich_distribution_plan_fields,
            enrich_development_plan_fields,
            enrich_domain_claim_fields,
        ):
            normalized = enrich(payload, normalized)
        for field in self.GENERALIZED_PLANNER_FIELDS:
            self.assertIn(field, normalized[1], field)
            self.assertTrue(str(normalized[1][field]).strip(), field)

    def test_backend_calls_an_enrich_step_for_each_field(self) -> None:
        # A field can also be lost by adding the enrich function and forgetting
        # to call it, which is how it was lost the first time.
        source = inspect.getsource(backend_module)
        block = re.search(
            r"def normalize_comment_plans\(.*?return apply_slot_distribution_schedule",
            source,
            re.S,
        )
        self.assertIsNotNone(block, "normalize_comment_plans not found")
        body = block.group()
        for enrich in (
            "enrich_distribution_plan_fields",
            "enrich_development_plan_fields",
            "enrich_domain_claim_fields",
        ):
            self.assertIn(enrich, body, enrich)

    def test_domain_claim_off_clears_a_model_returned_claim(self) -> None:
        payload = {
            "comment_plans": [
                {
                    "sample_id": "S1",
                    "domain_claim": "a fact the disabled arm must not retain",
                }
            ]
        }
        normalized = enrich_domain_claim_fields(payload, {1: {}}, enabled=False)
        self.assertEqual(normalized[1]["domain_claim"], "")

    def test_selective_schedule_clears_an_unscheduled_model_claim(self) -> None:
        payload = {
            "comment_plans": [
                {"sample_id": "S1", "domain_claim": "unscheduled fact"},
                {"sample_id": "S2", "domain_claim": "scheduled fact"},
            ]
        }
        normalized = enrich_domain_claim_fields(
            payload,
            {1: {}, 2: {}},
            allowed_sample_ids={2},
        )
        self.assertEqual(normalized[1]["domain_claim"], "")
        self.assertEqual(normalized[2]["domain_claim"], "scheduled fact")

    def test_planner_prompt_requests_every_field_it_relies_on(self) -> None:
        from generalized_card import prompts

        source = inspect.getsource(prompts)
        for field in self.GENERALIZED_PLANNER_FIELDS:
            self.assertIn(f'"{field}"', source, field)

    def test_discourse_contract_reaches_the_default_writer_end_to_end(self) -> None:
        """A planned rant must not collapse into an unspecified helpful turn."""

        from generalized_card.backend import (
            configure_generator_backend,
            load_generator_backend,
        )
        from generalized_card.domain import load_domain_config

        module = configure_generator_backend(
            load_generator_backend(),
            load_domain_config("camera"),
        )
        module.GENERALIZED_WRITER_PROMPT_MODE = "focused"
        self.assertFalse(
            module.real_text_allows_first_person_frame("I used this yesterday")
        )
        self.assertFalse(
            module.real_text_allows_uncertainty_frame("I think maybe it could work?")
        )
        branch = module.BranchPlan(
            branch_id=1,
            anchor_quote="visible shutter problem",
            anchor_source="seed",
            detour_type="none",
            branch_goal="react to one reliability failure",
            allowed_functions=("reaction",),
            evidence_modes=("none_assertion",),
            tone_palette=("annoyed",),
            story_modes=("no_story",),
            content_angles=("risk_reliability_support",),
        )
        raw_plan = {
            "sample_id": "S1",
            "branch_id": 1,
            "payload_type": "rant",
            "comment_function": "reaction",
            "content_angle": "risk_reliability_support",
            "evidence_mode": "none_assertion",
            "story_mode": "no_story",
            "voice": "annoyed",
            "speaker_role": "ranter",
            "semantic_move": "react to the sticky shutter failure",
            "local_topic": "shutter reliability",
            "reply_relation": "reacts_to_seed",
            "stance": "hard_disagree",
            "detail_focus": "sticky shutter end-to-end marker",
            "avoid_repeating": "generic troubleshooting end-to-end marker",
            "claim_family": "miscellaneous",
            "claim_key": "sticky_shutter_failure",
            "perspective_id": "seed_local",
            "domain_intent": "vent about the failed repair end-to-end marker",
            "decision_boundary": "whether the failure is acceptable",
            "opening_style": "lead with the failure",
            "context_aperture": "full_seed",
            "tone_class": "impolite",
            "affect_role": "anger",
            "development_plan": "none",
            "domain_claim": "none",
        }
        normalized = module.normalize_comment_move_plans(
            {"comment_plans": [raw_plan]},
            branches=[branch],
        )
        seed = module.SeedPost(
            index=0,
            title="Sticky shutter question",
            body="The shutter keeps sticking after repair.",
            content="Sticky shutter question\nThe shutter keeps sticking after repair.",
            source_raw_post_id="field-survival-seed",
            real_num_comments=1,
            metadata={},
        )
        tasks = module.expand_matched_real_sample_to_tasks(
            branches=[branch],
            target=module.ThreadTarget(1, 1, 0, "quiet", "matched"),
            seed_post=seed,
            matched_real_thread={
                "comments": [
                    {
                        "body": "Side note: Thanks, I appreciate it. I think maybe this could work. "
                        + " ".join(f"shape{index}" for index in range(33)),
                        "comment_id": "real_1",
                        "comment_fullname": "t1_real_1",
                        "parent_id": "t3_seed",
                    }
                ]
            },
            matched_real_comments=100,
            comment_plans=normalized,
            rng=random.Random(42),
        )
        self.assertEqual(len(tasks), 1)
        task = tasks[0]
        self.assertEqual(task.surface_texture, "plain")
        self.assertEqual(task.real_surface_shape, "full_answer")
        self.assertNotEqual(task.real_tone_slot, "pure_acknowledgement")
        self.assertFalse(task.allow_first_person_frame)
        self.assertFalse(task.allow_uncertainty_frame)
        for field in (
            "payload_type",
            "comment_function",
            "content_angle",
            "evidence_mode",
            "speaker_role",
            "voice",
            "stance",
            "detail_focus",
            "domain_intent",
            "avoid_repeating",
        ):
            self.assertEqual(getattr(task, field), normalized[1][field], field)
        rendered = module.build_writer_prompt(
            profile="gpt54_reddit_writer",
            seed_post=seed,
            task=module.finalize_rebalanced_task(task),
            parent_comment=None,
            previous_comments=[],
            recent_openings=[],
        )
        for expected in (
            "- payload form: rant",
            "- speaker role: ranter",
            "- stance: hard_disagree",
            "- specific detail: sticky shutter end-to-end marker",
            "- decision intent: vent about the failed repair end-to-end marker",
            "- content to avoid: generic troubleshooting end-to-end marker",
        ):
            self.assertEqual(rendered.count(expected), 1, expected)

    def test_reply_planner_asks_for_claim_family_by_enumeration(self) -> None:
        """A closed vocabulary must be enumerated wherever it is requested.

        The root request interpolates the full family list; the reply request
        asked for "one generic claim family" with no list anywhere in it. Every
        answer therefore fell outside the vocabulary and normalized to
        `miscellaneous` -- 61 of 61 reply slots in v70 against 14 distinct
        families across roots -- which silently disabled the per-thread
        claim-family share cap for the 45% of the thread that replies make up.
        """

        from generalized_card import prompts
        from generalized_card.reply_planning import render_direct_reply_planner_prompt

        families = prompts.GENERIC_CLAIM_FAMILIES
        root = {
            "body": "word " * 40,
            "depth": 0,
            "comment_fullname": "t1_p",
            "parent_id": "t3_seed",
        }
        reply = {
            "body": "word " * 30,
            "depth": 1,
            "comment_fullname": "t1_c",
            "parent_id": "t1_p",
        }
        rendered = render_direct_reply_planner_prompt(
            config=SimpleNamespace(community_context="a camera community"),
            backend=SimpleNamespace(
                compact=lambda value, limit: str(value)[:limit],
                CLAIM_FAMILIES=families,
            ),
            seed_post=SimpleNamespace(title="t", body="b", content="c"),
            comments=[reply],
            all_comments=[root, reply],
            sample_offset=1,
            prior_plans=[{"sample_id": "S1", "semantic_move": "the parent point"}],
            slot_distribution="- S2: tone_class=neutral",
        )
        for family in families:
            self.assertIn(family, rendered, family)
        self.assertNotIn('"claim_family": "one generic claim family"', rendered)


class SpeakerIdSurvivalTest(unittest.TestCase):
    """`speaker_id` has to reach the Writer through the real pipeline.

    Written for the same reason as the class above. `semantic_move` was set by
    the Planner and then silently overwritten in 347 of 347 reply slots because
    it was missing from one tuple; the mechanism shipped, ran, and looked healthy
    in every log. `speaker_id` is set once during task expansion and then passes
    through the surface rebalancer, the planner-contract restore, and the task
    finalizer before anything reads it, so each of those is asserted here rather
    than assumed.
    """

    def _task(self, core):
        return core.CommentTask(
            local_task_id=3,
            local_parent_task_id=None,
            depth=0,
            branch_id=1,
            branch_goal="grip comfort",
            visible_scope="seed",
            local_anchor="grip",
            comment_function="verdict_evaluation",
            content_angle="fit_use_case",
            evidence_mode="firsthand_experience",
            story_mode="no_story",
            voice="casual_neutral",
            payload_type="soft_helpful",
            length_bucket="long",
            speaker_role="datapoint_only",
            utterance_mode="one_datapoint",
            surface_texture="plain",
            allow_first_person_frame=True,
            allow_uncertainty_frame=False,
            planner_intent="one local verdict",
            must_not_do="",
            real_sample_id=3,
            real_word_count=120,
            speaker_id="S007",
        )

    def test_speaker_id_survives_the_task_finalizer(self) -> None:
        from generalized_card.backend import load_generator_backend

        core = load_generator_backend()
        finalized = core.finalize_rebalanced_task(self._task(core))
        self.assertEqual(finalized.speaker_id, "S007")

    def test_speaker_id_survives_the_planner_contract_restore(self) -> None:
        from generalized_card.backend import load_generator_backend
        from generalized_card.task_distribution import restore_planner_task_contract

        core = load_generator_backend()
        restored = restore_planner_task_contract(
            self._task(core),
            {"semantic_move": "a different point", "stance": "disagree"},
            core=core,
        )
        self.assertEqual(restored.speaker_id, "S007")

    def test_speaker_id_is_a_slot_invariant(self) -> None:
        from generalized_card.task_distribution import (
            PLANNER_AND_SLOT_INVARIANTS,
            PLANNER_OWNED_TASK_FIELDS,
        )

        self.assertIn("speaker_id", PLANNER_AND_SLOT_INVARIANTS)
        # It is a matched-slot fact, not something the Planner may reassign.
        self.assertNotIn("speaker_id", PLANNER_OWNED_TASK_FIELDS)

    def test_the_author_name_falls_back_when_no_speaker_is_resolved(self) -> None:
        """`--speaker-identity off` must reproduce the pre-v77 naming exactly."""

        import inspect

        from generalized_card.backend import load_generator_backend

        source = inspect.getsource(load_generator_backend().generate_post_from_tasks)
        self.assertIn(
            'f"sampled_user_{run_index}_{post_slot}_{task.local_task_id}"', source
        )
        self.assertIn(
            'f"sampled_user_{run_index}_{post_slot}_{task.speaker_id}"', source
        )


if __name__ == "__main__":
    unittest.main()
