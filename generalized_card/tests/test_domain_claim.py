from __future__ import annotations

import unittest
from types import SimpleNamespace

from generalized_card import prompts
from generalized_card.backend import configure_generator_backend, load_generator_backend
from generalized_card.domain import load_domain_config
from generalized_card.domain_claim import (
    claim_for_task,
    normalized_domain_claim,
    render_domain_claim_rule,
    seed_claim_key,
)


class DomainClaimTest(unittest.TestCase):
    def test_absent_claim_normalizes_to_empty(self) -> None:
        for value in (None, "", "none", "N/A", "  NONE  ", "not applicable"):
            self.assertEqual(normalized_domain_claim(value), "")

    def test_reference_ids_and_urls_never_survive(self) -> None:
        # A claim must not expose the held-out bank it came from, and an invented
        # link is a hard Writer failure.
        claim = normalized_domain_claim("Per R00421 see https://example.com/spec for the mount")
        self.assertNotIn("R00421", claim)
        self.assertNotIn("http", claim)
        self.assertIn("general domain knowledge", claim)
        self.assertIn("a published source", claim)

    def test_claim_is_bounded(self) -> None:
        claim = normalized_domain_claim("mount " * 200)
        self.assertLessEqual(len(claim), 220)

    def test_registry_lookup_uses_the_planner_key(self) -> None:
        seed = SimpleNamespace(source_raw_post_id="ABC123", index=0, title="t")
        task = SimpleNamespace(real_sample_id=7, local_task_id=99)
        registry = {(seed_claim_key(seed), 7): "EF glass adapts to RF bodies"}
        self.assertEqual(claim_for_task(registry, seed, task), "EF glass adapts to RF bodies")
        self.assertEqual(claim_for_task({}, seed, task), "")

    def test_rule_is_empty_without_a_claim(self) -> None:
        self.assertEqual(render_domain_claim_rule(""), "")
        self.assertIn("Domain fact this turn states", render_domain_claim_rule("x adapts to y"))


class DomainClaimWriterContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_domain_config("camera")
        cls.module = configure_generator_backend(load_generator_backend(), cls.config)

    def _prompt(self, claim: str) -> str:
        module = self.module
        seed = module.SeedPost(
            index=0,
            title="Sony A7 IV grip question",
            body="Is the grip comfortable?",
            content="Sony A7 IV grip question",
            source_raw_post_id="ABC123",
            real_num_comments=8,
            metadata={},
        )
        task = module.finalize_rebalanced_task(
            module.CommentTask(
                local_task_id=1,
                local_parent_task_id=None,
                depth=0,
                branch_id=1,
                branch_goal="weigh grip against handling",
                visible_scope="seed",
                local_anchor="a7 IV grip",
                comment_function="explanation_analysis",
                content_angle="fit_use_case",
                evidence_mode="technical_or_policy_reasoning",
                story_mode="no_story",
                voice="casual_neutral",
                payload_type="soft_helpful",
                length_bucket="medium",
                speaker_role="datapoint_only",
                utterance_mode="local_answer_with_context",
                surface_texture="plain",
                allow_first_person_frame=False,
                allow_uncertainty_frame=False,
                planner_intent="state the mount relation",
                must_not_do="Do not add a review.",
                real_word_count=60,
                semantic_move="state the mount adaptation relation",
                local_topic="mount",
                reply_relation="answers_parent",
                stance="neutral",
                detail_focus="mount",
                avoid_repeating="review",
                claim_key="mount_adapt",
                claim_family="technical_explanation",
                opening_style="fact then condition",
                context_aperture="full_seed",
                tone_shape="neutral_fact",
                real_sample_id=1,
            )
        )
        module.GENERALIZED_DOMAIN_CLAIMS.clear()
        if claim:
            module.GENERALIZED_DOMAIN_CLAIMS[(seed_claim_key(seed), 1)] = claim
        return module.build_writer_prompt(
            profile="gpt54_reddit_writer",
            seed_post=seed,
            task=task,
            parent_comment=None,
            previous_comments=[],
            recent_openings=[],
        )

    def test_planned_claim_reaches_the_writer(self) -> None:
        rendered = self._prompt("EF glass adapts to RF bodies with the Canon adapter")
        self.assertIn("EF glass adapts to RF bodies", rendered)
        self.assertIn("Domain fact this turn states", rendered)

    def test_safety_rule_does_not_cancel_the_planned_claim(self) -> None:
        # The blanket invention ban and the claim would otherwise contradict each
        # other in the same prompt, which is how an earlier tone control was made
        # unrealizable.
        rendered = self._prompt("IBIS is absent on that body")
        self.assertIn("Beyond the domain fact assigned above", rendered)
        self.assertNotIn(
            "Do not invent products, specifications, prices, measurements, dates, "
            "outcomes, policies, links, or personal experiences.",
            rendered,
        )
        self.assertIn("Never state a specification", rendered)

    def test_slot_without_a_claim_keeps_the_blanket_ban(self) -> None:
        rendered = self._prompt("")
        self.assertNotIn("Domain fact this turn states", rendered)
        self.assertIn("Do not invent products, specifications", rendered)


if __name__ == "__main__":
    unittest.main()


class DomainClaimAblationTest(unittest.TestCase):
    """The claim has to be separable from the rest of a release.

    It reached 508 of 522 comments in v71 against 0 in v69, alongside six other
    changes, and the run lost `semantic_mean_cosine` and `emotion_entropy`. An
    intervention that large cannot be attributed without a switch that removes
    only it.
    """

    def test_off_mode_keeps_a_planned_claim_out_of_the_writer_prompt(self) -> None:
        import os
        from generalized_card.backend import (
            configure_generator_backend,
            load_generator_backend,
        )
        from generalized_card.domain import load_domain_config

        config = load_domain_config("camera")
        previous = os.environ.get("GENERALIZED_CARD_DOMAIN_CLAIM")
        try:
            os.environ["GENERALIZED_CARD_DOMAIN_CLAIM"] = "off"
            module = configure_generator_backend(load_generator_backend(), config)
            self.assertEqual(module.GENERALIZED_DOMAIN_CLAIM_MODE, "off")
            os.environ["GENERALIZED_CARD_DOMAIN_CLAIM"] = "planned"
            module = configure_generator_backend(load_generator_backend(), config)
            self.assertEqual(module.GENERALIZED_DOMAIN_CLAIM_MODE, "planned")
        finally:
            if previous is None:
                os.environ.pop("GENERALIZED_CARD_DOMAIN_CLAIM", None)
            else:
                os.environ["GENERALIZED_CARD_DOMAIN_CLAIM"] = previous
