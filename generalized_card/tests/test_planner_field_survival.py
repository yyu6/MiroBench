from __future__ import annotations

import inspect
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

    def test_planner_prompt_requests_every_field_it_relies_on(self) -> None:
        from generalized_card import prompts

        source = inspect.getsource(prompts)
        for field in self.GENERALIZED_PLANNER_FIELDS:
            self.assertIn(f'"{field}"', source, field)

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


if __name__ == "__main__":
    unittest.main()
