"""The two Planner schemas must not drift apart again.

`reply_planning.py` had silently lost `perspective_id` and `domain_intent` and
had downgraded `content_angle` to free text that `normalize_plan_rows` then
folded to `unclear_mixed`. Nothing failed: the fields simply held their code
fallbacks for every reply, which is 49% of a thread, and the writer prompt read
`decision intent: one seed-grounded local move` on 178 of 364 slots (G202).

A rendering bug that pins a control to a constant produces no error, so the only
thing that catches it is a test that reads both rendered schemas and compares
them against the field list the task builder consumes.
"""
from __future__ import annotations

import re
import unittest

from generalized_card import plan_vocabulary as pv


class PlanVocabularyParityTest(unittest.TestCase):
    def tearDown(self) -> None:
        pv.set_plan_vocabulary("closed")

    def _reply_schema(self) -> str:
        import inspect

        from generalized_card import reply_planning

        return inspect.getsource(reply_planning)

    def _root_schema(self) -> str:
        import inspect

        from generalized_card import prompts

        return inspect.getsource(prompts)

    def test_open_mode_gives_both_schemas_every_shared_field(self) -> None:
        pv.set_plan_vocabulary("open")
        root = self._root_schema()
        # The reply schema's two repaired fields are rendered, not literal, so
        # read them from the renderer the schema interpolates.
        reply = self._reply_schema() + pv.reply_shared_field_lines()
        for name, text in (("reply", reply), ("root", root)):
            missing = pv.missing_fields(text)
            self.assertEqual(
                [], missing,
                f"{name} planner schema does not ask for {missing} under open; "
                f"those slots hold their code fallbacks and nothing reports it",
            )

    def test_closed_mode_still_reproduces_the_v150_reply_schema(self) -> None:
        # The drift repair is deliberately gated: under the closed taxonomy a
        # reply would receive a P## from a shopping menu. `closed` must stay an
        # exact reproduction so recorded comparisons against it hold.
        self.assertEqual("", pv.reply_shared_field_lines())
        missing = pv.missing_fields(self._reply_schema())
        self.assertEqual(["perspective_id", "domain_intent"], missing)

    def test_shared_and_root_only_fields_are_disjoint(self) -> None:
        overlap = set(pv.SHARED_PLAN_FIELDS) & set(pv.ROOT_ONLY_PLAN_FIELDS)
        self.assertEqual(set(), overlap)

    def test_closed_mode_hints_are_unchanged(self) -> None:
        self.assertEqual(
            "one P## from the frozen domain profile, or seed_local",
            pv.perspective_schema_hint(),
        )
        self.assertEqual(
            "one P## from the frozen domain profile",
            pv.perspective_schema_hint(allow_seed_local=False),
        )
        self.assertEqual("a | b", pv.content_angle_schema_hint("a | b"))
        self.assertEqual("", pv.abstraction_block("- P01: x"))

    def test_closed_mode_reproduces_the_v150_prompt_text(self) -> None:
        """Every string this arm made mode-aware, as v150 rendered it.

        Five sites in `comment_planner_prompt` moved from literals in the
        f-string to helper calls. Under `closed` each must return the same
        characters, or `closed` is not the reproduction the arm claims and no
        comparison recorded against v150 still holds. These are the literals as
        `git show <v150>:prompts.py` carried them.
        """
        from generalized_card import prompts

        self.assertEqual(
            "Frozen domain-neutral decision lenses:", prompts._lens_framing()
        )
        self.assertEqual(
            "Each P## states how a comment reasons about the local topic. It is "
            "not the topic,\nentity, product, feature, event, or claim itself. "
            "Derive the actual local move\nfrom the visible seed/parent and the "
            "non-test reference-comment pattern below.",
            prompts._lens_note(),
        )
        self.assertEqual(
            "Use a frozen ``perspective_id`` only when it fits the visible seed "
            "or parent; otherwise use ``seed_local``.",
            prompts._perspective_field_rule(),
        )
        self.assertEqual(
            "one P## from the frozen domain profile",
            pv.perspective_schema_hint(allow_seed_local=False),
        )
        self.assertEqual(
            "one P## from the frozen domain profile, or seed_local",
            pv.perspective_schema_hint(),
        )

    def test_open_mode_withdraws_the_escape_hatch(self) -> None:
        pv.set_plan_vocabulary("open")
        hint = pv.perspective_schema_hint()
        self.assertNotIn("P##", hint.replace("never write seed_local or a P##", ""))
        self.assertIn("named by you", hint)
        self.assertNotEqual("a | b", pv.content_angle_schema_hint("a | b"))
        block = pv.abstraction_block("- P01: needs and constraints")
        self.assertIn("DERIVE THE LENSES", block)
        self.assertIn("no 'none of these' option", block)
        self.assertIn("P01", block)

    def test_canonical_lens_collapses_reorderings(self) -> None:
        self.assertEqual(
            pv.canonical_lens("media framing and trust"),
            pv.canonical_lens("The trust in media framing"),
        )
        self.assertNotEqual(
            pv.canonical_lens("media framing and trust"),
            pv.canonical_lens("who the audience believes"),
        )

    def test_bare_identifiers_are_not_lenses(self) -> None:
        for bad in ("B3", "S12", "P01", "R7", "  b 2 "):
            self.assertEqual(
                "FB", pv.normalize_open_control(bad, fallback="FB", limit=48), bad
            )
        self.assertEqual(
            "whether the framing is doing the work",
            pv.normalize_open_control(
                "  whether the framing   is doing the work ", fallback="FB", limit=48
            ),
        )

    def test_open_control_is_length_capped(self) -> None:
        out = pv.normalize_open_control("x" * 200, fallback="FB", limit=48)
        self.assertEqual(48, len(out))


class CanonicalizerGateTest(unittest.TestCase):
    """The gate that would have made this arm inert.

    `_canonicalize_plan_controls` maps any lens outside the frozen P## set to
    `seed_local`. Left unchanged it deletes every lens the Planner names, and
    the run records `plan_vocabulary: open` while generating nothing new -- the
    failure v143obs shipped.
    """

    def tearDown(self) -> None:
        pv.set_plan_vocabulary("closed")

    def _run(self, raw: str) -> str:
        from generalized_card.backend import _canonicalize_plan_controls

        plans = {1: {"perspective_id": raw}}
        _canonicalize_plan_controls(plans, perspective_ids={"P01", "P02"})
        return str(plans[1]["perspective_id"])

    def test_closed_folds_an_unlisted_lens(self) -> None:
        self.assertEqual("seed_local", self._run("whether the framing carries it"))
        self.assertEqual("P01", self._run("p01"))

    def test_open_keeps_a_named_lens(self) -> None:
        pv.set_plan_vocabulary("open")
        self.assertEqual(
            "whether the framing carries it", self._run("whether the framing carries it")
        )

    def test_open_still_repairs_a_bare_identifier(self) -> None:
        pv.set_plan_vocabulary("open")
        self.assertEqual("seed_local", self._run("B1"))
        self.assertEqual("seed_local", self._run(""))


if __name__ == "__main__":
    unittest.main()
