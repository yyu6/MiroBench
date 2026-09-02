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


class NamedLensHandoffTest(unittest.TestCase):
    """Root slots and reply slots are planned by two different prompts.

    The root prompt derives the lens set and carries it forward through a ledger
    summary; the reply prompt was given a `perspective_id` field under `open`
    and nothing to fill it from. Measured on a 97-comment thread: root batches
    converged to 0-2 new lenses per batch by batch 8, then the first reply batch
    added 8 fresh ones and the next two added 8 each, ending at 54 lenses for
    one thread against a real corpus at 5-12.
    """

    def tearDown(self) -> None:
        pv.set_plan_vocabulary("closed")

    def _plans(self):
        return [
            {"perspective_id": "practical burden lens"},
            {"perspective_id": "practical burden lens"},
            {"perspective_id": "reputation management"},
            {"perspective_id": "seed_local"},
            {"perspective_id": ""},
        ]

    def test_closed_mode_adds_nothing(self) -> None:
        self.assertEqual("", pv.named_lens_block(self._plans()))

    def test_open_mode_lists_named_lenses_with_counts(self) -> None:
        pv.set_plan_vocabulary("open")
        block = pv.named_lens_block(self._plans())
        self.assertIn("practical burden lens (used 2x)", block)
        self.assertIn("reputation management (used 1x)", block)
        self.assertNotIn("seed_local", block)
        # The list must not read as a menu. Real threads of this size hold 22
        # to 40 distinct positions, so telling a reply planner to stay inside
        # an existing set suppresses the variety the arm exists to create --
        # which is what the first version of this block said.
        self.assertIn("NOT a menu", block)
        self.assertIn("must name a new one", block)
        self.assertIn("same position does not get two names", block)

    def test_open_mode_is_empty_before_anything_is_named(self) -> None:
        pv.set_plan_vocabulary("open")
        self.assertEqual("", pv.named_lens_block([]))
        self.assertEqual("", pv.named_lens_block([{"perspective_id": "seed_local"}]))

    def test_same_position_under_a_different_suffix_counts_once(self) -> None:
        """The reply planner wrote "practical burden" for the root planner's
        "practical burden lens"; those are one position, not two."""
        self.assertEqual(
            pv.canonical_lens("practical burden lens"),
            pv.canonical_lens("practical burden"),
        )
        self.assertEqual(
            pv.canonical_lens("background privilege angle"),
            pv.canonical_lens("privilege background"),
        )
        self.assertNotEqual(
            pv.canonical_lens("practical burden lens"),
            pv.canonical_lens("reputation management"),
        )

    def test_a_bare_head_noun_still_canonicalises(self) -> None:
        """Stripping must not empty a lens that is only a head noun."""
        self.assertEqual("lens", pv.canonical_lens("lens"))
        self.assertEqual("angle lens", pv.canonical_lens("the lens and the angle"))


class RealPositionCountTest(unittest.TestCase):
    """The lens-count sentence must come from the thread, not from me.

    The prompt told the Planner a thread holds "typically five to twelve"
    lenses. That figure was invented. Agglomerative clustering at the project's
    existing 0.35 unrelatedness threshold puts the ten celebrity seeds at 34
    positions for 97 comments, 40 for 61, and 22 for 29 -- low by a factor of
    three to seven, and suppressing the variety the arm exists to create.
    """

    @staticmethod
    def _unit(*pairs):
        import math

        out = []
        for x, y in pairs:
            n = math.hypot(x, y) or 1.0
            out.append([x / n, y / n])
        return out

    def test_counts_positions_at_the_project_threshold(self) -> None:
        vecs = self._unit((1, 0), (1, 0.02), (0, 1), (0, 1), (-1, 0))
        self.assertEqual(
            3, pv.real_position_count(list("abcde"), lambda _t: vecs)
        )

    def test_near_duplicates_are_one_position(self) -> None:
        vecs = self._unit((1, 0), (1, 0.01), (1, 0.02), (1, 0.03))
        self.assertEqual(1, pv.real_position_count(list("abcd"), lambda _t: vecs))

    def test_too_few_comments_returns_zero(self) -> None:
        self.assertEqual(0, pv.real_position_count(["a", "b", "c"], lambda _t: []))
        self.assertEqual(0, pv.real_position_count([], lambda _t: []))

    def test_a_missing_model_degrades_to_silence_not_a_guess(self) -> None:
        def boom(_texts):
            raise RuntimeError("no embedding model")

        self.assertEqual(
            0, pv.real_position_count(["q1", "q2", "q3", "q4", "q5"], boom)
        )
        # and the sentence is omitted rather than inventing a range
        pv.set_plan_vocabulary("open")
        try:
            self.assertNotIn("distinct semantic positions", pv.abstraction_block("", 0))
            self.assertIn("distinct semantic positions", pv.abstraction_block("", 34))
            self.assertNotIn("five to twelve", pv.abstraction_block("", 34))
        finally:
            pv.set_plan_vocabulary("closed")


class LensListCompletenessTest(unittest.TestCase):
    """A planner told not to duplicate a list it cannot see will duplicate it.

    The block capped its list at 24 named lenses. On a 97-comment thread the
    root batches named 36, so the first reply batch was shown 24 of them and
    added 1 new lens -- the fix working -- and the next two batches, shown 24 of
    37 and 24 of 43, added 7 each. The cap was the leak.
    """

    def setUp(self) -> None:
        pv.set_plan_vocabulary("open")

    def tearDown(self) -> None:
        pv.set_plan_vocabulary("closed")

    def test_a_realistic_thread_is_never_truncated(self) -> None:
        plans = [{"perspective_id": f"lens number {i}"} for i in range(97)]
        block = pv.named_lens_block(plans)
        self.assertNotIn("not shown", block)
        for i in (0, 50, 96):
            self.assertIn(f"lens number {i} ", block)

    def test_truncation_is_disclosed_when_it_happens(self) -> None:
        plans = [{"perspective_id": f"lens number {i}"} for i in range(200)]
        block = pv.named_lens_block(plans)
        self.assertIn("more not shown", block)
        self.assertIn("even if it is not listed here", block)

    def test_most_used_lenses_survive_a_truncation(self) -> None:
        plans = [{"perspective_id": "dominant one"}] * 5
        plans += [{"perspective_id": f"rare {i}"} for i in range(200)]
        block = pv.named_lens_block(plans, limit=3)
        self.assertIn("dominant one (used 5x)", block)
