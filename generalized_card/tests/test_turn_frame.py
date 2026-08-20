"""The adjudication frame and the entry-grammar exclusion, on the active path.

A definition is not evidence that code executes, so these assertions render the
Writer prompt through the configured backend rather than calling the gate
directly.
"""

from __future__ import annotations

import os
import unittest
from types import SimpleNamespace
from unittest import mock

from generalized_card.semantic_realization import (
    set_turn_frame,
    turn_settles_a_question,
)
from test_generalized_card import FocusedWriterPromptTest


BOUNDARY_LINE = "The question your turn settles"


class TurnFrameWriterPromptTest(unittest.TestCase):
    def setUp(self) -> None:
        self.helper = FocusedWriterPromptTest()
        self.helper.setUp()

    def tearDown(self) -> None:
        set_turn_frame("adjudicative_only")

    def _prompt(self, mode: str = "focused", **overrides: object) -> str:
        overrides.setdefault(
            "decision_boundary", "whether the grip stays comfortable over a long shoot"
        )
        return self.helper._prompt(mode, "impolite", **overrides)

    def test_an_adjudicating_turn_still_receives_its_boundary(self) -> None:
        prompt = self._prompt(
            comment_function="correction_caveat", payload_type="correction"
        )
        self.assertIn(BOUNDARY_LINE, prompt)

    def test_a_datapoint_turn_is_not_told_which_question_it_settles(self) -> None:
        # The frame was in 29.1% of v96 personal-datapoint comments, its worst
        # function, against effectively zero in matched real text.
        prompt = self._prompt(
            comment_function="personal_datapoint", payload_type="fragment_datapoint"
        )
        self.assertNotIn(BOUNDARY_LINE, prompt)

    def test_a_story_slot_is_never_told_which_question_it_settles(self) -> None:
        # Checked on the gate rather than the rendered prompt: the shared task
        # fixture's dependent controls normalize an unsupported story mode away
        # before the Writer prompt is built, so a prompt assertion here would
        # pass for the wrong reason.
        story = SimpleNamespace(
            story_mode="specific_personal_story",
            payload_type="soft_helpful",
            comment_function="verdict_evaluation",
        )
        self.assertFalse(turn_settles_a_question(story))
        no_story = SimpleNamespace(
            story_mode="no_story",
            payload_type="soft_helpful",
            comment_function="verdict_evaluation",
        )
        self.assertTrue(turn_settles_a_question(no_story))

    def test_universal_arm_reproduces_the_v96_prompt(self) -> None:
        # The arm is read from the environment every time the backend is
        # configured, so selecting it in-process is not enough.
        with mock.patch.dict(
            os.environ, {"GENERALIZED_CARD_TURN_FRAME": "universal"}
        ):
            prompt = self._prompt(
                comment_function="personal_datapoint",
                payload_type="fragment_datapoint",
            )
        self.assertIn(BOUNDARY_LINE, prompt)

    def test_the_gate_applies_to_the_full_prompt_arm_too(self) -> None:
        # Applying a fix to one of two Writer paths makes a run unattributable.
        gated = self._prompt(
            "full",
            comment_function="personal_datapoint",
            payload_type="fragment_datapoint",
        )
        self.assertNotIn("decision boundary:", gated)
        adjudicating = self._prompt(
            "full", comment_function="correction_caveat", payload_type="correction"
        )
        self.assertIn("decision boundary:", adjudicating)

    def test_the_planned_move_survives_the_gate(self) -> None:
        # Only the boundary is withheld. Dropping the slot's own proposition
        # would remove the contract instead of the frame.
        prompt = self._prompt(
            comment_function="personal_datapoint", payload_type="fragment_datapoint"
        )
        self.assertIn("commit to a verdict on grip comfort", prompt)


class OpenerExclusionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.helper = FocusedWriterPromptTest()
        self.helper.setUp()

    def test_a_non_polarity_opener_forbids_the_measured_default(self) -> None:
        # v96 opened 20.7% of comments with a bare polarity token against 6.8%
        # of matched real comments and 5.3% scheduled.
        prompt = self.helper._prompt(
            "focused", "impolite", opener_type="content_phrase"
        )
        self.assertIn("Opening grammar for this turn: content_phrase", prompt)
        self.assertIn("Do not open with a bare agreement", prompt)

    def test_a_polarity_opener_is_not_forbidden_from_itself(self) -> None:
        prompt = self.helper._prompt(
            "focused", "impolite", opener_type="polarity_token"
        )
        self.assertIn("Opening grammar for this turn: polarity_token", prompt)
        self.assertNotIn("Do not open with a bare agreement", prompt)


if __name__ == "__main__":
    unittest.main()
