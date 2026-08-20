"""What a `no_story` slot is barred from, on the active Writer path.

v96 and v97 barred any past action or event on 453 of 532 slots. Measured on
those slots against their matched real comments, past-tense verbs appeared in
0.181 against a real 0.543, future in 0.031 against 0.226, and present perfect in
0.031 against 0.167. The thread lexicon fell to 2,670 distinct types against a
real 3,645, which is the whole `self_bertscore_mean_f1` gap.
"""

from __future__ import annotations

import os
import unittest
from unittest import mock

from generalized_card.story_scope import (
    SEQUENCE_BAN_INSTRUCTION,
    TENSE_BAN_INSTRUCTION,
    no_story_instruction,
    set_no_story_scope,
)
from generalized_card.backend import (
    configure_generator_backend,
    load_generator_backend,
)
from test_generalized_card import FocusedWriterPromptTest


class ScopeTest(unittest.TestCase):
    def tearDown(self) -> None:
        set_no_story_scope("sequence")

    def test_the_sequence_arm_still_bars_a_narrative(self) -> None:
        # StorySeeker scores narrative sequence. Loosening past tense must not
        # loosen what the passing metric depends on.
        text = no_story_instruction().lower()
        for barred in ("second event", "then/after pacing", "before/after change", "story arc"):
            self.assertIn(barred, text)

    def test_the_sequence_arm_permits_ordinary_tense(self) -> None:
        text = no_story_instruction().lower()
        self.assertIn("ordinary past and future tense", text)
        self.assertNotIn("no past action", text)

    def test_the_tense_arm_reproduces_the_v97_instruction(self) -> None:
        set_no_story_scope("tense")
        self.assertEqual(no_story_instruction(), TENSE_BAN_INSTRUCTION)

    def test_the_two_arms_are_different_instructions(self) -> None:
        self.assertNotEqual(SEQUENCE_BAN_INSTRUCTION, TENSE_BAN_INSTRUCTION)

    def test_an_unknown_mode_selects_the_current_default(self) -> None:
        self.assertTrue(set_no_story_scope(""))
        self.assertEqual(no_story_instruction(), SEQUENCE_BAN_INSTRUCTION)


class WriterPromptTest(unittest.TestCase):
    def setUp(self) -> None:
        self.helper = FocusedWriterPromptTest()
        self.helper.setUp()

    def tearDown(self) -> None:
        set_no_story_scope("sequence")

    def _prompt(self, mode: str = "focused") -> str:
        return self.helper._prompt(mode, "neutral", story_mode="no_story")

    def test_the_sequence_scope_reaches_the_focused_prompt(self) -> None:
        prompt = self._prompt()
        self.assertIn("ordinary past and future tense", prompt)
        self.assertNotIn("no past action", prompt)

    def test_it_reaches_the_full_writer_prompt_too(self) -> None:
        self.assertIn("ordinary past and future tense", self._prompt("full"))

    def test_the_tense_arm_reproduces_the_v97_prompt(self) -> None:
        with mock.patch.dict(os.environ, {"GENERALIZED_CARD_NO_STORY_SCOPE": "tense"}):
            prompt = self._prompt()
        self.assertIn("no past action, event, before/after", prompt)

    def test_the_substitution_still_belongs_to_the_coherence_arm(self) -> None:
        # Asserted through the coherence arm rather than by passing a story
        # mode: the shared fixture's dependent controls normalize an unsupported
        # story mode back to `no_story` before the prompt is built, so a story
        # assertion here would pass for the wrong reason. Same reason
        # `test_turn_frame` checks its gate directly.
        module = configure_generator_backend(load_generator_backend(), self.helper.config)
        module.GENERALIZED_SOCIAL_CONTRACT_COHERENCE = "off"
        with mock.patch.dict(
            os.environ, {"GENERALIZED_CARD_SOCIAL_CONTRACT_COHERENCE": "off"}
        ):
            prompt = self._prompt()
        self.assertNotIn(SEQUENCE_BAN_INSTRUCTION, prompt)
        self.assertNotIn(TENSE_BAN_INSTRUCTION, prompt)


if __name__ == "__main__":
    unittest.main()
