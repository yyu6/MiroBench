"""The reused mid-comment route ledger, on the focused Writer path.

`used_sentence_routes` has fed the `full` Writer arm since v66. `focused` has
been the active arm since v82 and rendered only comment openings, short
utterances, and semantic coverage, so a slot could be shown 24 openings and 21
short lines and nothing about a phrase seven of its predecessors had already
used. Measured over the v97 N=10 output, the adjudication frame persisted at
0.144 on slots that never received the boundary line at all.
"""

from __future__ import annotations

import os
import unittest
from unittest import mock

from generalized_card.semantic_realization import (
    reused_sentence_routes,
    set_route_ledger,
    used_sentence_routes,
)
from test_generalized_card import FocusedWriterPromptTest


HEADING = "Sentence routes already reused in this thread"
FRAME = "that's the part that"


def _comments(texts: list[str]) -> list[dict[str, str]]:
    return [{"content": text} for text in texts]


REPEATED = _comments(
    [
        "The grip holds up fine. That's the part that actually matters here.",
        "Battery life is fine, but that's the part that actually matters most.",
        "That's the part that actually matters, honestly, once you shoot all day.",
        "A different observation entirely about the strap mount and nothing else.",
    ]
)


class ReusedRouteTest(unittest.TestCase):
    def tearDown(self) -> None:
        set_route_ledger("on")

    def test_only_routes_the_thread_actually_reused_are_returned(self) -> None:
        # `used_sentence_routes` pads its list with single-use routes so the full
        # arm's ledger has a fixed size. A ledger headed "already reused" that
        # carried them would list ordinary phrasing as a habit.
        padded = used_sentence_routes(REPEATED, limit=16)
        self.assertTrue(any("(used " not in value for value in padded))
        reused = reused_sentence_routes(REPEATED, limit=16)
        self.assertTrue(reused)
        self.assertTrue(all("(used " in value for value in reused))

    def test_the_entry_carries_how_entrenched_the_route_is(self) -> None:
        reused = reused_sentence_routes(REPEATED, limit=16)
        # Comments 1 and 3 enter on this route; comment 2 enters on "but that's
        # the part", a different clause path, which is the distinction the
        # segmenter is for.
        self.assertIn(f"{FRAME} (used 2x)", reused)

    def test_an_early_slot_has_nothing_to_report(self) -> None:
        self.assertEqual(reused_sentence_routes(_comments(["one single comment here"])), [])

    def test_off_arm_returns_nothing(self) -> None:
        set_route_ledger("off")
        self.assertEqual(reused_sentence_routes(REPEATED, limit=16), [])


class FocusedPromptTest(unittest.TestCase):
    def setUp(self) -> None:
        self.helper = FocusedWriterPromptTest()
        self.helper.setUp()

    def tearDown(self) -> None:
        set_route_ledger("on")

    def _prompt(self, previous: list[dict[str, str]]) -> str:
        return self.helper._prompt("focused", "neutral", previous_comments=previous)

    def test_the_reused_route_reaches_the_focused_prompt(self) -> None:
        prompt = self._prompt(REPEATED)
        self.assertIn(HEADING, prompt)
        self.assertIn(FRAME, prompt)

    def test_the_section_is_absent_when_nothing_has_been_reused(self) -> None:
        # An empty section on every early slot is prompt mass with no content.
        self.assertNotIn(HEADING, self._prompt(_comments(["a single opening comment"])))

    def test_off_arm_reproduces_the_v97_focused_prompt(self) -> None:
        with mock.patch.dict(os.environ, {"GENERALIZED_CARD_ROUTE_LEDGER": "off"}):
            prompt = self._prompt(REPEATED)
        self.assertNotIn(HEADING, prompt)

    def test_the_other_two_ledgers_survive(self) -> None:
        prompt = self._prompt(REPEATED)
        self.assertIn("Short utterances already used anywhere in this thread", prompt)
        self.assertIn("Semantic contributions already covered in this thread", prompt)


if __name__ == "__main__":
    unittest.main()
