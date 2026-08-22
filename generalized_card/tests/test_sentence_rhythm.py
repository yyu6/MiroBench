"""Per-slot typing habits, on the active Writer path.

The mechanism these assertions protect is that two slots of the same size get
*different* habits. Through v97 every surface cue was a function of the slot's
size, so same-size generated comments converged on one function-word skeleton --
cosine 0.502 against a real 0.368 -- and that is where `self_bertscore_mean_f1`
lives.
"""

from __future__ import annotations

import os
import unittest
from collections import Counter
from unittest import mock

from generalized_card.sentence_rhythm import (
    band_row,
    build_rhythm_profile,
    habit_names,
    rhythm_guidance,
    sentence_words,
    set_active_rhythm_profile,
    set_digit_cue_guard,
    set_sentence_rhythm,
    slot_habits,
    slot_uses_habit,
)
from test_generalized_card import FocusedWriterPromptTest


def _threads(rows: list[tuple[str, int]]) -> list[dict[str, object]]:
    return [{"comments": [{"body": body} for body, count in rows for _ in range(count)]}]


MEASURED = {
    "available": True,
    "bands": {
        "medium": {
            "median_words_per_sentence": 14.0,
            "median_sentences": 3,
            "shares": {
                "short_sentence": 0.401,
                "exclamation": 0.101,
                "parenthetical": 0.145,
                "ellipsis": 0.058,
                "digit": 0.575,
                "semicolon": 0.016,
                "dash_clause": 0.043,
            },
        },
        "micro": {
            "median_words_per_sentence": 4.0,
            "median_sentences": 1,
            "shares": {"short_sentence": 1.0, "exclamation": 0.160, "digit": 0.239},
        },
    },
}


class ProfileTest(unittest.TestCase):
    def test_profile_needs_enough_samples(self) -> None:
        self.assertFalse(build_rhythm_profile(_threads([("a b c", 10)]))["available"])

    def test_a_band_with_too_few_samples_is_omitted_rather_than_guessed(self) -> None:
        profile = build_rhythm_profile(
            _threads([("word " * 30, 300), ("word " * 400, 3)])
        )
        self.assertIn("medium", profile["bands"])
        self.assertNotIn("essay", profile["bands"])
        self.assertEqual(band_row(profile, 400), {})

    def test_profile_measures_pacing_and_habits(self) -> None:
        body = "One sentence here. Yes! Another sentence follows on."
        profile = build_rhythm_profile(_threads([(body, 300)]))
        row = profile["bands"]["micro"]
        self.assertEqual(row["median_sentences"], 3)
        self.assertAlmostEqual(row["shares"]["exclamation"], 1.0)
        self.assertAlmostEqual(row["shares"]["semicolon"], 0.0)

    def test_short_sentence_is_measured_only_where_it_is_a_choice(self) -> None:
        # A one-sentence comment trivially contains its own shortest sentence, so
        # counting it would report a share of 1.0 and ask a four-word comment for
        # a shorter sentence inside itself.
        single = "word " * 30
        uneven = "Yes. " + "word " * 28
        profile = build_rhythm_profile(_threads([(single, 200), (uneven, 200)]))
        row = profile["bands"]["medium"]
        self.assertEqual(row["multi_sentence_count"], 200)
        self.assertAlmostEqual(row["shares"]["short_sentence"], 1.0)

    def test_sentence_split_counts_newline_separated_lines(self) -> None:
        self.assertEqual(sentence_words("one two\nthree"), [2, 1])
        self.assertEqual(sentence_words("Yes! Then more words here."), [1, 4])


class DrawTest(unittest.TestCase):
    def tearDown(self) -> None:
        set_sentence_rhythm("measured")
        set_active_rhythm_profile({})

    def test_the_draw_reproduces_every_measured_share(self) -> None:
        counts: Counter[str] = Counter()
        trials = 4000
        for index in range(trials):
            for name, drawn in slot_habits(
                MEASURED, slot_key=f"seed:{index}", word_count=35
            ):
                if drawn:
                    counts[name] += 1
        for name, share in MEASURED["bands"]["medium"]["shares"].items():
            self.assertAlmostEqual(counts[name] / trials, share, delta=0.03, msg=name)

    def test_same_size_slots_get_different_habits(self) -> None:
        # This is the mechanism, not a side effect of it.
        sets = {
            slot_habits(MEASURED, slot_key=f"seed:{index}", word_count=35)
            for index in range(60)
        }
        self.assertGreater(len(sets), 5)

    def test_the_draw_is_reproducible_for_one_slot(self) -> None:
        first = slot_habits(MEASURED, slot_key="seed:7", word_count=35)
        self.assertEqual(first, slot_habits(MEASURED, slot_key="seed:7", word_count=35))

    def test_a_one_sentence_band_is_never_asked_for_a_shorter_sentence(self) -> None:
        self.assertFalse(
            any(
                slot_uses_habit(
                    MEASURED, slot_key=f"seed:{i}", habit="short_sentence", word_count=4
                )
                for i in range(200)
            )
        )

    def test_an_unmeasured_band_draws_nothing(self) -> None:
        self.assertEqual(slot_habits(MEASURED, slot_key="seed:1", word_count=900), ())
        self.assertEqual(rhythm_guidance(MEASURED, slot_key="seed:1", word_count=900), "")


class GuidanceTest(unittest.TestCase):
    def tearDown(self) -> None:
        set_sentence_rhythm("measured")
        set_digit_cue_guard("off")

    def _cues(self, word_count: int, trials: int = 80) -> list[str]:
        return [
            rhythm_guidance(MEASURED, slot_key=f"seed:{index}", word_count=word_count)
            for index in range(trials)
        ]

    def test_over_used_constructions_are_suppressed_rather_than_asked_for(self) -> None:
        # The Writer emits a semicolon at 4.5x and a dash-joined clause at 4x the
        # real rate, so for these two the measured share is the share that is
        # *allowed*, and the cue on the rest of the slots is a prohibition.
        cues = self._cues(35)
        self.assertGreater(sum("no semicolons" in cue for cue in cues), 60)
        self.assertGreater(sum("dash" in cue for cue in cues), 60)

    def test_the_measured_habits_all_reach_some_slot(self) -> None:
        joined = " ".join(self._cues(35, trials=400))
        for fragment in (
            "under five words",
            "exclamation mark",
            "in parentheses",
            "trail off",
            "written as a figure",
        ):
            self.assertIn(fragment, joined)

    def test_a_one_sentence_band_gets_no_pacing_claim(self) -> None:
        for cue in self._cues(4):
            self.assertNotIn("they are uneven", cue)

    def test_the_pacing_number_is_the_measured_one(self) -> None:
        cue = next(c for c in self._cues(35) if "Typing rhythm" in c)
        self.assertIn("about 14 words", cue)

    def test_a_slot_whose_skeleton_names_its_sentence_count_is_not_given_another(
        self,
    ) -> None:
        # A 115-word slot was told "about 12 sentences" by its skeleton and
        # "about 16 words" per sentence by this rule, which is 192 words against
        # a 133-word ask. The slot-specific count wins.
        cue = rhythm_guidance(
            MEASURED, slot_key="seed:1", word_count=35, slot_names_sentence_count=True
        )
        self.assertNotIn("words here", cue)
        self.assertIn("keep those sentences uneven", cue)

    def test_the_cue_never_asks_for_invented_facts(self) -> None:
        joined = " ".join(self._cues(35, trials=400))
        self.assertIn("rather than inventing one", joined)
        self.assertIn("only a number you are allowed to name above", joined.lower())

    def test_cue_carries_no_domain_vocabulary(self) -> None:
        joined = " ".join(self._cues(35, trials=200)).lower()
        for term in ("camera", "lens", "photo", "card", "phone"):
            self.assertNotIn(term, joined)

    def test_off_arm_renders_nothing(self) -> None:
        set_sentence_rhythm("off")
        self.assertEqual(rhythm_guidance(MEASURED, slot_key="seed:1", word_count=35), "")

    def test_digit_cue_guard_default_is_the_legacy_wording(self) -> None:
        joined = " ".join(self._cues(35, trials=400))
        self.assertIn("written as a figure rather than described in", joined)
        self.assertNotIn("ordinary word", joined)

    def test_digit_cue_guard_on_excludes_ordinary_quantifiers(self) -> None:
        set_digit_cue_guard("on")
        joined = " ".join(self._cues(35, trials=400))
        self.assertIn("ordinary word", joined)
        self.assertIn("citing a real quantity", joined)
        # The underlying instruction is still there -- this adds an exclusion,
        # it does not replace the ask for a genuine figure.
        self.assertIn("rather than inventing one", joined)

    def test_digit_cue_guard_carries_no_domain_vocabulary(self) -> None:
        set_digit_cue_guard("on")
        joined = " ".join(self._cues(35, trials=200)).lower()
        for term in ("camera", "lens", "photo", "card", "phone"):
            self.assertNotIn(term, joined)

    def test_every_habit_the_module_defines_is_measurable(self) -> None:
        # A habit with no pattern and no computed share would render never.
        measured = set(MEASURED["bands"]["medium"]["shares"])
        self.assertEqual(measured, set(habit_names()))


class WriterPromptTest(unittest.TestCase):
    """Rendered through the configured backend, not by calling the renderer."""

    def setUp(self) -> None:
        self.helper = FocusedWriterPromptTest()
        self.helper.setUp()

    def tearDown(self) -> None:
        set_active_rhythm_profile({})
        set_sentence_rhythm("measured")
        set_digit_cue_guard("off")

    def _prompt(self, mode: str = "focused", **overrides: object) -> str:
        overrides.setdefault("real_word_count", 35)
        return self.helper._prompt(
            mode, "neutral", rhythm_profile=MEASURED, **overrides
        )

    def test_the_rule_reaches_the_focused_writer_prompt(self) -> None:
        self.assertIn("Typing rhythm:", self._prompt())

    def test_the_rule_reaches_the_full_writer_prompt_too(self) -> None:
        # Applying a surface fix to one of two Writer paths makes a run
        # unattributable; v74 left 106 of 522 slots on the old path.
        self.assertIn("Typing rhythm:", self._prompt("full"))

    def test_the_off_arm_removes_the_rule_from_the_prompt(self) -> None:
        with mock.patch.dict(os.environ, {"GENERALIZED_CARD_SENTENCE_RHYTHM": "off"}):
            self.assertNotIn("Typing rhythm:", self._prompt())

    def test_digit_cue_guard_env_var_reaches_the_writer_prompt(self) -> None:
        with mock.patch.dict(os.environ, {"GENERALIZED_CARD_DIGIT_CUE_GUARD": "on"}):
            found = any(
                "ordinary word" in self._prompt(local_task_id=index)
                for index in range(1, 25)
            )
        self.assertTrue(found)

    def test_two_slots_of_one_size_receive_different_rules(self) -> None:
        rendered = {
            self._prompt(local_task_id=index).split("Typing rhythm:")[1].split("\n")[0]
            for index in range(1, 25)
        }
        self.assertGreater(len(rendered), 3)


if __name__ == "__main__":
    unittest.main()
