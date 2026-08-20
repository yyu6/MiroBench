from __future__ import annotations

import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
EVALUATION_DIR = REPO_ROOT / "scripts" / "evaluation"
if str(EVALUATION_DIR) not in sys.path:
    sys.path.insert(0, str(EVALUATION_DIR))

from score_thread_self_bleu import tokenize as evaluator_tokenize  # noqa: E402

from generalized_card.surface_typography import (  # noqa: E402
    TYPOGRAPHY_CLASSES,
    apply_final_punctuation_habit,
    apply_keyboard_typography,
    build_final_punctuation_profile,
    build_typography_profile,
    set_reddit_typography,
    speaker_uses_typographic,
)


ALL_KEYBOARD = {
    "available": True,
    "shares": {spec["name"]: 0.0 for spec in TYPOGRAPHY_CLASSES},
}
ALL_TYPOGRAPHIC = {
    "available": True,
    "shares": {spec["name"]: 1.0 for spec in TYPOGRAPHY_CLASSES},
}


class TypographyProfileTest(unittest.TestCase):
    def tearDown(self) -> None:
        set_reddit_typography("on")

    def test_profile_needs_enough_samples(self) -> None:
        thin = build_typography_profile([{"comments": [{"body": "it’s fine"}]}])
        self.assertFalse(thin["available"])

    def test_share_is_a_comment_ratio_not_a_character_ratio(self) -> None:
        # One comment full of typographic dashes must not outweigh many
        # comments that each use one keyboard dash: the draw it feeds is per
        # speaker, so the quantity measured is how many comments show the form.
        threads = [
            {
                "comments": [{"body": "a — b — c — d — e — f"}]
                + [{"body": "a - b"} for _ in range(299)]
            }
        ]
        profile = build_typography_profile(threads)
        self.assertTrue(profile["available"])
        self.assertAlmostEqual(profile["shares"]["dash"], 1 / 300, places=4)

    def test_measured_shares_are_per_class(self) -> None:
        threads = [
            {
                "comments": [{"body": "it’s \"quoted\""} for _ in range(100)]
                + [{"body": "it's “quoted”"} for _ in range(100)]
                + [{"body": "plain text"} for _ in range(100)]
            }
        ]
        profile = build_typography_profile(threads)
        self.assertAlmostEqual(profile["shares"]["apostrophe"], 0.5, places=2)
        self.assertAlmostEqual(profile["shares"]["double_quote"], 0.5, places=2)


class KeyboardTypographyTest(unittest.TestCase):
    def tearDown(self) -> None:
        set_reddit_typography("on")

    def test_keyboard_speaker_gets_ascii_punctuation(self) -> None:
        shaped = apply_keyboard_typography(
            "it’s fine — really… “sure”",
            speaker_key="seed:S001",
            profile=ALL_KEYBOARD,
        )
        self.assertEqual(shaped, 'it\'s fine - really... "sure"')

    def test_typographic_speaker_is_left_byte_identical(self) -> None:
        text = "it’s fine — really… “sure”"
        self.assertIs(
            apply_keyboard_typography(
                text, speaker_key="seed:S001", profile=ALL_TYPOGRAPHIC
            ),
            text,
        )

    def test_arm_off_reproduces_the_model_output(self) -> None:
        set_reddit_typography("off")
        text = "it’s fine"
        self.assertIs(
            apply_keyboard_typography(
                text, speaker_key="seed:S001", profile=ALL_KEYBOARD
            ),
            text,
        )

    def test_missing_profile_changes_nothing(self) -> None:
        text = "it’s fine"
        self.assertIs(
            apply_keyboard_typography(text, speaker_key="seed:S001", profile=None),
            text,
        )

    def test_draw_is_stable_per_speaker_and_varies_across_speakers(self) -> None:
        half = {"available": True, "shares": {"apostrophe": 0.5}}
        first = [
            speaker_uses_typographic(half, speaker_key="s:S001", class_name="apostrophe")
            for _ in range(5)
        ]
        self.assertEqual(len(set(first)), 1)
        drawn = {
            speaker_uses_typographic(
                half, speaker_key=f"s:S{index:03d}", class_name="apostrophe"
            )
            for index in range(40)
        }
        self.assertEqual(drawn, {True, False})

    def test_draw_reproduces_the_measured_share(self) -> None:
        profile = {"available": True, "shares": {"apostrophe": 0.27}}
        hits = sum(
            speaker_uses_typographic(
                profile, speaker_key=f"seed:S{index:04d}", class_name="apostrophe"
            )
            for index in range(4000)
        )
        # A hash draw is deterministic but not stratified, so this checks the
        # share is reproduced within ordinary sampling error at this size.
        self.assertAlmostEqual(hits / 4000, 0.27, delta=0.02)

    def test_paragraph_structure_survives_shaping(self) -> None:
        shaped = apply_keyboard_typography(
            "first line — one\n\nsecond line… two",
            speaker_key="seed:S001",
            profile=ALL_KEYBOARD,
        )
        self.assertEqual(shaped, "first line - one\n\nsecond line... two")


class SelfBleuTokenizationTest(unittest.TestCase):
    """The metric-level reason this module exists, checked with the scorer's own
    tokenizer rather than a local approximation."""

    def test_typographic_contraction_costs_two_extra_shared_tokens(self) -> None:
        typographic = evaluator_tokenize("that’s the part that")
        keyboard = evaluator_tokenize("that's the part that")
        self.assertEqual(len(typographic), len(keyboard) + 2)
        self.assertIn("’", typographic)
        self.assertNotIn("’", keyboard)




class FinalPunctuationHabitTest(unittest.TestCase):
    """A declarative ending is a typing habit; a question mark is a choice."""

    PROFILE = {
        "available": True,
        "bare_share_by_band": {
            "micro": 1.0,
            "short": 0.308,
            "medium": 0.198,
            "long": 0.0,
        },
    }

    def tearDown(self) -> None:
        set_reddit_typography("on")

    def test_profile_measures_the_bare_to_period_ratio_per_band(self) -> None:
        threads = [
            {
                "comments": [{"body": "word " * 4 + "end"} for _ in range(120)]
                + [{"body": "word " * 4 + "end."} for _ in range(80)]
                + [{"body": "word " * 40 + "end."} for _ in range(200)]
            }
        ]
        profile = build_final_punctuation_profile(threads)
        self.assertTrue(profile["available"])
        self.assertAlmostEqual(profile["bare_share_by_band"]["micro"], 0.6, places=2)
        self.assertAlmostEqual(profile["bare_share_by_band"]["medium"], 0.0, places=2)

    def test_a_question_mark_is_never_stripped(self) -> None:
        text = "So which one actually matters?"
        self.assertIs(
            apply_final_punctuation_habit(
                text, speaker_key="s:S001", profile=self.PROFILE
            ),
            text,
        )

    def test_an_exclamation_is_never_stripped(self) -> None:
        text = "That helped a lot!"
        self.assertIs(
            apply_final_punctuation_habit(
                text, speaker_key="s:S001", profile=self.PROFILE
            ),
            text,
        )

    def test_an_ellipsis_is_never_stripped(self) -> None:
        text = "Not sure about that one..."
        self.assertIs(
            apply_final_punctuation_habit(
                text, speaker_key="s:S001", profile=self.PROFILE
            ),
            text,
        )

    def test_a_period_is_dropped_where_the_band_says_it_is(self) -> None:
        self.assertEqual(
            apply_final_punctuation_habit(
                "Nope, that is the filter.",
                speaker_key="s:S001",
                profile=self.PROFILE,
            ),
            "Nope, that is the filter",
        )

    def test_a_band_measured_at_zero_keeps_its_period(self) -> None:
        long_text = "word " * 90 + "and that is the whole point."
        self.assertIs(
            apply_final_punctuation_habit(
                long_text, speaker_key="s:S001", profile=self.PROFILE
            ),
            long_text,
        )

    def test_one_speaker_is_consistent_and_speakers_differ(self) -> None:
        profile = {"available": True, "bare_share_by_band": {"short": 0.5}}
        text = "word " * 14 + "so that is the answer."
        first = {
            apply_final_punctuation_habit(
                text, speaker_key="s:S001", profile=profile
            )
            for _ in range(5)
        }
        self.assertEqual(len(first), 1)
        outcomes = {
            apply_final_punctuation_habit(
                text, speaker_key=f"s:S{index:03d}", profile=profile
            ).endswith(".")
            for index in range(40)
        }
        self.assertEqual(outcomes, {True, False})

    def test_arm_off_reproduces_the_model_output(self) -> None:
        set_reddit_typography("off")
        text = "Nope, that is the filter."
        self.assertIs(
            apply_final_punctuation_habit(
                text, speaker_key="s:S001", profile=self.PROFILE
            ),
            text,
        )


if __name__ == "__main__":
    unittest.main()
