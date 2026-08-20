"""The length cue asks for what realizes the slot, not for the slot's own count.

Through v97 the two were identical and the Writer regressed every slot toward its
own preferred length: realized/target 1.42x at the shortest slots, 0.71x at
251-400 words. The thread's mean length survived and its spread collapsed.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from generalized_card.length_calibration import (
    MAX_ASK_MULTIPLIER,
    MIN_ASK_MULTIPLIER,
    ask_multiplier,
    calibrated_word_ask,
    set_length_calibration,
)
from generalized_card.length_policy import (
    soft_length_guidance,
    writer_provider_token_budget,
)


class CalibrationTest(unittest.TestCase):
    def tearDown(self) -> None:
        set_length_calibration("measured")

    def test_short_slots_are_asked_for_less_and_long_slots_for_more(self) -> None:
        self.assertLess(calibrated_word_ask(7), 7)
        self.assertLess(calibrated_word_ask(20), 20)
        self.assertGreater(calibrated_word_ask(120), 120)
        self.assertGreater(calibrated_word_ask(300), 300)

    def test_the_crossover_is_where_the_measurement_put_it(self) -> None:
        # Realized/target crossed 1.0 between the 26-40 and 41-60 word bands.
        self.assertEqual(calibrated_word_ask(35), 35)

    def test_the_ask_is_monotone_in_the_target(self) -> None:
        # A calibration that reordered two slots would trade one distribution
        # defect for another.
        asks = [calibrated_word_ask(value) for value in range(1, 900)]
        self.assertEqual(asks, sorted(asks))

    def test_the_multiplier_never_leaves_its_clamp(self) -> None:
        for target in (1, 2, 5, 40, 300, 845, 5000):
            multiplier = ask_multiplier(target)
            self.assertGreaterEqual(multiplier, MIN_ASK_MULTIPLIER)
            self.assertLessEqual(multiplier, MAX_ASK_MULTIPLIER)

    def test_the_clamp_does_not_bind_inside_the_fitted_range(self) -> None:
        # The fit's support is 1-845 words. A clamp that bound inside it would be
        # discarding measured signal rather than bounding extrapolation.
        for target in (2, 13, 40, 220, 845):
            self.assertLess(ask_multiplier(target), MAX_ASK_MULTIPLIER)
            self.assertGreater(ask_multiplier(target), MIN_ASK_MULTIPLIER)

    def test_an_ask_is_never_zero_words(self) -> None:
        self.assertEqual(calibrated_word_ask(1), 1)
        self.assertEqual(calibrated_word_ask(0), 0)

    def test_off_arm_reproduces_the_pre_v98_cue(self) -> None:
        set_length_calibration("off")
        self.assertEqual(calibrated_word_ask(300), 300)
        self.assertIn(
            "roughly 300 words",
            soft_length_guidance(SimpleNamespace(real_word_count=300)),
        )


class RenderedCueTest(unittest.TestCase):
    def tearDown(self) -> None:
        set_length_calibration("measured")

    def test_the_cue_carries_the_calibrated_number(self) -> None:
        rendered = soft_length_guidance(SimpleNamespace(real_word_count=220))
        self.assertIn(f"roughly {calibrated_word_ask(220)} words", rendered)
        self.assertNotIn("roughly 220 words", rendered)

    def test_a_one_word_ask_is_not_pluralized(self) -> None:
        self.assertIn("roughly 1 word.", soft_length_guidance(SimpleNamespace(real_word_count=1)))

    def test_the_token_ceiling_clears_the_calibrated_ask(self) -> None:
        # Asking for more words than the provider will emit turns the calibration
        # into a truncation.
        for target in (120, 300, 845):
            task = SimpleNamespace(real_word_count=target)
            self.assertGreater(
                writer_provider_token_budget(task, configured_max=260),
                calibrated_word_ask(target) * 1.7,
                target,
            )

    def test_the_cue_direction_follows_the_measured_crossover(self) -> None:
        # The old cutoff was a written-down 100 words, so every slot between 35
        # and 100 was told "do not pad past it" while its measured error was
        # undershoot -- on the v98 seed-2 gate the 56-80 word slots realized
        # 0.48 to 0.74 of their target. The cue and the ask now read the same
        # curve, so they cannot disagree again.
        for words in (12, 25, 34):
            self.assertIn(
                "Do not pad past it",
                soft_length_guidance(SimpleNamespace(real_word_count=words)),
                words,
            )
        for words in (40, 70, 100, 300):
            self.assertIn(
                "do not trim toward a medium-length answer",
                soft_length_guidance(SimpleNamespace(real_word_count=words)),
                words,
            )

    def test_the_crossover_is_where_the_ask_multiplier_turns(self) -> None:
        flip = next(w for w in range(1, 200) if ask_multiplier(w) >= 1.0)
        self.assertIn(
            "do not trim", soft_length_guidance(SimpleNamespace(real_word_count=flip))
        )
        self.assertIn(
            "Do not pad", soft_length_guidance(SimpleNamespace(real_word_count=flip - 1))
        )

    def test_the_layout_and_beat_cues_keep_the_matched_size(self) -> None:
        # Only the number in the length cue is calibrated. The layout describes
        # what a comment of the slot's real size looks like, so calibrating it
        # too would ask a 300-word comment for a 388-word comment's paragraphs.
        from generalized_card.comment_structure import (
            expected_paragraphs,
            set_active_structure_profile,
        )

        profile = {
            "available": True,
            "buckets": {
                "essay": {
                    "median_paragraphs": 6,
                    "p90_paragraphs": 14,
                    "median_words_per_paragraph": 50.0,
                }
            },
        }
        set_active_structure_profile(profile)
        try:
            rendered = soft_length_guidance(SimpleNamespace(real_word_count=300))
            self.assertIn(
                f"about {expected_paragraphs(profile, 300)} short paragraphs", rendered
            )
        finally:
            set_active_structure_profile({})


if __name__ == "__main__":
    unittest.main()
