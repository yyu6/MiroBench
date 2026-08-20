from __future__ import annotations

import unittest
from types import SimpleNamespace

from generalized_card.comment_structure import (
    build_structure_profile,
    expected_paragraphs,
    layout_guidance,
    set_active_structure_profile,
    set_long_form_layout,
    structure_bucket,
)
from generalized_card.length_calibration import calibrated_word_ask
from generalized_card.length_policy import soft_length_guidance


def _threads(rows: list[tuple[str, int]]) -> list[dict[str, object]]:
    return [{"comments": [{"body": body} for body, count in rows for _ in range(count)]}]


class StructureProfileTest(unittest.TestCase):
    def tearDown(self) -> None:
        set_active_structure_profile({})
        set_long_form_layout("measured")

    def test_buckets_are_size_bands(self) -> None:
        self.assertEqual(structure_bucket(4), "micro")
        self.assertEqual(structure_bucket(20), "short")
        self.assertEqual(structure_bucket(59), "medium")
        self.assertEqual(structure_bucket(119), "long")
        self.assertEqual(structure_bucket(249), "very_long")
        self.assertEqual(structure_bucket(900), "essay")

    def test_profile_needs_enough_samples(self) -> None:
        thin = build_structure_profile([{"comments": [{"body": "a b c"}]}])
        self.assertFalse(thin["available"])

    def test_profile_measures_paragraphs_per_band(self) -> None:
        short = "word " * 30
        essay_body = ("word " * 60 + "\n\n") * 5
        profile = build_structure_profile(
            _threads([(short, 200), (essay_body, 100)])
        )
        self.assertTrue(profile["available"])
        self.assertEqual(profile["buckets"]["medium"]["median_paragraphs"], 1)
        self.assertEqual(profile["buckets"]["essay"]["median_paragraphs"], 5)
        self.assertAlmostEqual(
            profile["buckets"]["essay"]["median_words_per_paragraph"], 60.0, places=1
        )

    def test_profile_measures_list_and_quote_share(self) -> None:
        listed = "- one\n- two\n" + "word " * 300
        quoted = "> parent said this\n" + "word " * 300
        plain = "word " * 300
        profile = build_structure_profile(
            _threads([(listed, 100), (quoted, 100), (plain, 200)])
        )
        essay = profile["buckets"]["essay"]
        self.assertAlmostEqual(essay["list_share"], 0.25, places=2)
        self.assertAlmostEqual(essay["quote_share"], 0.25, places=2)

    def test_a_band_with_too_few_samples_is_omitted_rather_than_guessed(self) -> None:
        profile = build_structure_profile(
            _threads([("word " * 30, 300), ("word " * 400, 3)])
        )
        self.assertIn("medium", profile["buckets"])
        self.assertNotIn("essay", profile["buckets"])
        self.assertEqual(expected_paragraphs(profile, 400), 0)


class LayoutGuidanceTest(unittest.TestCase):
    def tearDown(self) -> None:
        set_active_structure_profile({})
        set_long_form_layout("measured")

    def _profile(self) -> dict[str, object]:
        return {
            "available": True,
            "buckets": {
                "medium": {
                    "median_paragraphs": 1,
                    "p90_paragraphs": 2,
                    "median_words_per_paragraph": 30.0,
                },
                "long": {
                    "median_paragraphs": 2,
                    "p90_paragraphs": 4,
                    "median_words_per_paragraph": 40.0,
                },
                "essay": {
                    "median_paragraphs": 6,
                    "p90_paragraphs": 14,
                    "median_words_per_paragraph": 53.6,
                },
            },
        }

    def test_the_open_ended_band_scales_within_itself(self) -> None:
        # Words per paragraph is nearly flat inside a band while the paragraph
        # count scales with length: real comments of 500-700 words are laid out
        # in 10 paragraphs against 6 for 250-350. One band median would ask an
        # 845-word slot for a 300-word slot's layout.
        profile = self._profile()
        self.assertEqual(expected_paragraphs(profile, 300), 6)
        self.assertEqual(expected_paragraphs(profile, 539), 10)
        self.assertEqual(expected_paragraphs(profile, 845), 14)

    def test_the_scaled_count_never_leaves_the_measured_range(self) -> None:
        profile = self._profile()
        self.assertEqual(expected_paragraphs(profile, 2000), 14)
        self.assertEqual(expected_paragraphs(profile, 251), 6)
        self.assertEqual(expected_paragraphs(profile, 30), 1)

    def test_a_band_without_the_rate_falls_back_to_its_median(self) -> None:
        profile = {"available": True, "buckets": {"essay": {"median_paragraphs": 6}}}
        self.assertEqual(expected_paragraphs(profile, 845), 6)

    def test_a_one_paragraph_slot_gets_no_cue(self) -> None:
        self.assertEqual(layout_guidance(self._profile(), 40), "")

    def test_a_long_slot_is_asked_for_its_measured_paragraph_count(self) -> None:
        cue = layout_guidance(self._profile(), 300)
        self.assertIn("about 6 short paragraphs", cue)
        self.assertIn("blank line", cue)

    def test_the_cue_permits_a_side_point_and_prescribes_no_entry_route(self) -> None:
        # Prescribing an opening or clause order gives every comment in one
        # register a shared entry route, which moved both self-similarity
        # metrics the wrong way in an earlier release.
        cue = layout_guidance(self._profile(), 300).lower()
        self.assertIn("side point", cue)
        for banned in ("open with", "start with", "lead with", "first sentence"):
            self.assertNotIn(banned, cue)

    def test_cue_carries_no_domain_vocabulary(self) -> None:
        cue = layout_guidance(self._profile(), 300).lower()
        for term in ("camera", "lens", "photo", "product", "card", "phone"):
            self.assertNotIn(term, cue)

    def test_length_rule_renders_the_layout_for_a_long_slot(self) -> None:
        set_active_structure_profile(self._profile())
        rendered = soft_length_guidance(
            SimpleNamespace(real_word_count=300, development_plan="")
        )
        self.assertIn(f"roughly {calibrated_word_ask(300)} words", rendered)
        self.assertIn("about 6 short paragraphs", rendered)

    def test_beats_only_arm_reproduces_the_pre_v97_length_rule(self) -> None:
        set_active_structure_profile(self._profile())
        set_long_form_layout("beats_only")
        rendered = soft_length_guidance(
            SimpleNamespace(real_word_count=300, development_plan="")
        )
        self.assertNotIn("paragraphs", rendered)


if __name__ == "__main__":
    unittest.main()
