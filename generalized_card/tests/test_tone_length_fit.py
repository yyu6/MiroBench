from __future__ import annotations

import unittest
from collections import Counter

from generalized_card.comment_structure import structure_bucket
from generalized_card.generation_distribution import TONE_CLASSES
from generalized_card.planner_distribution import build_slot_distribution_schedule
from generalized_card.tone_length_fit import (
    band_shares,
    build_tone_length_profile,
    fit_tone_labels,
    realized_joint,
    set_tone_length_fit,
)


# The camera domain's measured conditional, rounded. Used as a fixture so the
# test states the dependence it expects rather than re-reading the corpus.
MEASURED = {
    "available": True,
    "conditional": {
        "micro": {"polite": 0.251, "somewhat_polite": 0.064, "neutral": 0.281, "impolite": 0.404},
        "short": {"polite": 0.162, "somewhat_polite": 0.123, "neutral": 0.234, "impolite": 0.481},
        "medium": {"polite": 0.263, "somewhat_polite": 0.115, "neutral": 0.145, "impolite": 0.476},
        "long": {"polite": 0.520, "somewhat_polite": 0.071, "neutral": 0.059, "impolite": 0.350},
        "very_long": {"polite": 0.638, "somewhat_polite": 0.038, "neutral": 0.035, "impolite": 0.289},
        "essay": {"polite": 0.720, "somewhat_polite": 0.021, "neutral": 0.029, "impolite": 0.230},
    },
}


def _slots(sizes: list[int]) -> list[dict[str, object]]:
    return [
        {"sample_id": index, "words": words, "depth": 1, "surface": "ordinary_turn"}
        for index, words in enumerate(sizes, start=1)
    ]


class ToneLengthProfileTest(unittest.TestCase):
    def test_missing_band_falls_back_to_uniform_rather_than_zero(self) -> None:
        shares = band_shares(MEASURED, "no_such_band", TONE_CLASSES)
        self.assertEqual(set(shares), set(TONE_CLASSES))
        for value in shares.values():
            self.assertAlmostEqual(value, 1 / len(TONE_CLASSES))

    def test_absent_label_stays_assignable(self) -> None:
        profile = {"available": True, "conditional": {"micro": {"impolite": 1.0}}}
        shares = band_shares(profile, "micro", TONE_CLASSES)
        self.assertGreater(shares["polite"], 0.0)

    def test_profile_needs_enough_samples(self) -> None:
        thin = build_tone_length_profile(
            "does/not/exist", reference_thread_ids=["t1"], tone_classes=TONE_CLASSES
        )
        self.assertFalse(thin["available"])


class FitTest(unittest.TestCase):
    def tearDown(self) -> None:
        set_tone_length_fit("conditional")

    def test_template_counts_are_exact(self) -> None:
        slots = _slots([5, 15, 40, 90, 200, 400] * 10)
        totals = Counter({"polite": 18, "impolite": 30, "neutral": 9, "somewhat_polite": 3})
        assignments, unassigned = fit_tone_labels(
            slots, totals, profile=MEASURED, tone_classes=TONE_CLASSES
        )
        self.assertEqual(unassigned, [])
        self.assertEqual(len(assignments), len(slots))
        self.assertEqual(Counter(assignments.values()), totals)

    def test_polite_share_rises_with_slot_size(self) -> None:
        # v96 planned 74% impolite on 120-250 word slots and 100% on slots over
        # 250 words, where excluded real comments of that size are 72% polite.
        # The bands compared here are the ones the measurement separates: micro
        # and medium differ by 0.012 in the reference data, which integer cell
        # counts cannot resolve and this fit does not pretend to.
        slots = _slots([8] * 60 + [40] * 60 + [300] * 60)
        totals = Counter({"polite": 60, "impolite": 90, "neutral": 20, "somewhat_polite": 10})
        assignments, _ = fit_tone_labels(
            slots, totals, profile=MEASURED, tone_classes=TONE_CLASSES
        )
        joint = realized_joint(slots, assignments)
        polite = {
            band: row.get("polite", 0) / sum(row.values())
            for band, row in joint.items()
        }
        self.assertLess(polite["micro"], polite["essay"])
        self.assertLess(polite["medium"], polite["essay"])
        impolite = {
            band: row.get("impolite", 0) / sum(row.values())
            for band, row in joint.items()
        }
        self.assertGreater(impolite["short" if "short" in impolite else "medium"], impolite["essay"])

    def test_fitted_shares_track_the_measured_ratios(self) -> None:
        slots = _slots([8] * 60 + [40] * 60 + [300] * 60)
        totals = Counter({"polite": 60, "impolite": 90, "neutral": 20, "somewhat_polite": 10})
        assignments, _ = fit_tone_labels(
            slots, totals, profile=MEASURED, tone_classes=TONE_CLASSES
        )
        joint = realized_joint(slots, assignments)
        # Both margins are fixed, so the fitted shares cannot equal the measured
        # conditional; they must stay proportional to it up to that constraint.
        for band, row in joint.items():
            fitted = row.get("polite", 0) / sum(row.values())
            measured = MEASURED["conditional"][band]["polite"]
            self.assertLess(abs(fitted - measured), 0.12, band)

    def test_fit_does_not_exceed_the_measured_dependence(self) -> None:
        # A min-cost assignment drove the largest band to 100% polite. The
        # proportional fit must not invent a sharper dependence than measured.
        slots = _slots([8] * 60 + [40] * 60 + [300] * 60)
        totals = Counter({"polite": 60, "impolite": 90, "neutral": 20, "somewhat_polite": 10})
        assignments, _ = fit_tone_labels(
            slots, totals, profile=MEASURED, tone_classes=TONE_CLASSES
        )
        joint = realized_joint(slots, assignments)
        essay = joint["essay"]
        self.assertLess(essay.get("polite", 0) / sum(essay.values()), 0.95)
        self.assertGreater(essay.get("impolite", 0), 0)

    def test_a_short_slot_can_still_be_warm(self) -> None:
        # A quarter of real comments under ten words are labelled polite; the
        # pre-v97 hard exclusion made that impossible.
        slots = _slots([6] * 40)
        totals = Counter({"polite": 10, "impolite": 20, "neutral": 10})
        assignments, _ = fit_tone_labels(
            slots, totals, profile=MEASURED, tone_classes=TONE_CLASSES
        )
        self.assertIn("polite", set(assignments.values()))

    def test_more_labels_than_slots_reports_the_shortfall(self) -> None:
        slots = _slots([40, 40])
        totals = Counter({"polite": 3, "impolite": 3})
        assignments, unassigned = fit_tone_labels(
            slots, totals, profile=MEASURED, tone_classes=TONE_CLASSES
        )
        self.assertEqual(len(assignments), 2)
        self.assertEqual(len(unassigned), 4)


class ScheduleArmTest(unittest.TestCase):
    def tearDown(self) -> None:
        set_tone_length_fit("conditional")

    def _schedule(self, arm: str) -> dict[str, object]:
        set_tone_length_fit(arm)
        comments = (
            [{"body": "word " * 6, "depth": 1} for _ in range(20)]
            + [{"body": "word " * 40, "depth": 1} for _ in range(20)]
            + [{"body": "word " * 300, "depth": 1} for _ in range(20)]
        )
        template = {
            "comment_count": 60,
            "story_count": 6,
            "polite_rate": 0.30,
            "impolite_rate": 0.45,
            "neutral_rate": 0.15,
            "somewhat_polite_rate": 0.10,
            "dominant_emotion_counts": {"neutral": 40, "approval": 20},
        }
        return build_slot_distribution_schedule(
            template=template,
            comments=comments,
            total_comments=60,
            tone_length_profile=MEASURED,
        )

    def test_schedule_records_the_arm_and_the_realized_joint(self) -> None:
        schedule = self._schedule("conditional")
        self.assertEqual(schedule["tone_length_fit"], "conditional")
        joint = schedule["tone_length_joint"]
        self.assertIn("essay", joint)
        self.assertGreater(joint["essay"].get("polite", 0), 0)

    def test_conditional_arm_puts_more_warmth_on_long_slots_than_median_arm(
        self,
    ) -> None:
        conditional = self._schedule("conditional")["tone_length_joint"]
        median = self._schedule("median")["tone_length_joint"]
        self.assertGreater(
            conditional["essay"].get("polite", 0),
            median["essay"].get("polite", 0),
        )

    def test_every_slot_keeps_a_tone_contract(self) -> None:
        schedule = self._schedule("conditional")
        assigned = [
            row
            for row in schedule["assignments"].values()
            if row.get("tone_class")
        ]
        self.assertEqual(len(assigned), 60)
        self.assertEqual(schedule["unassigned_tone_labels"], [])


class BandTest(unittest.TestCase):
    def test_bands_match_the_layout_bands(self) -> None:
        # One band definition serves the layout profile and the tone conditional,
        # so a change to one cannot silently desynchronize the other.
        self.assertEqual(structure_bucket(300), "essay")
        self.assertIn("essay", MEASURED["conditional"])


if __name__ == "__main__":
    unittest.main()
