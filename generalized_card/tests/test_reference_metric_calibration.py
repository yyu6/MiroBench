from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from generalized_card.reference_metric_calibration import (
    build_reference_metric_calibration,
    select_reference_template,
)


class ReferenceMetricCalibrationTest(unittest.TestCase):
    def test_uses_only_reference_ids_and_stores_no_text(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metric_rows = [
                {
                    "thread_id": "seed",
                    "comment_count": 10,
                    "story_count": 9,
                    "story_rate": 0.9,
                },
                {
                    "thread_id": "ref-a",
                    "thread_title": "private title",
                    "comment_count": 10,
                    "story_count": 1,
                    "story_rate": 0.1,
                    "mean_story_probability": 0.12,
                    "emotion_entropy": 1.1,
                    "self_bleu_4": 0.03,
                    "semantic_mean_cosine": 0.31,
                    "polite_rate": 0.4,
                    "impolite_rate": 0.3,
                    "neutral_rate": 0.2,
                },
            ]
            emotion_rows = {
                "threads": [
                    {
                        "thread_id": "ref-a",
                        "comment_count": 10,
                        "comments": [{"text": "private comment"}],
                        "dominant_emotion_counts": {
                            "neutral": 7,
                            "approval": 3,
                        },
                    }
                ]
            }
            (root / "thread_metrics_summary.json").write_text(
                json.dumps(metric_rows), encoding="utf-8"
            )
            (root / "go_emotions_results.json").write_text(
                json.dumps(emotion_rows), encoding="utf-8"
            )
            (root / "storyseeker_results.json").write_text(
                json.dumps(
                    {
                        "threads": [
                            {
                                "thread_id": "ref-a",
                                "comments": [
                                    {"story_probability": 0.01, "text": "secret one"},
                                    {"story_probability": 0.10, "text": "secret two"},
                                    {"story_probability": 0.40, "text": "secret three"},
                                    {"story_probability": 0.60, "text": "secret four"},
                                    {"story_probability": 0.90, "text": "secret five"},
                                ],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            calibration = build_reference_metric_calibration(
                root,
                reference_thread_ids={"ref-a"},
                excluded_seed_ids={"seed"},
            )

            self.assertTrue(calibration["available"])
            self.assertEqual(calibration["reference_thread_count"], 1)
            self.assertEqual(calibration["seed_reference_overlap_count"], 0)
            self.assertFalse(calibration["raw_text_included"])
            serialized = json.dumps(calibration)
            self.assertNotIn("private title", serialized)
            self.assertNotIn("private comment", serialized)
            self.assertNotIn("secret one", serialized)
            selected = select_reference_template(
                calibration, comment_count=10, selector=0
            )
            self.assertEqual(selected["dominant_emotion_counts"]["neutral"], 7)
            self.assertEqual(selected["semantic_mean_cosine"], 0.31)
            self.assertEqual(
                selected["story_probability_tier_counts"],
                {
                    "ambiguous": 1,
                    "low": 1,
                    "story_high": 1,
                    "story_mid": 1,
                    "very_low": 1,
                },
            )
            self.assertEqual(
                calibration["metric_bands_by_size"]["small"]
                ["semantic_mean_cosine"]["median"],
                0.31,
            )

    def test_rejects_seed_reference_overlap(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                build_reference_metric_calibration(
                    Path(directory),
                    reference_thread_ids={"same"},
                    excluded_seed_ids={"same"},
                )

    def test_pairwise_metric_bands_exclude_single_comment_threads(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [
                {
                    "thread_id": "single",
                    "comment_count": 1,
                    "self_bleu_4": 0.0,
                    "semantic_mean_cosine": 0.0,
                },
                {
                    "thread_id": "pair",
                    "comment_count": 2,
                    "self_bleu_4": 0.04,
                    "semantic_mean_cosine": 0.40,
                },
            ]
            (root / "thread_metrics_summary.json").write_text(
                json.dumps(rows), encoding="utf-8"
            )
            calibration = build_reference_metric_calibration(
                root,
                reference_thread_ids={"single", "pair"},
                excluded_seed_ids={"seed"},
            )
            semantic = calibration["metric_bands_by_size"]["tiny"]
            semantic = semantic["semantic_mean_cosine"]
            self.assertEqual(semantic["sample_count"], 1)
            self.assertEqual(semantic["median"], 0.40)


if __name__ == "__main__":
    unittest.main()
