from __future__ import annotations

import csv
import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from generalized_card.content_profile import build_content_profile, render_markdown
from generalized_card.content_profile_analysis import repeated_ngram_share
from generalized_card.content_profile_data import EVALUATION_METRICS


class MatchedContentProfileTest(unittest.TestCase):
    def test_uses_exact_matched_thread_and_marks_n1_descriptive(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir, config = self._fixture(root)
            report = build_content_profile(run_dir, config)

            self.assertEqual(report["scope"]["paired_threads"], 1)
            self.assertEqual(report["scope"]["generated_comments"], 2)
            properties = {
                row["property"]: row for row in report["model_scored_properties"]
            }
            self.assertAlmostEqual(
                properties["dominant-emotion entropy"]["real"], math.log(2), places=6
            )
            self.assertAlmostEqual(properties["mean story probability"]["real"], 0.2)
            self.assertNotAlmostEqual(properties["mean story probability"]["real"], 0.99)

            metrics = {row["metric"]: row for row in report["metrics"]["rows"]}
            self.assertEqual(metrics["self_bleu_4"]["inferential_status"], "descriptive_only_n1")
            self.assertAlmostEqual(metrics["self_bleu_4"]["gap"], 0.1)
            self.assertIn("cannot validate MWU/KS", report["statistical_interpretation"])

    def test_joins_planner_controls_to_saved_comment_models(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report = build_content_profile(*self._fixture(Path(directory)))
            realized = report["planner_writer_realization"]
            self.assertEqual(realized["tone"]["covered"], 2)
            self.assertEqual(realized["tone"]["aligned"], 1)
            self.assertAlmostEqual(realized["tone"]["exact_rate"], 0.5)
            self.assertAlmostEqual(
                realized["story"]["planned_story"]["mean_probability"], 0.8
            )
            self.assertAlmostEqual(
                realized["story"]["planned_no_story"]["mean_probability"], 0.7
            )
            self.assertAlmostEqual(
                realized["planned_surface"]["recommendation_advice_share"], 0.5
            )
            self.assertEqual(len(report["examples"]["story_mismatches"]), 2)

    def test_surface_probes_remain_explicitly_weak(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report = build_content_profile(*self._fixture(Path(directory)))
            self.assertEqual(
                report["surface_diagnostics"]["provenance"],
                "matched-side lexical regex; diagnostic only",
            )
            self.assertIn("never semantic ground truth", report["evidence_boundaries"]["weak_surface_probes"])
            markdown = render_markdown(report)
            self.assertIn("Weak surface probes (not semantic labels)", markdown)
            self.assertIn("- No, this is bullshit.", markdown)
            self.assertNotIn("- : No, this is bullshit.", markdown)

    def test_repeated_ngram_share_counts_cross_comment_reuse(self) -> None:
        texts = [
            "same narrow phrase repeats inside same narrow phrase",
            "same narrow phrase appears elsewhere",
        ]
        self.assertGreater(repeated_ngram_share(texts, 3), 0.0)
        self.assertEqual(repeated_ngram_share([texts[0]], 3), 0.0)

    def test_rejects_a_content_cohort_without_matched_metric_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir, config = self._fixture(Path(directory))
            (run_dir / "matched_evaluation" / "matched_real_thread_scores.csv").unlink()
            with self.assertRaisesRegex(ValueError, "same cohort"):
                build_content_profile(run_dir, config)

    @staticmethod
    def _fixture(root: Path) -> tuple[Path, SimpleNamespace]:
        raw = root / "raw" / "product"
        raw.mkdir(parents=True)
        (raw / "product.jsonl").write_text(
            json.dumps({"id": "real1", "title": "Matched post"}) + "\n",
            encoding="utf-8",
        )
        real_comments = [
            {"post_id": "real1", "comment_id": "r1", "body": "No, this is bullshit.", "created_utc": 1},
            {"post_id": "real1", "comment_id": "r2", "body": "You should check that part.", "created_utc": 2},
        ]
        (raw / "product.comments.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in real_comments), encoding="utf-8"
        )
        write_threads(
            raw / "go_emotions_results.json",
            [
                model_thread("real1", [{"comment_id": "r1", "dominant_emotion": "neutral"}, {"comment_id": "r2", "dominant_emotion": "anger"}]),
                model_thread("other", [{"comment_id": "x", "dominant_emotion": "joy"}] * 10),
            ],
        )
        write_threads(
            raw / "storyseeker_results.json",
            [
                model_thread("real1", [{"comment_id": "r1", "story_probability": 0.1}, {"comment_id": "r2", "story_probability": 0.3}]),
                model_thread("other", [{"comment_id": "x", "story_probability": 0.99}]),
            ],
        )
        write_threads(raw / "politeness_results.json", [model_thread("real1", [])])

        run_dir = root / "runs" / "fixture"
        generated = run_dir / "generated" / "run_00_sampled_reddit"
        cleaned = run_dir / "cleaned" / "run_00"
        matched = run_dir / "matched_evaluation"
        generated.mkdir(parents=True)
        cleaned.mkdir(parents=True)
        matched.mkdir(parents=True)
        seed_pool = root / "seed.json"
        seed_pool.write_text(
            json.dumps(
                {
                    "seed_posts": [
                        {
                            "seed_index": 0,
                            "source_raw_post_id": "real1",
                            "source_product_dir": "product",
                            "source_file": str(raw / "product.jsonl"),
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        (run_dir / "run_config.json").write_text(
            json.dumps({"seed_pool": str(seed_pool)}), encoding="utf-8"
        )
        records = [
            {
                "post_id": "gen1",
                "seed_index": 0,
                "task": {
                    "tone_target": "impolite",
                    "affect_role": "anger",
                    "story_mode": "specific_personal_story",
                    "comment_function": "recommendation_advice",
                    "payload_type": "advice",
                    "speaker_role": "advisor",
                },
                "comment": {"comment_id": "1", "content": "No, this is bullshit."},
            },
            {
                "post_id": "gen1",
                "seed_index": 0,
                "task": {
                    "tone_target": "polite",
                    "affect_role": "neutral",
                    "story_mode": "no_story",
                    "comment_function": "reaction",
                    "payload_type": "soft_helpful",
                    "speaker_role": "side_observer",
                },
                "comment": {"comment_id": "2", "content": "You should check that part."},
            },
        ]
        (generated / "generation_records.json").write_text(json.dumps(records), encoding="utf-8")
        write_threads(
            cleaned / "go_emotions_results.json",
            [model_thread("gen1", [{"comment_id": "1", "dominant_emotion": "anger"}, {"comment_id": "2", "dominant_emotion": "neutral"}])],
        )
        write_threads(
            cleaned / "storyseeker_results.json",
            [model_thread("gen1", [{"comment_id": "1", "story_probability": 0.8}, {"comment_id": "2", "story_probability": 0.7}])],
        )
        write_threads(
            cleaned / "politeness_results.json",
            [model_thread("gen1", [{"comment_id": "1", "pred_label": "impolite"}, {"comment_id": "2", "pred_label": "impolite"}])],
        )

        write_metric_csv(matched / "matched_real_thread_scores.csv", 0.2)
        write_metric_csv(matched / "matched_generated_thread_scores.csv", 0.3)
        (matched / "matched_seed_group_eval.json").write_text(
            json.dumps({metric: {"mwu_p_value": 1.0, "ks_p_value": 1.0} for metric in EVALUATION_METRICS}),
            encoding="utf-8",
        )
        config = SimpleNamespace(
            raw_discussions_dir=raw.parent,
            technical_terms=("sensor",),
            protected_entity_terms=("Canon",),
        )
        return run_dir, config


def model_thread(thread_id: str, comments: list[dict[str, object]]) -> dict[str, object]:
    return {"thread_id": thread_id, "comments": comments}


def write_threads(path: Path, threads: list[dict[str, object]]) -> None:
    path.write_text(json.dumps({"threads": threads}), encoding="utf-8")


def write_metric_csv(path: Path, value: float) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(EVALUATION_METRICS))
        writer.writeheader()
        writer.writerow({metric: value for metric in EVALUATION_METRICS})


if __name__ == "__main__":
    unittest.main()
