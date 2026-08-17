from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "run_evaluate.py"
SPEC = importlib.util.spec_from_file_location("generalized_card_run_evaluate", SCRIPT)
assert SPEC and SPEC.loader
RUN_EVALUATE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUN_EVALUATE)


class EvaluationSnapshotTest(unittest.TestCase):
    def test_n1_metric_status_is_never_pass(self) -> None:
        row = {"mwu_p_value": 1.0, "ks_p_value": 1.0}
        self.assertEqual(RUN_EVALUATE._metric_status(row, sample_size=1), "DESCRIPTIVE")
        self.assertEqual(RUN_EVALUATE._metric_status(row, sample_size=10), "PASS")

    def test_staging_preserves_writer_discussion_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "generated"
            run = source / "run_00_sampled_reddit"
            run.mkdir(parents=True)
            discussion = {
                "posts": [
                    {
                        "post_id": "post-1",
                        "comments": [
                            {
                                "comment_id": 1,
                                "parent_comment_id": None,
                                "depth": 0,
                                "content": "wtf, this is the Writer output",
                                "replies": [],
                            }
                        ],
                    }
                ]
            }
            discussion_path = run / "discussion.json"
            discussion_path.write_text(
                json.dumps(discussion, ensure_ascii=False, indent=1) + "\n",
                encoding="utf-8",
            )
            records = run / "generation_records.json"
            records.write_text('[{"raw":"unchanged"}]\n', encoding="utf-8")
            target = root / "cleaned"

            RUN_EVALUATE._stage_generated_snapshot(
                source=source,
                target=target,
                complete=lambda: RUN_EVALUATE._cleaned_complete(target, 1),
                resume=False,
            )

            copied = target / run.name
            self.assertEqual(
                (copied / "discussion.json").read_bytes(), discussion_path.read_bytes()
            )
            self.assertEqual(
                (copied / "generation_records.json").read_bytes(), records.read_bytes()
            )

    def test_staging_rejects_noncanonical_tree_without_repair(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "generated"
            run = source / "run_00_sampled_reddit"
            run.mkdir(parents=True)
            discussion = {
                "posts": [
                    {
                        "post_id": "post-1",
                        "comments": [
                            {
                                "comment_id": 1,
                                "parent_comment_id": 999,
                                "depth": 7,
                                "content": "do not normalize me",
                                "replies": [],
                            }
                        ],
                    }
                ]
            }
            (run / "discussion.json").write_text(
                json.dumps(discussion), encoding="utf-8"
            )
            target = root / "cleaned"

            with self.assertRaisesRegex(RuntimeError, "non-canonical"):
                RUN_EVALUATE._stage_generated_snapshot(
                    source=source,
                    target=target,
                    complete=lambda: RUN_EVALUATE._cleaned_complete(target, 1),
                    resume=False,
                )
            staged = json.loads(
                (target / run.name / "discussion.json").read_text(encoding="utf-8")
            )
            self.assertEqual(staged["posts"][0]["comments"][0]["depth"], 7)


if __name__ == "__main__":
    unittest.main()
