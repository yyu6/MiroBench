"""Every finished comment must be on disk before the next one is paid for.

`write_discussion_bundle` runs once per post, after every comment in it. A
97-comment thread holds 85 minutes of paid generation in memory and writes
nothing until the last one lands. Two runs were lost exactly there -- one at 56
of 97, one with all 97 written and killed before the flush -- and each left a
run directory holding logs and reproducibility snapshots and no comments.
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "sampling_generator"))
import run_sampled_reddit_generator as G  # noqa: E402


class IncrementalRecordsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.dir = Path(tempfile.mkdtemp())
        self.path = self.dir / "records.partial.jsonl"
        self._saved = G.GENERALIZED_INCREMENTAL_RECORDS_PATH

    def tearDown(self) -> None:
        G.GENERALIZED_INCREMENTAL_RECORDS_PATH = self._saved

    def test_disabled_by_default_writes_nothing(self) -> None:
        G.GENERALIZED_INCREMENTAL_RECORDS_PATH = None
        G._append_incremental_record(1, {"raw": "x"})
        self.assertEqual([], list(self.dir.iterdir()))

    def test_one_line_per_comment_with_the_seed_it_belongs_to(self) -> None:
        G.GENERALIZED_INCREMENTAL_RECORDS_PATH = self.path
        for i in range(3):
            G._append_incremental_record(7, {"raw": f"c{i}", "task": {"local_task_id": i}})
        lines = self.path.read_text().strip().splitlines()
        self.assertEqual(3, len(lines))
        rows = [json.loads(x) for x in lines]
        self.assertEqual(["c0", "c1", "c2"], [r["raw"] for r in rows])
        self.assertEqual([7, 7, 7], [r["seed_index"] for r in rows])
        # the plan has to survive too, or a recovered run cannot be analysed
        self.assertEqual(1, rows[1]["task"]["local_task_id"])

    def test_a_write_failure_never_costs_the_comment(self) -> None:
        """The comment is already in `records`; this log must not raise."""
        G.GENERALIZED_INCREMENTAL_RECORDS_PATH = "/nonexistent-dir-xyz/a.jsonl"
        G._append_incremental_record(1, {"raw": "y"})

    def test_unserialisable_content_does_not_corrupt_earlier_lines(self) -> None:
        G.GENERALIZED_INCREMENTAL_RECORDS_PATH = self.path
        G._append_incremental_record(1, {"raw": "kept"})
        G._append_incremental_record(1, {"raw": object()})
        lines = self.path.read_text().strip().splitlines()
        self.assertEqual(1, len(lines))
        self.assertEqual("kept", json.loads(lines[0])["raw"])


if __name__ == "__main__":
    unittest.main()
