from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

from generalized_card.thread_metric_suite import (
    load_thread_metrics,
    metric_commands,
    run_metric_commands,
)


class ThreadMetricSuiteTest(unittest.TestCase):
    def test_loader_excludes_synthetic_summary_row(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "thread_metrics_summary.csv").write_text(
                "thread_id,self_bleu_4\npost-1,0.1\n__summary_mean__,0.9\n",
                encoding="utf-8",
            )
            frame = load_thread_metrics(root)
            self.assertEqual(frame["thread_id"].tolist(), ["post-1"])

    def test_command_manifest_covers_all_metric_outputs(self) -> None:
        commands = metric_commands(
            scripts=Path("scripts/evaluation"),
            sim_dir=Path("artifact"),
            python="python",
            device="cpu",
        )
        outputs = {name for name, _ in commands}
        self.assertEqual(len(outputs), 9)
        self.assertIn("storyseeker_results.json", outputs)
        self.assertIn("go_emotions_results.json", outputs)
        self.assertIn("thread_structure_results.json", outputs)

    def test_failed_metric_retries_serially(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "metric.json"
            marker = root / "attempted"
            code = (
                "import json,pathlib,sys;"
                f"o=pathlib.Path({str(output)!r});m=pathlib.Path({str(marker)!r});"
                "first=not m.exists();m.touch();"
                "sys.exit(3) if first else o.write_text(json.dumps({'ok':1}))"
            )
            run_metric_commands(
                [(output.name, [sys.executable, "-c", code])],
                sim_dir=root,
                max_workers=1,
            )
            self.assertEqual(json.loads(output.read_text(encoding="utf-8")), {"ok": 1})
            self.assertTrue((root / "metric_logs" / "metric.retry.stderr.log").exists())


if __name__ == "__main__":
    unittest.main()
