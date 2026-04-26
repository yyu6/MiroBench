"""
Tests for calibration/runner.py — run_candidates() function.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from calibration.runner import run_candidates


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SAMPLE_OVERLAY = {"temperature": 0.8, "top_p": 0.95}

REFERENCE_CONFIG = {
    "input_file": "products.json",
    "agents": 10,
    "hours": 24,
    "rounds": 5,
    "seed_posts": 2,
    "seed": 42,
    "hint": None,
    "few_shot_source": None,
    "few_shot_count": 0,
    "few_shot_comments": 2,
}


def _make_subprocess_mock(returncode: int = 0, create_sim_dir: bool = True):
    """Return a mock for subprocess.run that optionally creates a sim directory."""

    def _side_effect(cmd, **kwargs):
        if create_sim_dir and returncode == 0:
            # Parse --output-dir from the command args
            try:
                idx = cmd.index("--output-dir")
                output_dir = Path(cmd[idx + 1])
                output_dir.mkdir(parents=True, exist_ok=True)
                # Create a fake sim subdirectory (mimics run_discussion.py behaviour)
                sim_subdir = output_dir / "fake_sim_20240101_120000"
                sim_subdir.mkdir(parents=True, exist_ok=True)
            except (ValueError, IndexError):
                pass

        result = MagicMock()
        result.returncode = returncode
        result.stdout = b"stdout text"
        result.stderr = b"stderr text"
        return result

    return _side_effect


# ---------------------------------------------------------------------------
# test_creates_candidate_dirs
# ---------------------------------------------------------------------------

class TestCreatesCandidateDirs:
    def test_creates_candidate_dirs(self, tmp_path):
        """Candidate directories and overlay.json are created for each overlay."""
        overlays = [SAMPLE_OVERLAY, {"temperature": 0.5}]
        iter_dir = tmp_path / "iter_0"

        with patch("subprocess.run", side_effect=_make_subprocess_mock(returncode=0)):
            results = run_candidates(
                overlays=overlays,
                iter_dir=iter_dir,
                reference_run_config=REFERENCE_CONFIG,
                parallel=1,
            )

        assert len(results) == 2

        for i, result in enumerate(results):
            candidate_dir = Path(result["candidate_dir"])
            assert candidate_dir.exists(), f"candidate_dir {candidate_dir} does not exist"

            overlay_path = candidate_dir / "overlay.json"
            assert overlay_path.exists(), f"overlay.json missing in {candidate_dir}"

            saved = json.loads(overlay_path.read_text())
            assert saved == overlays[i]

            assert result["candidate_id"] == i
            assert "candidate_dir" in result
            assert "sim_dir" in result

    def test_candidate_dirs_named_correctly(self, tmp_path):
        """Candidate directories follow the candidate_N naming convention."""
        overlays = [SAMPLE_OVERLAY]
        iter_dir = tmp_path / "iter_0"

        with patch("subprocess.run", side_effect=_make_subprocess_mock(returncode=0)):
            results = run_candidates(
                overlays=overlays,
                iter_dir=iter_dir,
                reference_run_config=REFERENCE_CONFIG,
            )

        candidate_dir = Path(results[0]["candidate_dir"])
        assert candidate_dir.name == "candidate_0"
        assert candidate_dir.parent.name == "candidates"
        assert candidate_dir.parent.parent == iter_dir

    def test_logs_written(self, tmp_path):
        """stdout.log and stderr.log are written to each candidate dir."""
        overlays = [SAMPLE_OVERLAY]
        iter_dir = tmp_path / "iter_1"

        with patch("subprocess.run", side_effect=_make_subprocess_mock(returncode=0)):
            results = run_candidates(
                overlays=overlays,
                iter_dir=iter_dir,
                reference_run_config=REFERENCE_CONFIG,
            )

        candidate_dir = Path(results[0]["candidate_dir"])
        assert (candidate_dir / "stdout.log").exists()
        assert (candidate_dir / "stderr.log").exists()

    def test_success_marked_true(self, tmp_path):
        """success=True when subprocess returns returncode=0."""
        overlays = [SAMPLE_OVERLAY]
        iter_dir = tmp_path / "iter_2"

        with patch("subprocess.run", side_effect=_make_subprocess_mock(returncode=0)):
            results = run_candidates(
                overlays=overlays,
                iter_dir=iter_dir,
                reference_run_config=REFERENCE_CONFIG,
            )

        assert results[0]["success"] is True
        assert results[0]["returncode"] == 0


# ---------------------------------------------------------------------------
# test_failed_simulation_marked
# ---------------------------------------------------------------------------

class TestFailedSimulationMarked:
    def test_failed_simulation_marked(self, tmp_path):
        """success=False when subprocess returns a non-zero returncode."""
        overlays = [SAMPLE_OVERLAY]
        iter_dir = tmp_path / "iter_fail"

        with patch(
            "subprocess.run",
            side_effect=_make_subprocess_mock(returncode=1, create_sim_dir=False),
        ):
            results = run_candidates(
                overlays=overlays,
                iter_dir=iter_dir,
                reference_run_config=REFERENCE_CONFIG,
            )

        assert len(results) == 1
        assert results[0]["success"] is False
        assert results[0]["returncode"] == 1

    def test_failed_has_no_sim_dir(self, tmp_path):
        """sim_dir is None when the subprocess fails."""
        overlays = [SAMPLE_OVERLAY]
        iter_dir = tmp_path / "iter_fail2"

        with patch(
            "subprocess.run",
            side_effect=_make_subprocess_mock(returncode=1, create_sim_dir=False),
        ):
            results = run_candidates(
                overlays=overlays,
                iter_dir=iter_dir,
                reference_run_config=REFERENCE_CONFIG,
            )

        assert results[0]["sim_dir"] is None

    def test_multiple_overlays_partial_failure(self, tmp_path):
        """One failure and one success are both tracked correctly."""
        overlays = [SAMPLE_OVERLAY, {"temperature": 0.3}]
        iter_dir = tmp_path / "iter_partial"
        call_count = {"n": 0}

        def _mixed_side_effect(cmd, **kwargs):
            n = call_count["n"]
            call_count["n"] += 1
            rc = 0 if n == 0 else 1
            return _make_subprocess_mock(returncode=rc, create_sim_dir=(rc == 0))(
                cmd, **kwargs
            )

        with patch("subprocess.run", side_effect=_mixed_side_effect):
            results = run_candidates(
                overlays=overlays,
                iter_dir=iter_dir,
                reference_run_config=REFERENCE_CONFIG,
            )

        successes = [r for r in results if r["success"]]
        failures = [r for r in results if not r["success"]]
        assert len(successes) == 1
        assert len(failures) == 1


# ---------------------------------------------------------------------------
# test_reference_config_fields_passed
# ---------------------------------------------------------------------------

class TestReferenceConfigFieldsPassed:
    def test_input_file_in_command(self, tmp_path):
        """input_file from reference_run_config is included in subprocess cmd."""
        captured = {}

        def _capture(cmd, **kwargs):
            captured["cmd"] = cmd
            result = MagicMock()
            result.returncode = 0
            result.stdout = b""
            result.stderr = b""
            return result

        overlays = [SAMPLE_OVERLAY]
        iter_dir = tmp_path / "iter_ref"

        with patch("subprocess.run", side_effect=_capture):
            run_candidates(
                overlays=overlays,
                iter_dir=iter_dir,
                reference_run_config=REFERENCE_CONFIG,
            )

        cmd = captured["cmd"]
        assert "products.json" in cmd

    def test_overlay_flag_present(self, tmp_path):
        """--overlay flag pointing to overlay.json is included in subprocess cmd."""
        captured = {}

        def _capture(cmd, **kwargs):
            captured["cmd"] = cmd
            result = MagicMock()
            result.returncode = 0
            result.stdout = b""
            result.stderr = b""
            return result

        overlays = [SAMPLE_OVERLAY]
        iter_dir = tmp_path / "iter_overlay"

        with patch("subprocess.run", side_effect=_capture):
            run_candidates(
                overlays=overlays,
                iter_dir=iter_dir,
                reference_run_config=REFERENCE_CONFIG,
            )

        cmd = captured["cmd"]
        assert "--overlay" in cmd
        overlay_idx = cmd.index("--overlay")
        overlay_path = Path(cmd[overlay_idx + 1])
        assert overlay_path.name == "overlay.json"
