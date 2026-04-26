"""
Tests for calibration/orchestrator.py — CalibrationState only.
(The orchestrator loop is integration-heavy and tested separately.)
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from calibration.orchestrator import CalibrationState


class TestCalibrationState:
    def test_init_fresh(self, tmp_path: Path) -> None:
        state = CalibrationState(output_dir=tmp_path / "cal_run")
        assert state.current_best_overlay == {}
        assert state.current_best_score is None
        assert state.current_best_diagnostic is None
        assert state.completed_iterations == 0

    def test_state_path_set(self, tmp_path: Path) -> None:
        out = tmp_path / "cal_run"
        state = CalibrationState(output_dir=out)
        assert state.state_path == out / "calibration_state.json"
        assert state.output_dir == out

    def test_save_and_load(self, tmp_path: Path) -> None:
        out = tmp_path / "cal_run"
        state = CalibrationState(output_dir=out)
        state.current_best_overlay = {"persona.sentiment_bias_min": -0.4}
        state.current_best_score = {"fail_rate": 0.2, "mean_abs_delta": 0.15}
        state.current_best_diagnostic = {"fail_rate": 0.2, "per_metric": {}}
        state.completed_iterations = 3
        state.save()

        assert state.state_path.exists()

        loaded = CalibrationState(output_dir=out)
        assert loaded.current_best_overlay == {"persona.sentiment_bias_min": -0.4}
        assert loaded.current_best_score["fail_rate"] == 0.2
        assert loaded.current_best_score["mean_abs_delta"] == 0.15
        assert loaded.current_best_diagnostic["fail_rate"] == 0.2
        assert loaded.completed_iterations == 3

    def test_save_creates_output_dir(self, tmp_path: Path) -> None:
        out = tmp_path / "nested" / "cal_run"
        assert not out.exists()
        state = CalibrationState(output_dir=out)
        state.completed_iterations = 1
        state.save()
        assert out.exists()
        assert state.state_path.exists()

    def test_load_partial_state(self, tmp_path: Path) -> None:
        """Load from a manually written JSON with partial fields."""
        out = tmp_path / "cal_run"
        out.mkdir(parents=True)
        data = {
            "current_best_overlay": {"persona.toxicity_max": 0.5},
            "current_best_score": None,
            "current_best_diagnostic": None,
            "completed_iterations": 7,
        }
        (out / "calibration_state.json").write_text(json.dumps(data), encoding="utf-8")

        loaded = CalibrationState(output_dir=out)
        assert loaded.current_best_overlay == {"persona.toxicity_max": 0.5}
        assert loaded.current_best_score is None
        assert loaded.completed_iterations == 7

    def test_idempotent_save_load(self, tmp_path: Path) -> None:
        """Multiple save/load cycles should be stable."""
        out = tmp_path / "cal_run"
        state = CalibrationState(output_dir=out)
        state.current_best_overlay = {"a": 1}
        state.completed_iterations = 2
        state.save()

        for _ in range(3):
            s = CalibrationState(output_dir=out)
            assert s.current_best_overlay == {"a": 1}
            assert s.completed_iterations == 2
            s.save()

    def test_no_file_on_init(self, tmp_path: Path) -> None:
        """CalibrationState should not create the JSON file on __init__ alone."""
        out = tmp_path / "cal_run"
        CalibrationState(output_dir=out)
        # state file should NOT be written until .save() is called
        assert not (out / "calibration_state.json").exists()
