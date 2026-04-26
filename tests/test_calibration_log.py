"""Tests for calibration/log.py — CalibrationLog."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from calibration.log import CalibrationLog


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _make_entry(strategy_label: str, beat: bool, best_fail_rate: float = 0.3) -> dict:
    """Build a minimal log entry matching the expected schema."""
    return {
        "iteration": 1,
        "strategy_label": strategy_label,
        "best_fail_rate": best_fail_rate,
        "selection": {"beat_current_best": beat},
        "candidates": [{"prompt": "p1"}, {"prompt": "p2"}],
    }


# ---------------------------------------------------------------------------
# new_log_is_empty
# ---------------------------------------------------------------------------

def test_new_log_is_empty(tmp_path: Path) -> None:
    log = CalibrationLog(tmp_path / "log.json")
    assert log.entries() == []
    assert log.failed_strategies() == []
    assert log.trajectory() == []


# ---------------------------------------------------------------------------
# append_and_read
# ---------------------------------------------------------------------------

def test_append_and_read(tmp_path: Path) -> None:
    log = CalibrationLog(tmp_path / "log.json")
    entry = _make_entry("strategy_A", beat=True)
    log.append(entry)

    entries = log.entries()
    assert len(entries) == 1
    assert entries[0]["strategy_label"] == "strategy_A"
    assert entries[0]["selection"]["beat_current_best"] is True

    # Second append
    log.append(_make_entry("strategy_B", beat=False))
    assert len(log.entries()) == 2


# ---------------------------------------------------------------------------
# persistence
# ---------------------------------------------------------------------------

def test_persistence(tmp_path: Path) -> None:
    path = tmp_path / "log.json"

    # Write with first instance
    log1 = CalibrationLog(path)
    log1.append(_make_entry("strategy_X", beat=True, best_fail_rate=0.2))
    log1.append(_make_entry("strategy_Y", beat=False, best_fail_rate=0.4))

    # Read with second instance
    log2 = CalibrationLog(path)
    entries = log2.entries()
    assert len(entries) == 2
    assert entries[0]["strategy_label"] == "strategy_X"
    assert entries[1]["strategy_label"] == "strategy_Y"


# ---------------------------------------------------------------------------
# failed_strategies
# ---------------------------------------------------------------------------

def test_failed_strategies(tmp_path: Path) -> None:
    log = CalibrationLog(tmp_path / "log.json")
    log.append(_make_entry("good_strategy", beat=True))
    log.append(_make_entry("bad_strategy_1", beat=False))
    log.append(_make_entry("bad_strategy_2", beat=False))
    log.append(_make_entry("another_good", beat=True))

    failed = log.failed_strategies()
    assert "bad_strategy_1" in failed
    assert "bad_strategy_2" in failed
    assert "good_strategy" not in failed
    assert "another_good" not in failed
    assert len(failed) == 2


# ---------------------------------------------------------------------------
# trajectory
# ---------------------------------------------------------------------------

def test_trajectory(tmp_path: Path) -> None:
    log = CalibrationLog(tmp_path / "log.json")
    log.append(_make_entry("s1", beat=True, best_fail_rate=0.25))
    log.append(_make_entry("s2", beat=False, best_fail_rate=0.35))

    traj = log.trajectory()
    assert len(traj) == 2

    for t_entry in traj:
        # candidates key must be stripped
        assert "candidates" not in t_entry
        # other fields must be present
        assert "best_fail_rate" in t_entry
        assert "strategy_label" in t_entry

    assert traj[0]["best_fail_rate"] == 0.25
    assert traj[1]["best_fail_rate"] == 0.35


# ---------------------------------------------------------------------------
# entries() returns a copy (mutation safety)
# ---------------------------------------------------------------------------

def test_entries_returns_copy(tmp_path: Path) -> None:
    log = CalibrationLog(tmp_path / "log.json")
    log.append(_make_entry("s1", beat=True))

    entries = log.entries()
    entries.clear()  # mutate the returned list

    # Original log should be unaffected
    assert len(log.entries()) == 1
