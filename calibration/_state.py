"""CalibrationState: persistent state for resume support.

Extracted from orchestrator.py.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# CalibrationState
# ---------------------------------------------------------------------------

class CalibrationState:
    """Persistent state for a calibration run, with resume support.

    The state is serialised to ``output_dir/calibration_state.json``.
    """

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = Path(output_dir)
        self.state_path = self.output_dir / "calibration_state.json"
        self.current_best_overlay: dict = {}
        self.current_best_score: dict | None = None
        self.current_best_diagnostic: dict | None = None
        self.current_best_candidate_dir: str | None = None
        self.current_search_root_overlay: dict = {}
        self.current_search_root_diagnostic: dict | None = None
        self.current_search_root_candidate_dir: str | None = None
        self.current_search_root_mode: str = "global_best"
        self.current_search_root_reason: str = "global_best"
        self.frontier: dict[str, dict[str, Any]] = {}
        self.stagnation_count: int = 0
        self.completed_iterations: int = 0
        self.current_phase_name: str | None = None
        self.completed_phase_summaries: list[dict[str, Any]] = []
        self.manual_block_phase_name: str | None = None
        self.manual_block_best_overlay: dict = {}
        self.manual_block_best_score: dict | None = None
        self.manual_block_best_diagnostic: dict | None = None
        self.manual_block_best_candidate_dir: str | None = None

        if self.state_path.exists():
            self._load()

    def save(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        data = {
            "current_best_overlay": self.current_best_overlay,
            "current_best_score": self.current_best_score,
            "current_best_diagnostic": self.current_best_diagnostic,
            "current_best_candidate_dir": self.current_best_candidate_dir,
            "current_search_root_overlay": self.current_search_root_overlay,
            "current_search_root_diagnostic": self.current_search_root_diagnostic,
            "current_search_root_candidate_dir": self.current_search_root_candidate_dir,
            "current_search_root_mode": self.current_search_root_mode,
            "current_search_root_reason": self.current_search_root_reason,
            "frontier": self.frontier,
            "stagnation_count": self.stagnation_count,
            "completed_iterations": self.completed_iterations,
            "current_phase_name": self.current_phase_name,
            "completed_phase_summaries": self.completed_phase_summaries,
            "manual_block_phase_name": self.manual_block_phase_name,
            "manual_block_best_overlay": self.manual_block_best_overlay,
            "manual_block_best_score": self.manual_block_best_score,
            "manual_block_best_diagnostic": self.manual_block_best_diagnostic,
            "manual_block_best_candidate_dir": self.manual_block_best_candidate_dir,
        }
        self.state_path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _load(self) -> None:
        raw = json.loads(self.state_path.read_text(encoding="utf-8"))
        self.current_best_overlay = render_structured_overlay(raw.get("current_best_overlay", {}))
        self.current_best_score = raw.get("current_best_score")
        self.current_best_diagnostic = raw.get("current_best_diagnostic")
        self.current_best_candidate_dir = raw.get("current_best_candidate_dir")
        self.current_search_root_overlay = render_structured_overlay(raw.get(
            "current_search_root_overlay",
            self.current_best_overlay,
        ))
        self.current_search_root_diagnostic = raw.get(
            "current_search_root_diagnostic",
            self.current_best_diagnostic,
        )
        self.current_search_root_candidate_dir = raw.get(
            "current_search_root_candidate_dir",
            self.current_best_candidate_dir,
        )
        self.current_search_root_mode = raw.get("current_search_root_mode", "global_best")
        self.current_search_root_reason = raw.get("current_search_root_reason", "global_best")
        self.frontier = raw.get("frontier", {})
        self.stagnation_count = raw.get("stagnation_count", 0)
        self.completed_iterations = raw.get("completed_iterations", 0)
        self.current_phase_name = raw.get("current_phase_name")
        self.completed_phase_summaries = raw.get("completed_phase_summaries", [])
        self.manual_block_phase_name = raw.get("manual_block_phase_name")
        self.manual_block_best_overlay = render_structured_overlay(raw.get(
            "manual_block_best_overlay",
            self.current_best_overlay,
        ))
        self.manual_block_best_score = raw.get(
            "manual_block_best_score",
            self.current_best_score,
        )
        self.manual_block_best_diagnostic = raw.get(
            "manual_block_best_diagnostic",
            self.current_best_diagnostic,
        )
        self.manual_block_best_candidate_dir = raw.get(
            "manual_block_best_candidate_dir",
            self.current_best_candidate_dir,
        )


