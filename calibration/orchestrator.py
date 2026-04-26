"""
Orchestrator for the calibration system.

Components
----------
CalibrationState      : Persistent state for resume support.
run_calibration_loop  : Main calibration loop.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from openai import OpenAI

from .log import CalibrationLog
from .overlay import diff_overlay, merge_overlay, save_overlay
from .reasoner import (
    build_reasoner_prompt,
    call_reasoner,
    generate_variants,
    parse_reasoner_response,
)
from .registry import KnobRegistry
from .runner import run_candidates
from .scorer import (
    DEFAULT_METRICS,
    compute_real_baseline,
    load_thread_metrics,
    score_candidate,
    select_best_candidate,
)
from .stats import compare_before_after, evaluate_group_vs_real


# ---------------------------------------------------------------------------
# CalibrationState
# ---------------------------------------------------------------------------

class CalibrationState:
    """Persistent state for a calibration run, with resume support.

    The state is serialised to ``output_dir/calibration_state.json``.
    If that file already exists on construction the state is loaded from it,
    enabling resumption of a previously interrupted run.

    Attributes
    ----------
    output_dir : Path
    state_path : Path
    current_best_overlay : dict
    current_best_score : dict | None
    current_best_diagnostic : dict | None
    completed_iterations : int
    """

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = Path(output_dir)
        self.state_path = self.output_dir / "calibration_state.json"
        self.current_best_overlay: dict = {}
        self.current_best_score: dict | None = None
        self.current_best_diagnostic: dict | None = None
        self.completed_iterations: int = 0

        if self.state_path.exists():
            self._load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save(self) -> None:
        """Write current state to JSON."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        data = {
            "current_best_overlay": self.current_best_overlay,
            "current_best_score": self.current_best_score,
            "current_best_diagnostic": self.current_best_diagnostic,
            "completed_iterations": self.completed_iterations,
        }
        self.state_path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Read state from JSON."""
        raw = json.loads(self.state_path.read_text(encoding="utf-8"))
        self.current_best_overlay = raw.get("current_best_overlay", {})
        self.current_best_score = raw.get("current_best_score")
        self.current_best_diagnostic = raw.get("current_best_diagnostic")
        self.completed_iterations = raw.get("completed_iterations", 0)


# ---------------------------------------------------------------------------
# run_calibration_loop
# ---------------------------------------------------------------------------

def run_calibration_loop(
    output_dir: Path,
    real_dir: Path,
    reference_run_config: dict,
    max_iterations: int = 10,
    candidates_per_iter: int = 5,
    parallel: int = 1,
    calibration_model: str = "gpt-4o-mini",
    api_key: str = "",
    base_url: str = "https://api.openai.com/v1",
    seed: int = 42,
    python: str = sys.executable,
    repo_root: Path | None = None,
    metrics: list[str] | None = None,
    metric_definitions: str = "",
    device: str = "cpu",
) -> dict:
    """Main calibration loop.

    Parameters
    ----------
    output_dir : Path
        Root directory for all calibration artefacts.
    real_dir : Path
        Directory containing the real ``thread_metrics_summary.csv``.
    reference_run_config : dict
        Baseline run parameters forwarded to ``run_discussion.py``.
    max_iterations : int
        Maximum number of calibration iterations.
    candidates_per_iter : int
        Number of candidate overlays to evaluate per iteration (iteration 0
        runs this many copies of the default overlay).
    parallel : int
        Worker count for candidate simulation.
    calibration_model : str
        OpenAI model for the LLM reasoner.
    api_key : str
        OpenAI API key.
    base_url : str
        OpenAI API base URL.
    seed : int
        Random seed for reproducibility.
    python : str
        Python executable to use for subprocesses.
    repo_root : Path | None
        Repository root (inferred if None).
    metrics : list[str] | None
        Metrics to evaluate. Defaults to ``DEFAULT_METRICS``.
    metric_definitions : str
        Plain-text descriptions of each metric for the LLM prompt.
    device : str
        Device hint (e.g. ``"cpu"`` / ``"cuda"``).

    Returns
    -------
    dict
        Calibration summary with keys: ``best_overlay``, ``best_score``,
        ``completed_iterations``, ``output_dir``.
    """
    output_dir = Path(output_dir)
    real_dir = Path(real_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if metrics is None:
        metrics = DEFAULT_METRICS

    if repo_root is None:
        repo_root = Path(__file__).parent.parent

    # -----------------------------------------------------------------------
    # Initialise components
    # -----------------------------------------------------------------------
    registry = KnobRegistry()
    log = CalibrationLog(output_dir / "calibration_log.json")
    state = CalibrationState(output_dir=output_dir)
    client = OpenAI(api_key=api_key, base_url=base_url)

    # -----------------------------------------------------------------------
    # Compute real baseline
    # -----------------------------------------------------------------------
    real_baseline = compute_real_baseline(real_dir, metrics)

    # Save baseline medians for reference
    baseline_medians = {m: v["median"] for m, v in real_baseline.items()}
    (output_dir / "real_baseline_metrics.json").write_text(
        json.dumps(baseline_medians, indent=2),
        encoding="utf-8",
    )

    # -----------------------------------------------------------------------
    # Main loop
    # -----------------------------------------------------------------------
    for iteration in range(state.completed_iterations, max_iterations):
        iter_dir = output_dir / f"iter_{iteration:03d}"
        iter_dir.mkdir(parents=True, exist_ok=True)

        # -------------------------------------------------------------------
        # Build candidate overlays
        # -------------------------------------------------------------------
        if iteration == 0:
            # Default overlays — no LLM call
            strategy_label = "defaults"
            diagnosis = "Initial baseline run using default knob values."
            overlay_diff: dict = {}
            overlays = [dict(state.current_best_overlay) for _ in range(candidates_per_iter)]
        else:
            # Build prompt from current best state
            prompt = build_reasoner_prompt(
                registry=registry,
                current_overlay=state.current_best_overlay,
                current_diagnostic=state.current_best_diagnostic or {},
                real_baseline=baseline_medians,
                trajectory=log.trajectory(),
                failed_strategies=log.failed_strategies(),
                metric_definitions=metric_definitions,
            )

            raw_response = call_reasoner(client, calibration_model, prompt)
            parsed = parse_reasoner_response(raw_response)

            strategy_label = parsed["strategy_label"]
            diagnosis = parsed["diagnosis"]
            overlay_diff = parsed["overlay_diff"]

            overlays = generate_variants(
                current_overlay=state.current_best_overlay,
                base_diff=overlay_diff,
                prompt_alternatives=parsed.get("prompt_alternatives", {}),
                registry=registry,
                seed=seed + iteration,
            )

        # Save diagnosis for this iteration
        (iter_dir / "diagnosis.json").write_text(
            json.dumps(
                {
                    "iteration": iteration,
                    "strategy_label": strategy_label,
                    "diagnosis": diagnosis,
                    "overlay_diff": overlay_diff,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        # -------------------------------------------------------------------
        # Run candidates
        # -------------------------------------------------------------------
        candidate_results = run_candidates(
            overlays=overlays,
            iter_dir=iter_dir,
            reference_run_config=reference_run_config,
            parallel=parallel,
            python=python,
            repo_root=repo_root,
        )

        # -------------------------------------------------------------------
        # Score successful candidates
        # -------------------------------------------------------------------
        scored: list[dict] = []
        for result in candidate_results:
            if not result["success"] or result["sim_dir"] is None:
                continue
            sim_dir = Path(result["sim_dir"])
            try:
                sc = score_candidate(sim_dir, real_baseline, metrics)
                sc["candidate_id"] = result["candidate_id"]
                sc["candidate_dir"] = result["candidate_dir"]
                sc["overlay"] = overlays[result["candidate_id"]]
                scored.append(sc)
            except Exception:
                # Missing CSV or other scoring error — skip
                pass

        # -------------------------------------------------------------------
        # Select best candidate this iteration
        # -------------------------------------------------------------------
        if scored:
            winner = select_best_candidate(scored)
        else:
            winner = None

        # -------------------------------------------------------------------
        # Check if winner beats current best
        # -------------------------------------------------------------------
        beat_current_best = False
        if winner is not None:
            if state.current_best_score is None:
                beat_current_best = True
            else:
                prev_fail = state.current_best_score["fail_rate"]
                prev_delta = state.current_best_score["mean_abs_delta"]
                new_fail = winner["fail_rate"]
                new_delta = winner["mean_abs_delta"]
                if (new_fail, new_delta) < (prev_fail, prev_delta):
                    beat_current_best = True

            if beat_current_best:
                state.current_best_overlay = winner.get("overlay", {})
                state.current_best_score = {
                    "fail_rate": winner["fail_rate"],
                    "mean_abs_delta": winner["mean_abs_delta"],
                }
                state.current_best_diagnostic = {
                    k: v for k, v in winner.items()
                    if k not in ("candidate_id", "candidate_dir", "overlay")
                }

        # -------------------------------------------------------------------
        # Prepare log entry (candidates without thread-level data to keep log small)
        # -------------------------------------------------------------------
        def _slim_candidate(c: dict) -> dict:
            """Strip 'threads' from per_metric dicts to keep log small."""
            slim = {k: v for k, v in c.items() if k != "per_metric"}
            slim["per_metric_summary"] = {
                m: {sk: sv for sk, sv in md.items() if sk != "threads"}
                for m, md in c.get("per_metric", {}).items()
            }
            return slim

        slim_candidates = [_slim_candidate(c) for c in scored]

        best_fail_rate = winner["fail_rate"] if winner else None
        best_delta = winner["mean_abs_delta"] if winner else None

        log_entry: dict[str, Any] = {
            "iteration": iteration,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "strategy_label": strategy_label,
            "diagnosis": diagnosis,
            "overlay_diff": overlay_diff,
            "candidates": slim_candidates,
            "selection": {
                "winner_candidate_id": winner["candidate_id"] if winner else None,
                "beat_current_best": beat_current_best,
                "best_fail_rate": best_fail_rate,
                "best_mean_abs_delta": best_delta,
            },
        }
        log.append(log_entry)

        # -------------------------------------------------------------------
        # Advance state
        # -------------------------------------------------------------------
        state.completed_iterations = iteration + 1
        state.save()

    # -----------------------------------------------------------------------
    # Export final artefacts
    # -----------------------------------------------------------------------
    save_overlay(state.current_best_overlay, output_dir / "best_overlay.json")

    summary = {
        "best_overlay": state.current_best_overlay,
        "best_score": state.current_best_score,
        "completed_iterations": state.completed_iterations,
        "output_dir": str(output_dir),
    }
    (output_dir / "calibration_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return summary
