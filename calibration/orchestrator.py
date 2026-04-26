"""
Orchestrator for the calibration system.

Components
----------
CalibrationState      : Persistent state for resume support.
run_calibration_loop  : Main calibration loop.
"""
from __future__ import annotations

import json
import subprocess
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
    compute_baseline_from_csv,
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

    def save(self) -> None:
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

    def _load(self) -> None:
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
    real_train_csv: Path,
    real_val_csv: Path,
    real_test_csv: Path,
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
    final_sim_runs: int = 25,
) -> dict:
    """Main calibration loop with train/val/test splits.

    Parameters
    ----------
    output_dir : Path
        Root directory for all calibration artefacts.
    real_train_csv : Path
        Thread scores CSV for the train split — used by the LLM reasoner to
        build baseline context (medians) and diagnostics.
    real_val_csv : Path
        Thread scores CSV for the validation split — used to score candidates
        and select the best overlay each iteration.
    real_test_csv : Path
        Thread scores CSV for the test split — used only for the final
        post-calibration evaluation.
    reference_run_config : dict
        Simulation parameters forwarded to ``run_discussion.py``
        (keys: ``input_file``, ``agents``, ``hours``, ``rounds``,
        ``seed_posts``, ``seed``, ``hint``, ``discussion_backbone``,
        ``few_shot_source``, ``few_shot_count``).
    final_sim_runs : int
        Number of fresh simulations for the final test evaluation.

    Returns
    -------
    dict with keys: ``best_overlay``, ``best_score``, ``completed_iterations``,
    ``output_dir``, ``final_evaluation``.
    """
    output_dir = Path(output_dir)
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
    # Compute baselines from train and val splits
    # -----------------------------------------------------------------------
    train_baseline = compute_baseline_from_csv(real_train_csv, metrics)
    val_baseline = compute_baseline_from_csv(real_val_csv, metrics)

    # Save train medians (used by reasoner)
    train_medians = {m: v["median"] for m, v in train_baseline.items()}
    (output_dir / "real_train_baseline_metrics.json").write_text(
        json.dumps(train_medians, indent=2), encoding="utf-8",
    )
    val_medians = {m: v["median"] for m, v in val_baseline.items()}
    (output_dir / "real_val_baseline_metrics.json").write_text(
        json.dumps(val_medians, indent=2), encoding="utf-8",
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
        parsed: dict = {}
        if iteration == 0:
            strategy_label = "defaults"
            diagnosis = "Initial baseline run using default knob values."
            overlay_diff: dict = {}
            overlays = [dict(state.current_best_overlay) for _ in range(candidates_per_iter)]
        else:
            # Build prompt — uses TRAIN baseline for context
            prompt = build_reasoner_prompt(
                registry=registry,
                current_overlay=state.current_best_overlay,
                current_diagnostic=state.current_best_diagnostic or {},
                real_baseline=train_medians,
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
                conservative_diff=parsed.get("conservative_diff"),
            )

        # Save diagnosis
        (iter_dir / "diagnosis.json").write_text(
            json.dumps({
                "iteration": iteration,
                "strategy_label": strategy_label,
                "diagnosis": diagnosis,
                "overlay_diff": overlay_diff,
            }, indent=2),
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
        # Score candidates against VALIDATION baseline
        # -------------------------------------------------------------------
        scored: list[dict] = []
        for result in candidate_results:
            if not result["success"] or result["sim_dir"] is None:
                continue
            sim_dir = Path(result["sim_dir"])
            try:
                sc = score_candidate(sim_dir, val_baseline, metrics)
                sc["candidate_id"] = result["candidate_id"]
                sc["candidate_dir"] = result["candidate_dir"]
                sc["overlay"] = overlays[result["candidate_id"]]
                scored.append(sc)
            except Exception:
                pass

        # -------------------------------------------------------------------
        # Select best candidate
        # -------------------------------------------------------------------
        winner = select_best_candidate(scored) if scored else None

        # -------------------------------------------------------------------
        # Check if winner beats current best
        # -------------------------------------------------------------------
        beat_current_best = False
        if winner is not None:
            if state.current_best_score is None:
                beat_current_best = True
            else:
                prev = (state.current_best_score["fail_rate"],
                        state.current_best_score["mean_abs_delta"])
                new = (winner["fail_rate"], winner["mean_abs_delta"])
                if new < prev:
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
        # Log entry
        # -------------------------------------------------------------------
        def _slim(c: dict) -> dict:
            slim = {k: v for k, v in c.items() if k != "per_metric"}
            slim["per_metric_summary"] = {
                m: {sk: sv for sk, sv in md.items() if sk != "threads"}
                for m, md in c.get("per_metric", {}).items()
            }
            return slim

        log.append({
            "iteration": iteration,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "strategy_label": strategy_label,
            "primary_layer": parsed.get("primary_layer", ""),
            "diagnosis": diagnosis,
            "overlay_diff": overlay_diff,
            "candidate_rationale": parsed.get("candidate_rationale", []),
            "candidates": [_slim(c) for c in scored],
            "selection": {
                "winner_candidate_id": winner["candidate_id"] if winner else None,
                "beat_current_best": beat_current_best,
                "best_fail_rate": winner["fail_rate"] if winner else None,
                "best_mean_abs_delta": winner["mean_abs_delta"] if winner else None,
            },
        })

        state.completed_iterations = iteration + 1
        state.save()

    # -----------------------------------------------------------------------
    # Export best overlay
    # -----------------------------------------------------------------------
    save_overlay(state.current_best_overlay, output_dir / "best_overlay.json")

    # -----------------------------------------------------------------------
    # Final evaluation on TEST split
    # -----------------------------------------------------------------------
    final_eval = _run_final_evaluation(
        output_dir=output_dir,
        best_overlay=state.current_best_overlay,
        real_test_csv=real_test_csv,
        reference_run_config=reference_run_config,
        sim_runs=final_sim_runs,
        metrics=metrics,
        python=python,
        repo_root=repo_root,
    )

    summary = {
        "best_overlay": state.current_best_overlay,
        "best_score": state.current_best_score,
        "completed_iterations": state.completed_iterations,
        "output_dir": str(output_dir),
        "final_evaluation": final_eval,
    }
    (output_dir / "calibration_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary


# ---------------------------------------------------------------------------
# Final post-calibration evaluation
# ---------------------------------------------------------------------------

def _run_final_evaluation(
    output_dir: Path,
    best_overlay: dict,
    real_test_csv: Path,
    reference_run_config: dict,
    sim_runs: int,
    metrics: list[str],
    python: str,
    repo_root: Path,
) -> dict:
    """Generate fresh simulations with best_overlay, score against test split.

    Returns a dict with keys: fail_rate, mean_abs_delta, per_metric, sim_runs.
    """
    final_dir = output_dir / "final_evaluation"
    final_dir.mkdir(parents=True, exist_ok=True)

    # Save the overlay used
    overlay_path = final_dir / "overlay.json"
    save_overlay(best_overlay, overlay_path)

    # Build test baseline
    test_baseline = compute_baseline_from_csv(real_test_csv, metrics)

    # Run fresh simulations
    print(f"\n{'='*60}")
    print(f"FINAL EVALUATION: Generating {sim_runs} fresh simulations with best overlay")
    print(f"{'='*60}")

    overlays = [dict(best_overlay) for _ in range(sim_runs)]
    results = run_candidates(
        overlays=overlays,
        iter_dir=final_dir,
        reference_run_config=reference_run_config,
        parallel=1,
        python=python,
        repo_root=repo_root,
    )

    # Score all successful runs against test baseline
    scored: list[dict] = []
    for result in results:
        if not result["success"] or result["sim_dir"] is None:
            continue
        sim_dir = Path(result["sim_dir"])
        try:
            sc = score_candidate(sim_dir, test_baseline, metrics)
            sc["candidate_id"] = result["candidate_id"]
            scored.append(sc)
        except Exception:
            pass

    if not scored:
        print("[WARN] No successful final simulations to evaluate.")
        return {"fail_rate": None, "mean_abs_delta": None, "sim_runs": 0}

    # Aggregate across all scored runs
    all_fail_rates = [s["fail_rate"] for s in scored]
    all_abs_deltas = [s["mean_abs_delta"] for s in scored]

    final_result = {
        "fail_rate": float(np.mean(all_fail_rates)),
        "mean_abs_delta": float(np.mean(all_abs_deltas)),
        "sim_runs": len(scored),
        "per_run": [
            {"candidate_id": s["candidate_id"],
             "fail_rate": s["fail_rate"],
             "mean_abs_delta": s["mean_abs_delta"]}
            for s in scored
        ],
    }

    (final_dir / "test_evaluation.json").write_text(
        json.dumps(final_result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Final test fail rate:   {final_result['fail_rate']:.4f}")
    print(f"Final test |delta|:     {final_result['mean_abs_delta']:.4f}")
    print(f"Successful runs:        {final_result['sim_runs']}/{sim_runs}")

    return final_result
