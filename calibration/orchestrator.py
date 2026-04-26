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
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # type: ignore[assignment,misc]

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
    final_sim_runs: int = 12,
    vanilla_scores_csv: Path | None = None,
    min_sim_threads: int = 0,
) -> dict:
    """Main calibration loop with train/val/test splits.

    Phases
    ------
    Phase 0  Before-calibration group evaluation (vanilla vs real_test).
    Phase 1  Calibration loop (per-thread empirical p-value diagnostics).
    Phase 2  After-calibration group evaluation (calibrated vs real_test).
    Phase 3  Improvement analysis (before vs after).

    Parameters
    ----------
    real_train_csv : Path
        Thread scores CSV for the train split — used by the LLM reasoner.
    real_val_csv : Path
        Thread scores CSV for the validation split — used for candidate scoring.
    real_test_csv : Path
        Thread scores CSV for the test split — used for before/after evaluation.
    vanilla_scores_csv : Path | None
        Pre-existing vanilla simulation scores CSV.  If provided, used as the
        before-calibration baseline for the improvement analysis.  If absent,
        the improvement analysis is skipped.
    final_sim_runs : int
        Number of fresh simulations for the after-calibration evaluation.
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
    if OpenAI is not None:
        client = OpenAI(api_key=api_key, base_url=base_url)
    else:
        # openai < 1.0: store credentials on a simple namespace
        import types
        client = types.SimpleNamespace(api_key=api_key, base_url=base_url)

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
    # Extract train thread IDs for few-shot filtering (no val/test leakage)
    # -----------------------------------------------------------------------
    train_df = pd.read_csv(real_train_csv)
    if "thread_id" in train_df.columns:
        train_thread_ids = train_df["thread_id"].dropna().astype(str).tolist()
        train_ids_path = output_dir / "train_thread_ids.json"
        train_ids_path.write_text(
            json.dumps(train_thread_ids, ensure_ascii=False), encoding="utf-8",
        )
        reference_run_config["few_shot_thread_ids"] = str(train_ids_path)

    # -----------------------------------------------------------------------
    # Phase 0: Before-calibration group evaluation (vanilla vs real_test)
    # -----------------------------------------------------------------------
    before_eval: dict[str, dict] | None = None
    if vanilla_scores_csv is not None:
        before_eval = _run_before_calibration_evaluation(
            vanilla_scores_csv=vanilla_scores_csv,
            real_test_csv=real_test_csv,
            metrics=metrics,
            output_dir=output_dir,
        )

    # -----------------------------------------------------------------------
    # Phase 1: Calibration loop
    # -----------------------------------------------------------------------
    for iteration in range(state.completed_iterations, max_iterations):
        iter_dir = output_dir / f"iter_{iteration:03d}"
        iter_dir.mkdir(parents=True, exist_ok=True)

        # -------------------------------------------------------------------
        # Build candidate overlays
        # -------------------------------------------------------------------
        parsed: dict = {}
        if iteration == 0:
            # Iteration 0: single candidate with default overlay to establish
            # a baseline diagnostic for the reasoner.
            strategy_label = "defaults"
            diagnosis = "Initial baseline run using default knob values."
            overlay_diff: dict = {}
            overlays = [dict(state.current_best_overlay)]
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
        # Run candidates (different overlays share the same seed)
        # -------------------------------------------------------------------
        candidate_results = run_candidates(
            overlays=overlays,
            iter_dir=iter_dir,
            reference_run_config=reference_run_config,
            parallel=parallel,
            python=python,
            repo_root=repo_root,
            device=device,
        )

        # -------------------------------------------------------------------
        # Score candidates against VALIDATION baseline (per-thread empirical p)
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
    # Phase 2: After-calibration group evaluation (calibrated vs real_test)
    # -----------------------------------------------------------------------
    after_eval = _run_after_calibration_evaluation(
        output_dir=output_dir,
        best_overlay=state.current_best_overlay,
        real_test_csv=real_test_csv,
        reference_run_config=reference_run_config,
        sim_runs=final_sim_runs,
        metrics=metrics,
        python=python,
        repo_root=repo_root,
        device=device,
        min_sim_threads=min_sim_threads,
    )

    # -----------------------------------------------------------------------
    # Phase 3: Improvement analysis (before vs after)
    # -----------------------------------------------------------------------
    improvement: dict | None = None
    if before_eval is not None and after_eval.get("group_eval") is not None:
        improvement = compare_before_after(before_eval, after_eval["group_eval"])
        (output_dir / "before_after_improvement_summary.json").write_text(
            json.dumps(improvement, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        _print_improvement_summary(improvement)

    # -----------------------------------------------------------------------
    # Final summary
    # -----------------------------------------------------------------------
    summary = {
        "best_overlay": state.current_best_overlay,
        "best_score": state.current_best_score,
        "completed_iterations": state.completed_iterations,
        "output_dir": str(output_dir),
        "after_calibration_evaluation": {
            k: v for k, v in after_eval.items() if k != "group_eval"
        },
        "improvement": improvement,
    }
    (output_dir / "calibration_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary


# ---------------------------------------------------------------------------
# Phase 0: Before-calibration group evaluation
# ---------------------------------------------------------------------------

def _run_before_calibration_evaluation(
    vanilla_scores_csv: Path,
    real_test_csv: Path,
    metrics: list[str],
    output_dir: Path,
) -> dict[str, dict]:
    """Load pre-existing vanilla scores and evaluate against real_test.

    Returns the output of ``evaluate_group_vs_real()`` (dict[metric -> stats]).
    """
    print(f"\n{'='*60}")
    print("PHASE 0: Before-calibration group evaluation (vanilla vs real_test)")
    print(f"{'='*60}")

    real_test_df = pd.read_csv(real_test_csv)
    vanilla_df = pd.read_csv(vanilla_scores_csv)
    print(f"  real_test threads: {len(real_test_df)}")
    print(f"  vanilla threads:   {len(vanilla_df)}")

    before_eval = evaluate_group_vs_real(real_test_df, vanilla_df, metrics)

    (output_dir / "before_calibration_group_eval.json").write_text(
        json.dumps(before_eval, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"  → Saved before_calibration_group_eval.json")
    return before_eval


# ---------------------------------------------------------------------------
# Phase 2: After-calibration group evaluation
# ---------------------------------------------------------------------------

def _run_after_calibration_evaluation(
    output_dir: Path,
    best_overlay: dict,
    real_test_csv: Path,
    reference_run_config: dict,
    sim_runs: int,
    metrics: list[str],
    python: str,
    repo_root: Path,
    device: str = "cpu",
    min_sim_threads: int = 0,
) -> dict:
    """Generate fresh simulations with best_overlay, evaluate against real_test.

    Each run uses a different seed (base_seed + i) since the overlay is the
    same across all runs.

    Returns a dict with:
      - ``group_eval``: output of evaluate_group_vs_real (for improvement analysis)
      - ``fail_rate``, ``mean_abs_delta``: aggregated per-thread diagnostics
      - ``sim_runs``, ``total_threads``: counts
    """
    final_dir = output_dir / "after_calibration"
    final_dir.mkdir(parents=True, exist_ok=True)

    # Save the overlay used
    save_overlay(best_overlay, final_dir / "overlay.json")

    print(f"\n{'='*60}")
    print(f"PHASE 2: After-calibration evaluation ({sim_runs} fresh sims with best overlay)")
    print(f"{'='*60}")

    # Each run gets a different seed (same overlay → must differ by seed)
    overlays = [dict(best_overlay) for _ in range(sim_runs)]
    seed_offsets = list(range(sim_runs))

    results = run_candidates(
        overlays=overlays,
        iter_dir=final_dir,
        reference_run_config=reference_run_config,
        parallel=1,
        python=python,
        repo_root=repo_root,
        device=device,
        seed_offsets=seed_offsets,
        min_threads=min_sim_threads,
    )

    # Collect all thread metrics from successful runs into one DataFrame
    frames: list[pd.DataFrame] = []
    for result in results:
        if not result["success"] or result["sim_dir"] is None:
            continue
        sim_dir = Path(result["sim_dir"])
        try:
            df = load_thread_metrics(sim_dir)
            df["_run_id"] = result["candidate_id"]
            frames.append(df)
        except Exception:
            pass

    if not frames:
        print("[WARN] No successful after-calibration simulations.")
        return {
            "group_eval": None,
            "fail_rate": None,
            "mean_abs_delta": None,
            "sim_runs": 0,
            "total_threads": 0,
        }

    all_sim_df = pd.concat(frames, ignore_index=True)
    real_test_df = pd.read_csv(real_test_csv)

    print(f"  calibrated threads: {len(all_sim_df)}")
    print(f"  real_test threads:  {len(real_test_df)}")

    # Group-level evaluation: MWU, KS, Cliff's delta per metric
    group_eval = evaluate_group_vs_real(real_test_df, all_sim_df, metrics)

    (final_dir / "after_calibration_group_eval.json").write_text(
        json.dumps(group_eval, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Also compute aggregate per-thread empirical diagnostics
    test_baseline = compute_baseline_from_csv(real_test_csv, metrics)
    all_fail_rates: list[float] = []
    all_abs_deltas: list[float] = []
    for result in results:
        if not result["success"] or result["sim_dir"] is None:
            continue
        sim_dir = Path(result["sim_dir"])
        try:
            sc = score_candidate(sim_dir, test_baseline, metrics)
            all_fail_rates.append(sc["fail_rate"])
            all_abs_deltas.append(sc["mean_abs_delta"])
        except Exception:
            pass

    after_result = {
        "group_eval": group_eval,
        "fail_rate": float(np.mean(all_fail_rates)) if all_fail_rates else None,
        "mean_abs_delta": float(np.mean(all_abs_deltas)) if all_abs_deltas else None,
        "sim_runs": len(frames),
        "total_threads": len(all_sim_df),
    }

    (final_dir / "after_calibration_evaluation.json").write_text(
        json.dumps(
            {k: v for k, v in after_result.items() if k != "group_eval"},
            indent=2, ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(f"  avg fail rate:    {after_result['fail_rate']:.4f}" if after_result["fail_rate"] else "")
    print(f"  avg |delta|:      {after_result['mean_abs_delta']:.4f}" if after_result["mean_abs_delta"] else "")
    print(f"  successful runs:  {after_result['sim_runs']}/{sim_runs}")
    print(f"  → Saved after_calibration_group_eval.json")

    return after_result


# ---------------------------------------------------------------------------
# Improvement summary printer
# ---------------------------------------------------------------------------

def _print_improvement_summary(improvement: dict) -> None:
    """Print a concise terminal summary of before vs after improvement."""
    s = improvement.get("summary", {})

    print(f"\n{'='*60}")
    print("IMPROVEMENT ANALYSIS (before vs after calibration)")
    print(f"{'='*60}")
    print(f"  Metrics sig. different before: {s.get('metrics_sig_different_before', '?')}")
    print(f"  Metrics sig. different after:  {s.get('metrics_sig_different_after', '?')}")
    print(f"  Avg |Cliff's delta| before:    {s.get('avg_abs_cliffs_delta_before', 0):.4f}")
    print(f"  Avg |Cliff's delta| after:     {s.get('avg_abs_cliffs_delta_after', 0):.4f}")
    print(f"  Overall fail rate before:      {s.get('overall_fail_rate_before', 0):.4f}")
    print(f"  Overall fail rate after:       {s.get('overall_fail_rate_after', 0):.4f}")
    print(f"  Overall pass rate before:      {s.get('overall_pass_rate_before', 0):.4f}")
    print(f"  Overall pass rate after:       {s.get('overall_pass_rate_after', 0):.4f}")

    pm = improvement.get("per_metric", {})
    improved_count = sum(1 for v in pm.values() if v.get("improved"))
    print(f"\n  Metrics improved: {improved_count}/{len(pm)}")
