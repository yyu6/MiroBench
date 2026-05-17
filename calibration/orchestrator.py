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
from .overlay import (
    STRUCTURED_PHASE_BLOCKS_KEY,
    diff_overlay,
    merge_overlay,
    render_structured_overlay,
    save_overlay,
)
from .reasoner import (
    build_dedup_prompt,
    build_reasoner_prompt,
    build_text_materializer_prompt,
    call_reasoner,
    generate_variants,
    materializer_response_format,
    parse_reasoner_response,
    parse_text_materializer_response,
)
from .registry import KnobRegistry
from .runner import run_candidates
from .scorer import (
    DEFAULT_METRICS,
    PRIMARY_CALIBRATION_METRICS,
    candidate_selection_key,
    compute_baseline_from_csv,
    load_thread_metrics,
    score_candidate,
    select_best_candidate,
)
from .stats import compare_before_after, evaluate_group_vs_real

import math as _math

# ---------------------------------------------------------------------------
# Phase specs and metric guidance constants
# (moved to calibration/_phase_specs.py to keep this file under 1500 lines)
# ---------------------------------------------------------------------------
from ._phase_specs import (
    _HEADLINE_METRICS,
    _TARGET_METRICS_12,
    _STAGNATION_TRIGGER,
    _MANUAL_PHASE_MODE,
    _MANUAL_METRIC_GUIDANCE,
    _MANUAL_PHASE_SPECS,
    _DEDUP_ITERATION,
    _DEDUP_PHASE_SPEC,
)


# ---------------------------------------------------------------------------
# Manual-phase helpers (moved to calibration/_manual_phase.py)
# ---------------------------------------------------------------------------
from ._manual_phase import (
    _manual_total_edited_iterations,
    _is_dedup_iteration,
    _parse_dedup_response,
    _manual_phase_for_iteration,
    _manual_phase_context,
    _phase1_total_iterations,
    _phase1_reported_iteration_count,
    _manual_phase_prompt_trajectory,
    _manual_block_reference,
    _manual_start_block,
    _manual_commit_block_best,
    _subset_robust_stats,
    _subset_group_eval_stats,
    _manual_phase_metric_rows,
    _target_metric_eval_summary,
    _manual_guard_threshold,
    _manual_phase_guard_summary,
    _manual_metric_row_key,
    _manual_phase_score,
    _manual_phase_selection_key,
)




# ---------------------------------------------------------------------------
# Display/printing helpers (moved to calibration/_display.py)
# ---------------------------------------------------------------------------
from ._display import (
    _fmt,
    _fmt_signed,
    _print_candidate_score_summary,
    _print_group_eval_summary,
    _print_improvement_table,
    _selection_ranking_rows,
    _print_selection_ranking,
    _manual_phase_ranking_rows,
    _print_manual_phase_selection_ranking,
    _serialize_metric_rows,
    _print_phase_watch_metrics,
    _print_winner_selection_breakdown,
)


# ---------------------------------------------------------------------------
# Selection/frontier helpers (moved to calibration/_selection.py)
# ---------------------------------------------------------------------------
from ._selection import (
    _group_eval_selection_summary,
    _candidate_selection_key,
    _current_best_selection_reference,
    _group_score_key,
    _group_severity_key,
    _worst_group_order,
    _stagnation_count_from_entries,
    _slim_candidate_diagnostic,
    _make_frontier_entry,
    _update_frontier,
    _frontier_prompt_summary,
    _choose_search_root,
)


# ---------------------------------------------------------------------------
# Misc helpers (moved to calibration/_helpers.py)
# ---------------------------------------------------------------------------
from ._helpers import (
    _sanitize_overlay,
    _composite_thread_key,
    _format_terminal_value,
    _maybe_record_completed_phase_summary,
    _completed_phase_prompt_summary,
    _knob_runtime_location,
    _overlay_change_records,
    _print_candidate_change_preview,
    _extract_sample_real_thread,
    _extract_sample_sim_thread,
    _find_reusable_vanilla_sim_dir,
    _resolve_eval_thread_target,
    _make_reused_baseline_candidate_result,
    _load_iteration_checkpoint,
)


# ---------------------------------------------------------------------------
# CalibrationState (moved to calibration/_state.py)
# ---------------------------------------------------------------------------
from ._state import CalibrationState


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
    rerun_phase0_vanilla: bool = False,
    min_sim_threads: int = 0,
    metric_parallel: int = 2,
    calibration_reasoning_effort: str | None = None,
    simulation_reasoning_effort: str | None = None,
    stop_after_phase1: bool = False,
    calibration_rounds: int | None = None,
    combination_start_iteration: int | None = None,
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

    if _MANUAL_PHASE_MODE:
        manual_iterations = _manual_total_edited_iterations()
        if int(max_iterations) != manual_iterations:
            print(
                f"  [manual phase] overriding edited iterations "
                f"{int(max_iterations)} → {manual_iterations}",
                flush=True,
            )
        max_iterations = manual_iterations

    if metrics is None:
        metrics = DEFAULT_METRICS
    ranking_metrics = PRIMARY_CALIBRATION_METRICS
    if combination_start_iteration is None:
        combination_start_iteration = max_iterations // 2
    combination_start_iteration = max(1, min(int(combination_start_iteration), max_iterations))
    if repo_root is None:
        repo_root = Path(__file__).parent.parent

    # -----------------------------------------------------------------------
    # Initialise components
    # -----------------------------------------------------------------------
    registry = KnobRegistry()
    log = CalibrationLog(output_dir / "calibration_log.json")
    state = CalibrationState(output_dir=output_dir)
    sanitized_best_overlay, best_overlay_errors = _sanitize_overlay(
        registry, state.current_best_overlay
    )
    if best_overlay_errors:
        print("  [overlay validation] Sanitized persisted best overlay:")
        for err in best_overlay_errors:
            print(f"    - {err}")
        state.current_best_overlay = sanitized_best_overlay
        state.save()
    if state.current_best_overlay and not state.current_search_root_overlay:
        state.current_search_root_overlay = dict(state.current_best_overlay)
        state.current_search_root_diagnostic = state.current_best_diagnostic
        state.current_search_root_candidate_dir = state.current_best_candidate_dir
        state.current_search_root_mode = "global_best"
        state.current_search_root_reason = "global_best"
        state.save()
    if OpenAI is not None:
        client = OpenAI(api_key=api_key, base_url=base_url)
    else:
        # openai < 1.0: store credentials on a simple namespace
        import types
        client = types.SimpleNamespace(api_key=api_key, base_url=base_url)

    # -----------------------------------------------------------------------
    # Compute baselines from train and val splits
    # -----------------------------------------------------------------------
    val_df = pd.read_csv(real_val_csv)
    real_test_df = pd.read_csv(real_test_csv)
    train_baseline = compute_baseline_from_csv(real_train_csv, metrics)
    val_baseline = compute_baseline_from_csv(real_val_csv, metrics)
    before_generated_df = pd.read_csv(vanilla_scores_csv) if vanilla_scores_csv is not None else None
    reusable_vanilla_sim_dir = _find_reusable_vanilla_sim_dir(vanilla_scores_csv)

    def _baseline_summary(baseline: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
        summary: dict[str, dict[str, Any]] = {}
        for m, v in baseline.items():
            arr = np.asarray(v["values"], dtype=float)
            summary[m] = {
                "median": v["median"],
                "mean": v["mean"],
                "std": float(np.std(arr)) if arr.size > 0 else 0.0,
                "p10": float(np.percentile(arr, 10)) if arr.size > 0 else float("nan"),
                "p25": float(np.percentile(arr, 25)) if arr.size > 0 else float("nan"),
                "p75": float(np.percentile(arr, 75)) if arr.size > 0 else float("nan"),
                "p90": float(np.percentile(arr, 90)) if arr.size > 0 else float("nan"),
                "min": float(np.min(arr)) if arr.size > 0 else float("nan"),
                "max": float(np.max(arr)) if arr.size > 0 else float("nan"),
                "n": int(arr.size),
            }
        return summary

    # Train remains the qualitative source split; validation is the actual
    # reference distribution used for candidate scoring and ranking.
    train_summary = _baseline_summary(train_baseline)
    val_summary = _baseline_summary(val_baseline)

    (output_dir / "real_train_baseline_metrics.json").write_text(
        json.dumps(train_summary, indent=2), encoding="utf-8",
    )
    (output_dir / "real_val_baseline_metrics.json").write_text(
        json.dumps(val_summary, indent=2), encoding="utf-8",
    )

    # -----------------------------------------------------------------------
    # Extract train thread keys for few-shot filtering (no val/test leakage)
    # -----------------------------------------------------------------------
    train_df = pd.read_csv(real_train_csv)
    train_thread_ids: list[str] = []
    if "thread_id" in train_df.columns:
        if "product" in train_df.columns:
            train_pairs = train_df[["product", "thread_id"]].dropna(subset=["thread_id"])
            train_thread_ids = [
                _composite_thread_key(product, thread_id)
                for product, thread_id in zip(
                    train_pairs["product"].astype(str),
                    train_pairs["thread_id"].astype(str),
                )
                if str(thread_id).strip()
            ]
        else:
            train_thread_ids = train_df["thread_id"].dropna().astype(str).tolist()
        train_thread_ids = list(dict.fromkeys(train_thread_ids))
        train_ids_path = output_dir / "train_thread_ids.json"
        train_ids_path.write_text(
            json.dumps(train_thread_ids, ensure_ascii=False), encoding="utf-8",
        )
        reference_run_config["few_shot_thread_ids"] = str(train_ids_path)

    # -----------------------------------------------------------------------
    # Phase 0: Before-calibration group evaluation (vanilla vs real_test)
    # -----------------------------------------------------------------------
    before_eval: dict[str, dict] | None = None
    _before_eval_path = output_dir / "before_calibration_group_eval.json"
    _before_generated_scores_path = output_dir / "before_calibration_generated_scores.csv"
    _before_reused_sim_path = output_dir / "before_calibration_reused_sim_dir.txt"
    if rerun_phase0_vanilla:
        if _before_eval_path.exists() and _before_generated_scores_path.exists():
            before_eval = json.loads(_before_eval_path.read_text(encoding="utf-8"))
            before_generated_df = pd.read_csv(_before_generated_scores_path)
            if _before_reused_sim_path.exists():
                reusable_vanilla_sim_dir = Path(
                    _before_reused_sim_path.read_text(encoding="utf-8").strip()
                )
            print("\n" + "=" * 60)
            print("PHASE 0: Before-calibration group evaluation (skipped — already done)")
            print("=" * 60)
        else:
            before_reference_run_config = _force_vanilla_backbone(
                reference_run_config
            )
            before_eval, before_generated_df, reusable_vanilla_sim_dir = _run_before_calibration_evaluation(
                output_dir=output_dir,
                real_test_csv=real_test_csv,
                reference_run_config=before_reference_run_config,
                sim_runs=final_sim_runs,
                metrics=metrics,
                python=python,
                repo_root=repo_root,
                device=device,
                min_sim_threads=min_sim_threads,
                metric_parallel=metric_parallel,
                simulation_reasoning_effort=simulation_reasoning_effort,
            )
        if before_eval:
            print("  Vanilla vs real_test (key metrics):")
            _print_group_eval_summary(before_eval)
    elif vanilla_scores_csv is not None:
        if _before_eval_path.exists():
            before_eval = json.loads(_before_eval_path.read_text(encoding="utf-8"))
            print("\n" + "=" * 60)
            print("PHASE 0: Before-calibration group evaluation (skipped — already done)")
            print("=" * 60)
        else:
            before_eval, before_generated_df, _ = _run_before_calibration_evaluation(
                output_dir=output_dir,
                real_test_csv=real_test_csv,
                metrics=metrics,
                vanilla_scores_csv=vanilla_scores_csv,
            )
        if before_eval:
            print("  Vanilla vs real_test (key metrics):")
            _print_group_eval_summary(before_eval)

    # -----------------------------------------------------------------------
    # Phase 1: Calibration loop — resume status
    # -----------------------------------------------------------------------
    total_phase1_iterations = _phase1_total_iterations(max_iterations)
    starting_completed_iterations = state.completed_iterations
    reported_completed_iterations = _phase1_reported_iteration_count(state.completed_iterations)
    remaining = max(0, max_iterations - reported_completed_iterations)
    print(f"\n{'='*60}")
    print("PHASE 1: Calibration loop")
    print(f"{'='*60}")
    if state.completed_iterations > 0:
        best = state.current_best_score or {}
        print(f"  Resumed at iteration {reported_completed_iterations}/{max_iterations}")
        if best.get("quantile_fail_rate") is not None:
            print(
                f"  Best so far → quantile_fail={best['quantile_fail_rate']:.4f}  "
                f"pct_dist={best.get('mean_percentile_distance', float('nan')):.4f}  "
                f"robust_z={best.get('mean_abs_robust_z', float('nan')):.4f}"
            )
            print(
                f"                 legacy fail_rate={best.get('fail_rate', float('nan')):.4f}  "
                f"|delta|={best.get('mean_abs_delta', float('nan')):.4f}"
            )
        else:
            print(
                f"  Best so far → fail_rate={best.get('fail_rate', float('nan')):.4f}  "
                f"|delta|={best.get('mean_abs_delta', float('nan')):.4f}"
            )
        if state.current_search_root_mode != "global_best":
            print(
                f"  Search root → {state.current_search_root_mode} "
                f"({state.current_search_root_reason})"
            )
    else:
        print(f"  Starting fresh — {max_iterations} iterations planned")
    print(f"  Iterations remaining: {remaining}")

    # Build a separate run config for Phase 1 calibration iterations.
    # If --calibration-rounds is set, use fewer rounds during candidate
    # ranking (iterations 1+) for faster turnaround.  Iteration 0 (baseline)
    # and Phase 2 (final evaluation) always use the full rounds.
    phase1_run_config = dict(reference_run_config)
    if calibration_rounds is not None and calibration_rounds != reference_run_config.get("rounds"):
        phase1_run_config["rounds"] = calibration_rounds
        print(f"  Phase 1 calibration rounds: {calibration_rounds} (full: {reference_run_config.get('rounds')})")

    for iteration in range(state.completed_iterations, total_phase1_iterations):
        iter_dir = output_dir / f"iter_{iteration:03d}"
        iter_dir.mkdir(parents=True, exist_ok=True)
        iteration_checkpoint = _load_iteration_checkpoint(iter_dir)
        resume_iteration = iteration_checkpoint is not None
        manual_iteration = iteration - 1 if _MANUAL_PHASE_MODE else iteration
        manual_block_active = _MANUAL_PHASE_MODE and iteration > 0
        phase_context = (
            _manual_phase_context(manual_iteration)
            if manual_block_active else {}
        )
        if _MANUAL_PHASE_MODE:
            active_phase_name = str(phase_context.get("name", "")).strip()
            if iteration == 0:
                state.current_phase_name = None
                state.manual_block_phase_name = None
            else:
                if state.current_phase_name and state.current_phase_name != active_phase_name:
                    previous_phase = _manual_phase_context(manual_iteration - 1)
                    _manual_commit_block_best(state, previous_phase)
                if state.manual_block_phase_name != active_phase_name:
                    _manual_start_block(state, phase_context)
                state.current_phase_name = active_phase_name
            state.save()

        display_iteration = manual_iteration + 1 if _MANUAL_PHASE_MODE and iteration > 0 else 0
        print(f"\n[Iter {display_iteration}/{max_iterations}] ── {'baseline run' if iteration == 0 else 'LLM reasoner → generate candidates'}")
        if _MANUAL_PHASE_MODE:
            if iteration == 0:
                print(
                    "  → Manual phase warm-start baseline "
                    f"(not counted against the {max_iterations} edited iterations)"
                )
            else:
                print(
                    "  → Manual phase: "
                    f"{phase_context.get('label')} "
                    f"({phase_context.get('block_label')}, focus={phase_context.get('focus_metrics')})"
                )
        if resume_iteration:
            print("  → Resuming from saved iteration checkpoint...")

        # -------------------------------------------------------------------
        # Build candidate overlays
        # -------------------------------------------------------------------
        parsed: dict = {}
        candidates_list: list[dict] = []
        candidate_previews: list[dict[str, Any]] = []
        candidate_validation_errors: dict[int, list[str]] = {}
        validation_errors: list[str] = []
        reasoner_prompt_path: Path | None = None
        reasoner_response_path: Path | None = None
        materializer_prompt_path: Path | None = None
        materializer_response_path: Path | None = None
        if resume_iteration:
            assert iteration_checkpoint is not None
            strategy_label = iteration_checkpoint["strategy_label"]
            diagnosis = iteration_checkpoint["diagnosis"]
            overlay_diff = iteration_checkpoint["overlay_diff"]
            overlays = iteration_checkpoint["overlays"]
            candidate_previews = list(iteration_checkpoint["candidate_previews"])
            validation_errors = list(iteration_checkpoint["validation_errors"])
            candidate_validation_errors = {
                int(preview.get("candidate_id", idx)): list(preview.get("validation_errors", []))
                for idx, preview in enumerate(candidate_previews)
                if preview.get("validation_errors")
            }
            parsed = {
                "strategy_label": strategy_label,
                "diagnosis": diagnosis,
                "overlay_diff": overlay_diff,
                "primary_layer": "resumed",
            }
            print(f"  → Restored {len(overlays)} candidate overlay(s) from {iter_dir / 'diagnosis.json'}")
        elif iteration == 0:
            # Iteration 0: single candidate with default overlay to establish
            # a baseline diagnostic for the reasoner.
            strategy_label = "defaults"
            if reusable_vanilla_sim_dir is not None:
                diagnosis = "Initial baseline iteration reusing one pre-calibration vanilla simulation."
            else:
                diagnosis = "Initial baseline run using default knob values."
            overlay_diff: dict = {}
            overlays = [dict(state.current_best_overlay)]
            if reusable_vanilla_sim_dir is not None:
                print(f"  → 1 candidate (reused pre-calibration vanilla simulation)")
                print(f"    source: {reusable_vanilla_sim_dir}")
            else:
                print(f"  → 1 candidate (default overlay)")
        else:
            # Build prompt — validation is the scoring reference; train remains
            # the qualitative source split for sample-thread context.
            print(f"  → Calling {calibration_model} for strategy...", flush=True)
            if _MANUAL_PHASE_MODE:
                search_root_overlay = dict(state.manual_block_best_overlay or state.current_best_overlay)
                search_root_diagnostic = state.manual_block_best_diagnostic or state.current_best_diagnostic
                search_root_candidate_dir = state.manual_block_best_candidate_dir or state.current_best_candidate_dir
                search_root_mode = f"manual_phase:{phase_context.get('name')}"
                search_root_reason = (
                    "deterministic manual phase-block schedule; "
                    "branch from current block_best built on cumulative committed overlay"
                )
                phase_context["current_focus_metric_rows"] = _manual_phase_metric_rows(
                    search_root_diagnostic or {},
                    list(phase_context.get("focus_metrics", [])),
                )
                phase_context["current_protected_metric_rows"] = _manual_phase_metric_rows(
                    search_root_diagnostic or {},
                    list(phase_context.get("protected_metrics", [])),
                )
                state.current_search_root_overlay = dict(search_root_overlay)
                state.current_search_root_diagnostic = search_root_diagnostic
                state.current_search_root_candidate_dir = search_root_candidate_dir
                state.current_search_root_mode = search_root_mode
                state.current_search_root_reason = search_root_reason
                state.stagnation_count = 0
                state.save()
                print(f"  → Search mode: {search_root_mode}")
                print(f"    reason: {search_root_reason}")
                _print_phase_watch_metrics(
                    phase_context,
                    phase_context.get("current_focus_metric_rows", []),
                    phase_context.get("current_protected_metric_rows", []),
                )
            else:
                state.stagnation_count = _stagnation_count_from_entries(log.entries())
                (
                    search_root_overlay,
                    search_root_diagnostic,
                    search_root_candidate_dir,
                    search_root_mode,
                    search_root_reason,
                ) = _choose_search_root(state)
                state.current_search_root_overlay = dict(search_root_overlay)
                state.current_search_root_diagnostic = search_root_diagnostic
                state.current_search_root_candidate_dir = search_root_candidate_dir
                state.current_search_root_mode = search_root_mode
                state.current_search_root_reason = search_root_reason
                state.save()
                print(
                    f"  → Search mode: {search_root_mode} "
                    f"(stagnation_count={state.stagnation_count})"
                )
                if search_root_reason and search_root_reason != "global_best":
                    print(f"    reason: {search_root_reason}")

            # ── Dedup-only round: skip normal reasoner/materializer ────────
            is_dedup_round = bool(phase_context.get("is_dedup"))
            if is_dedup_round:
                print("  → DEDUP ROUND: deduplicating accumulated overlay text")
                dedup_source_overlay = dict(state.current_best_overlay)
                prompt = build_dedup_prompt(
                    current_overlay=dedup_source_overlay,
                    num_candidates=candidates_per_iter,
                )
                reasoner_prompt_path = iter_dir / "dedup_prompt.txt"
                reasoner_prompt_path.write_text(prompt, encoding="utf-8")

                raw_response = call_reasoner(
                    client,
                    calibration_model,
                    prompt,
                    reasoning_effort=calibration_reasoning_effort,
                    schema_kind=None,
                )
                reasoner_response_path = iter_dir / "dedup_raw_response.json"
                reasoner_response_path.write_text(raw_response, encoding="utf-8")

                # Parse dedup response: expect {candidate_0: {overlay}, ...}
                dedup_parsed = _parse_dedup_response(
                    raw_response, candidates_per_iter, dedup_source_overlay,
                )
                overlays = dedup_parsed["overlays"]
                strategy_label = "dedup_final"
                diagnosis = "Final deduplication round — removing overlay text redundancy."
                overlay_diff: dict = {}
                parsed = {
                    "strategy_label": strategy_label,
                    "diagnosis": diagnosis,
                    "overlay_diff": overlay_diff,
                    "primary_layer": "both",
                }
                candidates_list = [
                    {
                        "strategy_label": f"dedup_variant_{i}",
                        "mechanism_family": "dedup",
                        "primary_layer": "both",
                        "anti_incumbent": False,
                    }
                    for i in range(len(overlays))
                ]
                print(f"  → {len(overlays)} dedup candidate(s) generated")
                for ci, ov in enumerate(overlays):
                    persona_len = len(str(ov.get("persona.generation_guidance", "")))
                    prompt_len = len(str(ov.get("prompt.comment_style_guidance", "")))
                    print(f"      [{ci}] persona={persona_len} chars, prompt={prompt_len} chars")

            else:
                # ── Normal reasoner + materializer flow ──────────────────────

                # Extract sample threads for qualitative context
                _few_shot_dir = Path(reference_run_config.get("few_shot_source", ""))
                _best_cand_dir = (
                    Path(search_root_candidate_dir)
                    if search_root_candidate_dir else None
                )
                sample_real = _extract_sample_real_thread(
                    _few_shot_dir, train_thread_ids,
                ) if _few_shot_dir.exists() else ""
                sample_sim = _extract_sample_sim_thread(
                    _best_cand_dir,
                ) if _best_cand_dir else ""

                prompt_trajectory = (
                    _manual_phase_prompt_trajectory(log.trajectory(), phase_context, iteration)
                    if _MANUAL_PHASE_MODE else log.trajectory()
                )

                prompt = build_reasoner_prompt(
                    registry=registry,
                    current_overlay=search_root_overlay,
                    current_diagnostic=search_root_diagnostic or {},
                    real_baseline=val_summary,
                    trajectory=prompt_trajectory,
                    failed_strategies=log.failed_strategies(),
                    metric_definitions=metric_definitions,
                    sample_real_thread=sample_real,
                    sample_sim_thread=sample_sim,
                    iteration=manual_iteration if _MANUAL_PHASE_MODE else iteration,
                    max_iterations=max_iterations,
                    combination_start_iteration=combination_start_iteration,
                    global_best_overlay=state.current_best_overlay,
                    global_best_diagnostic=state.current_best_diagnostic or {},
                    frontier=_frontier_prompt_summary(state.frontier),
                    stagnation_count=state.stagnation_count,
                    search_mode=search_root_mode,
                    search_root_reason=search_root_reason,
                    phase_context=phase_context if _MANUAL_PHASE_MODE else None,
                    completed_phase_summaries=None,
                )
                reasoner_prompt_path = iter_dir / "reasoner_prompt.txt"
                reasoner_prompt_path.write_text(prompt, encoding="utf-8")

                raw_response = call_reasoner(
                    client,
                    calibration_model,
                    prompt,
                    reasoning_effort=calibration_reasoning_effort,
                    schema_kind="strategist",
                )
                reasoner_response_path = iter_dir / "reasoner_raw_response.json"
                reasoner_response_path.write_text(
                    raw_response,
                    encoding="utf-8",
                )
                parsed = parse_reasoner_response(raw_response)
                diagnosis_preview = " ".join(str(parsed.get("diagnosis", "")).split())
                if diagnosis_preview:
                    print(f"  → Diagnosis: {diagnosis_preview[:300]}")

                # Second-stage text materializer:
                # The strategist chooses what to modify; a second LLM call writes the
                # actual prompt/persona text blocks used at runtime.
                parsed_candidates = parsed.get("candidates", [])
                if _MANUAL_PHASE_MODE and parsed_candidates:
                    required_family = str(phase_context.get("required_mechanism_family", "")).strip()
                    for candidate in parsed_candidates[:5]:
                        candidate["primary_layer"] = "both"
                        if required_family:
                            candidate["mechanism_family"] = required_family
                if parsed_candidates:
                    materializer_prompt = build_text_materializer_prompt(
                        registry=registry,
                        current_overlay=search_root_overlay,
                        current_diagnostic=search_root_diagnostic or {},
                        diagnosis=parsed.get("diagnosis", ""),
                        candidates=parsed_candidates,
                        real_baseline=val_summary,
                        trajectory=prompt_trajectory,
                        failed_strategies=log.failed_strategies(),
                        metric_definitions=metric_definitions,
                        sample_real_thread=sample_real,
                        sample_sim_thread=sample_sim,
                        iteration=manual_iteration if _MANUAL_PHASE_MODE else iteration,
                        max_iterations=max_iterations,
                        combination_start_iteration=combination_start_iteration,
                        global_best_overlay=state.current_best_overlay,
                        global_best_diagnostic=state.current_best_diagnostic or {},
                        frontier=_frontier_prompt_summary(state.frontier),
                        stagnation_count=state.stagnation_count,
                        search_mode=search_root_mode,
                        search_root_reason=search_root_reason,
                        phase_context=phase_context if _MANUAL_PHASE_MODE else None,
                        completed_phase_summaries=None,
                    )
                    materializer_prompt_path = iter_dir / "materializer_prompt.txt"
                    materializer_prompt_path.write_text(materializer_prompt, encoding="utf-8")
                    expected_materialized_candidates = min(5, len(parsed_candidates))
                    raw_materialized = call_reasoner(
                        client,
                        calibration_model,
                        materializer_prompt,
                        reasoning_effort=calibration_reasoning_effort,
                        schema_kind="materializer",
                        response_format_override=materializer_response_format(
                            expected_materialized_candidates
                        ),
                    )
                    materializer_response_path = iter_dir / "materializer_raw_response.json"
                    materializer_response_path.write_text(
                        raw_materialized,
                        encoding="utf-8",
                    )
                    materialized = parse_text_materializer_response(
                        raw_materialized,
                        expected_candidates=expected_materialized_candidates,
                    )
                    for ci, candidate in enumerate(parsed_candidates[:5]):
                        text_diff = materialized.get(ci, {})
                        filtered_text_diff: dict[str, Any] = {}
                        for name, value in text_diff.items():
                            try:
                                if registry.get(name)["type"] == "text":
                                    filtered_text_diff[name] = value
                            except KeyError:
                                continue
                        if filtered_text_diff:
                            candidate["materialized_text_overlay_diff"] = filtered_text_diff
                            candidate["overlay_diff"] = merge_overlay(
                                candidate.get("overlay_diff", {}),
                                filtered_text_diff,
                            )
                            if ci == 0 and isinstance(parsed.get("overlay_diff"), dict):
                                parsed["overlay_diff"] = merge_overlay(
                                    parsed.get("overlay_diff", {}),
                                    filtered_text_diff,
                                )
                if "overlay_diff" in parsed:
                    parsed["overlay_diff"], overlay_errors = _sanitize_overlay(
                        registry,
                        parsed.get("overlay_diff", {}),
                    )
                    candidate_validation_errors.setdefault(0, []).extend(overlay_errors)
                    validation_errors.extend(
                        [f"base overlay_diff: {err}" for err in overlay_errors]
                    )
                for ci, candidate in enumerate(parsed_candidates[:5]):
                    cleaned_diff, candidate_errors = _sanitize_overlay(
                        registry,
                        candidate.get("overlay_diff", {}),
                    )
                    candidate["overlay_diff"] = cleaned_diff
                    if _MANUAL_PHASE_MODE:
                        candidate["primary_layer"] = "both"
                        required_family = str(phase_context.get("required_mechanism_family", "")).strip()
                        if required_family:
                            candidate["mechanism_family"] = required_family
                    if candidate_errors:
                        candidate_validation_errors.setdefault(ci, []).extend(candidate_errors)
                        candidate["validation_errors"] = list(
                            dict.fromkeys(candidate_validation_errors[ci])
                        )
                        validation_errors.extend(
                            [f"candidate_{ci}: {err}" for err in candidate_errors]
                        )

                strategy_label = parsed["strategy_label"]
                diagnosis = parsed["diagnosis"]
                overlay_diff = parsed.get("overlay_diff", {})
                candidates_list = parsed_candidates
                if _MANUAL_PHASE_MODE:
                    parsed["primary_layer"] = "both"

                overlays = generate_variants(
                    current_overlay=search_root_overlay,
                    base_diff=overlay_diff,
                    prompt_alternatives=parsed.get("prompt_alternatives", {}),
                    registry=registry,
                    seed=seed + iteration,
                    conservative_diff=parsed.get("conservative_diff"),
                    parsed_candidates=candidates_list,
                    append_text_mode=_MANUAL_PHASE_MODE,
                    structured_phase_name=str(phase_context.get("name", "")).strip() if _MANUAL_PHASE_MODE else None,
                    structured_phase_label=str(phase_context.get("label", "")).strip() if _MANUAL_PHASE_MODE else None,
                    structured_phase_order=int(phase_context.get("phase_index", 0)) if _MANUAL_PHASE_MODE else None,
                )
                if len(candidates_list) >= 5:
                    print(f"  → 5 independent strategies from LLM:")
                    for ci, c in enumerate(candidates_list[:5]):
                        print(f"      [{ci}] {c.get('strategy_label','?')} ({c.get('primary_layer','?')})")
                else:
                    print(f"  → Strategy: {strategy_label}")
                print(f"  → {len(overlays)} candidate(s) generated")

        sanitized_overlays: list[dict[str, Any]] = []
        for ci, overlay in enumerate(overlays):
            cleaned_overlay, overlay_errors = _sanitize_overlay(registry, overlay)
            sanitized_overlays.append(cleaned_overlay)
            if overlay_errors:
                candidate_validation_errors.setdefault(ci, []).extend(overlay_errors)
            validation_errors.extend(
                [f"candidate_{ci} merged overlay: {err}" for err in overlay_errors]
            )
        overlays = sanitized_overlays
        candidate_validation_errors = {
            ci: list(dict.fromkeys(errors))
            for ci, errors in candidate_validation_errors.items()
        }
        validation_errors = list(dict.fromkeys(validation_errors))
        if validation_errors:
            print("  [overlay validation] Dropped invalid candidate overlay entries:")
            for err in validation_errors:
                print(f"    - {err}")

        print("  → Candidate guidance changes:")
        if candidate_previews:
            for preview in candidate_previews:
                _print_candidate_change_preview(
                    candidate_id=preview.get("candidate_id", 0),
                    strategy_label=preview.get("strategy_label", "strategy"),
                    primary_layer=preview.get("primary_layer", "both"),
                    strategy=preview.get("strategy", ""),
                    rationale=preview.get("rationale", ""),
                    changes=preview.get("effective_changes", []),
                    validation_errors=preview.get("validation_errors"),
                )
        elif candidates_list and len(candidates_list) >= 5:
            for ci, cand in enumerate(candidates_list[:5]):
                effective_changes = _overlay_change_records(
                    registry,
                    search_root_overlay,
                    overlays[ci],
                )
                preview = {
                    "candidate_id": ci,
                    "strategy_label": cand.get("strategy_label", f"candidate_{ci}"),
                    "strategy": cand.get("strategy", ""),
                    "primary_layer": cand.get("primary_layer", "both"),
                    "mechanism_family": cand.get("mechanism_family", "mixed"),
                    "anti_incumbent": bool(cand.get("anti_incumbent", False)),
                    "rationale": cand.get("rationale", ""),
                    "overlay_diff": cand.get("overlay_diff", {}),
                    "materialized_text_overlay_diff": cand.get("materialized_text_overlay_diff", {}),
                    "effective_changes": effective_changes,
                }
                if candidate_validation_errors.get(ci):
                    preview["validation_errors"] = candidate_validation_errors[ci]
                candidate_previews.append(preview)
                _print_candidate_change_preview(
                    candidate_id=ci,
                    strategy_label=preview["strategy_label"],
                    primary_layer=preview["primary_layer"],
                    strategy=preview["strategy"],
                    rationale=preview["rationale"],
                    changes=effective_changes,
                    validation_errors=preview.get("validation_errors"),
                )
        else:
            base_overlay = state.current_best_overlay
            if iteration > 0:
                base_overlay = state.current_search_root_overlay or state.current_best_overlay
            overlay = overlays[0] if overlays else dict(base_overlay)
            effective_changes = _overlay_change_records(registry, base_overlay, overlay)
            preview = {
                "candidate_id": 0,
                "strategy_label": strategy_label,
                "strategy": parsed.get("strategy", diagnosis),
                "primary_layer": parsed.get("primary_layer", "both"),
                "mechanism_family": parsed.get("mechanism_family", "mixed"),
                "anti_incumbent": bool(parsed.get("anti_incumbent", False)),
                "rationale": parsed.get("rationale", ""),
                "overlay_diff": overlay_diff,
                "materialized_text_overlay_diff": parsed.get("materialized_text_overlay_diff", {}),
                "effective_changes": effective_changes,
            }
            if candidate_validation_errors.get(0):
                preview["validation_errors"] = candidate_validation_errors[0]
            candidate_previews.append(preview)
            _print_candidate_change_preview(
                candidate_id=0,
                strategy_label=preview["strategy_label"],
                primary_layer=preview["primary_layer"],
                strategy=preview["strategy"],
                rationale=preview["rationale"],
                changes=effective_changes,
                validation_errors=preview.get("validation_errors"),
            )

        # Save diagnosis
        diag_payload: dict = {
            "iteration": iteration,
            "strategy_label": strategy_label,
            "diagnosis": diagnosis,
        }
        if manual_block_active:
            diag_payload["manual_phase_context"] = dict(phase_context)
            diag_payload["watch_metrics"] = {
                "focus_metric_rows": _serialize_metric_rows(
                    list(phase_context.get("current_focus_metric_rows", []))
                ),
                "protected_metric_rows": _serialize_metric_rows(
                    list(phase_context.get("current_protected_metric_rows", []))
                ),
            }
        diag_payload["artifacts"] = {
            "reasoner_prompt": str(reasoner_prompt_path) if reasoner_prompt_path else None,
            "reasoner_response": str(reasoner_response_path) if reasoner_response_path else None,
            "materializer_prompt": str(materializer_prompt_path) if materializer_prompt_path else None,
            "materializer_response": str(materializer_response_path) if materializer_response_path else None,
        }
        diag_payload["candidates"] = candidate_previews
        if not (candidates_list and len(candidates_list) >= 5):
            diag_payload["overlay_diff"] = overlay_diff
        if validation_errors:
            diag_payload["validation_errors"] = validation_errors
        (iter_dir / "diagnosis.json").write_text(
            json.dumps(diag_payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        # -------------------------------------------------------------------
        # Run candidates (different overlays share the same seed)
        # -------------------------------------------------------------------
        if iteration == 0 and reusable_vanilla_sim_dir is not None:
            print("  → Reusing 1 precomputed vanilla simulation...", flush=True)
            candidate_results = [
                _make_reused_baseline_candidate_result(
                    iter_dir=iter_dir,
                    overlay=overlays[0],
                    source_sim_dir=reusable_vanilla_sim_dir,
                )
            ]
        else:
            # Iteration 0 uses full rounds (baseline); iterations 1+ use
            # phase1_run_config which may have reduced rounds for speed.
            active_run_config = reference_run_config if iteration == 0 else phase1_run_config
            print(f"  → Running {len(overlays)} simulation(s)...", flush=True)
            candidate_results = run_candidates(
                overlays=overlays,
                iter_dir=iter_dir,
                reference_run_config=active_run_config,
                parallel=len(overlays),
                python=python,
                repo_root=repo_root,
                device=device,
                metric_parallel=metric_parallel,
                simulation_reasoning_effort=simulation_reasoning_effort,
                reuse_existing=resume_iteration,
            )
        succeeded = sum(1 for r in candidate_results if r["success"])
        print(f"  → {succeeded}/{len(overlays)} simulation(s) succeeded")

        # -------------------------------------------------------------------
        # Score candidates against VALIDATION baseline (per-thread empirical p)
        # -------------------------------------------------------------------
        print(f"  → Scoring candidates against val baseline...", flush=True)
        scored: list[dict] = []
        preview_by_id = {
            int(preview.get("candidate_id", idx)): preview
            for idx, preview in enumerate(candidate_previews)
        }
        manual_reference_payload = _manual_block_reference(state) if manual_block_active else None
        for result in candidate_results:
            if not result["success"] or result["sim_dir"] is None:
                continue
            sim_dir = Path(result["sim_dir"])
            try:
                sim_df = load_thread_metrics(sim_dir)
                sc = score_candidate(
                    sim_dir,
                    val_baseline,
                    metrics,
                    ranking_metrics=ranking_metrics,
                )
                group_eval = evaluate_group_vs_real(val_df, sim_df, metrics)
                sc.update(_group_eval_selection_summary(group_eval))
                sc["group_eval_per_metric"] = group_eval.get("per_metric", group_eval)
                sc["candidate_id"] = result["candidate_id"]
                sc["candidate_dir"] = result["candidate_dir"]
                sc["overlay"] = overlays[result["candidate_id"]]
                preview = preview_by_id.get(int(result["candidate_id"]), {})
                sc["strategy_label"] = preview.get("strategy_label", "")
                sc["mechanism_family"] = preview.get("mechanism_family", "mixed")
                sc["primary_layer"] = preview.get("primary_layer", "both")
                sc["anti_incumbent"] = bool(preview.get("anti_incumbent", False))
                if manual_block_active:
                    sc["manual_phase_context"] = dict(phase_context)
                    sc["manual_phase_score"] = _manual_phase_score(sc, phase_context)
                    sc["manual_phase_guard"] = _manual_phase_guard_summary(
                        sc,
                        manual_reference_payload,
                        phase_context,
                    )
                scored.append(sc)
            except Exception:
                pass
        print(f"  → {len(scored)} candidate(s) scored")
        state.frontier = _update_frontier(
            state.frontier,
            scored,
            preview_by_id,
            iteration,
        )

        # -------------------------------------------------------------------
        # Select best candidate
        # -------------------------------------------------------------------
        if manual_block_active and scored:
            winner = min(
                scored,
                key=lambda candidate: _manual_phase_selection_key(candidate, phase_context),
            )
        else:
            winner = select_best_candidate(scored) if scored else None

        # -------------------------------------------------------------------
        # Check if winner beats current best
        # -------------------------------------------------------------------
        beat_current_best = False
        winner_target_eval: dict[str, Any] | None = None
        if winner is not None:
            if manual_block_active:
                prev_payload = _manual_block_reference(state)
                if prev_payload is None:
                    beat_current_best = True
                else:
                    prev = _manual_phase_selection_key(prev_payload, phase_context)
                    new = _manual_phase_selection_key(winner, phase_context)
                    if new < prev:
                        beat_current_best = True
            elif state.current_best_score is None:
                beat_current_best = True
            else:
                prev_payload = _current_best_selection_reference(state)
                prev = _candidate_selection_key(prev_payload or state.current_best_score)
                new = _candidate_selection_key(winner)
                if new < prev:
                    beat_current_best = True

            if beat_current_best:
                winner_score_payload = {
                    "fail_rate": winner["fail_rate"],
                    "mean_abs_delta": winner["mean_abs_delta"],
                    "ranking_fail_rate": winner.get("ranking_fail_rate"),
                    "ranking_mean_abs_delta": winner.get("ranking_mean_abs_delta"),
                    "quantile_fail_rate": winner.get("quantile_fail_rate"),
                    "mean_percentile_distance": winner.get("mean_percentile_distance"),
                    "mean_abs_robust_z": winner.get("mean_abs_robust_z"),
                    "mean_group_percentile_distance": winner.get("mean_group_percentile_distance"),
                    "group_scores": winner.get("group_scores"),
                    "selection_family_scores": winner.get("selection_family_scores"),
                    "group_mean_abs_cliffs_delta": winner.get("group_mean_abs_cliffs_delta"),
                    "group_overall_fail_rate": winner.get("group_overall_fail_rate"),
                    "group_metrics_sig_different": winner.get("group_metrics_sig_different"),
                    "manual_phase_context": winner.get("manual_phase_context"),
                    "manual_phase_score": winner.get("manual_phase_score"),
                }
                winner_diagnostic_payload = {
                    k: v for k, v in winner.items()
                    if k not in ("candidate_id", "candidate_dir", "overlay")
                }
                if manual_block_active:
                    state.manual_block_best_overlay = winner.get("overlay", {})
                    state.manual_block_best_score = winner_score_payload
                    state.manual_block_best_diagnostic = winner_diagnostic_payload
                    state.manual_block_best_candidate_dir = winner.get("candidate_dir")
                    state.current_search_root_overlay = dict(state.manual_block_best_overlay)
                    state.current_search_root_diagnostic = state.manual_block_best_diagnostic
                    state.current_search_root_candidate_dir = state.manual_block_best_candidate_dir
                    state.current_search_root_mode = f"manual_phase:{phase_context.get('name')}"
                    state.current_search_root_reason = "winner became current block_best incumbent"
                else:
                    state.current_best_overlay = winner.get("overlay", {})
                    state.current_best_score = winner_score_payload
                    state.current_best_diagnostic = winner_diagnostic_payload
                    state.current_best_candidate_dir = winner.get("candidate_dir")
                    state.current_search_root_overlay = dict(state.current_best_overlay)
                    state.current_search_root_diagnostic = state.current_best_diagnostic
                    state.current_search_root_candidate_dir = state.current_best_candidate_dir
                    state.current_search_root_mode = "global_best"
                    state.current_search_root_reason = "global_best"
                state.stagnation_count = 0
            else:
                if manual_block_active:
                    state.stagnation_count = 0
                    state.current_search_root_overlay = dict(
                        state.manual_block_best_overlay or state.current_best_overlay
                    )
                    state.current_search_root_diagnostic = (
                        state.manual_block_best_diagnostic or state.current_best_diagnostic
                    )
                    state.current_search_root_candidate_dir = (
                        state.manual_block_best_candidate_dir or state.current_best_candidate_dir
                    )
                    state.current_search_root_mode = f"manual_phase:{phase_context.get('name')}"
                    state.current_search_root_reason = "no update; keep current block_best incumbent"
                else:
                    state.stagnation_count = _stagnation_count_from_entries(log.entries()) + 1
                    (
                        state.current_search_root_overlay,
                        state.current_search_root_diagnostic,
                        state.current_search_root_candidate_dir,
                        state.current_search_root_mode,
                        state.current_search_root_reason,
                    ) = _choose_search_root(state)

        if winner:
            winner_target_eval = _target_metric_eval_summary(winner)
            (iter_dir / "winner_target_metric_eval.json").write_text(
                json.dumps(winner_target_eval, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            if manual_block_active:
                improved_str = " ✓ new block best" if beat_current_best else ""
            else:
                improved_str = " ✓ new best" if beat_current_best else ""
            if manual_block_active:
                phase_score = winner.get("manual_phase_score") or _manual_phase_score(winner, phase_context)
                focus_rows = phase_score.get("focus_metric_rows", [])
                print(
                    f"  → Winner: candidate_{winner['candidate_id']}  "
                    f"focus_metrics={phase_context.get('focus_metrics', [])}{improved_str}"
                )
                for metric_row in focus_rows:
                    print(
                        "    "
                        f"{metric_row['metric']}: "
                        f"W={_fmt(metric_row['wasserstein'], '.4f')}  "
                        f"Q={_fmt(metric_row['quantile_error'], '.4f')}  "
                        f"fail={_fmt(metric_row['empirical_fail_rate'], '.4f')}  "
                        f"|med|={_fmt(metric_row['abs_median_gap'], '.4f')}  "
                        f"|cd|={_fmt(metric_row['abs_cliffs_delta'], '.4f')}  "
                        f"mwu_p={_fmt(metric_row['mwu_p_value'], '.4f')}  "
                        f"ks_p={_fmt(metric_row['ks_p_value'], '.4f')}  "
                        f"oor={metric_row['out_of_range']}  "
                        f"pct={_fmt(metric_row['percentile_distance'], '.4f')}  "
                        f"raw_z={_fmt(metric_row['abs_raw_robust_z'], '.4f')}"
                    )
                print(
                    "    target-12 summary: "
                    f"MWU>0.05 {winner_target_eval['mwu_pass_count']}/{len(winner_target_eval['metrics'])}  "
                    f"KS>0.05 {winner_target_eval['ks_pass_count']}/{len(winner_target_eval['metrics'])}  "
                    f"meanW={_fmt(winner_target_eval['mean_wasserstein'], '.4f')}  "
                    f"meanQ={_fmt(winner_target_eval['mean_quantile_error'], '.4f')}  "
                    f"mean|cd|={_fmt(winner_target_eval['mean_abs_cliffs_delta'], '.4f')}  "
                    f"meanFail={_fmt(winner_target_eval['mean_empirical_fail_rate'], '.4f')}"
                )
                _print_manual_phase_selection_ranking(scored, phase_context)
            elif winner.get("quantile_fail_rate") is not None:
                print(
                    f"  → Winner: candidate_{winner['candidate_id']}  "
                    f"quantile_fail={winner['quantile_fail_rate']:.4f}  "
                    f"pct_dist={winner['mean_percentile_distance']:.4f}  "
                    f"robust_z={winner['mean_abs_robust_z']:.4f}{improved_str}"
                )
                print(
                    f"    legacy fail_rate={winner['fail_rate']:.4f}  "
                    f"|delta|={winner['mean_abs_delta']:.4f}"
                )
                _print_winner_selection_breakdown(winner)
                _print_selection_ranking(scored)
            else:
                print(
                    f"  → Winner: candidate_{winner['candidate_id']}  "
                    f"fail_rate={winner['fail_rate']:.4f}  "
                    f"|delta|={winner['mean_abs_delta']:.4f}{improved_str}"
                )
            if winner.get("group_mean_abs_cliffs_delta") is not None:
                print(
                    f"    group |delta|={winner['group_mean_abs_cliffs_delta']:.4f}  "
                    f"group fail={winner['group_overall_fail_rate']:.4f}"
                )
            _print_candidate_score_summary(winner)

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

        if winner_target_eval is not None:
            diag_payload["winner_target_metric_eval"] = winner_target_eval
        if winner and winner.get("manual_phase_guard") is not None:
            diag_payload["winner_manual_phase_guard"] = winner.get("manual_phase_guard", {})
        (iter_dir / "diagnosis.json").write_text(
            json.dumps(diag_payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        # Build per-candidate strategy metadata for trajectory
        candidate_strategies: list[dict] = []
        for preview in candidate_previews:
            cs_entry: dict = {
                "candidate_id": preview.get("candidate_id", 0),
                "strategy_label": preview.get("strategy_label", "strategy"),
                "strategy": preview.get("strategy", ""),
                "primary_layer": preview.get("primary_layer", "both"),
                "mechanism_family": preview.get("mechanism_family", "mixed"),
                "anti_incumbent": bool(preview.get("anti_incumbent", False)),
                "overlay_diff": preview.get("overlay_diff", {}),
                "rationale": preview.get("rationale", ""),
                "effective_changes": preview.get("effective_changes", []),
            }
            if preview.get("materialized_text_overlay_diff"):
                cs_entry["materialized_text_overlay_diff"] = preview["materialized_text_overlay_diff"]
            matched = [
                s for s in scored
                if s.get("candidate_id") == cs_entry["candidate_id"]
            ]
            if matched:
                cs_entry["fail_rate"] = matched[0]["fail_rate"]
                cs_entry["mean_abs_delta"] = matched[0]["mean_abs_delta"]
                cs_entry["ranking_fail_rate"] = matched[0].get("ranking_fail_rate")
                cs_entry["ranking_mean_abs_delta"] = matched[0].get("ranking_mean_abs_delta")
                cs_entry["quantile_fail_rate"] = matched[0].get("quantile_fail_rate")
                cs_entry["mean_percentile_distance"] = matched[0].get("mean_percentile_distance")
                cs_entry["mean_abs_robust_z"] = matched[0].get("mean_abs_robust_z")
                cs_entry["group_mean_abs_cliffs_delta"] = matched[0].get("group_mean_abs_cliffs_delta")
                cs_entry["group_overall_fail_rate"] = matched[0].get("group_overall_fail_rate")
                cs_entry["group_scores"] = matched[0].get("group_scores", {})
                cs_entry["selection_family_scores"] = matched[0].get("selection_family_scores", {})
                if matched[0].get("manual_phase_score") is not None:
                    cs_entry["manual_phase_score"] = matched[0].get("manual_phase_score")
                if matched[0].get("manual_phase_guard") is not None:
                    cs_entry["manual_phase_guard"] = matched[0].get("manual_phase_guard")
                cs_entry["target_metric_eval"] = _target_metric_eval_summary(matched[0])
                # Store headline metric values for trajectory visibility
                m_pm = matched[0].get("per_metric", {})
                headline_vals: dict[str, dict] = {}
                for hkey, _hlabel in _HEADLINE_METRICS:
                    hinfo = m_pm.get(hkey, {})
                    if not hinfo:
                        continue
                    headline_vals[hkey] = {
                        "sim_median": hinfo.get("sim_median"),
                        "real_median": hinfo.get("real_median"),
                        "percentile_distance": hinfo.get("percentile_distance"),
                        "robust_z": hinfo.get("robust_z"),
                        "status": hinfo.get("status"),
                    }
                if headline_vals:
                    cs_entry["headline_metrics"] = headline_vals
            if preview.get("validation_errors"):
                cs_entry["validation_errors"] = preview["validation_errors"]
            candidate_strategies.append(cs_entry)

        log.upsert_iteration({
            "iteration": iteration,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "strategy_label": strategy_label,
            "primary_layer": parsed.get("primary_layer", ""),
            "diagnosis": diagnosis,
            "overlay_diff": overlay_diff,
            "artifacts": {
                "iteration_dir": str(iter_dir),
                "diagnosis_path": str(iter_dir / "diagnosis.json"),
                "reasoner_prompt": str(reasoner_prompt_path) if reasoner_prompt_path else None,
                "reasoner_response": str(reasoner_response_path) if reasoner_response_path else None,
                "materializer_prompt": str(materializer_prompt_path) if materializer_prompt_path else None,
                "materializer_response": str(materializer_response_path) if materializer_response_path else None,
            },
            "validation_errors": validation_errors,
            "candidate_strategies": candidate_strategies,
            "candidates": [_slim(c) for c in scored],
            "selection": {
                "winner_candidate_id": winner["candidate_id"] if winner else None,
                "beat_current_best": beat_current_best,
                "best_fail_rate": winner["fail_rate"] if winner else None,
                "best_mean_abs_delta": winner["mean_abs_delta"] if winner else None,
                "best_ranking_fail_rate": winner.get("ranking_fail_rate") if winner else None,
                "best_ranking_mean_abs_delta": winner.get("ranking_mean_abs_delta") if winner else None,
                "best_quantile_fail_rate": winner.get("quantile_fail_rate") if winner else None,
                "best_mean_percentile_distance": (
                    winner.get("mean_percentile_distance") if winner else None
                ),
                "best_mean_abs_robust_z": winner.get("mean_abs_robust_z") if winner else None,
                "best_group_mean_abs_cliffs_delta": (
                    winner.get("group_mean_abs_cliffs_delta") if winner else None
                ),
                "best_group_overall_fail_rate": (
                    winner.get("group_overall_fail_rate") if winner else None
                ),
                "winner_selection_family_scores": (
                    winner.get("selection_family_scores", {}) if winner else {}
                ),
                "winner_manual_phase_score": (
                    winner.get("manual_phase_score", {}) if winner else {}
                ),
                "winner_manual_phase_guard": (
                    winner.get("manual_phase_guard", {}) if winner else {}
                ),
                "winner_target_metric_eval": winner_target_eval or {},
                "selection_ranking": (
                    _manual_phase_ranking_rows(scored, phase_context)
                    if _MANUAL_PHASE_MODE else _selection_ranking_rows(scored)
                ),
            },
            "search_state": {
                "stagnation_count": state.stagnation_count,
                "search_root_mode": state.current_search_root_mode,
                "search_root_reason": state.current_search_root_reason,
                "search_root_candidate_dir": state.current_search_root_candidate_dir,
                "manual_phase_context": phase_context if _MANUAL_PHASE_MODE else {},
                "watch_metrics": (
                    {
                        "focus_metric_rows": _serialize_metric_rows(
                            list(phase_context.get("current_focus_metric_rows", []))
                        ),
                        "protected_metric_rows": _serialize_metric_rows(
                            list(phase_context.get("current_protected_metric_rows", []))
                        ),
                    }
                    if manual_block_active else {}
                ),
            },
        })

        state.completed_iterations = iteration + 1
        state.save()

    if _MANUAL_PHASE_MODE and state.completed_iterations > 1:
        final_phase = _manual_phase_context(state.completed_iterations - 2)
        _manual_commit_block_best(state, final_phase)
        state.save()

    if stop_after_phase1:
        summary = {
            "best_overlay": state.current_best_overlay,
            "best_score": state.current_best_score,
            "completed_iterations": _phase1_reported_iteration_count(state.completed_iterations),
            "output_dir": str(output_dir),
            "after_calibration_evaluation": {},
            "improvement": None,
            "stopped_after_phase1": True,
        }
        (output_dir / "calibration_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"\n{'='*60}")
        print("STOPPED AFTER PHASE 1")
        print(f"{'='*60}")
        print("  Phase 2 and Phase 3 were skipped by request.")
        return summary

    # -----------------------------------------------------------------------
    # Export best overlay
    # -----------------------------------------------------------------------
    save_overlay(state.current_best_overlay, output_dir / "best_overlay.json")

    # -----------------------------------------------------------------------
    # Phase 2: After-calibration group evaluation (calibrated vs real_test)
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("PHASE 2: After-calibration group evaluation")
    print(f"{'='*60}")
    _after_eval_path = output_dir / "after_calibration" / "after_calibration_evaluation.json"
    ran_new_iterations = state.completed_iterations > starting_completed_iterations
    if _after_eval_path.exists() and not ran_new_iterations:
        print(f"  Skipped — already done (found {_after_eval_path})")
        after_eval = json.loads(_after_eval_path.read_text(encoding="utf-8"))
        # reload group_eval too
        _group_eval_path = output_dir / "after_calibration" / "after_calibration_group_eval.json"
        if _group_eval_path.exists():
            after_eval["group_eval"] = json.loads(_group_eval_path.read_text(encoding="utf-8"))
    else:
        print(f"  Running {final_sim_runs} fresh simulation(s) with best overlay "
              f"(stop at {min_sim_threads} threads)...", flush=True)
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
            metric_parallel=metric_parallel,
            simulation_reasoning_effort=simulation_reasoning_effort,
        )

    if after_eval.get("group_eval"):
        print("  Calibrated vs real_test (key metrics):")
        _print_group_eval_summary(after_eval["group_eval"])

    # -----------------------------------------------------------------------
    # Phase 3: Improvement analysis (before vs after)
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("PHASE 3: Improvement analysis")
    print(f"{'='*60}")
    improvement: dict | None = None
    if before_eval is not None and after_eval.get("group_eval") is not None:
        improvement = compare_before_after(
            before_eval,
            after_eval["group_eval"],
            real_df=real_test_df,
            before_df=before_generated_df,
            after_df=after_eval.get("_all_sim_df"),
            metrics=metrics,
        )
        (output_dir / "before_after_improvement_summary.json").write_text(
            json.dumps(improvement, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        _print_improvement_summary(improvement)
        _print_improvement_table(improvement)

    # -----------------------------------------------------------------------
    # Final summary
    # -----------------------------------------------------------------------
    summary = {
        "best_overlay": state.current_best_overlay,
        "best_score": state.current_best_score,
        "completed_iterations": _phase1_reported_iteration_count(state.completed_iterations),
        "output_dir": str(output_dir),
        "after_calibration_evaluation": {
            k: v for k, v in after_eval.items() if k not in {"group_eval", "_all_sim_df"}
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
    output_dir: Path,
    real_test_csv: Path,
    metrics: list[str],
    vanilla_scores_csv: Path | None = None,
    reference_run_config: dict[str, Any] | None = None,
    sim_runs: int = 12,
    python: str = sys.executable,
    repo_root: Path | None = None,
    device: str = "cpu",
    min_sim_threads: int = 0,
    metric_parallel: int = 2,
    simulation_reasoning_effort: str | None = None,
) -> tuple[dict[str, dict], pd.DataFrame, Path | None]:
    """Evaluate a before-calibration vanilla baseline against real_test.

    If ``reference_run_config`` is provided, run fresh vanilla simulations with the
    current config. Otherwise load the precomputed ``vanilla_scores_csv``.
    """
    print(f"\n{'='*60}")
    print("PHASE 0: Before-calibration group evaluation (vanilla vs real_test)")
    print(f"{'='*60}")

    real_test_df = pd.read_csv(real_test_csv)
    reused_sim_dir: Path | None = None
    target_threads = _resolve_eval_thread_target(len(real_test_df), min_sim_threads)

    if reference_run_config is not None:
        if repo_root is None:
            repo_root = Path(__file__).parent.parent

        before_dir = output_dir / "before_calibration"
        before_dir.mkdir(parents=True, exist_ok=True)
        overlays = [{} for _ in range(sim_runs)]
        seed_offsets = list(range(sim_runs))

        print(
            f"  Running {sim_runs} fresh vanilla simulation(s) "
            f"(target {target_threads if target_threads > 0 else 'unbounded'} threads)...",
            flush=True,
        )
        results = run_candidates(
            overlays=overlays,
            iter_dir=before_dir,
            reference_run_config=reference_run_config,
            python=python,
            repo_root=repo_root,
            device=device,
            seed_offsets=seed_offsets,
            min_threads=target_threads,
            metric_parallel=metric_parallel,
            batch_schedule=[sim_runs],
            simulation_reasoning_effort=simulation_reasoning_effort,
        )

        frames: list[pd.DataFrame] = []
        for result in results:
            if not result["success"] or result["sim_dir"] is None:
                continue
            sim_dir = Path(result["sim_dir"])
            if reused_sim_dir is None:
                reused_sim_dir = sim_dir
            try:
                df = load_thread_metrics(sim_dir)
                df["_run_id"] = result["candidate_id"]
                frames.append(df)
            except Exception:
                pass

        if not frames:
            raise RuntimeError("No successful vanilla baseline simulations for Phase 0.")

        vanilla_df = pd.concat(frames, ignore_index=True)
        if target_threads > 0 and len(vanilla_df) > target_threads:
            vanilla_df = vanilla_df.iloc[:target_threads].copy()
        vanilla_df.to_csv(output_dir / "before_calibration_generated_scores.csv", index=False)
        if reused_sim_dir is not None:
            (output_dir / "before_calibration_reused_sim_dir.txt").write_text(
                str(reused_sim_dir),
                encoding="utf-8",
            )
    else:
        if vanilla_scores_csv is None:
            raise ValueError("Either vanilla_scores_csv or reference_run_config must be provided.")
        vanilla_df = pd.read_csv(vanilla_scores_csv)

    print(f"  real_test threads: {len(real_test_df)}")
    print(f"  vanilla threads:   {len(vanilla_df)}")

    before_eval = evaluate_group_vs_real(real_test_df, vanilla_df, metrics)

    (output_dir / "before_calibration_group_eval.json").write_text(
        json.dumps(before_eval, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"  → Saved before_calibration_group_eval.json")
    return before_eval, vanilla_df, reused_sim_dir


def _force_vanilla_backbone(
    reference_run_config: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Return a copy of ``reference_run_config`` that always uses vanilla OASIS."""

    if reference_run_config is None:
        return None
    forced = dict(reference_run_config)
    forced["discussion_backbone"] = "vanilla_oasis"
    return forced


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
    metric_parallel: int = 2,
    simulation_reasoning_effort: str | None = None,
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
    real_test_df = pd.read_csv(real_test_csv)
    target_threads = _resolve_eval_thread_target(len(real_test_df), min_sim_threads)

    # Save the overlay used
    save_overlay(best_overlay, final_dir / "overlay.json")

    print(f"\n{'='*60}")
    print(f"PHASE 2: After-calibration evaluation ({sim_runs} fresh sims with best overlay)")
    print(f"{'='*60}")
    print(
        f"  target threads: {target_threads if target_threads > 0 else 'unbounded'}",
        flush=True,
    )

    # Each run gets a different seed (same overlay → must differ by seed)
    overlays = [dict(best_overlay) for _ in range(sim_runs)]
    seed_offsets = list(range(sim_runs))

    results = run_candidates(
        overlays=overlays,
        iter_dir=final_dir,
        reference_run_config=reference_run_config,
        python=python,
        repo_root=repo_root,
        device=device,
        seed_offsets=seed_offsets,
        min_threads=target_threads,
        metric_parallel=metric_parallel,
        batch_schedule=[4, 3, 2] + [1] * max(0, sim_runs - 9),
        simulation_reasoning_effort=simulation_reasoning_effort,
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
            "_all_sim_df": None,
            "fail_rate": None,
            "mean_abs_delta": None,
            "sim_runs": 0,
            "total_threads": 0,
            "target_threads": target_threads,
        }

    all_sim_df = pd.concat(frames, ignore_index=True)
    if target_threads > 0 and len(all_sim_df) > target_threads:
        all_sim_df = all_sim_df.iloc[:target_threads].copy()

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
        "_all_sim_df": all_sim_df,
        "fail_rate": float(np.mean(all_fail_rates)) if all_fail_rates else None,
        "mean_abs_delta": float(np.mean(all_abs_deltas)) if all_abs_deltas else None,
        "sim_runs": len(frames),
        "total_threads": len(all_sim_df),
        "target_threads": target_threads,
    }

    (final_dir / "after_calibration_evaluation.json").write_text(
        json.dumps(
            {k: v for k, v in after_result.items() if k not in {"group_eval", "_all_sim_df"}},
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
# _print_improvement_summary moved to calibration/_display.py
# ---------------------------------------------------------------------------
from ._display import _print_improvement_summary
