"""Manual-phase helper functions.

Extracted from orchestrator.py to keep file size manageable. Logic
unchanged — these are the same functions, just in a separate module.
"""
from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import numpy as np

from ._phase_specs import (
    _DEDUP_ITERATION,
    _DEDUP_PHASE_SPEC,
    _MANUAL_METRIC_GUIDANCE,
    _MANUAL_PHASE_MODE,
    _MANUAL_PHASE_SPECS,
    _TARGET_METRICS_12,
)

if TYPE_CHECKING:
    from .orchestrator import CalibrationState


def _manual_total_edited_iterations() -> int:
    """Return the fixed number of edited iterations in manual-phase mode.

    Includes the 4 regular phase blocks (12 iterations) plus 1 dedup round = 13.
    """
    return _DEDUP_ITERATION + 1


def _is_dedup_iteration(manual_iteration: int) -> bool:
    """Return True if this manual-iteration is the dedup-only round."""
    return manual_iteration == _DEDUP_ITERATION


def _parse_dedup_response(
    raw: str,
    num_candidates: int,
    fallback_overlay: dict[str, Any],
) -> dict[str, Any]:
    """Parse the dedup LLM response into a list of full overlay dicts.

    Returns ``{"overlays": [overlay_0, overlay_1, ...]}``.  Each overlay is a
    complete overlay dict (same shape as the input) ready to be simulated.
    If a candidate is missing or malformed, the original overlay is used as-is.
    """
    import re as _re

    # Strip markdown fencing if present
    cleaned = raw.strip()
    fence_match = _re.search(r"```(?:json)?\s*\n?(.*?)```", cleaned, _re.DOTALL)
    if fence_match:
        cleaned = fence_match.group(1).strip()

    overlays: list[dict[str, Any]] = []
    try:
        data = json.loads(cleaned)
    except (json.JSONDecodeError, ValueError):
        print("  [dedup] WARNING: could not parse LLM response as JSON; using original overlay for all candidates")
        return {"overlays": [dict(fallback_overlay)] * num_candidates}

    if not isinstance(data, dict):
        return {"overlays": [dict(fallback_overlay)] * num_candidates}

    for i in range(num_candidates):
        key = f"candidate_{i}"
        candidate_data = data.get(key)
        if not isinstance(candidate_data, dict):
            overlays.append(dict(fallback_overlay))
            continue
        # Build a complete overlay from the candidate data
        overlay: dict[str, Any] = {}
        for field in ("persona.generation_guidance", "prompt.comment_style_guidance"):
            overlay[field] = str(candidate_data.get(field, fallback_overlay.get(field, "")))
        # Preserve _manual_phase_blocks if provided, otherwise keep original
        blocks = candidate_data.get("_manual_phase_blocks")
        if isinstance(blocks, dict) and blocks:
            overlay["_manual_phase_blocks"] = blocks
        else:
            orig_blocks = fallback_overlay.get("_manual_phase_blocks")
            if orig_blocks:
                overlay["_manual_phase_blocks"] = orig_blocks
        overlays.append(overlay)

    return {"overlays": overlays}


def _manual_phase_for_iteration(iteration: int) -> dict[str, Any]:
    """Return the deterministic phase spec for the manual schedule.

    Iterations 0-11 map to the 4 regular phase blocks.
    Iteration 12 is the dedup-only round.
    """
    if _is_dedup_iteration(iteration):
        phase = dict(_DEDUP_PHASE_SPEC)
        phase["phase_index"] = len(_MANUAL_PHASE_SPECS)
        phase["block_index"] = len(_MANUAL_PHASE_SPECS)
        return phase
    for idx, spec in enumerate(_MANUAL_PHASE_SPECS):
        if spec["iteration_start"] <= iteration <= spec["iteration_end"]:
            phase = dict(spec)
            phase["phase_index"] = idx
            phase["block_index"] = idx
            return phase
    phase = dict(_MANUAL_PHASE_SPECS[-1])
    phase["phase_index"] = len(_MANUAL_PHASE_SPECS) - 1
    phase["block_index"] = len(_MANUAL_PHASE_SPECS) - 1
    return phase


def _manual_phase_context(iteration: int) -> dict[str, Any]:
    """Return phase metadata plus protected metrics for the current iteration."""
    phase = _manual_phase_for_iteration(iteration)
    if phase.get("is_dedup"):
        # Dedup round: all metrics from all phases are protected; no focus metrics.
        all_metrics: list[str] = []
        for spec in _MANUAL_PHASE_SPECS:
            for metric in spec["focus_metrics"]:
                if metric not in all_metrics:
                    all_metrics.append(metric)
        phase["focus_metrics"] = []
        phase["protected_metrics"] = all_metrics
        phase["focus_metric_guidance"] = []
        phase["protected_metric_guidance"] = [
            {
                "metric": metric,
                "guidance": _MANUAL_METRIC_GUIDANCE.get(metric, ""),
            }
            for metric in all_metrics
            if metric in _MANUAL_METRIC_GUIDANCE
        ]
        phase["iteration_label"] = f"iter_{iteration + 1}_dedup"
        phase["block_label"] = f"iter_{iteration + 1} (dedup)"
        phase["candidate_plan"] = []
        return phase

    protected_metrics: list[str] = []
    for earlier in _MANUAL_PHASE_SPECS[: phase["phase_index"]]:
        for metric in earlier["focus_metrics"]:
            if metric not in protected_metrics:
                protected_metrics.append(metric)
    phase["focus_metrics"] = list(phase["focus_metrics"])
    phase["protected_metrics"] = protected_metrics
    phase["focus_metric_guidance"] = [
        {
            "metric": metric,
            "guidance": _MANUAL_METRIC_GUIDANCE.get(metric, ""),
        }
        for metric in phase["focus_metrics"]
    ]
    phase["protected_metric_guidance"] = [
        {
            "metric": metric,
            "guidance": _MANUAL_METRIC_GUIDANCE.get(metric, ""),
        }
        for metric in protected_metrics
        if metric in _MANUAL_METRIC_GUIDANCE
    ]
    phase["iteration_label"] = f"iter_{iteration + 1}"
    phase["block_label"] = f"iter_{phase['iteration_start'] + 1}-{phase['iteration_end'] + 1}"
    return phase


def _phase1_total_iterations(max_iterations: int) -> int:
    """Return total Phase-1 loop iterations, including baseline in manual mode."""
    return max_iterations + 1 if _MANUAL_PHASE_MODE else max_iterations


def _phase1_reported_iteration_count(completed_iterations: int) -> int:
    """Return the user-facing edited-iteration count for Phase 1 progress reporting."""
    if not _MANUAL_PHASE_MODE:
        return completed_iterations
    return max(0, completed_iterations - 1)


def _manual_phase_prompt_trajectory(
    trajectory: list[dict[str, Any]],
    phase_context: dict[str, Any],
    iteration: int,
) -> list[dict[str, Any]]:
    """Return all previous iterations inside the same manual phase block."""
    phase_name = str(phase_context.get("name", "")).strip()
    phase_start = int(phase_context.get("iteration_start", 0))
    phase_end = int(phase_context.get("iteration_end", phase_start))
    current_manual_iteration = max(0, iteration - 1)
    if not phase_name or current_manual_iteration <= phase_start:
        return []
    filtered: list[dict[str, Any]] = []
    for entry in trajectory:
        manual_ctx = ((entry.get("search_state", {}) or {}).get("manual_phase_context", {}) or {})
        if str(manual_ctx.get("name", "")).strip() != phase_name:
            continue
        entry_iteration_label = str(manual_ctx.get("iteration_label", "")).strip()
        if entry_iteration_label.startswith("iter_"):
            try:
                entry_manual_iteration = int(entry_iteration_label.split("_", 1)[1]) - 1
            except ValueError:
                continue
        else:
            continue
        if not (phase_start <= entry_manual_iteration <= phase_end):
            continue
        if entry_manual_iteration >= current_manual_iteration:
            continue
        filtered.append(entry)
    return filtered


def _manual_block_reference(state: "CalibrationState") -> dict[str, Any] | None:
    """Return the current manual phase-block incumbent payload, if any."""
    if state.manual_block_best_diagnostic and "quantile_fail_rate" in state.manual_block_best_diagnostic:
        return state.manual_block_best_diagnostic
    return state.manual_block_best_score


def _manual_start_block(
    state: "CalibrationState",
    phase_context: dict[str, Any],
) -> None:
    """Initialize a new phase block with no incumbent so the first iteration always wins.

    The base overlay (from previous phases) is kept as the search root for
    candidate generation, but the block-best score/diagnostic are cleared so
    that ``_manual_block_reference`` returns ``None``.  This guarantees the
    first iteration's winner is always saved — fulfilling the requirement that
    every phase must produce a block overlay.
    """
    state.manual_block_phase_name = str(phase_context.get("name", "")).strip() or None
    state.manual_block_best_overlay = dict(state.current_best_overlay)
    # Clear score/diagnostic so the first iteration has no incumbent to beat.
    state.manual_block_best_score = None
    state.manual_block_best_diagnostic = None
    state.manual_block_best_candidate_dir = None
    state.current_search_root_overlay = dict(state.current_best_overlay)
    state.current_search_root_diagnostic = state.current_best_diagnostic
    state.current_search_root_candidate_dir = state.current_best_candidate_dir
    state.current_search_root_mode = f"manual_phase:{phase_context.get('name')}"
    state.current_search_root_reason = "start new manual phase block from cumulative committed overlay"


def _manual_commit_block_best(
    state: "CalibrationState",
    phase_context: dict[str, Any],
) -> None:
    """Commit the current block incumbent into the cumulative overlay state."""
    phase_name = str(phase_context.get("name", "")).strip()
    if not phase_name:
        return
    if str(state.manual_block_phase_name or "").strip() != phase_name:
        return
    state.current_best_overlay = dict(state.manual_block_best_overlay)
    state.current_best_score = (
        dict(state.manual_block_best_score) if isinstance(state.manual_block_best_score, dict) else state.manual_block_best_score
    )
    state.current_best_diagnostic = (
        dict(state.manual_block_best_diagnostic) if isinstance(state.manual_block_best_diagnostic, dict) else state.manual_block_best_diagnostic
    )
    state.current_best_candidate_dir = state.manual_block_best_candidate_dir
    state.current_search_root_overlay = dict(state.current_best_overlay)
    state.current_search_root_diagnostic = state.current_best_diagnostic
    state.current_search_root_candidate_dir = state.current_best_candidate_dir
    state.current_search_root_mode = f"manual_phase:{phase_name}"
    state.current_search_root_reason = "committed block_best into cumulative overlay"
    _maybe_record_completed_phase_summary(state, phase_context)


def _subset_robust_stats(
    per_metric: dict[str, dict[str, Any]],
    metrics: list[str],
) -> dict[str, float | int]:
    """Aggregate robust scoring fields for an arbitrary metric subset."""
    items = [
        per_metric[m]
        for m in metrics
        if m in per_metric and per_metric[m].get("status") != "missing"
    ]
    if not items:
        return {
            "metric_count": 0,
            "out_of_range_count": 0,
            "mean_percentile_distance": float("inf"),
            "max_percentile_distance": float("inf"),
            "mean_abs_raw_robust_z": float("inf"),
            "max_abs_raw_robust_z": float("inf"),
        }
    percentile_distances = [float(item.get("percentile_distance", 0.0)) for item in items]
    raw_robust_zs = [float(item.get("abs_robust_z", 0.0)) for item in items]
    return {
        "metric_count": len(items),
        "out_of_range_count": sum(1 for item in items if item.get("status") != "in_range"),
        "mean_percentile_distance": float(np.mean(percentile_distances)),
        "max_percentile_distance": float(np.max(percentile_distances)),
        "mean_abs_raw_robust_z": float(np.mean(raw_robust_zs)),
        "max_abs_raw_robust_z": float(np.max(raw_robust_zs)),
    }


def _subset_group_eval_stats(
    per_metric: dict[str, dict[str, Any]],
    metrics: list[str],
) -> dict[str, float | int]:
    """Aggregate group-vs-real statistics for an arbitrary metric subset."""
    items = [per_metric[m] for m in metrics if m in per_metric]
    if not items:
        return {
            "metric_count": 0,
            "mwu_sig_count": 0,
            "ks_sig_count": 0,
            "mwu_pass_count": 0,
            "ks_pass_count": 0,
            "mean_wasserstein": float("inf"),
            "mean_quantile_error": float("inf"),
            "mean_empirical_fail_rate": float("inf"),
            "mean_abs_median_gap": float("inf"),
            "mean_abs_cliffs_delta": float("inf"),
        }
    wasserstein = [float(item.get("wasserstein_distance", float("inf"))) for item in items]
    quantile_error = [float(item.get("quantile_error", float("inf"))) for item in items]
    empirical_fail = [float(item.get("empirical_fail_rate", float("inf"))) for item in items]
    abs_median_gap = [abs(float(item.get("median_gap", item.get("abs_median_gap", float("inf"))))) for item in items]
    abs_cliffs = [abs(float(item.get("cliffs_delta", float("inf")))) for item in items]
    mwu_sig_count = sum(1 for item in items if float(item.get("mwu_p_value", 1.0)) <= 0.05)
    ks_sig_count = sum(1 for item in items if float(item.get("ks_p_value", 1.0)) <= 0.05)
    return {
        "metric_count": len(items),
        "mwu_sig_count": mwu_sig_count,
        "ks_sig_count": ks_sig_count,
        "mwu_pass_count": len(items) - mwu_sig_count,
        "ks_pass_count": len(items) - ks_sig_count,
        "mean_wasserstein": float(np.mean(wasserstein)),
        "mean_quantile_error": float(np.mean(quantile_error)),
        "mean_empirical_fail_rate": float(np.mean(empirical_fail)),
        "mean_abs_median_gap": float(np.mean(abs_median_gap)),
        "mean_abs_cliffs_delta": float(np.mean(abs_cliffs)),
    }


def _manual_phase_metric_rows(
    candidate: dict[str, Any],
    metrics: list[str],
) -> list[dict[str, Any]]:
    """Return ordered per-metric comparison rows for manual-phase selection."""
    robust_per_metric = candidate.get("per_metric", {}) or {}
    group_eval_per_metric = candidate.get("group_eval_per_metric", {}) or {}
    rows: list[dict[str, Any]] = []
    for metric in metrics:
        robust = robust_per_metric.get(metric, {}) or {}
        group = group_eval_per_metric.get(metric, {}) or {}
        status = str(robust.get("status", "missing"))
        mwu_p = float(group.get("mwu_p_value", 1.0))
        ks_p = float(group.get("ks_p_value", 1.0))
        rows.append(
            {
                "metric": metric,
                "wasserstein": float(group.get("wasserstein_distance", float("inf"))),
                "quantile_error": float(group.get("quantile_error", float("inf"))),
                "empirical_fail_rate": float(group.get("empirical_fail_rate", float("inf"))),
                "abs_median_gap": abs(float(group.get("median_gap", group.get("abs_median_gap", float("inf"))))),
                "abs_cliffs_delta": abs(float(group.get("cliffs_delta", float("inf")))),
                "mwu_sig": int(mwu_p <= 0.05),
                "ks_sig": int(ks_p <= 0.05),
                "mwu_p_value": mwu_p,
                "ks_p_value": ks_p,
                "out_of_range": 0 if status == "in_range" else 1,
                "percentile_distance": float(robust.get("percentile_distance", float("inf"))),
                "abs_raw_robust_z": float(robust.get("abs_robust_z", float("inf"))),
                "status": status,
            }
        )
    return rows


def _target_metric_eval_summary(
    candidate: dict[str, Any],
    metrics: list[str] | None = None,
) -> dict[str, Any]:
    """Return a compact all-target snapshot for one candidate or winner."""
    target_metrics = list(metrics or _TARGET_METRICS_12)
    rows = _manual_phase_metric_rows(candidate, target_metrics)
    if not rows:
        return {
            "metrics": target_metrics,
            "rows": [],
            "mwu_pass_count": 0,
            "ks_pass_count": 0,
            "mean_wasserstein": float("inf"),
            "mean_quantile_error": float("inf"),
            "mean_empirical_fail_rate": float("inf"),
            "mean_abs_median_gap": float("inf"),
            "mean_abs_cliffs_delta": float("inf"),
        }
    return {
        "metrics": target_metrics,
        "rows": rows,
        "mwu_pass_count": sum(1 for row in rows if float(row.get("mwu_p_value", 0.0)) > 0.05),
        "ks_pass_count": sum(1 for row in rows if float(row.get("ks_p_value", 0.0)) > 0.05),
        "mean_wasserstein": float(np.mean([float(row.get("wasserstein", float("inf"))) for row in rows])),
        "mean_quantile_error": float(np.mean([float(row.get("quantile_error", float("inf"))) for row in rows])),
        "mean_empirical_fail_rate": float(np.mean([float(row.get("empirical_fail_rate", float("inf"))) for row in rows])),
        "mean_abs_median_gap": float(np.mean([float(row.get("abs_median_gap", float("inf"))) for row in rows])),
        "mean_abs_cliffs_delta": float(np.mean([float(row.get("abs_cliffs_delta", float("inf"))) for row in rows])),
    }


def _manual_guard_threshold(
    baseline: float,
    *,
    multiplier: float,
    additive_floor: float,
) -> float:
    """Return a bounded regression threshold around a protected baseline value."""
    return max(baseline * multiplier, baseline + additive_floor)


def _manual_phase_guard_summary(
    candidate: dict[str, Any],
    reference_payload: dict[str, Any] | None,
    phase_context: dict[str, Any],
) -> dict[str, Any]:
    """Detect regressions on protected metrics using Cliff's delta and
    Wasserstein distance only (no p-values).

    A violation fires when a protected metric's |Cliff's delta| or Wasserstein
    distance increases by more than a relative tolerance compared to the
    reference (previous best).  This ensures earlier gains are preserved while
    the current phase optimizes its focus metrics.

    Tolerance: a protected metric is violated when its candidate value exceeds
    ``max(ref * 1.5, ref + 0.05)`` for either |Cliff's delta| or Wasserstein.
    """
    protected_metrics = list(phase_context.get("protected_metrics", []))
    if not protected_metrics or not reference_payload:
        return {
            "protected_metric_count": len(protected_metrics),
            "violation_count": 0,
            "max_severity": 0.0,
            "violations": [],
        }

    candidate_rows = {
        row["metric"]: row
        for row in _manual_phase_metric_rows(candidate, protected_metrics)
    }
    reference_rows = {
        row["metric"]: row
        for row in _manual_phase_metric_rows(reference_payload, protected_metrics)
    }

    violations: list[dict[str, Any]] = []
    max_severity = 0.0

    for metric in protected_metrics:
        cand_row = candidate_rows.get(metric)
        ref_row = reference_rows.get(metric)
        if not cand_row or not ref_row:
            continue

        ref_cd = float(ref_row.get("abs_cliffs_delta", float("inf")))
        cand_cd = float(cand_row.get("abs_cliffs_delta", float("inf")))
        ref_w = float(ref_row.get("wasserstein", float("inf")))
        cand_w = float(cand_row.get("wasserstein", float("inf")))

        triggered: list[str] = []
        # Check Cliff's delta regression
        cd_threshold = _manual_guard_threshold(ref_cd, multiplier=1.5, additive_floor=0.05)
        if cand_cd > cd_threshold and ref_cd < float("inf"):
            triggered.append("cliffs_delta_regressed")
        # Check Wasserstein regression
        w_threshold = _manual_guard_threshold(ref_w, multiplier=1.5, additive_floor=0.05)
        if cand_w > w_threshold and ref_w < float("inf"):
            triggered.append("wasserstein_regressed")

        if triggered:
            # Severity = how much worse the candidate is relative to reference
            cd_ratio = (cand_cd / max(ref_cd, 1e-9)) if ref_cd < float("inf") else 1.0
            w_ratio = (cand_w / max(ref_w, 1e-9)) if ref_w < float("inf") else 1.0
            severity = max(cd_ratio, w_ratio)
            max_severity = max(max_severity, severity)
            violations.append(
                {
                    "metric": metric,
                    "triggered_fields": triggered,
                    "severity": severity,
                    "reference": ref_row,
                    "candidate": cand_row,
                }
            )

    return {
        "protected_metric_count": len(protected_metrics),
        "violation_count": len(violations),
        "max_severity": float(max_severity),
        "violations": violations,
    }


def _manual_metric_row_key(row: dict[str, Any]) -> tuple[float, ...]:
    """Return the comparison key for one metric.

    Lower key = better candidate.  Ranking uses only Cliff's delta and
    Wasserstein distance — both should be driven toward 0.

    1. |Cliff's delta| — lower is better (closer to 0 = distributions match).
    2. Wasserstein distance — lower is better (full-distribution shape match).
    3. |median gap| — tie-break on center-location mismatch.
    """
    return (
        float(row.get("abs_cliffs_delta", float("inf"))),
        float(row.get("wasserstein", float("inf"))),
        float(row.get("abs_median_gap", float("inf"))),
    )


def _manual_phase_score(
    candidate: dict[str, Any],
    phase_context: dict[str, Any],
) -> dict[str, Any]:
    """Build a phase-specific score using only the targeted metrics plus protected ones."""
    per_metric = candidate.get("per_metric", {}) or {}
    group_eval_per_metric = candidate.get("group_eval_per_metric", {}) or {}
    focus_metrics = list(phase_context.get("focus_metrics", []))
    protected_metrics = list(phase_context.get("protected_metrics", []))
    return {
        "phase_name": phase_context.get("name"),
        "focus_metrics": focus_metrics,
        "protected_metrics": protected_metrics,
        "focus_metric_rows": _manual_phase_metric_rows(candidate, focus_metrics),
        "protected_metric_rows": _manual_phase_metric_rows(candidate, protected_metrics),
        "focus_robust": _subset_robust_stats(per_metric, focus_metrics),
        "focus_group_eval": _subset_group_eval_stats(group_eval_per_metric, focus_metrics),
        "protected_robust": _subset_robust_stats(per_metric, protected_metrics),
        "protected_group_eval": _subset_group_eval_stats(group_eval_per_metric, protected_metrics),
    }


def _manual_phase_selection_key(candidate: dict[str, Any], phase_context: dict[str, Any]) -> tuple[float, ...]:
    """Return the deterministic selection key for the current manual phase.

    Selection uses only Cliff's delta and Wasserstein distance — no p-values.
    The goal is to drive both statistics toward 0 for ALL tracked metrics.

    Ranking tiers (lower is better for every component):
    1. Guard violation count — protected metrics whose |cd| or Wasserstein
       regressed beyond tolerance.  Fewer violations first.
    2. Mean |Cliff's delta| across ALL tracked metrics — overall effect-size
       proximity to real distribution.  Lower is better.
    3. Mean Wasserstein across ALL tracked metrics — overall distributional
       shape match.  Lower is better.
    4. Per-focus-metric tie-break using _manual_metric_row_key (cd, W, |med|).
    """
    phase_score = candidate.get("manual_phase_score") or _manual_phase_score(candidate, phase_context)
    guard = candidate.get("manual_phase_guard") or {}

    # Global mean |Cliff's delta| and mean Wasserstein across ALL group_eval metrics
    group_eval = candidate.get("group_eval_per_metric", {}) or {}
    cd_values: list[float] = []
    w_values: list[float] = []
    for _metric_name, metric_info in group_eval.items():
        cd = abs(float(metric_info.get("cliffs_delta", float("inf"))))
        w = float(metric_info.get("wasserstein_distance", float("inf")))
        if cd < float("inf"):
            cd_values.append(cd)
        if w < float("inf"):
            w_values.append(w)

    mean_cd = float(np.mean(cd_values)) if cd_values else float("inf")
    mean_w = float(np.mean(w_values)) if w_values else float("inf")

    focus_rows = phase_score.get("focus_metric_rows", [])

    key: list[float] = [
        float(guard.get("violation_count", 0)),   # fewer guard violations first
        mean_cd,                                    # lower mean |Cliff's delta| first
        mean_w,                                     # lower mean Wasserstein first
    ]
    # Per-focus-metric tie-break
    for row in focus_rows:
        key.extend(_manual_metric_row_key(row))
    return tuple(key)
