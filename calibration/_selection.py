"""Candidate scoring, frontier, and search-root selection.

Extracted from orchestrator.py to keep file size manageable.
Logic unchanged.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from .orchestrator import CalibrationState


def _group_eval_selection_summary(group_eval: dict) -> dict[str, float | int]:
    """Summarize group-level evaluation into candidate selection metrics."""
    per_metric = group_eval.get("per_metric", group_eval)
    if not per_metric:
        return {
            "group_mean_abs_cliffs_delta": float("inf"),
            "group_overall_fail_rate": float("inf"),
            "group_metrics_sig_different": 0,
        }

    abs_deltas = [abs(float(info.get("cliffs_delta", 0.0))) for info in per_metric.values()]
    fail_rates = [float(info.get("empirical_fail_rate", 0.0)) for info in per_metric.values()]
    sig_count = sum(
        1
        for info in per_metric.values()
        if float(info.get("mwu_p_value", 1.0)) < 0.05
        or float(info.get("ks_p_value", 1.0)) < 0.05
    )
    return {
        "group_mean_abs_cliffs_delta": float(np.mean(abs_deltas)) if abs_deltas else 0.0,
        "group_overall_fail_rate": float(np.mean(fail_rates)) if fail_rates else 0.0,
        "group_metrics_sig_different": sig_count,
    }


def _candidate_selection_key(candidate: dict[str, Any]) -> tuple[float, ...]:
    """Return the comparison key for candidate selection and incumbent checks."""
    if "quantile_fail_rate" in candidate:
        return candidate_selection_key(candidate)

    if (
        "group_mean_abs_cliffs_delta" in candidate
        or "group_overall_fail_rate" in candidate
    ):
        return (
            float(candidate.get("group_mean_abs_cliffs_delta", candidate.get("mean_abs_delta", float("inf")))),
            float(candidate.get("group_overall_fail_rate", candidate.get("fail_rate", float("inf")))),
            float(candidate.get("mean_abs_delta", float("inf"))),
            float(candidate.get("fail_rate", float("inf"))),
        )

    return (
        float(candidate.get("fail_rate", float("inf"))),
        float(candidate.get("mean_abs_delta", float("inf"))),
    )


def _current_best_selection_reference(state: "CalibrationState") -> dict[str, Any] | None:
    """Return the richest persisted incumbent payload for winner comparison."""
    if state.current_best_diagnostic and "quantile_fail_rate" in state.current_best_diagnostic:
        return state.current_best_diagnostic
    return state.current_best_score


def _group_score_key(group_info: dict[str, Any] | None) -> tuple[float, ...]:
    """Return a stable comparison key for one metric-group summary."""
    if not group_info:
        return (float("inf"), float("inf"), float("inf"))
    return (
        float(group_info.get("quantile_fail_rate", float("inf"))),
        float(group_info.get("mean_percentile_distance", float("inf"))),
        float(group_info.get("mean_abs_robust_z", float("inf"))),
    )


def _group_severity_key(group_info: dict[str, Any] | None) -> tuple[float, ...]:
    """Return a severity key where larger values mean the group is worse."""
    if not group_info:
        return (float("-inf"), float("-inf"), float("-inf"))
    return (
        float(group_info.get("quantile_fail_rate", 0.0)),
        float(group_info.get("mean_percentile_distance", 0.0)),
        float(group_info.get("mean_abs_robust_z", 0.0)),
    )


def _worst_group_order(score: dict[str, Any] | None) -> list[str]:
    """Return group names sorted from worst to best for the provided score."""
    if not score:
        return []
    group_scores = score.get("group_scores", {}) or {}
    return [
        name
        for name, _info in sorted(
            group_scores.items(),
            key=lambda item: _group_severity_key(item[1]),
            reverse=True,
        )
    ]


def _stagnation_count_from_entries(entries: list[dict[str, Any]]) -> int:
    """Return the number of consecutive iterations without a new best."""
    count = 0
    for entry in reversed(entries):
        if entry.get("selection", {}).get("beat_current_best", False):
            break
        count += 1
    return count


def _slim_candidate_diagnostic(candidate: dict[str, Any]) -> dict[str, Any]:
    """Return a persisted diagnostic payload for a scored candidate."""
    return {
        k: v
        for k, v in candidate.items()
        if k not in {"candidate_id", "candidate_dir", "overlay"}
    }


def _make_frontier_entry(
    candidate: dict[str, Any],
    preview: dict[str, Any] | None,
    iteration: int,
) -> dict[str, Any]:
    """Persist a scored candidate as a reusable frontier/search-root entry."""
    return {
        "iteration": iteration,
        "candidate_id": candidate.get("candidate_id"),
        "candidate_dir": candidate.get("candidate_dir"),
        "overlay": candidate.get("overlay", {}),
        "diagnostic": _slim_candidate_diagnostic(candidate),
        "strategy_label": (preview or {}).get("strategy_label", candidate.get("strategy_label", "")),
        "strategy": (preview or {}).get("strategy", ""),
        "primary_layer": (preview or {}).get("primary_layer", "both"),
        "mechanism_family": (preview or {}).get("mechanism_family", candidate.get("mechanism_family", "mixed")),
        "anti_incumbent": bool((preview or {}).get("anti_incumbent", False)),
    }


def _update_frontier(
    frontier: dict[str, dict[str, Any]],
    scored: list[dict[str, Any]],
    preview_by_id: dict[int, dict[str, Any]],
    iteration: int,
) -> dict[str, dict[str, Any]]:
    """Update the per-group frontier with any newly superior candidates."""
    updated = dict(frontier or {})
    for candidate in scored:
        group_scores = candidate.get("group_scores", {}) or {}
        preview = preview_by_id.get(int(candidate.get("candidate_id", -1)), {})
        for group_name, group_info in group_scores.items():
            existing = updated.get(group_name)
            candidate_key = _group_score_key(group_info) + _candidate_selection_key(candidate)
            existing_key = (
                _group_score_key((existing or {}).get("diagnostic", {}).get("group_scores", {}).get(group_name))
                + _candidate_selection_key((existing or {}).get("diagnostic", {}))
                if existing
                else (float("inf"),) * 8
            )
            if existing is None or candidate_key < existing_key:
                updated[group_name] = _make_frontier_entry(candidate, preview, iteration)
    return updated


def _frontier_prompt_summary(frontier: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Return a compact frontier summary suitable for the reasoner prompt."""
    summary: dict[str, dict[str, Any]] = {}
    for group_name, entry in (frontier or {}).items():
        diagnostic = entry.get("diagnostic", {})
        group_info = (diagnostic.get("group_scores", {}) or {}).get(group_name, {})
        summary[group_name] = {
            "iteration": entry.get("iteration"),
            "candidate_id": entry.get("candidate_id"),
            "strategy_label": entry.get("strategy_label"),
            "mechanism_family": entry.get("mechanism_family"),
            "primary_layer": entry.get("primary_layer"),
            "anti_incumbent": bool(entry.get("anti_incumbent", False)),
            "group_quantile_fail_rate": group_info.get("quantile_fail_rate"),
            "group_mean_percentile_distance": group_info.get("mean_percentile_distance"),
            "group_mean_abs_robust_z": group_info.get("mean_abs_robust_z"),
            "overall_quantile_fail_rate": diagnostic.get("quantile_fail_rate"),
            "overall_mean_percentile_distance": diagnostic.get("mean_percentile_distance"),
            "overall_mean_abs_robust_z": diagnostic.get("mean_abs_robust_z"),
        }
    return summary


def _choose_search_root(
    state: "CalibrationState",
) -> tuple[dict[str, Any], dict[str, Any] | None, str | None, str, str]:
    """Choose the overlay/diagnostic that the next iteration should branch from."""
    default_reason = "global_best"
    if not state.current_best_overlay:
        return {}, None, None, "global_best", default_reason

    if state.stagnation_count < _STAGNATION_TRIGGER:
        return (
            state.current_best_overlay,
            state.current_best_diagnostic,
            state.current_best_candidate_dir,
            "global_best",
            default_reason,
        )

    current_best_score = state.current_best_score or {}
    current_best_dir = state.current_best_candidate_dir
    frontier = state.frontier or {}
    worst_groups = _worst_group_order(current_best_score)
    for group_name in worst_groups:
        frontier_entry = frontier.get(group_name)
        if not frontier_entry:
            continue
        if frontier_entry.get("candidate_dir") == current_best_dir:
            continue
        challenger_group = (
            frontier_entry.get("diagnostic", {})
            .get("group_scores", {})
            .get(group_name, {})
        )
        incumbent_group = current_best_score.get("group_scores", {}).get(group_name, {})
        if _group_score_key(challenger_group) < _group_score_key(incumbent_group):
            return (
                frontier_entry.get("overlay", {}) or state.current_best_overlay,
                frontier_entry.get("diagnostic") or state.current_best_diagnostic,
                frontier_entry.get("candidate_dir") or state.current_best_candidate_dir,
                "challenger_root",
                (
                    f"stagnation_count={state.stagnation_count}; branching from frontier"
                    f" best for worst group '{group_name}' via "
                    f"{frontier_entry.get('strategy_label', 'candidate')}"
                ),
            )

    return (
        state.current_best_overlay,
        state.current_best_diagnostic,
        state.current_best_candidate_dir,
        "global_best",
        f"stagnation_count={state.stagnation_count}; no challenger frontier entry beat global_best on its target group",
    )


