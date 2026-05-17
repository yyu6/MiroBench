"""Small formatting helpers for reasoner prompts.

Extracted from reasoner.py to keep file size manageable.
"""
from __future__ import annotations

from typing import Any


def _fmt_float(value: Any, digits: int = 4) -> str:
    """Format *value* as a float string when possible, else return ``N/A``."""
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "N/A"


def _fmt_delta_vs_previous(
    current: Any,
    previous: Any,
    *,
    digits: int = 4,
    lower_is_better: bool = True,
) -> str:
    """Return a compact delta string vs. the previous iteration."""
    try:
        current_f = float(current)
        previous_f = float(previous)
    except (TypeError, ValueError):
        return "vs prev: N/A"

    delta = current_f - previous_f
    if abs(delta) < (10 ** (-digits)):
        status = "flat"
    else:
        improved = delta < 0 if lower_is_better else delta > 0
        status = "better" if improved else "worse"
    return f"vs prev: Δ={delta:+.{digits}f} ({status})"


def _winner_headline_metrics(entry: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Return headline metrics for the selected winner in a trajectory entry."""
    winner_id = entry.get("selection", {}).get("winner_candidate_id")
    for candidate in entry.get("candidate_strategies", []):
        if candidate.get("candidate_id") == winner_id:
            metrics = candidate.get("headline_metrics", {})
            if isinstance(metrics, dict):
                return metrics
    return {}


def _headline_metric_delta_lines(
    current_entry: dict[str, Any],
    previous_entry: dict[str, Any] | None,
) -> list[str]:
    """Summarize winner headline-metric movement vs. the previous iteration."""
    if previous_entry is None:
        return []

    current_metrics = _winner_headline_metrics(current_entry)
    previous_metrics = _winner_headline_metrics(previous_entry)
    if not current_metrics or not previous_metrics:
        return []

    lines: list[str] = []
    for metric_name in sorted(set(current_metrics) & set(previous_metrics)):
        curr = current_metrics.get(metric_name, {})
        prev = previous_metrics.get(metric_name, {})
        try:
            curr_sim = float(curr.get("sim_median"))
            prev_sim = float(prev.get("sim_median"))
            curr_real = float(curr.get("real_median"))
            prev_real = float(prev.get("real_median"))
        except (TypeError, ValueError):
            continue

        prev_gap = abs(prev_sim - prev_real)
        curr_gap = abs(curr_sim - curr_real)
        improvement = prev_gap - curr_gap
        if abs(improvement) < 1e-9:
            movement = "flat"
        else:
            movement = "closer_to_real" if improvement > 0 else "farther_from_real"

        lines.append(
            "        "
            f"{metric_name}: sim {prev_sim:.3f}->{curr_sim:.3f} "
            f"(real≈{curr_real:.3f}, gap {prev_gap:.3f}->{curr_gap:.3f}, "
            f"Δgap={-improvement:+.3f}, {movement})"
        )
    return lines


def _rank_groups_by_severity(diagnostic: dict[str, Any]) -> list[str]:
    """Return metric-group names ordered from worst to best."""
    group_scores = diagnostic.get("group_scores", {}) or {}
    return [
        name
        for name, _info in sorted(
            group_scores.items(),
            key=lambda item: (
                float(item[1].get("quantile_fail_rate", 0.0)),
                float(item[1].get("mean_percentile_distance", 0.0)),
                float(item[1].get("mean_abs_robust_z", 0.0)),
            ),
            reverse=True,
        )
    ]
