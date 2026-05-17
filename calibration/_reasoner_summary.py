"""Trajectory and family learning summary for reasoner prompts.

Extracted from reasoner.py.
"""
from __future__ import annotations

from typing import Any


def _family_learning_summary(trajectory: list[dict[str, Any]]) -> dict[str, Any]:
    """Compress trajectory history into mechanism-family learnings."""
    summary: dict[str, Any] = {}

    for family in _MECHANISM_FAMILIES:
        attempts = 0
        winner_count = 0
        best_overall: dict[str, Any] | None = None
        best_group_fits: dict[str, dict[str, Any]] = {}

        for entry in trajectory:
            winner_id = entry.get("selection", {}).get("winner_candidate_id")
            iteration = entry.get("iteration")
            for candidate in entry.get("candidate_strategies", []):
                if candidate.get("mechanism_family") != family:
                    continue
                attempts += 1
                if candidate.get("candidate_id") == winner_id:
                    winner_count += 1

                candidate_key = (
                    float(candidate.get("quantile_fail_rate", float("inf"))),
                    float(candidate.get("mean_percentile_distance", float("inf"))),
                    float(candidate.get("mean_abs_robust_z", float("inf"))),
                )
                best_key = (
                    float(best_overall.get("quantile_fail_rate", float("inf"))),
                    float(best_overall.get("mean_percentile_distance", float("inf"))),
                    float(best_overall.get("mean_abs_robust_z", float("inf"))),
                ) if best_overall else (float("inf"), float("inf"), float("inf"))
                if best_overall is None or candidate_key < best_key:
                    best_overall = {
                        "iteration": iteration,
                        "candidate_id": candidate.get("candidate_id"),
                        "strategy_label": candidate.get("strategy_label"),
                        "primary_layer": candidate.get("primary_layer"),
                        "quantile_fail_rate": candidate.get("quantile_fail_rate"),
                        "mean_percentile_distance": candidate.get("mean_percentile_distance"),
                        "mean_abs_robust_z": candidate.get("mean_abs_robust_z"),
                    }

                for group_name, group_info in (candidate.get("group_scores", {}) or {}).items():
                    group_key = (
                        float(group_info.get("quantile_fail_rate", float("inf"))),
                        float(group_info.get("mean_percentile_distance", float("inf"))),
                        float(group_info.get("mean_abs_robust_z", float("inf"))),
                    )
                    existing = best_group_fits.get(group_name)
                    existing_key = (
                        float(existing.get("quantile_fail_rate", float("inf"))),
                        float(existing.get("mean_percentile_distance", float("inf"))),
                        float(existing.get("mean_abs_robust_z", float("inf"))),
                    ) if existing else (float("inf"), float("inf"), float("inf"))
                    if existing is None or group_key < existing_key:
                        best_group_fits[group_name] = {
                            "iteration": iteration,
                            "candidate_id": candidate.get("candidate_id"),
                            "strategy_label": candidate.get("strategy_label"),
                            "primary_layer": candidate.get("primary_layer"),
                            "quantile_fail_rate": group_info.get("quantile_fail_rate"),
                            "mean_percentile_distance": group_info.get("mean_percentile_distance"),
                            "mean_abs_robust_z": group_info.get("mean_abs_robust_z"),
                        }

        if attempts == 0:
            continue

        summary[family] = {
            "attempts": attempts,
            "winner_count": winner_count,
            "win_rate": winner_count / attempts,
            "best_overall": best_overall or {},
            "best_group_fits": best_group_fits,
        }

    return summary
