"""Display/printing helpers for the calibration orchestrator.

Pure formatting and console output — extracted from orchestrator.py to keep
file sizes manageable. Logic unchanged.
"""
from __future__ import annotations

import math as _math
from typing import Any

from ._phase_specs import _HEADLINE_METRICS
from ._manual_phase import _manual_phase_score, _manual_phase_selection_key


def _fmt(v: float, fmt: str = ".4f") -> str:
    return f"{v:{fmt}}" if not _math.isnan(v) else "  N/A"


def _fmt_signed(v: float, fmt: str = ".1f") -> str:
    return f"{v:+{fmt}}" if not _math.isnan(v) else "   N/A"


def _print_candidate_score_summary(score: dict) -> None:
    """Print a compact robust-distribution summary for headline metrics."""
    pm = score.get("per_metric", {})
    if not pm:
        return

    has_robust = any("sim_median" in pm.get(key, {}) for key, _ in _HEADLINE_METRICS if key in pm)
    if not has_robust:
        print(f"  {'Metric':<28} {'real_med':>8} {'gen_med':>8} {'fail%':>6} {'direction'}")
        print(f"  {'-'*28} {'-'*8} {'-'*8} {'-'*6} {'-'*16}")
        for key, label in _HEADLINE_METRICS:
            if key not in pm:
                continue
            m = pm[key]
            real_med = float(m.get("real_median", float("nan")))
            gen_med = float(m.get("generated_median", float("nan")))
            fail_pct = float(m.get("fail_rate", 0.0)) * 100
            direction = m.get("direction", "")
            print(f"  {label:<28} {_fmt(real_med):>8} {_fmt(gen_med):>8} {fail_pct:>5.1f}% {direction}")
        return

    print(
        f"  {'metric':<28} {'sim_med':>8} {'real_p10':>10} {'real_p50':>10} "
        f"{'real_p90':>10} {'pct_rank':>9} {'robust_z':>10} {'status'}"
    )
    print(
        f"  {'-'*28} {'-'*8} {'-'*10} {'-'*10} "
        f"{'-'*10} {'-'*9} {'-'*10} {'-'*10}"
    )
    for key, label in _HEADLINE_METRICS:
        if key not in pm:
            continue
        m = pm[key]
        sim_med = float(m.get("sim_median", m.get("generated_median", float("nan"))))
        real_p10 = float(m.get("real_p10", float("nan")))
        real_p50 = float(m.get("real_median", float("nan")))
        real_p90 = float(m.get("real_p90", float("nan")))
        pct_rank = float(m.get("percentile_rank", float("nan")))
        robust_z = float(m.get("robust_z", float("nan")))
        status = m.get("status", "N/A")
        print(
            f"  {label:<28} {_fmt(sim_med):>8} {_fmt(real_p10):>10} {_fmt(real_p50):>10} "
            f"{_fmt(real_p90):>10} {_fmt(pct_rank, '.2f'):>9} {_fmt_signed(robust_z):>10} {status}"
        )


def _print_group_eval_summary(group_eval: dict, label_a: str = "group_a", label_b: str = "group_b") -> None:
    """Print Cliff's delta and significance for headline metrics from evaluate_group_vs_real output.

    Accepts both the flat {metric: stats} shape returned by evaluate_group_vs_real
    and the wrapped {"per_metric": {metric: stats}} shape.
    """
    pm = group_eval.get("per_metric", group_eval)
    if not pm:
        return
    print(f"  {'Metric':<28} {'cliff_d':>8} {'mwu_p':>10} {'sig?':>5}")
    print(f"  {'-'*28} {'-'*8} {'-'*10} {'-'*5}")
    for key, label in _HEADLINE_METRICS:
        if key not in pm:
            continue
        m = pm[key]
        cd    = float(m.get("cliffs_delta", float("nan")))
        mwu_p = float(m.get("mwu_p_value", float("nan")))
        sig   = "YES" if not _math.isnan(mwu_p) and mwu_p < 0.05 else "no"
        p_str = f"{mwu_p:.2e}" if not _math.isnan(mwu_p) else "    N/A"
        print(f"  {label:<28} {_fmt(cd):>8} {p_str:>10} {sig:>5}")


def _print_improvement_table(improvement: dict) -> None:
    """Print before→after per-headline-metric improvement table."""
    pm = improvement.get("per_metric", {})
    if not pm:
        return
    print(f"  {'Metric':<28} {'cd_before':>9} {'cd_after':>9} {'Δfail%':>7} {'impr?':>5}")
    print(f"  {'-'*28} {'-'*9} {'-'*9} {'-'*7} {'-'*5}")
    for key, label in _HEADLINE_METRICS:
        if key not in pm:
            continue
        m = pm[key]
        cd_b  = float(m.get("before_cliffs_delta", float("nan")))
        cd_a  = float(m.get("after_cliffs_delta",  float("nan")))
        dfail = float(m.get("fail_rate_reduction") or 0.0) * 100
        impr  = "YES" if m.get("improved") else "no"
        d_str = f"{dfail:+.1f}%" if not _math.isnan(dfail) else "  N/A"
        print(f"  {label:<28} {_fmt(cd_b):>9} {_fmt(cd_a):>9} {d_str:>7} {impr:>5}")


def _selection_ranking_rows(scored_candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return candidates sorted by the active selection key with compact fields."""
    ranked = sorted(scored_candidates, key=_candidate_selection_key)
    rows: list[dict[str, Any]] = []
    for rank, candidate in enumerate(ranked, start=1):
        family_scores = candidate.get("selection_family_scores", {}) or {}
        guardrail = family_scores.get("guardrail_core", {}) or {}
        semantic = family_scores.get("semantic_core", {}) or {}
        engagement = family_scores.get("engagement_core", {}) or {}
        length = family_scores.get("length_core", {}) or {}
        rows.append(
            {
                "rank": rank,
                "candidate_id": candidate.get("candidate_id"),
                "strategy_label": candidate.get("strategy_label", ""),
                "mechanism_family": candidate.get("mechanism_family", ""),
                "primary_layer": candidate.get("primary_layer", ""),
                "guardrail_out_of_range_count": int(guardrail.get("out_of_range_count", 0)),
                "guardrail_max_percentile_distance": float(
                    guardrail.get("max_percentile_distance", float("nan"))
                ),
                "semantic_out_of_range_count": int(semantic.get("out_of_range_count", 0)),
                "semantic_mean_percentile_distance": float(
                    semantic.get("mean_percentile_distance", float("nan"))
                ),
                "semantic_max_percentile_distance": float(
                    semantic.get("max_percentile_distance", float("nan"))
                ),
                "semantic_mean_abs_robust_z": float(
                    semantic.get("mean_abs_robust_z", float("nan"))
                ),
                "semantic_mean_abs_raw_robust_z": float(
                    semantic.get("mean_abs_raw_robust_z", float("nan"))
                ),
                "engagement_out_of_range_count": int(engagement.get("out_of_range_count", 0)),
                "engagement_mean_percentile_distance": float(
                    engagement.get("mean_percentile_distance", float("nan"))
                ),
                "engagement_max_percentile_distance": float(
                    engagement.get("max_percentile_distance", float("nan"))
                ),
                "engagement_mean_abs_robust_z": float(
                    engagement.get("mean_abs_robust_z", float("nan"))
                ),
                "engagement_mean_abs_raw_robust_z": float(
                    engagement.get("mean_abs_raw_robust_z", float("nan"))
                ),
                "length_mean_percentile_distance": float(
                    length.get("mean_percentile_distance", float("nan"))
                ),
                "length_mean_abs_raw_robust_z": float(
                    length.get("mean_abs_raw_robust_z", float("nan"))
                ),
                "quantile_fail_rate": float(candidate.get("quantile_fail_rate", float("nan"))),
                "mean_percentile_distance": float(
                    candidate.get("mean_percentile_distance", float("nan"))
                ),
                "mean_abs_robust_z": float(candidate.get("mean_abs_robust_z", float("nan"))),
                "ranking_mean_abs_delta": float(
                    candidate.get("ranking_mean_abs_delta", float("nan"))
                ),
                "ranking_fail_rate": float(candidate.get("ranking_fail_rate", float("nan"))),
            }
        )
    return rows


def _print_selection_ranking(scored_candidates: list[dict[str, Any]]) -> None:
    """Print the actual selection ordering used for best-candidate choice."""
    rows = _selection_ranking_rows(scored_candidates)
    if not rows:
        return

    print("    selection ranking (best→worst by actual winner key):")
    print(
        f"    {'cand':<6} {'g_oor':>5} {'s_oor':>5} {'s_mean':>7} {'s_max':>6} "
        f"{'s_rawz':>7} {'e_oor':>5} {'e_mean':>7} {'e_rawz':>7} {'l_mean':>7} "
        f"{'qfail':>7} {'pct':>7} {'r_z':>6} {'r|d|':>6} {'r_fail':>7}"
    )
    print(
        f"    {'-'*6} {'-'*5} {'-'*5} {'-'*7} {'-'*6} "
        f"{'-'*7} {'-'*5} {'-'*7} {'-'*7} {'-'*7} "
        f"{'-'*7} {'-'*6} {'-'*6} {'-'*7}"
    )
    for row in rows:
        print(
            f"    c{row['candidate_id']!s:<5} "
            f"{row['guardrail_out_of_range_count']:>5d} "
            f"{row['semantic_out_of_range_count']:>5d} "
            f"{_fmt(row['semantic_mean_percentile_distance'], '.2f'):>7} "
            f"{_fmt(row['semantic_max_percentile_distance'], '.2f'):>6} "
            f"{_fmt(row['semantic_mean_abs_raw_robust_z'], '.2f'):>7} "
            f"{row['engagement_out_of_range_count']:>5d} "
            f"{_fmt(row['engagement_mean_percentile_distance'], '.2f'):>7} "
            f"{_fmt(row['engagement_mean_abs_raw_robust_z'], '.2f'):>7} "
            f"{_fmt(row['length_mean_percentile_distance'], '.2f'):>7} "
            f"{_fmt(row['quantile_fail_rate'], '.4f'):>7} "
            f"{_fmt(row['mean_percentile_distance'], '.4f'):>7} "
            f"{_fmt(row['mean_abs_robust_z'], '.2f'):>6} "
            f"{_fmt(row['ranking_mean_abs_delta'], '.4f'):>6} "
            f"{_fmt(row['ranking_fail_rate'], '.4f'):>7}"
        )


def _manual_phase_ranking_rows(
    scored_candidates: list[dict[str, Any]],
    phase_context: dict[str, Any],
) -> list[dict[str, Any]]:
    """Return candidates sorted by the active manual-phase selection key."""
    ranked = sorted(
        scored_candidates,
        key=lambda candidate: _manual_phase_selection_key(candidate, phase_context),
    )
    rows: list[dict[str, Any]] = []
    for rank, candidate in enumerate(ranked, start=1):
        phase_score = candidate.get("manual_phase_score") or _manual_phase_score(candidate, phase_context)
        rows.append(
            {
                "rank": rank,
                "candidate_id": candidate.get("candidate_id"),
                "strategy_label": candidate.get("strategy_label", ""),
                "primary_layer": candidate.get("primary_layer", ""),
                "manual_phase_guard": candidate.get("manual_phase_guard", {}),
                "focus_metric_rows": phase_score.get("focus_metric_rows", []),
                "protected_metric_rows": phase_score.get("protected_metric_rows", []),
            }
        )
    return rows


def _print_manual_phase_selection_ranking(
    scored_candidates: list[dict[str, Any]],
    phase_context: dict[str, Any],
) -> None:
    """Print the actual ranking used in deterministic manual phase mode."""
    rows = _manual_phase_ranking_rows(scored_candidates, phase_context)
    if not rows:
        return
    print("    manual phase ranking (best→worst by active block metrics):")
    for row in rows:
        print(
            f"    c{row['candidate_id']!s:<5} "
            f"strategy={row['strategy_label'] or 'candidate'} "
            f"layer={row['primary_layer'] or 'both'}"
        )
        guard = row.get("manual_phase_guard", {}) or {}
        if guard:
            print(
                "      guard "
                f"violations={int(guard.get('violation_count', 0))} "
                f"max_severity={_fmt(float(guard.get('max_severity', 0.0)), '.3f')}"
            )
        for metric_row in row.get("focus_metric_rows", []):
            print(
                "      focus "
                f"{metric_row['metric']:<24} "
                f"W={_fmt(metric_row['wasserstein'], '.4f')} "
                f"Q={_fmt(metric_row['quantile_error'], '.4f')} "
                f"fail={_fmt(metric_row['empirical_fail_rate'], '.4f')} "
                f"|med|={_fmt(metric_row['abs_median_gap'], '.4f')} "
                f"|cd|={_fmt(metric_row['abs_cliffs_delta'], '.4f')} "
                f"mwu_p={_fmt(metric_row['mwu_p_value'], '.4f')} "
                f"ks_p={_fmt(metric_row['ks_p_value'], '.4f')} "
                f"oor={metric_row['out_of_range']} "
                f"pct={_fmt(metric_row['percentile_distance'], '.4f')} "
                f"raw_z={_fmt(metric_row['abs_raw_robust_z'], '.4f')}"
            )
        for metric_row in row.get("protected_metric_rows", []):
            print(
                "      prot  "
                f"{metric_row['metric']:<24} "
                f"W={_fmt(metric_row['wasserstein'], '.4f')} "
                f"Q={_fmt(metric_row['quantile_error'], '.4f')} "
                f"fail={_fmt(metric_row['empirical_fail_rate'], '.4f')} "
                f"|med|={_fmt(metric_row['abs_median_gap'], '.4f')} "
                f"|cd|={_fmt(metric_row['abs_cliffs_delta'], '.4f')} "
                f"mwu_p={_fmt(metric_row['mwu_p_value'], '.4f')} "
                f"ks_p={_fmt(metric_row['ks_p_value'], '.4f')} "
                f"oor={metric_row['out_of_range']} "
                f"pct={_fmt(metric_row['percentile_distance'], '.4f')} "
                f"raw_z={_fmt(metric_row['abs_raw_robust_z'], '.4f')}"
            )


def _serialize_metric_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return JSON-friendly copies of per-metric comparison rows."""
    serialized: list[dict[str, Any]] = []
    for row in rows:
        serialized.append(
            {
                "metric": row.get("metric"),
                "wasserstein": float(row.get("wasserstein", float("inf"))),
                "quantile_error": float(row.get("quantile_error", float("inf"))),
                "empirical_fail_rate": float(row.get("empirical_fail_rate", float("inf"))),
                "abs_median_gap": float(row.get("abs_median_gap", float("inf"))),
                "abs_cliffs_delta": float(row.get("abs_cliffs_delta", float("inf"))),
                "mwu_sig": int(row.get("mwu_sig", 1)),
                "ks_sig": int(row.get("ks_sig", 1)),
                "mwu_p_value": float(row.get("mwu_p_value", 0.0)),
                "ks_p_value": float(row.get("ks_p_value", 0.0)),
                "out_of_range": int(row.get("out_of_range", 1)),
                "percentile_distance": float(row.get("percentile_distance", float("inf"))),
                "abs_raw_robust_z": float(row.get("abs_raw_robust_z", float("inf"))),
                "status": row.get("status", "missing"),
            }
        )
    return serialized


def _print_phase_watch_metrics(
    phase_context: dict[str, Any],
    focus_rows: list[dict[str, Any]],
    protected_rows: list[dict[str, Any]],
) -> None:
    """Print the metrics that matter for the current manual phase iteration."""
    print("  → Watch metrics this iteration:")
    print(f"    focus     : {phase_context.get('focus_metrics', [])}")
    if phase_context.get("protected_metrics"):
        print(f"    protected : {phase_context.get('protected_metrics', [])}")
    if focus_rows:
        print("    current focus metric stats (search-root / block incumbent):")
        for row in focus_rows:
            print(
                "      "
                f"{row['metric']}: "
                f"W={_fmt(row['wasserstein'], '.4f')}  "
                f"Q={_fmt(row['quantile_error'], '.4f')}  "
                f"fail={_fmt(row['empirical_fail_rate'], '.4f')}  "
                f"|med|={_fmt(row['abs_median_gap'], '.4f')}  "
                f"|cd|={_fmt(row['abs_cliffs_delta'], '.4f')}  "
                f"mwu_p={_fmt(row['mwu_p_value'], '.4f')}  "
                f"ks_p={_fmt(row['ks_p_value'], '.4f')}  "
                f"oor={row['out_of_range']}  "
                f"pct={_fmt(row['percentile_distance'], '.4f')}  "
                f"raw_z={_fmt(row['abs_raw_robust_z'], '.4f')}"
            )
    if protected_rows:
        print("    protected metric stats to preserve:")
        for row in protected_rows:
            print(
                "      "
                f"{row['metric']}: "
                f"W={_fmt(row['wasserstein'], '.4f')}  "
                f"Q={_fmt(row['quantile_error'], '.4f')}  "
                f"fail={_fmt(row['empirical_fail_rate'], '.4f')}  "
                f"|med|={_fmt(row['abs_median_gap'], '.4f')}  "
                f"|cd|={_fmt(row['abs_cliffs_delta'], '.4f')}  "
                f"mwu_p={_fmt(row['mwu_p_value'], '.4f')}  "
                f"ks_p={_fmt(row['ks_p_value'], '.4f')}  "
                f"oor={row['out_of_range']}  "
                f"pct={_fmt(row['percentile_distance'], '.4f')}  "
                f"raw_z={_fmt(row['abs_raw_robust_z'], '.4f')}"
            )


def _print_winner_selection_breakdown(winner: dict[str, Any]) -> None:
    """Print the family-level fields that actually determined the winner."""
    family_scores = winner.get("selection_family_scores", {}) or {}
    guardrail = family_scores.get("guardrail_core", {}) or {}
    semantic = family_scores.get("semantic_core", {}) or {}
    engagement = family_scores.get("engagement_core", {}) or {}
    length = family_scores.get("length_core", {}) or {}

    print("    selection breakdown:")
    print(
        "      guardrail_core: "
        f"oor={int(guardrail.get('out_of_range_count', 0))} "
        f"max_pct={_fmt(float(guardrail.get('max_percentile_distance', float('nan'))), '.3f')} "
        f"raw_z={_fmt(float(guardrail.get('mean_abs_raw_robust_z', guardrail.get('mean_abs_robust_z', float('nan')))), '.3f')}"
    )
    print(
        "      semantic_core : "
        f"oor={int(semantic.get('out_of_range_count', 0))} "
        f"mean_pct={_fmt(float(semantic.get('mean_percentile_distance', float('nan'))), '.3f')} "
        f"max_pct={_fmt(float(semantic.get('max_percentile_distance', float('nan'))), '.3f')} "
        f"raw_z={_fmt(float(semantic.get('mean_abs_raw_robust_z', semantic.get('mean_abs_robust_z', float('nan')))), '.3f')}"
    )
    print(
        "      engagement_core: "
        f"oor={int(engagement.get('out_of_range_count', 0))} "
        f"mean_pct={_fmt(float(engagement.get('mean_percentile_distance', float('nan'))), '.3f')} "
        f"max_pct={_fmt(float(engagement.get('max_percentile_distance', float('nan'))), '.3f')} "
        f"raw_z={_fmt(float(engagement.get('mean_abs_raw_robust_z', engagement.get('mean_abs_robust_z', float('nan')))), '.3f')}"
    )
    print(
        "      length_core   : "
        f"mean_pct={_fmt(float(length.get('mean_percentile_distance', float('nan'))), '.3f')} "
        f"raw_z={_fmt(float(length.get('mean_abs_raw_robust_z', length.get('mean_abs_robust_z', float('nan')))), '.3f')}"
    )
    print(
        "      overall       : "
        f"quantile_fail={_fmt(float(winner.get('quantile_fail_rate', float('nan'))), '.4f')} "
        f"pct_dist={_fmt(float(winner.get('mean_percentile_distance', float('nan'))), '.4f')} "
        f"robust_z={_fmt(float(winner.get('mean_abs_robust_z', float('nan'))), '.4f')} "
        f"ranking_|delta|={_fmt(float(winner.get('ranking_mean_abs_delta', float('nan'))), '.4f')} "
        f"ranking_fail={_fmt(float(winner.get('ranking_fail_rate', float('nan'))), '.4f')}"
    )




def _print_improvement_summary(improvement: dict) -> None:
    """Print a concise terminal summary of before vs after improvement."""
    s = improvement.get("summary", {})
    pm = improvement.get("per_metric", {})

    def _yn(flag: bool) -> str:
        return "YES" if flag else "no"

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
    print(
        f"  Avg Wasserstein before/after:  "
        f"{s.get('avg_wasserstein_distance_before', float('nan')):.4f} → "
        f"{s.get('avg_wasserstein_distance_after', float('nan')):.4f}"
    )
    print(
        f"  Avg quantile err before/after: "
        f"{s.get('avg_quantile_error_before', float('nan')):.4f} → "
        f"{s.get('avg_quantile_error_after', float('nan')):.4f}"
    )

    metric_count = len(pm)
    print(
        "\n  Strict improved"
        f" (|Cliff's delta|↓ AND fail_rate↓): {s.get('strict_improved_count', 0)}/{metric_count}"
    )
    print(
        "  Closer-to-real by Wasserstein:       "
        f"{s.get('closer_by_wasserstein_count', 0)}/{metric_count}"
    )
    print(
        "  Closer-to-real by quantile error:    "
        f"{s.get('closer_by_quantile_count', 0)}/{metric_count}"
    )
    print(
        "  Closer-to-real by mean gap:          "
        f"{s.get('closer_by_mean_gap_count', 0)}/{metric_count}"
    )
    print(
        "  Closer-to-real by abs median gap:    "
        f"{s.get('closer_by_abs_median_gap_count', 0)}/{metric_count}"
    )

    print(
        "\n  Note:"
        "\n    - 'strict improved' is the conservative pass/fail view used by phase 3."
        "\n    - the 'closer-to-real' counts show directional numeric movement even when strict improved stays false."
    )

    if not pm:
        return

    print(
        f"\n  {'Metric':<28} {'cd_before':>9} {'cd_after':>9} {'Δfail%':>7} "
        f"{'strict':>6} {'wass':>6} {'quant':>6} {'mean':>6}"
    )
    print(
        f"  {'-'*28} {'-'*9} {'-'*9} {'-'*7} {'-'*6} {'-'*6} {'-'*6} {'-'*6}"
    )
    for key, label in _HEADLINE_METRICS:
        info = pm.get(key)
        if not info:
            continue
        print(
            f"  {label:<28} "
            f"{float(info.get('before_cliffs_delta', float('nan'))):>9.4f} "
            f"{float(info.get('after_cliffs_delta', float('nan'))):>9.4f} "
            f"{float(info.get('fail_rate_reduction', float('nan'))) * 100:>+6.1f}% "
            f"{_yn(bool(info.get('improved'))):>6} "
            f"{_yn(bool(info.get('closer_by_wasserstein'))):>6} "
            f"{_yn(bool(info.get('closer_by_quantile'))):>6} "
            f"{_yn(bool(info.get('closer_by_mean_gap'))):>6}"
        )
