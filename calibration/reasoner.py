"""
Reasoner for the calibration system.

Responsible for:
- Assembling LLM prompts from registry state and diagnostics
- Calling the OpenAI API to get overlay diffs and reasoning
- Parsing structured JSON responses
- Generating candidate overlay variants for evaluation
"""
from __future__ import annotations

import json
import math
import random
from typing import Any

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # type: ignore[assignment,misc]

from .overlay import (
    append_text_overlay,
    apply_structured_phase_overlay,
    merge_overlay,
)
from .registry import KnobRegistry
from ._reasoner_constants import (
    _REQUIRED_TEXT_KNOBS,
    _MECHANISM_FAMILIES,
    _STAGNATION_TRIGGER,
    METRIC_INTERPRETATION,
    CALIBRATION_COMPARISON_STATS_GUIDE,
    CALIBRATION_PRINCIPLES,
    TEXT_MATERIALIZER_PRINCIPLES,
    REALISM_RULES,
    _CHARS_PER_TOKEN_ESTIMATE,
    _DEFAULT_MAX_PROMPT_TOKENS,
)


# ---------------------------------------------------------------------------
# Small formatting helpers (moved to _reasoner_format.py)
# ---------------------------------------------------------------------------
from ._reasoner_format import (
    _fmt_float,
    _fmt_delta_vs_previous,
    _winner_headline_metrics,
    _headline_metric_delta_lines,
    _rank_groups_by_severity,
)




# ---------------------------------------------------------------------------
# _family_learning_summary moved to _reasoner_summary.py
# ---------------------------------------------------------------------------
from ._reasoner_summary import _family_learning_summary




# _REQUIRED_TEXT_KNOBS, _MECHANISM_FAMILIES, _STAGNATION_TRIGGER moved to _reasoner_constants.py



# ---------------------------------------------------------------------------
# Response schemas + text-knob normalization (moved to _reasoner_schema.py)
# ---------------------------------------------------------------------------
from ._reasoner_schema import (
    _required_text_overlay_schema,
    materializer_response_format,
    _response_format_for,
    _normalize_text_knob_block,
)




# Large prompt text constants moved to _reasoner_constants.py


# ---------------------------------------------------------------------------
# Metric interpretation helpers
# ---------------------------------------------------------------------------

# Domain lookup: metric name prefix → human-readable domain
_METRIC_DOMAIN: dict[str, str] = {
    "self_bleu": "content repetitiveness",
    "self_bertscore": "content repetitiveness",
    "semantic": "content repetitiveness",
    "mean_story_probability": "story/anecdote likelihood",
    "toxicity": "toxicity",
    "severe_toxicity": "toxicity",
    "obscene": "toxicity",
    "threat": "toxicity",
    "insult": "toxicity",
    "identity_attack": "toxicity",
    "aggression": "aggressiveness",
    "length": "comment length diversity",
    "max_depth": "thread structure",
    "avg_depth": "thread structure",
    "avg_branching": "thread structure",
    "structural_virality": "thread structure",
}


def _metric_interpretation(metric: str, direction: str, tier: str) -> str:
    """Generate a short human-readable interpretation for the reasoner."""
    if tier == "acceptable":
        return "Within acceptable range of real baseline."

    # Find domain
    domain = "metric"
    for prefix, d in _METRIC_DOMAIN.items():
        if metric.startswith(prefix):
            domain = d
            break

    if direction == "generated_higher":
        return f"Generated discussions have {domain} values too high compared to real."
    elif direction == "generated_lower":
        return f"Generated discussions have {domain} values too low compared to real."
    return f"Generated discussions deviate from real baseline on {domain}."


def _format_phase_context_section(phase_context: dict[str, Any] | None) -> str:
    """Render the deterministic phase schedule for the reasoner/materializer prompts."""
    if not phase_context:
        return ""

    dominant_emphasis = phase_context.get("dominant_emphasis", phase_context.get("preferred_layer"))
    lines: list[str] = []
    lines.append("## Active Manual Calibration Phase\n")
    lines.append(
        json.dumps(
            {
                "phase_name": phase_context.get("name"),
                "phase_label": phase_context.get("label"),
                "block_label": phase_context.get("block_label"),
                "focus_metrics": phase_context.get("focus_metrics", []),
                "protected_metrics": phase_context.get("protected_metrics", []),
                "required_mechanism_family": phase_context.get("required_mechanism_family"),
                "dominant_emphasis": dominant_emphasis,
                "summary": phase_context.get("summary", ""),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    lines.append("")
    lines.append("### Phase Rules")
    for rule in phase_context.get("reasoner_rules", []):
        lines.append(f"- {rule}")
    lines.append("")
    focus_metric_guidance = phase_context.get("focus_metric_guidance", [])
    if focus_metric_guidance:
        lines.append("### Focus Metric Reading Guide")
        for entry in focus_metric_guidance:
            metric = str(entry.get("metric", "")).strip()
            guidance = str(entry.get("guidance", "")).strip()
            if metric and guidance:
                lines.append(f"- {metric}: {guidance}")
        lines.append("")
    protected_metric_guidance = phase_context.get("protected_metric_guidance", [])
    if protected_metric_guidance:
        lines.append("### Protected Metric Preservation Guide")
        lines.append(
            "- These are earlier-block metrics that should be preserved while you optimize the current focus metrics."
        )
        for entry in protected_metric_guidance:
            metric = str(entry.get("metric", "")).strip()
            guidance = str(entry.get("guidance", "")).strip()
            if metric and guidance:
                lines.append(f"- {metric}: {guidance}")
        lines.append("")
    example_moves = phase_context.get("example_moves", [])
    if example_moves:
        lines.append("### Example Directions")
        for move in example_moves:
            lines.append(f"- {move}")
        lines.append("")
    lines.append("### Candidate Slot Plan")
    for slot in phase_context.get("candidate_plan", []):
        lines.append(f"- {slot}")
    lines.append("")
    return "\n".join(lines)


def _format_local_candidate_metric_feedback(
    trajectory: list[dict[str, Any]] | None,
    phase_context: dict[str, Any] | None,
) -> str:
    """Render prior candidate metric outcomes for manual-phase refinement."""
    if not trajectory or not phase_context:
        return ""

    focus_metrics = list(phase_context.get("focus_metrics", []))
    protected_metrics = list(phase_context.get("protected_metrics", []))
    phase_name = str(phase_context.get("name", "")).strip()
    lines: list[str] = []
    lines.append("## Previous Phase-Block Candidate Metric Feedback\n")
    lines.append(
        "Use these candidate-level results from earlier iterations inside this same manual phase block.\n"
        "Keep or strengthen mechanisms that improved the active focus metrics. Avoid, weaken, or replace mechanisms "
        "that made the focus metrics worse. Do not look only at the winner; inspect the other candidates too, because "
        "some of them may have improved one target metric even if they lost overall.\n"
    )
    lines.append("")

    def _cmp_label(value: float, reference: float, lower_is_better: bool) -> str:
        if not math.isfinite(value) or not math.isfinite(reference):
            return "mixed"
        if abs(value - reference) <= 1e-9:
            return "mixed"
        if lower_is_better:
            return "improved" if value < reference else "worsened"
        return "improved" if value > reference else "worsened"

    def _row_reference_map(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        return {str(row.get("metric", "")).strip(): row for row in rows if row.get("metric")}

    current_focus_reference = _row_reference_map((phase_context.get("current_focus_metric_rows") or []))
    current_protected_reference = _row_reference_map((phase_context.get("current_protected_metric_rows") or []))

    for entry in trajectory:
        manual_ctx = ((entry.get("search_state", {}) or {}).get("manual_phase_context", {}) or {})
        if str(manual_ctx.get("name", "")).strip() != phase_name:
            continue
        iteration_label = str(manual_ctx.get("iteration_label", "")).strip() or f"iter_{int(entry.get('iteration', -1)) + 1}"
        lines.append(f"### {iteration_label}")
        candidate_strategies = entry.get("candidate_strategies", []) or []
        if not candidate_strategies:
            lines.append("- No candidate strategy details recorded.\n")
            continue
        for cs in candidate_strategies:
            cid = cs.get("candidate_id", "?")
            label = cs.get("strategy_label", "candidate")
            family = cs.get("mechanism_family", "mixed")
            layer = cs.get("primary_layer", "both")
            anti_incumbent = " anti-incumbent" if cs.get("anti_incumbent") else ""
            lines.append(
                f"- candidate_{cid} | {label} | family={family} | layer={layer}{anti_incumbent}"
            )
            manual_phase_score = cs.get("manual_phase_score", {}) or {}
            focus_rows = manual_phase_score.get("focus_metric_rows", []) or []
            protected_rows = manual_phase_score.get("protected_metric_rows", []) or []
            if not focus_rows and not protected_rows:
                lines.append("  - No per-metric manual-phase stats recorded.")
                continue
            for row in focus_rows:
                metric = str(row.get("metric", "")).strip()
                if metric not in focus_metrics:
                    continue
                ref = current_focus_reference.get(metric, {})
                stat_labels = {
                    "|cd|": _cmp_label(float(row.get("abs_cliffs_delta", float("inf"))), float(ref.get("abs_cliffs_delta", float("inf"))), True),
                    "W": _cmp_label(float(row.get("wasserstein", float("inf"))), float(ref.get("wasserstein", float("inf"))), True),
                    "Q": _cmp_label(float(row.get("quantile_error", float("inf"))), float(ref.get("quantile_error", float("inf"))), True),
                    "fail": _cmp_label(float(row.get("empirical_fail_rate", float("inf"))), float(ref.get("empirical_fail_rate", float("inf"))), True),
                    "|med|": _cmp_label(float(row.get("abs_median_gap", float("inf"))), float(ref.get("abs_median_gap", float("inf"))), True),
                    "oor": _cmp_label(float(row.get("out_of_range", 1)), float(ref.get("out_of_range", 1)), True),
                    "pct": _cmp_label(float(row.get("percentile_distance", float("inf"))), float(ref.get("percentile_distance", float("inf"))), True),
                    "raw_z": _cmp_label(float(row.get("abs_raw_robust_z", float("inf"))), float(ref.get("abs_raw_robust_z", float("inf"))), True),
                }
                labels = list(stat_labels.values())
                overall = "mixed"
                if any(label == "improved" for label in labels) and not any(label == "worsened" for label in labels):
                    overall = "improved"
                elif any(label == "worsened" for label in labels) and not any(label == "improved" for label in labels):
                    overall = "worsened"
                lines.append(
                    "  - focus "
                    f"{metric}: overall={overall}; "
                    f"|cd|={float(row.get('abs_cliffs_delta', float('inf'))):.4f}({stat_labels['|cd|']}), "
                    f"W={float(row.get('wasserstein', float('inf'))):.4f}({stat_labels['W']}), "
                    f"Q={float(row.get('quantile_error', float('inf'))):.4f}({stat_labels['Q']}), "
                    f"fail={float(row.get('empirical_fail_rate', float('inf'))):.4f}({stat_labels['fail']}), "
                    f"|med|={float(row.get('abs_median_gap', float('inf'))):.4f}({stat_labels['|med|']}), "
                    f"oor={int(row.get('out_of_range', 1))}({stat_labels['oor']}), "
                    f"pct={float(row.get('percentile_distance', float('inf'))):.4f}({stat_labels['pct']}), "
                    f"raw_z={float(row.get('abs_raw_robust_z', float('inf'))):.4f}({stat_labels['raw_z']})"
                )
            for row in protected_rows:
                metric = str(row.get("metric", "")).strip()
                if metric not in protected_metrics:
                    continue
                ref = current_protected_reference.get(metric, {})
                stat_labels = {
                    "|cd|": _cmp_label(float(row.get("abs_cliffs_delta", float("inf"))), float(ref.get("abs_cliffs_delta", float("inf"))), True),
                    "W": _cmp_label(float(row.get("wasserstein", float("inf"))), float(ref.get("wasserstein", float("inf"))), True),
                    "Q": _cmp_label(float(row.get("quantile_error", float("inf"))), float(ref.get("quantile_error", float("inf"))), True),
                    "fail": _cmp_label(float(row.get("empirical_fail_rate", float("inf"))), float(ref.get("empirical_fail_rate", float("inf"))), True),
                    "|med|": _cmp_label(float(row.get("abs_median_gap", float("inf"))), float(ref.get("abs_median_gap", float("inf"))), True),
                    "oor": _cmp_label(float(row.get("out_of_range", 1)), float(ref.get("out_of_range", 1)), True),
                    "pct": _cmp_label(float(row.get("percentile_distance", float("inf"))), float(ref.get("percentile_distance", float("inf"))), True),
                    "raw_z": _cmp_label(float(row.get("abs_raw_robust_z", float("inf"))), float(ref.get("abs_raw_robust_z", float("inf"))), True),
                }
                labels = list(stat_labels.values())
                overall = "mixed"
                if any(label == "improved" for label in labels) and not any(label == "worsened" for label in labels):
                    overall = "improved"
                elif any(label == "worsened" for label in labels) and not any(label == "improved" for label in labels):
                    overall = "worsened"
                lines.append(
                    "  - protected "
                    f"{metric}: overall={overall}; "
                    f"|cd|={float(row.get('abs_cliffs_delta', float('inf'))):.4f}({stat_labels['|cd|']}), "
                    f"W={float(row.get('wasserstein', float('inf'))):.4f}({stat_labels['W']}), "
                    f"Q={float(row.get('quantile_error', float('inf'))):.4f}({stat_labels['Q']}), "
                    f"fail={float(row.get('empirical_fail_rate', float('inf'))):.4f}({stat_labels['fail']}), "
                    f"|med|={float(row.get('abs_median_gap', float('inf'))):.4f}({stat_labels['|med|']}), "
                    f"oor={int(row.get('out_of_range', 1))}({stat_labels['oor']}), "
                    f"pct={float(row.get('percentile_distance', float('inf'))):.4f}({stat_labels['pct']}), "
                    f"raw_z={float(row.get('abs_raw_robust_z', float('inf'))):.4f}({stat_labels['raw_z']})"
                )
            materialized = cs.get("materialized_text_overlay_diff", {})
            if materialized:
                lines.append(
                    "  - materialized_text_overlay_diff: "
                    f"{json.dumps(materialized, ensure_ascii=False)}"
                )
            else:
                overlay_diff = cs.get("overlay_diff", {})
                if overlay_diff:
                    lines.append(
                        "  - overlay_diff: "
                        f"{json.dumps(overlay_diff, ensure_ascii=False)}"
                    )
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Trajectory rendering (extracted for budget-aware truncation)
# ---------------------------------------------------------------------------

# Approximate characters-per-token ratio for budget estimation.
# _CHARS_PER_TOKEN_ESTIMATE, _DEFAULT_MAX_PROMPT_TOKENS moved to _reasoner_constants.py


def _render_trajectory_lines(
    trajectory: list[dict],
    *,
    skip_iterations: set[int] | None = None,
) -> list[str]:
    """Render the calibration trajectory into prompt lines.

    Parameters
    ----------
    trajectory : list[dict]
        Per-iteration history entries.
    skip_iterations : set[int] | None
        Iteration numbers (0-indexed) whose *detailed* entries should be
        replaced by a single summary line.  ``None`` means render everything.

    Returns
    -------
    list[str]
        Lines ready to be appended to the sections list.
    """
    if not trajectory:
        return ["  (no history yet)"]

    skip_iterations = skip_iterations or set()
    lines: list[str] = []
    previous_entry: dict[str, Any] | None = None

    # Collect compact summaries for skipped iterations
    skipped_summaries: list[str] = []

    for entry in trajectory:
        iter_num = int(entry.get("iteration", -1))

        if iter_num in skip_iterations:
            # Emit a one-line summary instead of full detail
            sel = entry.get("selection", {})
            beat = sel.get("beat_current_best", False)
            status = "✓ NEW BEST" if beat else "✗"
            winner_id = sel.get("winner_candidate_id")
            fr = sel.get("best_fail_rate")
            fr_str = f"{fr:.4f}" if fr is not None else "N/A"
            skipped_summaries.append(
                f"iter {iter_num}: winner=cand_{winner_id}, "
                f"fail_rate={fr_str}, {status}"
            )
            previous_entry = entry
            continue

        # Flush any accumulated skipped summaries before the first non-skipped entry
        if skipped_summaries:
            lines.append(
                "  [earlier iterations truncated to save context — "
                f"{len(skipped_summaries)} entries condensed]"
            )
            for s in skipped_summaries:
                lines.append(f"    {s}")
            lines.append("")
            skipped_summaries = []

        # ── Full detail for non-skipped iterations ──────────────────────
        sel = entry.get("selection", {})
        beat = sel.get("beat_current_best", False)
        status = "✓ NEW BEST" if beat else "✗ no improvement"
        winner_id = sel.get("winner_candidate_id")
        fr = sel.get("best_fail_rate")
        ad = sel.get("best_mean_abs_delta")
        qf = sel.get("best_quantile_fail_rate")
        mpd = sel.get("best_mean_percentile_distance")
        mrz = sel.get("best_mean_abs_robust_z")
        fr_str = f"{fr:.4f}" if fr is not None else "N/A"
        ad_str = f"{ad:.4f}" if ad is not None else "N/A"
        qf_str = f"{qf:.4f}" if qf is not None else "N/A"
        mpd_str = f"{mpd:.4f}" if mpd is not None else "N/A"
        mrz_str = f"{mrz:.4f}" if mrz is not None else "N/A"
        lines.append(
            f"  iter {entry.get('iteration','?')}: "
            f"winner=candidate_{winner_id}, "
            f"fail_rate={fr_str}, |delta|={ad_str}, "
            f"quantile_fail={qf_str}, pct_dist={mpd_str}, robust_z={mrz_str}, "
            f"result={status}"
        )
        if previous_entry is not None:
            prev_sel = previous_entry.get("selection", {})
            lines.append(
                "    iteration_delta: "
                f"fail_rate {_fmt_delta_vs_previous(fr, prev_sel.get('best_fail_rate'))}; "
                f"|delta| {_fmt_delta_vs_previous(ad, prev_sel.get('best_mean_abs_delta'))}; "
                f"quantile_fail {_fmt_delta_vs_previous(qf, prev_sel.get('best_quantile_fail_rate'))}; "
                f"pct_dist {_fmt_delta_vs_previous(mpd, prev_sel.get('best_mean_percentile_distance'))}; "
                f"robust_z {_fmt_delta_vs_previous(mrz, prev_sel.get('best_mean_abs_robust_z'))}"
            )
        # Show diagnosis summary
        diag = entry.get("diagnosis", "")
        if diag:
            lines.append(f"    diagnosis: {diag}")

        # Show per-candidate strategy details (new format)
        cand_strats = entry.get("candidate_strategies", [])
        if cand_strats:
            for cs in cand_strats:
                cid = cs.get("candidate_id", "?")
                slabel = cs.get("strategy_label", "?")
                layer = cs.get("primary_layer", "?")
                family = cs.get("mechanism_family", "?")
                anti_incumbent = " anti-incumbent" if cs.get("anti_incumbent") else ""
                cs_fr = cs.get("fail_rate")
                cs_ad = cs.get("mean_abs_delta")
                cs_qf = cs.get("quantile_fail_rate")
                cs_mpd = cs.get("mean_percentile_distance")
                cs_mrz = cs.get("mean_abs_robust_z")
                cs_fr_str = f"{cs_fr:.4f}" if cs_fr is not None else "n/a"
                cs_ad_str = f"{cs_ad:.4f}" if cs_ad is not None else "n/a"
                cs_qf_str = f"{cs_qf:.4f}" if cs_qf is not None else "n/a"
                cs_mpd_str = f"{cs_mpd:.4f}" if cs_mpd is not None else "n/a"
                cs_mrz_str = f"{cs_mrz:.4f}" if cs_mrz is not None else "n/a"
                is_winner = " ← winner" if cid == winner_id else ""
                lines.append(
                    f"    [{cid}] {slabel} (family={family}, layer={layer}{anti_incumbent}) "
                    f"fail_rate={cs_fr_str} |delta|={cs_ad_str} "
                    f"quantile_fail={cs_qf_str} pct_dist={cs_mpd_str} robust_z={cs_mrz_str}{is_winner}"
                )
                # Show per-group scores
                gs = cs.get("group_scores", {})
                if gs:
                    gs_parts = []
                    for gname, ginfo in sorted(gs.items()):
                        gpd = ginfo.get("mean_percentile_distance")
                        gqf = ginfo.get("quantile_fail_rate")
                        gpd_str = f"{gpd:.3f}" if gpd is not None else "?"
                        gqf_str = f"{gqf:.3f}" if gqf is not None else "?"
                        gs_parts.append(f"{gname}(pct_dist={gpd_str},fail={gqf_str})")
                    lines.append(f"        groups: {', '.join(gs_parts)}")
                # Show headline metric values per candidate
                hm = cs.get("headline_metrics", {})
                if hm:
                    hm_parts = []
                    for hkey, hinfo in sorted(hm.items()):
                        sim_v = hinfo.get("sim_median")
                        real_v = hinfo.get("real_median")
                        hm_status = hinfo.get("status", "?")
                        sim_str = f"{sim_v:.3f}" if sim_v is not None else "?"
                        real_str = f"{real_v:.3f}" if real_v is not None else "?"
                        hm_parts.append(f"{hkey}={sim_str}(real={real_str},{hm_status})")
                    lines.append(f"        metrics: {', '.join(hm_parts)}")
                # Show what was actually changed
                cs_od = cs.get("overlay_diff", {})
                if cs_od:
                    lines.append(f"        overlay_diff: {json.dumps(cs_od, ensure_ascii=False)}")
                text_od = cs.get("materialized_text_overlay_diff", {})
                if text_od:
                    lines.append(
                        "        materialized_text_overlay_diff: "
                        f"{json.dumps(text_od, ensure_ascii=False)}"
                    )
        else:
            # Legacy format — show overlay_diff
            od = entry.get("overlay_diff", {})
            if od:
                lines.append(f"    overlay_diff: {json.dumps(od, ensure_ascii=False)}")
        metric_delta_lines = _headline_metric_delta_lines(entry, previous_entry)
        if metric_delta_lines:
            lines.append("    winner_metric_delta_vs_previous:")
            lines.extend(metric_delta_lines)
        lines.append("")
        previous_entry = entry

    # Flush remaining skipped summaries (if all entries were skipped)
    if skipped_summaries:
        lines.append(
            "  [earlier iterations truncated to save context — "
            f"{len(skipped_summaries)} entries condensed]"
        )
        for s in skipped_summaries:
            lines.append(f"    {s}")
        lines.append("")

    return lines


# ---------------------------------------------------------------------------
# Prompt assembly
# ---------------------------------------------------------------------------

def build_reasoner_prompt(
    registry: KnobRegistry,
    current_overlay: dict,
    current_diagnostic: dict,
    real_baseline: dict,
    trajectory: list[dict],
    failed_strategies: list[str],
    metric_definitions: str,
    sample_real_thread: str = "",
    sample_sim_thread: str = "",
    iteration: int = 0,
    max_iterations: int = 10,
    combination_start_iteration: int | None = None,
    global_best_overlay: dict | None = None,
    global_best_diagnostic: dict | None = None,
    frontier: dict[str, dict[str, Any]] | None = None,
    stagnation_count: int = 0,
    search_mode: str = "global_best",
    search_root_reason: str = "global_best",
    phase_context: dict[str, Any] | None = None,
    completed_phase_summaries: list[dict[str, Any]] | None = None,
) -> str:
    """Assemble a full reasoner prompt for the LLM.

    Parameters
    ----------
    registry : KnobRegistry
    current_overlay : dict
    current_diagnostic : dict
        Output of score_candidate: {fail_rate, mean_abs_delta, per_metric}.
    real_baseline : dict
        Median values from real Reddit data {metric_name: float}.
    trajectory : list[dict]
        Per-iteration history.
    failed_strategies : list[str]
        Strategy labels already tried without improvement.
    metric_definitions : str
        Optional external metric reference (may be empty).

    Returns
    -------
    str
    """
    sections: list[str] = []

    # ── Metric interpretation and calibration principles ──────────────────
    sections.append(METRIC_INTERPRETATION)
    sections.append("")
    sections.append(CALIBRATION_COMPARISON_STATS_GUIDE)
    sections.append("")
    sections.append(CALIBRATION_PRINCIPLES)
    sections.append("")
    sections.append(REALISM_RULES)
    sections.append("")

    # ── Optional external metric definitions ──────────────────────────────
    if metric_definitions and metric_definitions.strip():
        sections.append("## Additional Metric Reference\n")
        sections.append(metric_definitions.strip())
        sections.append("")

    # ── Tunable knobs ─────────────────────────────────────────────────────
    sections.append("## Tunable Knobs\n")
    sections.append(registry.for_llm_context().strip())
    sections.append("")

    global_best_overlay = global_best_overlay or current_overlay
    global_best_diagnostic = global_best_diagnostic or current_diagnostic
    frontier = frontier or {}
    completed_phase_summaries = completed_phase_summaries or []
    if combination_start_iteration is None:
        combination_start_iteration = max_iterations // 2
    combination_start_iteration = max(1, min(int(combination_start_iteration), max_iterations))

    search_state = {
        "search_mode": search_mode,
        "stagnation_count": stagnation_count,
        "search_root_reason": search_root_reason,
        "combination_start_iteration": combination_start_iteration,
    }
    sections.append("## Search State\n")
    sections.append(json.dumps(search_state, indent=2, ensure_ascii=False))
    sections.append("")

    # ── Current overlay (strip _manual_phase_blocks — it's redundant with
    #    the rendered persona.generation_guidance / prompt.comment_style_guidance) ──
    sections.append("## Active Search Root Overlay\n")
    _display_overlay = {k: v for k, v in current_overlay.items() if k != "_manual_phase_blocks"}
    sections.append(json.dumps(_display_overlay, indent=2, ensure_ascii=False))
    sections.append("")

    if global_best_overlay != current_overlay:
        sections.append("## Global Best Overlay\n")
        _display_global = {k: v for k, v in global_best_overlay.items() if k != "_manual_phase_blocks"}
        sections.append(json.dumps(_display_global, indent=2, ensure_ascii=False))
        sections.append("")
        sections.append("## Global Best Diagnostic Summary\n")
        sections.append(
            json.dumps(
                {
                    "fail_rate": global_best_diagnostic.get("fail_rate"),
                    "mean_abs_delta": global_best_diagnostic.get("mean_abs_delta"),
                    "quantile_fail_rate": global_best_diagnostic.get("quantile_fail_rate"),
                    "mean_percentile_distance": global_best_diagnostic.get("mean_percentile_distance"),
                    "mean_abs_robust_z": global_best_diagnostic.get("mean_abs_robust_z"),
                    "group_scores": global_best_diagnostic.get("group_scores", {}),
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        sections.append("")

    if frontier:
        sections.append("## Frontier Candidates\n")
        sections.append(
            "These are the best known branch points for each metric-group family. "
            "Use them to combine partial wins or to branch away from a stuck incumbent."
        )
        sections.append(json.dumps(frontier, indent=2, ensure_ascii=False))
        sections.append("")

    phase_section = _format_phase_context_section(phase_context)
    if phase_section:
        sections.append(phase_section)

    if completed_phase_summaries:
        sections.append("## Completed Phase Summary\n")
        sections.append(
            "Previous manual phase blocks that have been committed. "
            "Their prompt contributions are already baked into the Active Search Root Overlay above. "
            "Later phases should add to them rather than erase them."
        )
        # Strip `overlay` (already in the Active Search Root Overlay above) and
        # per-metric detail from `diagnostic` (keep only top-level scores).
        compact_summaries = []
        for ps in completed_phase_summaries:
            compact: dict[str, Any] = {
                "phase_name": ps.get("phase_name"),
                "phase_label": ps.get("phase_label"),
                "block_label": ps.get("block_label"),
                "focus_metrics": ps.get("focus_metrics"),
            }
            diag = ps.get("diagnostic") or {}
            if isinstance(diag, dict):
                compact["best_scores"] = {
                    k: diag.get(k)
                    for k in (
                        "fail_rate", "mean_abs_delta", "quantile_fail_rate",
                        "mean_percentile_distance", "mean_abs_robust_z",
                        "group_mean_abs_cliffs_delta", "group_overall_fail_rate",
                    )
                    if k in diag
                }
                # Keep group_scores (small) for per-group visibility
                if "group_scores" in diag:
                    compact["group_scores"] = diag["group_scores"]
            compact_summaries.append(compact)
        sections.append(json.dumps(compact_summaries, indent=2, ensure_ascii=False))
        sections.append("")

    previous_candidate_feedback = _format_local_candidate_metric_feedback(trajectory, phase_context)
    if previous_candidate_feedback:
        sections.append(previous_candidate_feedback)

    family_summary = _family_learning_summary(trajectory)
    if family_summary:
        sections.append("## Family Learning Summary\n")
        sections.append(
            "Compressed history by mechanism family. Use this to decide which families are "
            "worth refining, which ones are weak, and which cross-family combinations are plausible."
        )
        sections.append(json.dumps(family_summary, indent=2, ensure_ascii=False))
        sections.append("")

    worst_group_order = _rank_groups_by_severity(current_diagnostic)
    if worst_group_order:
        sections.append("## Current Worst Metric Groups\n")
        sections.append(
            json.dumps(
                [
                    {
                        "group": group_name,
                        "group_score": (current_diagnostic.get("group_scores", {}) or {}).get(group_name, {}),
                    }
                    for group_name in worst_group_order
                ],
                indent=2,
                ensure_ascii=False,
            )
        )
    sections.append("")

    # ── Combined per-metric diagnostic ──────────────────────────────────────
    # Merges validation reference stats + generated stats + diagnostic into
    # one view so the LLM sees the full picture per metric.
    sections.append("## Current Diagnostic\n")
    fail_rate = current_diagnostic.get("fail_rate", "N/A")
    mean_abs_delta = current_diagnostic.get("mean_abs_delta", "N/A")
    sections.append(f"overall_validation_fail_rate : {fail_rate}")
    sections.append(f"validation_mean_abs_delta    : {mean_abs_delta}")
    if current_diagnostic.get("quantile_fail_rate") is not None:
        sections.append(
            f"validation_quantile_fail_rate : {current_diagnostic.get('quantile_fail_rate')}"
        )
    if current_diagnostic.get("mean_percentile_distance") is not None:
        sections.append(
            "validation_mean_percentile_distance : "
            f"{current_diagnostic.get('mean_percentile_distance')}"
        )
    if current_diagnostic.get("mean_abs_robust_z") is not None:
        sections.append(
            f"validation_mean_abs_robust_z  : {current_diagnostic.get('mean_abs_robust_z')}"
        )
    sections.append("")

    per_metric = current_diagnostic.get("per_metric", {})
    sections.append("## Per-Metric Analysis (real validation reference + generated + diagnostic)\n")
    sections.append(
        "Each entry combines: real validation distribution stats, current generated "
        "distribution stats, robust position inside the validation distribution, "
        "and the diagnostic (fail_rate, effect size, tier).\n"
        "Sorted by severity: CRITICAL first, then SECONDARY, then acceptable."
    )

    metric_entries: list[dict] = []
    for metric in sorted(per_metric.keys()):
        if metric == "threads":
            continue
        info = per_metric[metric]
        fr = info.get("fail_rate", 0.0)
        cd = info.get("cliffs_delta", 0.0)
        direction = info.get("direction", "")
        robust_status = str(info.get("status", "missing") or "missing")
        percentile_distance = float(info.get("percentile_distance", 0.0) or 0.0)
        abs_robust_z = abs(float(info.get("robust_z", 0.0) or 0.0))

        # Classify tier
        if (
            robust_status != "in_range"
            and (
                percentile_distance >= 0.75
                or abs_robust_z >= 2.5
                or fr > 0.50
                or abs(cd) >= 0.474
            )
        ):
            tier = "CRITICAL"
        elif (
            robust_status != "in_range"
            or percentile_distance >= 0.40
            or abs_robust_z >= 1.0
            or fr >= 0.20
            or abs(cd) >= 0.33
        ):
            tier = "SECONDARY"
        else:
            tier = "acceptable"

        # Interpretation based on metric domain and direction
        interpretation = _metric_interpretation(metric, direction, tier)

        # Validation reference stats (from real_baseline dict)
        real_stats = real_baseline.get(metric, {})
        if not isinstance(real_stats, dict):
            real_stats = {"median": real_stats}

        # Generated stats (from scorer's generated_summary)
        gen_summary = info.get("generated_summary", {})
        if not gen_summary:
            gen_summary = {
                "mean": info.get("generated_median", ""),
                "median": info.get("generated_median", ""),
            }

        entry = {
            "metric": metric,
            "robust_validation_position": {
                "sim_median": round(float(info.get("sim_median", float("nan"))), 4),
                "real_p10": round(float(info.get("real_p10", float("nan"))), 4),
                "real_p50": round(float(info.get("real_median", float("nan"))), 4),
                "real_p90": round(float(info.get("real_p90", float("nan"))), 4),
                "percentile_rank": round(float(info.get("percentile_rank", float("nan"))), 4),
                "percentile_distance": round(percentile_distance, 4),
                "robust_z": round(float(info.get("robust_z", float("nan"))), 4),
                "status": robust_status,
            },
            "real_validation_summary": {
                k: round(v, 4) if isinstance(v, float) else v
                for k, v in real_stats.items()
            },
            "current_generated_summary": {
                k: round(v, 4) if isinstance(v, float) else v
                for k, v in gen_summary.items()
            },
            "diagnostic": {
                "fail_rate": round(fr, 3),
                "cliffs_delta": round(cd, 3),
                "direction": direction,
                "tier": tier,
                "interpretation": interpretation,
            },
        }
        metric_entries.append(entry)

    # Sort: CRITICAL first, then SECONDARY, then acceptable
    tier_order = {"CRITICAL": 0, "SECONDARY": 1, "acceptable": 2}
    metric_entries.sort(key=lambda r: (
        tier_order.get(r["diagnostic"]["tier"], 3),
        -float(r.get("robust_validation_position", {}).get("percentile_distance", 0.0)),
        -abs(float(r.get("robust_validation_position", {}).get("robust_z", 0.0))),
    ))
    sections.append(json.dumps(metric_entries, indent=2))
    sections.append("")

    # ── Calibration trajectory ────────────────────────────────────────────
    # Rendered via helper; placeholder tag lets us swap it for a truncated
    # version if the assembled prompt exceeds the token budget.
    _TRAJECTORY_PLACEHOLDER = "<<__TRAJECTORY_SECTION__>>"
    trajectory_header = (
        "## Calibration Trajectory\n\n"
        "History of all iterations: what was tried, what happened, and whether it improved.\n"
        "Learn from this — avoid repeating strategies that failed, build on what worked.\n\n"
    )
    trajectory_lines_full = _render_trajectory_lines(trajectory)
    sections.append(_TRAJECTORY_PLACEHOLDER)
    sections.append("")

    # ── Failed strategies ─────────────────────────────────────────────────
    sections.append("## Failed Strategies (do not repeat these)\n")
    if failed_strategies:
        for s in failed_strategies:
            sections.append(f"  - {s}")
    else:
        sections.append("  (none yet)")
    sections.append("")

    # ── Sample threads (real vs simulated) ──────────────────────────────
    if sample_real_thread or sample_sim_thread:
        sections.append("## Sample Threads (for qualitative comparison)\n")
        sections.append(
            "Below are excerpted threads to help you understand the qualitative "
            "differences between real Reddit discussions and the current simulation output. "
            "Use these to inform your diagnosis — the metrics above tell you WHAT is wrong, "
            "these samples show you WHY. When targeting a specific metric family, explicitly look for "
            "the analogous behavior in the real samples before proposing edits.\n"
            "\n"
            "How to use the real few-shot samples:\n"
            "- For story / personal-experience metrics, inspect how real threads use lived detail: how long the anecdote is, how specific it is, and how naturally it is inserted.\n"
            "- For diversity metrics, inspect how real comments vary in syntax, openings, cadence, evidence mode, and stance instead of repeating one polished template.\n"
            "- For length variation metrics, inspect the real mix of short, medium, and long comments and how different shapes coexist inside one thread.\n"
            "- For conflict / toxicity metrics, inspect how real threads express impatience, profanity, dismissal, sarcasm, hostility, or escalation when they do appear.\n"
            "- Do not copy the content of the few-shot examples. Extract their style, shape, and realism level and reproduce that pattern on new content.\n"
        )
        if sample_real_thread:
            sections.append("### Real Reddit Threads (from train set, qualitative reference only)\n")
            sections.append(sample_real_thread.strip())
            sections.append("")
        if sample_sim_thread:
            sections.append("### Current Best Simulated Threads\n")
            sections.append(sample_sim_thread.strip())
            sections.append("")

    # ── Task instructions ─────────────────────────────────────────────────
    # Dynamic slot allocation based on calibration phase
    is_exploration_phase = iteration < combination_start_iteration
    family_order = list(dict.fromkeys(_rank_groups_by_severity(current_diagnostic) + list(_MECHANISM_FAMILIES)))
    wildcard_family = family_order[-1] if family_order else "tone_civility"

    if phase_context:
        slot_instructions = (
            "DETERMINISTIC MANUAL PHASE MODE:\n"
            "This iteration belongs to a fixed manual phase block with explicit target metrics.\n"
            "All 5 candidates must stay inside the active phase objective.\n"
            "Do not branch to unrelated metric families just because they also look weak.\n"
            "Preserve the current overlay's prior wins unless the active phase explicitly requires a bounded refinement.\n"
        )
    elif stagnation_count >= _STAGNATION_TRIGGER:
        slot_instructions = (
            "STAGNATION MODE (no new best for {stagnation} consecutive iterations):\n"
            "The search is stuck near a local optimum. Do NOT generate small variations of the incumbent.\n"
            "Use these 5 mechanism-first slots:\n"
            "  - Candidate 0: target the WORST current family = {fam0}\n"
            "  - Candidate 1: target the SECOND-WORST current family = {fam1}\n"
            "  - Candidate 2: target a DIFFERENT family = {fam2}; must not rely on the incumbent's main hypothesis\n"
            "  - Candidate 3: branch from a FRONTIER candidate, not from the incumbent, targeting {fam3}\n"
            "  - Candidate 4: anti-incumbent wildcard targeting {fam4}; explicitly oppose the incumbent causal story\n"
            "Hard rules in stagnation mode:\n"
            "  - At least 3 candidates must use mechanism families DIFFERENT from the incumbent's apparent family.\n"
            "  - Candidate 4 MUST set anti_incumbent=true.\n"
            "  - If a family has failed repeatedly, only revisit it if you change the causal mechanism, not just the wording.\n"
        ).format(
            stagnation=stagnation_count,
            fam0=family_order[0],
            fam1=family_order[1],
            fam2=family_order[2],
            fam3=family_order[3],
            fam4=wildcard_family,
        )
    elif is_exploration_phase:
        slot_instructions = (
            "EXPLORATION PHASE (iteration {iter}/{max_iter} — before combination-heavy search begins at iteration {combo_start}):\n"
            "Keep the early search close to the original exploration pattern while still using family evidence:\n"
            "  - Candidate 0: persona-led single-family probe targeting {fam0}\n"
            "  - Candidate 1: persona-led single-family probe targeting {fam1}\n"
            "  - Candidate 2: prompt-led single-family probe targeting {fam2}\n"
            "  - Candidate 3: prompt-led single-family probe targeting {fam3}\n"
            "  - Candidate 4: one light combination / wildcard candidate using history, frontier, or anti-incumbent logic targeting {fam4}\n"
            "Focus on discovering WHICH mechanism family moves WHICH metric group and WHICH layer carries that effect.\n"
            "Even in early iterations, use trajectory, frontier, and family history to avoid weak repeats — but keep only one explicit combination candidate."
        ).format(
            iter=iteration + 1,
            max_iter=max_iterations,
            combo_start=combination_start_iteration + 1,
            fam0=family_order[0],
            fam1=family_order[1],
            fam2=family_order[2],
            fam3=family_order[3],
            fam4=wildcard_family,
        )
    else:
        slot_instructions = (
            "COMBINATION-HEAVY PHASE (iteration {iter}/{max_iter} — from iteration {combo_start} onward):\n"
            "Shift into later-stage combination search while still keeping one persona-only and one prompt-only probe alive:\n"
            "  - Candidate 0: persona-led refine candidate targeting {fam0}\n"
            "  - Candidate 1: prompt-led refine candidate targeting {fam1}\n"
            "  - Candidate 2: history-based combination candidate that combines a persona-side local win with a prompt-side local win from different families\n"
            "  - Candidate 3: frontier-branch combination candidate that mixes a non-incumbent family with the incumbent's strongest family\n"
            "  - Candidate 4: aggressive combination candidate targeting the remaining failure family = {fam4}\n"
            "The later phase should be explicitly more combination-heavy: three of the five candidates must be real combinations grounded in family-level evidence."
        ).format(
            iter=iteration + 1,
            max_iter=max_iterations,
            combo_start=combination_start_iteration + 1,
            fam0=family_order[0],
            fam1=family_order[1],
            fam4=wildcard_family,
        )

    sections.append("## Task\n")
    if phase_context:
        focus_metrics_json = json.dumps(phase_context.get("focus_metrics", []), ensure_ascii=False)
        protected_metrics_json = json.dumps(phase_context.get("protected_metrics", []), ensure_ascii=False)
        required_mechanism_family = phase_context.get("required_mechanism_family")
        dominant_emphasis = phase_context.get("dominant_emphasis", phase_context.get("preferred_layer", "both"))
        required_family_line = (
            f"- Required mechanism family for ALL 5 candidates in this block: {required_mechanism_family}\n"
            if required_mechanism_family else ""
        )
        required_family_rule = (
            f"  - All 5 candidates must keep mechanism_family='{required_mechanism_family}'. They may differ in mechanism, but they must stay inside that family.\n"
            if required_mechanism_family else ""
        )
        sections.append(
            "You are a calibration reasoner for a Reddit discussion simulator. "
            "This run is in DETERMINISTIC MANUAL PHASE MODE. Your job is to close the gap "
            "between simulated and real Reddit metrics, but ONLY through the active phase objective.\n"
            "\n"
            "Active-phase requirements:\n"
            f"- Focus metrics for this manual phase block: {focus_metrics_json}\n"
            f"- Protected metrics from earlier blocks: {protected_metrics_json}\n"
            f"{required_family_line}"
            "Protected behavior interpretation:\n"
            "- If mean_story_probability is protected, preserve short first-hand or observed datapoints. Do not delete anecdotes to improve diversity.\n"
            "- If semantic diversity metrics are protected, preserve comment-function diversity and anti-template gates. Do not return to repeated paraphrases.\n"
            "- If structure metrics are protected, preserve reply-to-parent behavior and parent-specific reactions. Do not flatten into top-level standalone advice.\n"
            "- If length_cv is protected, preserve a mixture of one-liners, short replies, medium advice, and occasional long explanations.\n"
            "- If civility/conflict metrics are protected, preserve sparse topical disagreement without increasing unsafe hostility.\n"
            f"- Dominant emphasis in this phase: {dominant_emphasis}\n"
            "- EVERY candidate must edit BOTH knobs: persona.generation_guidance and prompt.comment_style_guidance.\n"
            "- Treat dominant emphasis only as a steering hint. It is NOT a license to modify only one knob.\n"
            "- Do not remove previously working instructions from earlier blocks. Treat them as preserved gains.\n"
            "- The active overlay is phase-structured when available. Earlier block text should be treated as named preserved sections, not as disposable free text.\n"
            "- Write append-only patch text for the CURRENT phase block. Assume the system will store this inside the active phase section instead of replacing earlier sections.\n"
            "- The winner for this iteration will be judged mainly on the active focus metrics, then on whether protected metrics remain stable.\n"
            "- Only the final integrated block is allowed to optimize across all tracked metrics at once. Earlier blocks should optimize the active focus metrics and only preserve protected metrics.\n"
            "- Even in this focused block, keep the true end goal in mind: drive every targeted metric's |Cliff's delta| and Wasserstein distance toward 0.\n"
            "- Use the example directions in the active phase block as concrete templates for the kinds of changes to propose.\n"
            "- If the target behavior requires less politeness, less validation, rougher grammar, more clipped syntax, or sharper disagreement, encode that explicitly in the candidate overlay_diff. Do not assume the base runtime prompt will supply that style for free.\n"
            "- For history, only reference earlier iterations inside this same manual phase block when they exist. Do not use older blocks as strategy examples.\n"
            "- If previous-iteration candidate metric feedback is shown, read all candidates in that feedback, not only the prior winner.\n"
            "- Keep and strengthen mechanisms whose previous candidate rows improved the active focus metrics.\n"
            "- Reduce, avoid, or replace mechanisms whose previous candidate rows worsened the active focus metrics, even if the wording sounded plausible.\n"
            "- If one previous candidate helped one focus metric but hurt another, treat it as a partial ingredient to combine carefully rather than copying it wholesale.\n"
            "- Use the real few-shot sample threads as the realism anchor for the active metric family: match the way real threads tell stories, vary syntax, vary length, or escalate tone.\n"
            "- Study the few-shot threads deeply before proposing edits: understand how the real thread actually expresses the target behavior, then push the simulation toward that same distributional shape.\n"
            "- For diversity and conflict blocks specifically, be literal about surface form: if you want social-media short replies, write that; if you want fragments, lowercase starts, no-subject replies, or direct call-outs, write that. Do not hide those moves behind abstract wording like 'vary tone' or 'be less polite'.\n"
            "- For length and structure metrics, do not collapse the simulation toward one narrow safe median. Real validation threads often keep a broad spread. Matching that spread, or staying slightly broader than a bland under-dispersed simulation, is acceptable as long as relevance is preserved.\n"
            "\n"
            "Step 1 — DIAGNOSE: Explain only the failures relevant to the active focus metrics first. "
            "You may mention protected-metric risk, but do not spend most of the diagnosis on unrelated metrics.\n"
            "\n"
            "Step 2 — DESIGN 5 CONTROLLED CANDIDATES INSIDE THIS PHASE:\n"
            "Do not make 5 unrelated independent ideas. Each candidate must optimize the active focus metrics while preserving protected metrics from earlier phase blocks.\n"
            "Use this fixed candidate allocation:\n"
            "\n"
            "- Candidate 0: conservative_refinement\n"
            "Make the smallest change that targets the active metrics while preserving all protected behaviors.\n"
            "\n"
            "- Candidate 1: direct_active_metric_push\n"
            "Apply the strongest direct mechanism for the active metrics, but include explicit guardrails for protected metrics.\n"
            "\n"
           " - Candidate 2: protected_metric_guard\n"
           " Start from the protected metric behavior that worked earlier, then modify only the parts needed for the active metrics.\n"
            "\n"
            "- Candidate 3: alternative_causal_mechanism\n"
            "Try a different causal mechanism for the active metrics if the current mechanism risks damaging protected metrics.\n"
            "\n"
            "- Candidate 4: integrated_best_guess\n"
            "Combine the strongest active-metric mechanism with the strongest protected-metric preservation rule.\n"
            "\n"
            "For every candidate, explicitly include:\n"
            "1. active metrics targeted,\n"
            "2. protected metrics that must not regress,\n"
            "3. what behavior from earlier phases must be preserved,\n"
            "4. what behavior is allowed to change,\n"
            "5. what behavior is forbidden because it would damage protected metrics,\n"
            "6. expected metric movement,\n"
            "7. side effects to avoid.\n"

            "A candidate is invalid if it improves the active metric by removing or weakening a behavior that made a protected metric pass earlier.\n"
            "\n"
            f"{slot_instructions}\n"
            "\n"
            "CRITICAL RULE — READ THE DIRECTION CAREFULLY:\n"
            "  - If a metric is generated_lower, your fix must increase it.\n"
            "  - If a metric is generated_higher, your fix must decrease it.\n"
            "  - Keep track of which earlier metrics are protected so you do not erase them accidentally.\n"
            "\n"
            "CRITICAL RULES for strategy diversity in manual phase mode:\n"
            "  - Do NOT make 5 minor rewrites of the same candidate.\n"
            "  - Within the active phase, each candidate must test a distinct causal mechanism.\n"
            f"{required_family_rule}"
            "  - Preserve earlier block instructions unless this phase explicitly says to tighten or locally refine them.\n"
            "  - Every candidate must still write BOTH knobs, even when one side carries the dominant emphasis.\n"
            "  - New text should be append-only patch content, not a rewritten full overlay.\n"
            "  - Use concrete examples, sample phrasings, persona traits, and reply-shape instructions rather than abstract slogans.\n"
            "  - Ground those examples in the real few-shot threads: if real stories are short and awkwardly specific, do not propose grand polished anecdotes; if real diversity comes from jagged syntax, do not propose five polished variants of the same template.\n"
            "  - If you want agents to stop sounding soothing or assistant-like, say so concretely in the candidate text: forbid empathy-openers, forbid validation-first disagreement, allow fragments, allow abrupt starts, allow blunt replies when realistic.\n"
            "  - If you want short social-media form, say it explicitly: one-line replies, no-subject fragments, lowercase starts, unfinished clauses, dismissive one-liners, rude subreplies, or short skeptical questions.\n"
            "\n"
            "Step 3 — For each candidate, provide a concise seed overlay_diff with the two text knobs. "
            "These are strategist blueprints, not final polished prompt paragraphs. Encode the exact type of persona or runtime behavior to add, preserve, or suppress using compact operational seed text.\n"
            "Keep the seed concise enough that a second-stage writer can substantially rewrite and expand it.\n"
            "\n"
            "Respond with a single JSON object with these keys:\n"
            "  - diagnosis          (str)           : root cause analysis\n"
            "  - candidates         (list[object])  : exactly 5 candidate objects, each with:\n"
            "      - strategy_label   (str)         : short snake_case identifier (must be unique)\n"
            "      - strategy         (str)         : what this candidate changes and why\n"
            "      - mechanism_family (str)         : one of semantic_diversity, story_anecdote, tone_civility, length_variation, structure\n"
            "      - anti_incumbent   (bool)        : keep false unless the phase explicitly needs a strong challenge\n"
            "      - primary_layer    (str)         : always set this to 'both' in manual phase mode\n"
            "      - overlay_diff     (dict)        : the two text-slot seed edits for this candidate\n"
            "      - rationale        (str)         : why this candidate differs from the others\n"
            "  - constraints        (list[str])     : any hard constraints to enforce (may be [])\n"
            "\n"
            "Return valid JSON only. No commentary outside the JSON object."
        )
    else:
        sections.append(
            "You are a calibration reasoner for a Reddit discussion simulator. "
            "Your job is to close the gap between simulated and real Reddit metrics.\n"
            "\n"
            "Step 1 — DIAGNOSE: For each CRITICAL and SECONDARY metric, explain why it is "
            "failing, referencing its tier, direction (too_high/too_low), and what that "
            "implies about the generated discussions. If sample threads are provided, use "
            "them to understand the qualitative root cause.\n"
            "\n"
            "Step 2 — DESIGN 5 INDEPENDENT STRATEGIES: Each candidate must explore a "
            "DIFFERENT approach to fixing the problems. Strategies must be mechanism-first, "
            "not layer-first. Use these mechanism families:\n"
            "  - semantic_diversity : reduce paraphrase, lexical overlap, semantic sameness\n"
            "  - story_anecdote     : increase natural personal incidents and concrete lived detail\n"
            "  - tone_civility      : adjust sharpness, disagreement style, safety, or emotional realism\n"
            "  - length_variation   : fix comment length shape, heavy tails, short/medium/long mix\n"
            "  - structure          : increase replies, follow-ups, depth, and back-and-forth chains\n"
            "There are ONLY two knobs available: persona.generation_guidance and "
            "prompt.comment_style_guidance. Every candidate MUST write both knobs with "
            "substantive, operational text.\n"
            "For each candidate, pick ONE primary mechanism_family and make that explicit.\n"
            "\n"
            f"{slot_instructions}\n"
            "\n"
            "LEARNING FROM TRAJECTORY:\n"
            "  - The Calibration Trajectory above shows per-candidate group scores and mechanism families.\n"
            "  - Each iteration now includes explicit iteration_delta lines vs the previous iteration.\n"
            "    Use those deltas directly instead of inferring progress by eye.\n"
            "  - winner_metric_delta_vs_previous shows how the selected strategy moved each\n"
            "    headline metric relative to the previous iteration's winner.\n"
            "  - Frontier Candidates shows the strongest known candidate for each metric-group family.\n"
            "    Use frontier entries as branch points when the incumbent is stuck.\n"
            "  - Family Learning Summary compresses historical attempts/wins by mechanism family.\n"
            "    Use it during later iterations to combine families that have complementary strengths.\n"
            "  - Historical information alone is not enough: you must convert it into a search policy.\n"
            "    If one family has stalled, switch family or branch from a frontier challenger.\n"
            "  - In the exploitation phase, mixed candidates should combine proven elements from different families,\n"
            "    not just lightly rewrite the current best.\n"
            "\n"
            "CRITICAL RULE — READ THE DIRECTION CAREFULLY:\n"
            "  - Each metric's diagnostic shows 'direction': generated_higher or generated_lower.\n"
            "  - If a metric is 'generated_lower' (e.g., toxicity too LOW, this means the past results is lower than expected), your fix must INCREASE it, not decrease it.\n"
            "  - If a metric is 'generated_higher' (e.g., self_bleu too HIGH, this means the past results is higher than expected), your fix must DECREASE it.\n"
            "  - Getting the direction wrong wastes an entire candidate. Double-check before writing overlay_diff.\n"
            "\n"
            "CRITICAL RULES for strategy diversity:\n"
            "  - Do NOT make 5 variations of the same idea. Each candidate must have a genuinely different mechanism hypothesis.\n"
            "  - If a direction has been tried before (see Calibration Trajectory), do NOT "
            "repeat it with the same overlay_diff values. The trajectory shows the exact "
            "knob values that were tried — if those values did not improve the score, "
            "you MUST use meaningfully different values or a completely different approach.\n"
            "  - If the same strategy has failed 2+ times, you MUST try a fundamentally "
            "different approach — not the same knobs with slightly tweaked numbers.\n"
            "  - History is diagnostic, not a search policy by itself. You must decide whether to exploit the incumbent,\n"
            "    branch from a frontier challenger, or run an anti-incumbent test.\n"
            "  - Look at partially effective strategies in the trajectory — a strategy that "
            "improved some metric groups but not the overall score is VALUABLE: combine its "
            "successful elements with other approaches.\n"
            "  - Each candidate should change BOTH knobs (persona.generation_guidance AND "
            "prompt.comment_style_guidance). Use one as the main lever if needed, but always "
            "coordinate the cast of people with the writing behavior.\n"
            "\n"
            "Step 3 — For each candidate, provide a complete overlay_diff with the two text knobs. "
            "These should be meaningful seed edits, not empty slogans. Even though a second-stage writer "
            "will expand them, your overlay_diff should already encode:\n"
            "  - the type of people to create or suppress\n"
            "  - the comment/reply behaviors to encourage or suppress\n"
            "  - the anti-patterns to avoid\n"
            "  - the kinds of concrete examples, anecdotes, or reply shapes that should appear\n"
            "Keep them concise but specific enough that another LLM can reliably turn them into final prompt text.\n"
            "\n"
            "Respond with a single JSON object with these keys:\n"
            "  - diagnosis          (str)           : root cause analysis\n"
            "  - candidates         (list[object])  : exactly 5 candidate objects, each with:\n"
            "      - strategy_label   (str)         : short snake_case identifier (must be unique)\n"
            "      - strategy         (str)         : what this candidate changes and why\n"
            "      - mechanism_family (str)         : one of semantic_diversity, story_anecdote, tone_civility, length_variation, structure\n"
            "      - anti_incumbent   (bool)        : true only if this candidate explicitly challenges the incumbent hypothesis\n"
            "      - primary_layer    (str)         : 'persona', 'prompt', or 'both'\n"
            "      - overlay_diff     (dict)        : the two text-slot seed edits for this candidate\n"
            "      - rationale        (str)         : why this approach differs from the others\n"
            "  - constraints        (list[str])     : any hard constraints to enforce (may be [])\n"
            "\n"
            "Return valid JSON only. No commentary outside the JSON object."
        )

    # ── Budget-aware trajectory substitution ────────────────────────────
    # First pass: full trajectory
    full_trajectory_text = trajectory_header + "\n".join(trajectory_lines_full) + "\n"
    prompt_text = "\n".join(
        full_trajectory_text if s == _TRAJECTORY_PLACEHOLDER else s
        for s in sections
    )

    max_chars = int(_DEFAULT_MAX_PROMPT_TOKENS * _CHARS_PER_TOKEN_ESTIMATE)
    if len(prompt_text) > max_chars and trajectory:
        # Truncate iterations 0-2 (the earliest phase) to save context
        skip_iters = {int(e.get("iteration", -1)) for e in trajectory[:3]}
        truncated_lines = _render_trajectory_lines(trajectory, skip_iterations=skip_iters)
        truncated_trajectory_text = trajectory_header + "\n".join(truncated_lines) + "\n"
        prompt_text = "\n".join(
            truncated_trajectory_text if s == _TRAJECTORY_PLACEHOLDER else s
            for s in sections
        )

    return prompt_text


def build_text_materializer_prompt(
    registry: KnobRegistry,
    current_overlay: dict,
    current_diagnostic: dict,
    diagnosis: str,
    candidates: list[dict],
    real_baseline: dict | None = None,
    trajectory: list[dict] | None = None,
    failed_strategies: list[str] | None = None,
    metric_definitions: str = "",
    sample_real_thread: str = "",
    sample_sim_thread: str = "",
    iteration: int = 0,
    max_iterations: int = 10,
    combination_start_iteration: int | None = None,
    global_best_overlay: dict | None = None,
    global_best_diagnostic: dict | None = None,
    frontier: dict[str, dict[str, Any]] | None = None,
    stagnation_count: int = 0,
    search_mode: str = "global_best",
    search_root_reason: str = "global_best",
    phase_context: dict[str, Any] | None = None,
    completed_phase_summaries: list[dict[str, Any]] | None = None,
) -> str:
    """Build the second-stage prompt that writes actual text-block modifications."""

    text_knob_context: list[str] = []
    current_text_overlay: dict[str, Any] = {}
    for name in registry.knob_names():
        knob = registry.get(name)
        if knob["type"] != "text":
            continue
        current_text_overlay[name] = current_overlay.get(name, knob["default"])
        text_knob_context.append(
            f"- {name}: {knob['description']} (default={json.dumps(knob['default'], ensure_ascii=False)})"
        )

    diagnostic_slice = {
        "fail_rate": current_diagnostic.get("fail_rate"),
        "mean_abs_delta": current_diagnostic.get("mean_abs_delta"),
        "per_metric": current_diagnostic.get("per_metric", {}),
    }
    real_baseline = real_baseline or {}
    trajectory = trajectory or []
    failed_strategies = failed_strategies or []
    global_best_overlay = global_best_overlay or current_overlay
    global_best_diagnostic = global_best_diagnostic or current_diagnostic
    frontier = frontier or {}
    completed_phase_summaries = completed_phase_summaries or []
    if combination_start_iteration is None:
        combination_start_iteration = max_iterations // 2
    combination_start_iteration = max(1, min(int(combination_start_iteration), max_iterations))

    candidate_payload = []
    for idx, candidate in enumerate(candidates[:5]):
        candidate_payload.append(
            {
                "candidate_id": idx,
                "strategy_label": candidate.get("strategy_label", f"candidate_{idx}"),
                "mechanism_family": candidate.get("mechanism_family", "semantic_diversity"),
                "anti_incumbent": bool(candidate.get("anti_incumbent", False)),
                "primary_layer": candidate.get("primary_layer", "both"),
                "strategy": candidate.get("strategy", ""),
                "rationale": candidate.get("rationale", ""),
                "overlay_diff": candidate.get("overlay_diff", {}),
            }
        )

    output_contract_lines = []
    for idx in range(len(candidate_payload)):
        comma = "," if idx < len(candidate_payload) - 1 else ""
        output_contract_lines.append(
            f'  "candidate_{idx}": {{"text_overlay_diff": {{"persona.generation_guidance": "...", "prompt.comment_style_guidance": "..."}}}}{comma}'
        )
    output_contract = "{\n" + "\n".join(output_contract_lines) + "\n}"
    required_candidate_keys = ", ".join(
        f"candidate_{idx}" for idx in range(len(candidate_payload))
    )

    search_state = {
        "iteration": iteration + 1,
        "max_iterations": max_iterations,
        "search_mode": search_mode,
        "stagnation_count": stagnation_count,
        "search_root_reason": search_root_reason,
        "combination_start_iteration": combination_start_iteration,
    }

    sections = [
        TEXT_MATERIALIZER_PRINCIPLES,
        "",
        REALISM_RULES,
        "",
        CALIBRATION_COMPARISON_STATS_GUIDE,
        "",
    ]
    if metric_definitions and metric_definitions.strip():
        sections.extend([
            "## Additional Metric Reference",
            metric_definitions.strip(),
            "",
        ])
    sections.extend([
        "## Allowed Text Knobs",
        "\n".join(text_knob_context),
        "",
        "## Search State",
        json.dumps(search_state, indent=2, ensure_ascii=False),
        "",
        "## Current Text Overlay",
        json.dumps(current_text_overlay, indent=2, ensure_ascii=False),
        "",
        "## Current Diagnostic",
        json.dumps(diagnostic_slice, indent=2, ensure_ascii=False),
        "",
    ])
    if real_baseline:
        sections.extend([
            "## Real Validation Reference Summary",
            json.dumps(real_baseline, indent=2, ensure_ascii=False),
            "",
        ])
    if global_best_overlay != current_overlay:
        sections.extend([
            "## Global Best Overlay",
            json.dumps(global_best_overlay, indent=2, ensure_ascii=False),
            "",
            "## Global Best Diagnostic Summary",
            json.dumps(
                {
                    "fail_rate": global_best_diagnostic.get("fail_rate"),
                    "mean_abs_delta": global_best_diagnostic.get("mean_abs_delta"),
                    "quantile_fail_rate": global_best_diagnostic.get("quantile_fail_rate"),
                    "mean_percentile_distance": global_best_diagnostic.get("mean_percentile_distance"),
                    "mean_abs_robust_z": global_best_diagnostic.get("mean_abs_robust_z"),
                    "group_scores": global_best_diagnostic.get("group_scores", {}),
                },
                indent=2,
                ensure_ascii=False,
            ),
            "",
        ])
    if frontier:
        sections.extend([
            "## Frontier Candidates",
            (
                "These are the strongest non-incumbent branch points seen so far. "
                "Use them to understand what partial wins should be preserved or combined "
                "when turning strategist seeds into final prompt text."
            ),
            json.dumps(frontier, indent=2, ensure_ascii=False),
            "",
        ])
    phase_section = _format_phase_context_section(phase_context)
    if phase_section:
        sections.extend([phase_section, ""])
    if completed_phase_summaries:
        sections.extend(
            [
                "## Completed Phase Best Overlays",
                json.dumps(completed_phase_summaries, indent=2, ensure_ascii=False),
                "",
            ]
        )
    previous_candidate_feedback = _format_local_candidate_metric_feedback(trajectory, phase_context)
    if previous_candidate_feedback:
        sections.extend([previous_candidate_feedback, ""])
    if trajectory:
        sections.extend([
            "## Local Calibration Trajectory",
            (
                "This is the same trajectory context used by the strategist for this step. "
                "Use it to understand what was just tried, what improved, and what should be "
                "preserved, strengthened, or avoided when writing the final text blocks."
            ),
            json.dumps(trajectory, indent=2, ensure_ascii=False),
            "",
        ])
    if failed_strategies:
        sections.extend([
            "## Failed Strategies",
            "\n".join(f"- {label}" for label in failed_strategies),
            "",
        ])
    if sample_real_thread or sample_sim_thread:
        sections.extend(
            [
                "## Sample Threads (qualitative anchors)",
                (
            "Use these few-shot threads as the realism anchor while writing the final text blocks. "
            "Do not copy content. Extract their tone, anecdote shape, diversity of syntax, length mix, "
            "and conflict style, then turn that into reusable guidance."
        ),
            ]
        )
        if sample_real_thread:
            sections.extend(
                [
                    "### Real Reddit Threads",
                    sample_real_thread.strip(),
                    "",
                ]
            )
        if sample_sim_thread:
            sections.extend(
                [
                    "### Current Best Simulated Threads",
                    sample_sim_thread.strip(),
                    "",
                ]
            )
    sections.extend([
        "## Strategist Diagnosis",
        diagnosis or "",
        "",
        "## Candidate Strategies To Materialize",
        json.dumps(candidate_payload, indent=2, ensure_ascii=False),
        "",
        "## Task",
        (
            "For each candidate, produce only the TEXT modifications that should be injected "
            "into the simulator. These text modifications should operationalize the strategy.\n"
            "\n"
            "Requirements:\n"
            "- For EVERY candidate, you MUST write BOTH persona.generation_guidance and prompt.comment_style_guidance.\n"
            "- In manual phase mode, all candidates are mixed both-knob candidates. Use dominant emphasis only to decide where to add more detail.\n"
            "- One block may be shorter when the candidate has a clear dominant emphasis, but both must be present and operational.\n"
            "- These are the ONLY two knobs available. Do not invent other knob names.\n"
            "- You MUST return one output block for every candidate id shown above. Do not omit any candidate.\n"
            "- Do NOT return a list. Return an object with exact keys candidate_0, candidate_1, ... in numeric order.\n"
            f"- The required top-level keys for this call are exactly: {required_candidate_keys}.\n"
            "- Each text block should be 1-3 paragraphs with concrete instructions, logic, anti-patterns, and examples.\n"
            "- The strategist overlay_diff is only a seed blueprint. You must rewrite and expand it into stronger final runtime text.\n"
            "- Do not copy long spans verbatim from the strategist seed. Add clearer operational rules, anti-patterns, and examples in your own wording.\n"
            "- A valid materialization should be noticeably richer than the seed: more concrete, more executable, and more specific to the target metrics.\n"
            "- Use the strategist diagnosis, local trajectory, failed strategies, frontier hints, and real validation reference together. "
            "Do not write blind generic text when the context already tells you what recently worked or failed.\n"
            "- Preserve previously working behavior from completed phases unless the active phase explicitly calls for a bounded refinement.\n"
            "- The current overlay may already be structured by named manual-phase sections. Treat those sections as stable preserved blocks and write new text for the active phase section rather than rewriting the whole accumulated prompt.\n"
            "- If a phase objective is present, keep the text tightly aligned with the active focus metrics and avoid rewriting unrelated sections.\n"
            "- Only the final integrated block may optimize broadly across all metrics. Earlier blocks should write for the active focus metrics first and merely preserve protected metrics.\n"
            "- Keep the true end goal visible while writing: the resulting runtime text should push the active target metrics toward the same real validation distribution — drive |Cliff's delta| and Wasserstein distance toward 0.\n"
            "- If you want less politeness, less validation, rougher grammar, sharper disagreement, or more clipped social-media syntax, write that explicitly into the calibration text itself. Do not assume the base runtime prompt will create that style for you.\n"
            "- If previous-iteration candidate metric feedback is shown, use it directly when writing the final text blocks:\n"
            "  keep and strengthen mechanisms that improved the active focus metrics;\n"
            "  avoid or weaken mechanisms that worsened those focus metrics;\n"
            "  and combine partial wins carefully instead of copying weak candidates wholesale.\n"
            "- Write append-only patch text for the active phase block. Assume the runtime will place your text into that named phase section instead of replacing earlier sections.\n"
            "- If the current overlay is already structured by phase blocks, extend only the active block's section and keep earlier block sections intact.\n"
            "- Reuse and specialize the example directions from the active phase block. Turn them into executable prompt language, not abstract summaries.\n"
            "- Use the real few-shot sample threads as style evidence for the target metrics:\n"
            "  for story, match how real threads tell short, specific lived experiences;\n"
            "  for diversity, match how real comments vary syntax, cadence, and template shape;\n"
            "  for length, match the real short/medium/long mix;\n"
            "  for conflict, match the real bluntness, sarcasm, profanity, or hostility level when those appear.\n"
            "- If the real examples sound less soothing than the current simulation, encode that concretely in calibration text: forbid default empathy-openers, forbid validation-first disagreement, and allow clipped fragments, abrupt starts, dropped subjects, messy punctuation, or direct hostility when they fit the persona.\n"
            "- For diversity and conflict blocks, do not stop at abstract wording like 'vary tone' or 'be more aggressive'. Write the actual surface forms you want to see: one-line replies, lowercase fragments, no-subject dismissals, sharper subreplies, rude questions, or blunt corrections.\n"
            "- Read the few-shot threads deeply enough to understand the real direction of the target metric before you write any text. Do not guess at what 'story', 'diversity', 'length', or 'conflict' should look like; infer it from the real examples.\n"
            "- For length and structure targets, avoid flattening the thread into a narrow safe middle. The real target often has a broader spread; a slightly broader simulation is preferable to an overly uniform assistant-like one.\n"
            "- Do not copy the sample content verbatim. Translate the observed realism pattern into instructions.\n"
            "- persona.generation_guidance should directly tell the persona generator what kinds of people to cast:\n"
            "  their repeated grievances, what they keep bringing up, how certain they sound, what product memories they carry,\n"
            "  what triggers them to reply, and what kind of narrow angle or bias they often push.\n"
            "- prompt.comment_style_guidance should directly tell the runtime writer how those people should comment:\n"
            "  what concrete angle to pick, how to avoid paraphrase, when to reply to a visible comment, how to disagree,\n"
            "  how much anecdotal detail to use, what lengths/forms are realistic, and what low-value patterns to avoid.\n"
            "- Make the blocks self-contained and ready to paste into a prompt. Do not write meta commentary like\n"
            "  'here is the revised guidance' or 'as requested'.\n"
            "- The materialized prompt.comment_style_guidance must be domain-general unless the active phase/sample threads clearly identify a specific domain.\n"
            "- Use domain-adaptive slots such as ownership_or_usage_history, purchase_context, failure_or_success_event, comparison_target, decision_constraint, usage_pattern, technical_or_value_detail, and confidence_level.\n"
            "- Do not hard-code credit-card-only terms such as AF, SUB, 5/24, FTF, APR, recon, or cashback unless the active sample/domain is credit cards.\n"
            "- Likewise, do not hard-code laptop, phone, camera, or headphone-specific jargon unless the active sample/domain supports it.\n"
            "- For prompt.comment_style_guidance, include an anti-template gate whenever the candidate targets diversity, structure, conflict, or the final integrated objective.\n"
            "- Do not write generic slogans. Every sentence should be actionable.\n"
            "Return valid JSON only in this format:\n"
            f"{output_contract}"
        ),
    ])
    return "\n".join(sections)


def build_dedup_prompt(
    current_overlay: dict,
    num_candidates: int = 5,
) -> str:
    """Build a prompt for the dedup-only round.

    The LLM receives the current structured overlay and must produce
    ``num_candidates`` deduplicated versions.  Each candidate removes
    repetition/redundancy but does NOT add, change, or remove any rule
    or behavioral instruction.
    """
    import copy

    overlay = copy.deepcopy(current_overlay)
    # Show the full overlay including _manual_phase_blocks
    overlay_json = json.dumps(overlay, indent=2, ensure_ascii=False)

    candidate_keys = ", ".join(f'"candidate_{i}"' for i in range(num_candidates))
    output_lines = []
    for i in range(num_candidates):
        comma = "," if i < num_candidates - 1 else ""
        output_lines.append(
            f'  "candidate_{i}": {{'
            f'"persona.generation_guidance": "...", '
            f'"prompt.comment_style_guidance": "...", '
            f'"_manual_phase_blocks": {{...}}'
            f'}}{comma}'
        )
    output_contract = "{\n" + "\n".join(output_lines) + "\n}"

    sections = [
        "## Task: Deduplicate Overlay Text\n",
        "You are a calibration text editor.  Your ONLY job is to remove "
        "redundant, duplicated, or near-duplicated content from the overlay "
        "below.  You must produce EXACTLY the same behavioral instructions "
        "in fewer characters.\n",
        "### Hard Rules\n"
        "1. Do NOT add any new rules, instructions, examples, or content.\n"
        "2. Do NOT remove any rule, instruction, or behavioral constraint — "
        "every unique rule must survive in the output.\n"
        "3. Do NOT change the meaning, intent, or specificity of any rule.\n"
        "4. Do NOT change number thresholds, percentages, distributions, or "
        "named constants.\n"
        "5. Do NOT merge rules from different phase blocks — each phase block "
        "must remain a self-contained section.\n"
        "6. If the same rule appears in multiple places (e.g., 'max 2 replies "
        "per parent' appears 5 times), keep the most complete version and "
        "replace the others with a short back-reference like "
        "'(per anti-template gate above)' or remove them entirely.\n"
        "7. If a rationale/explanation paragraph appears in both persona and "
        "prompt sections with the same content, keep it in only one place "
        "(prefer persona for persona-related rationale, prompt for prompt-related).\n"
        "8. Preserve the _manual_phase_blocks structure.  Each block must "
        "retain its phase_label, phase_order, persona.generation_guidance, "
        "and prompt.comment_style_guidance keys.\n"
        "9. The top-level persona.generation_guidance and "
        "prompt.comment_style_guidance should be the rendered (concatenated) "
        "version of the phase blocks — keep them consistent.\n"
        "10. Aim for a 30-50% character reduction while preserving ALL unique "
        "behavioral content.\n",
        f"### Candidate Diversity ({num_candidates} candidates)\n"
        "Each candidate should try a different dedup strategy:\n"
        "- Candidate 0: Conservative — only remove exact or near-exact duplicates.\n"
        "- Candidate 1: Moderate — also compress verbose explanations into "
        "terser versions while keeping all rules.\n"
        "- Candidate 2: Aggressive — maximum compression; use short references "
        "for repeated concepts.\n"
        "- Candidate 3: Structure-focused — reorganize within each phase block "
        "to group related rules, removing redundancy in the process.\n"
        "- Candidate 4: Best-judgment — your best guess at optimal "
        "deduplication balancing readability and compression.\n",
        "### Current Overlay to Deduplicate\n",
        "```json",
        overlay_json,
        "```\n",
        f"Return valid JSON with keys: {candidate_keys}\n"
        f"Each candidate value must contain the same keys as the input overlay.\n"
        f"Format:\n{output_contract}",
    ]
    return "\n".join(sections)


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def _supports_custom_temperature(model: str) -> bool:
    """Return whether the model accepts a non-default temperature override."""
    normalized = model.strip().lower()
    return not normalized.startswith("gpt-5")


def call_reasoner(
    client,
    model: str,
    prompt: str,
    reasoning_effort: str | None = None,
    schema_kind: str | None = None,
    response_format_override: dict[str, Any] | None = None,
) -> str:
    """Call the OpenAI-compatible API with the reasoner prompt.

    Supports both openai >= 1.0 (client.chat.completions.create) and
    openai < 1.0 (openai.ChatCompletion.create).
    """
    if OpenAI is not None and isinstance(client, OpenAI):
        # openai >= 1.0
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a calibration component. "
                        "Return only valid JSON that fully satisfies the required schema. "
                        "Do not omit required fields. Do not return partial answers."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "response_format": response_format_override or _response_format_for(schema_kind),
            "timeout": 120,
        }
        if _supports_custom_temperature(model):
            kwargs["temperature"] = 0.4
        if reasoning_effort:
            kwargs["reasoning_effort"] = reasoning_effort
        response = client.chat.completions.create(
            **kwargs,
        )
        return response.choices[0].message.content
    else:
        # openai < 1.0 — set module-level credentials
        import openai as _openai
        _api_key = getattr(client, "api_key", None)
        _base_url = getattr(client, "base_url", None)
        if _api_key:
            _openai.api_key = _api_key
        if _base_url:
            _openai.api_base = str(_base_url)
        response = _openai.ChatCompletion.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a calibration component. "
                        "Always respond with complete valid JSON. "
                        "Never omit required fields."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            **({"temperature": 0.4} if _supports_custom_temperature(model) else {}),
        )
        content = response["choices"][0]["message"]["content"]
        if not content or not content.strip():
            raise RuntimeError(f"LLM returned empty response. Full response: {response}")
        return content



# ---------------------------------------------------------------------------
# Response parsing (moved to calibration/_reasoner_parse.py)
# ---------------------------------------------------------------------------
from ._reasoner_parse import (
    _parse_jsonish_response,
    _fallback_candidate,
    parse_reasoner_response,
    parse_text_materializer_response,
    _extract_text_overlay_from_dict,
)



# ---------------------------------------------------------------------------
# Variant generation
# ---------------------------------------------------------------------------

def generate_variants(
    current_overlay: dict,
    base_diff: dict,
    prompt_alternatives: dict,
    registry: KnobRegistry,
    seed: int = 42,
    conservative_diff: dict | None = None,
    parsed_candidates: list[dict] | None = None,
    append_text_mode: bool = False,
    structured_phase_name: str | None = None,
    structured_phase_label: str | None = None,
    structured_phase_order: int | None = None,
) -> list[dict]:
    """Generate candidate overlays for evaluation.

    If *parsed_candidates* has 5 entries (new LLM format with independent
    strategies), each candidate's overlay_diff is merged directly with
    current_overlay. No mechanical jitter is applied.

    If *parsed_candidates* has fewer than 5 entries (legacy single-strategy
    format), the old variant-generation logic is used as fallback.
    """
    # ── New format: 5 independent strategies from LLM ──
    if append_text_mode and structured_phase_name:
        def merge_fn(base: dict, patch: dict) -> dict:
            return apply_structured_phase_overlay(
                base,
                patch,
                phase_name=structured_phase_name,
                phase_label=structured_phase_label,
                phase_order=structured_phase_order,
            )
    else:
        merge_fn = append_text_overlay if append_text_mode else merge_overlay

    if parsed_candidates and len(parsed_candidates) >= 5:
        overlays = []
        for c in parsed_candidates[:5]:
            overlays.append(merge_fn(current_overlay, c.get("overlay_diff", {})))
        return overlays

    # ── Legacy fallback: text-only world has no meaningful numeric jitter ──
    # Duplicate exact-merge candidates, optionally layering any prompt alternatives
    # that may still appear in old logs or legacy responses.
    candidate_0 = merge_fn(current_overlay, base_diff)
    candidate_1 = merge_fn(current_overlay, conservative_diff or base_diff)
    candidate_2 = merge_fn(current_overlay, base_diff)
    alt_keys = list(prompt_alternatives.keys())
    if len(alt_keys) >= 1:
        first_alt = {alt_keys[0]: prompt_alternatives[alt_keys[0]]}
        candidate_3 = merge_fn(candidate_0, first_alt)
    else:
        candidate_3 = merge_fn(current_overlay, base_diff)
    if len(alt_keys) >= 2:
        second_alt = {alt_keys[1]: prompt_alternatives[alt_keys[1]]}
        candidate_4 = merge_fn(candidate_0, second_alt)
    elif len(alt_keys) == 1:
        first_alt = {alt_keys[0]: prompt_alternatives[alt_keys[0]]}
        candidate_4 = merge_fn(candidate_0, first_alt)
    else:
        candidate_4 = merge_fn(current_overlay, base_diff)

    return [candidate_0, candidate_1, candidate_2, candidate_3, candidate_4]
