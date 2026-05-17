"""Misc helper functions for the calibration orchestrator.

Overlay sanitization, sample-thread extraction, checkpoint loading, etc.
Extracted from orchestrator.py to keep file sizes manageable.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .overlay import (
    STRUCTURED_PHASE_BLOCKS_KEY,
    diff_overlay,
    render_structured_overlay,
    save_overlay,
)
from .registry import KnobRegistry


def _sanitize_overlay(
    registry: KnobRegistry,
    overlay: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    """Sanitize one overlay against the registry and deduplicate errors."""
    cleaned, errors = registry.sanitize_overlay(overlay)
    if STRUCTURED_PHASE_BLOCKS_KEY in overlay:
        cleaned[STRUCTURED_PHASE_BLOCKS_KEY] = overlay[STRUCTURED_PHASE_BLOCKS_KEY]
        cleaned = render_structured_overlay(cleaned)
        structured_error = f"Unknown knob: '{STRUCTURED_PHASE_BLOCKS_KEY}'."
        errors = [err for err in errors if err != structured_error]
    deduped = list(dict.fromkeys(errors))
    return cleaned, deduped


def _composite_thread_key(product: str | None, thread_id: str | None) -> str:
    """Return a stable thread key compatible with split-aware few-shot filters."""
    product_str = str(product or "").strip()
    thread_str = str(thread_id or "").strip()
    if product_str and thread_str:
        return f"{product_str}::{thread_str}"
    return thread_str


def _format_terminal_value(value: Any) -> str:
    """Render a knob value for terminal output without truncation."""
    if isinstance(value, float):
        return f"{value:.4f}"
    if isinstance(value, (int, bool)):
        return str(value)
    if isinstance(value, str):
        return value.strip()

    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2)


def _maybe_record_completed_phase_summary(
    state: "CalibrationState",
    phase_context: dict[str, Any],
) -> None:
    """Persist the current cumulative best as the completed best for a phase block."""
    if not state.current_best_overlay or not state.current_best_diagnostic:
        return
    phase_name = str(phase_context.get("name", "")).strip()
    if not phase_name:
        return
    if any(summary.get("phase_name") == phase_name for summary in state.completed_phase_summaries):
        return
    summary = {
        "phase_name": phase_name,
        "phase_label": phase_context.get("label"),
        "block_label": phase_context.get("block_label"),
        "iteration_end": phase_context.get("iteration_end"),
        "focus_metrics": list(phase_context.get("focus_metrics", [])),
        "overlay": dict(state.current_best_overlay),
        "diagnostic": dict(state.current_best_diagnostic),
        "candidate_dir": state.current_best_candidate_dir,
    }
    state.completed_phase_summaries.append(summary)


def _completed_phase_prompt_summary(completed_phase_summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return a compact phase-best summary for prompt consumption."""
    prompt_rows: list[dict[str, Any]] = []
    for summary in completed_phase_summaries:
        diagnostic = summary.get("diagnostic", {}) or {}
        prompt_rows.append(
            {
                "phase_name": summary.get("phase_name"),
                "phase_label": summary.get("phase_label"),
                "block_label": summary.get("block_label"),
                "focus_metrics": summary.get("focus_metrics", []),
                "quantile_fail_rate": diagnostic.get("quantile_fail_rate"),
                "mean_percentile_distance": diagnostic.get("mean_percentile_distance"),
                "mean_abs_robust_z": diagnostic.get("mean_abs_robust_z"),
                "overlay": summary.get("overlay", {}),
            }
        )
    return prompt_rows


def _knob_runtime_location(name: str, knob: dict[str, Any]) -> str:
    """Describe where a persisted calibration text slot is consumed."""
    location_overrides = {
        "persona.generation_guidance": "persona generator / calibration persona guidance block",
        "prompt.comment_style_guidance": "system prompt + action prompt / calibration comment guidance block",
    }
    if name in location_overrides:
        return location_overrides[name]
    return f"{knob['layer']} runtime / {knob['domain']}"


def _overlay_change_records(
    registry: KnobRegistry,
    previous_overlay: dict[str, Any],
    candidate_overlay: dict[str, Any],
) -> list[dict[str, Any]]:
    """Return structured change records for the effective candidate changes."""
    changed = diff_overlay(previous_overlay, candidate_overlay)
    records: list[dict[str, Any]] = []

    def _sort_key(name: str) -> tuple[str, str]:
        try:
            knob = registry.get(name)
        except KeyError:
            return ("zz_internal", name)
        return (str(knob.get("layer", "")), name)

    for name in sorted(changed.keys(), key=_sort_key):
        try:
            knob = registry.get(name)
        except KeyError:
            continue
        old_value = previous_overlay.get(name, knob["default"])
        new_value = candidate_overlay.get(name, knob["default"])
        record: dict[str, Any] = {
            "name": name,
            "layer": knob["layer"],
            "domain": knob["domain"],
            "type": knob["type"],
            "description": knob["description"],
            "runtime_location": _knob_runtime_location(name, knob),
            "old_value": old_value,
            "new_value": new_value,
        }

        if knob["type"] == "distribution":
            old_dist = old_value if isinstance(old_value, dict) else {}
            new_dist = new_value if isinstance(new_value, dict) else {}
            subchanges: list[dict[str, Any]] = []
            for key in knob["keys"]:
                old_key_val = float(old_dist.get(key, 0.0))
                new_key_val = float(new_dist.get(key, 0.0))
                if abs(old_key_val - new_key_val) > 1e-12:
                    subchanges.append({
                        "key": key,
                        "old_value": old_key_val,
                        "new_value": new_key_val,
                    })
            record["changed_keys"] = subchanges

        records.append(record)

    return records


def _print_candidate_change_preview(
    candidate_id: int,
    strategy_label: str,
    primary_layer: str,
    strategy: str,
    rationale: str,
    changes: list[dict[str, Any]],
    validation_errors: list[str] | None = None,
) -> None:
    """Print one candidate's exact persisted text edits to stdout."""
    print(f"      [{candidate_id}] {strategy_label} (layer={primary_layer})")
    if strategy:
        print(f"          strategy : {strategy}")
    if rationale:
        print(f"          rationale: {rationale}")

    if not changes:
        print("          changes  : no effective edits after validation")
    else:
        print(f"          changes  : {len(changes)} applied edit(s)")
        for change in changes:
            header = (
                f"          - {change['name']} "
                f"[{change['layer']} | {change['domain']}]"
            )
            print(header)
            runtime_location = change.get(
                "runtime_location",
                f"{change['layer']} runtime / {change['domain']}",
            )
            print(f"            applies at: {runtime_location}")
            if change["type"] == "distribution":
                subchanges = change.get("changed_keys", [])
                if subchanges:
                    joined = "; ".join(
                        f"{entry['key']} {entry['old_value']:.4f} -> {entry['new_value']:.4f}"
                        for entry in subchanges
                    )
                    print(f"            {joined}")
                else:
                    print("            no distribution entries changed")
            else:
                old_value = _format_terminal_value(change["old_value"])
                new_value = _format_terminal_value(change["new_value"])
                if change["type"] == "text":
                    print("            old:")
                    for line in old_value.splitlines() or [""]:
                        print(f"              {line}")
                    print("            new:")
                    for line in new_value.splitlines() or [""]:
                        print(f"              {line}")
                else:
                    print(f"            {old_value} -> {new_value}")

    if validation_errors:
        print("          validation:")
        for err in validation_errors:
            print(f"            - {err}")


# ---------------------------------------------------------------------------
# Sample thread extraction (for reasoner prompt)
# ---------------------------------------------------------------------------

def _extract_sample_real_thread(
    few_shot_dir: Path,
    train_thread_ids: list[str],
    max_comments: int = 15,
    count: int = 2,
) -> str:
    """Extract *count* real Reddit threads as readable text samples.

    Picks random train-only threads from the few_shot_dir.
    Returns a formatted string suitable for inclusion in the reasoner prompt.
    """
    import random as _rnd

    # Find .comments.jsonl files
    comment_files: list[Path] = []
    for sub in sorted(few_shot_dir.iterdir()):
        if not sub.is_dir():
            continue
        for f in sub.iterdir():
            if f.name.endswith(".comments.jsonl"):
                comment_files.append(f)

    if not comment_files:
        return ""

    # Collect all eligible threads across files
    all_threads: list[tuple[str, list[dict]]] = []
    _rnd.shuffle(comment_files)
    for cf in comment_files[:20]:
        threads: dict[str, list[dict]] = {}
        try:
            for line in cf.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                obj = json.loads(line)
                tid = str(obj.get("post_id", ""))
                composite_id = _composite_thread_key(cf.parent.name, tid)
                if tid and (
                    not train_thread_ids
                    or tid in train_thread_ids
                    or composite_id in train_thread_ids
                ):
                    threads.setdefault(tid, []).append(obj)
        except Exception:
            continue

        for tid, comments in threads.items():
            if len(comments) >= 3:
                all_threads.append((tid, comments))

    if not all_threads:
        return ""

    # Shuffle and pick up to *count* threads
    _rnd.shuffle(all_threads)
    samples: list[str] = []
    for tid, comments in all_threads[:count]:
        lines = [f"[Thread ID: {tid}]"]
        title = comments[0].get("post_title", "")
        if title:
            lines.append(f"Title: {title}")
        lines.append("")
        for c in comments[:max_comments]:
            author = c.get("author", "anonymous")
            body = c.get("body", "").strip()
            depth = c.get("depth", 0)
            indent = "  " * int(depth)
            lines.append(f"{indent}[{author}] (depth={depth}): {body[:200]}")
        samples.append("\n".join(lines))

    return "\n\n---\n\n".join(samples)


def _extract_sample_sim_thread(
    best_candidate_dir: Path | None,
    max_comments: int = 15,
    count: int = 2,
) -> str:
    """Extract up to *count* simulated threads from the best candidate's discussion.json.

    Returns a formatted string suitable for inclusion in the reasoner prompt.
    """
    if best_candidate_dir is None:
        return ""

    # Allow the stored path to point directly at a simulation directory.
    direct_discussion = best_candidate_dir / "discussion.json"
    if direct_discussion.exists():
        discussion_path = direct_discussion
    else:
        discussion_path = None

    # Find discussion.json in sim_output
    sim_output = best_candidate_dir / "sim_output"
    if discussion_path is None and sim_output.exists():
        for sub in sim_output.iterdir():
            if sub.is_dir():
                dp = sub / "discussion.json"
                if dp.exists():
                    discussion_path = dp
                    break

    if discussion_path is None:
        return ""

    try:
        data = json.loads(discussion_path.read_text(encoding="utf-8"))
    except Exception:
        return ""

    posts = data.get("posts", [])
    if not posts:
        return ""

    # Pick up to *count* posts with enough comments
    samples: list[str] = []
    for post in posts:
        comments = post.get("comments", [])
        if len(comments) >= 3:
            lines = [f"[Simulated Thread]"]
            lines.append(f"Post: {post.get('content', '')[:200]}")
            lines.append("")
            for c in comments[:max_comments]:
                author = c.get("author", "agent")
                body = c.get("content", "").strip()
                depth = c.get("depth", 0)
                indent = "  " * int(depth)
                lines.append(f"{indent}[{author}] (depth={depth}): {body[:200]}")
            samples.append("\n".join(lines))
            if len(samples) >= count:
                break

    return "\n\n---\n\n".join(samples)


def _find_reusable_vanilla_sim_dir(vanilla_scores_csv: Path | None) -> Path | None:
    """Return one existing vanilla simulation directory for iter-0 reuse."""
    if vanilla_scores_csv is None:
        return None
    runs_dir = vanilla_scores_csv.parent / "runs"
    if not runs_dir.exists():
        return None
    candidates = sorted(
        path for path in runs_dir.iterdir()
        if path.is_dir() and (path / "thread_metrics_summary.csv").exists()
    )
    return candidates[0] if candidates else None


def _resolve_eval_thread_target(reference_thread_count: int, requested_cap: int) -> int:
    """Return the effective evaluation-thread target.

    A positive ``requested_cap`` is treated as an upper bound, so evaluation
    runs target ``min(reference_thread_count, requested_cap)`` threads.
    Non-positive values preserve the prior "no explicit cap" behavior.
    """

    if requested_cap <= 0:
        return 0
    if reference_thread_count <= 0:
        return 0
    return min(reference_thread_count, requested_cap)


def _make_reused_baseline_candidate_result(
    iter_dir: Path,
    overlay: dict[str, Any],
    source_sim_dir: Path,
) -> dict[str, Any]:
    """Create a pseudo candidate result that reuses a precomputed vanilla sim."""
    candidate_dir = iter_dir / "candidates" / "candidate_0"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    save_overlay(overlay, candidate_dir / "overlay.json")
    (candidate_dir / "reused_from.txt").write_text(
        str(source_sim_dir), encoding="utf-8",
    )
    return {
        "candidate_id": 0,
        "candidate_dir": str(source_sim_dir),
        "sim_dir": str(source_sim_dir),
        "success": True,
        "returncode": 0,
        "reused": True,
        "reused_from": str(source_sim_dir),
    }


def _load_iteration_checkpoint(iter_dir: Path) -> dict[str, Any] | None:
    """Load a partially-completed iteration's saved candidate set, if present."""
    diagnosis_path = iter_dir / "diagnosis.json"
    if not diagnosis_path.exists():
        return None

    try:
        payload = json.loads(diagnosis_path.read_text(encoding="utf-8"))
    except Exception:
        return None

    candidate_dir_root = iter_dir / "candidates"
    if not candidate_dir_root.exists():
        return None

    candidate_dirs = sorted(
        path for path in candidate_dir_root.iterdir()
        if path.is_dir() and path.name.startswith("candidate_")
    )
    if not candidate_dirs:
        return None

    overlays: list[dict[str, Any]] = []
    for candidate_dir in candidate_dirs:
        overlay_path = candidate_dir / "overlay.json"
        if not overlay_path.exists():
            return None
        try:
            overlays.append(json.loads(overlay_path.read_text(encoding="utf-8")))
        except Exception:
            return None

    candidate_previews = payload.get("candidates", [])
    validation_errors = payload.get("validation_errors", [])
    overlay_diff = payload.get("overlay_diff", {})

    return {
        "strategy_label": payload.get("strategy_label", "resumed_iteration"),
        "diagnosis": payload.get("diagnosis", ""),
        "candidate_previews": candidate_previews,
        "overlays": overlays,
        "overlay_diff": overlay_diff if isinstance(overlay_diff, dict) else {},
        "validation_errors": validation_errors if isinstance(validation_errors, list) else [],
    }


