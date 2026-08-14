from __future__ import annotations

import math
import re
from typing import Any, Iterable

from .lexical_quality import candidate_thread_bleu_diagnostics
from .reference_metric_calibration import thread_size_bucket


SNIPPET_SPACE_RE = re.compile(r"\s+")


def build_thread_distribution_target(
    template: dict[str, Any] | None,
    calibration: dict[str, Any] | None,
    *,
    generated_comment_count: int | None = None,
) -> dict[str, Any]:
    """Build a text-free target for one generated thread.

    The target comes from one same-size held-out real metric template. Metric
    bands aggregate other held-out real threads in the same size bucket.
    """

    row = dict(template or {})
    if not row:
        return {}
    template_comment_count = _safe_int(row.get("comment_count"), 0)
    comment_count = max(
        1,
        _safe_int(generated_comment_count, template_comment_count),
    )
    bucket = thread_size_bucket(comment_count)
    by_size = (calibration or {}).get("metric_bands_by_size") or {}
    return {
        "comment_count": comment_count,
        "template_comment_count": template_comment_count,
        "thread_size_bucket": bucket,
        "self_bleu_4": _finite_float(row.get("self_bleu_4")),
        "semantic_mean_cosine": _finite_float(
            row.get("semantic_mean_cosine")
        ),
        "metric_bands": dict(by_size.get(bucket) or {}),
        "source": "evaluation-excluded same-size aggregate metric template",
        "raw_text_included": False,
    }


def distribution_target_with_slot_progress(
    target: dict[str, Any] | None,
    *,
    local_task_id: Any,
) -> dict[str, Any]:
    """Attach generated-slot progress without changing the frozen target."""

    result = dict(target or {})
    processed = _safe_int(local_task_id, 0)
    if processed > 0:
        total = max(processed, _safe_int(result.get("comment_count"), processed))
        result["processed_slot_count"] = min(processed, total)
    return result


def joint_candidate_diagnostics(
    *,
    text: str,
    previous_texts: Iterable[str],
    lexical_calibration: dict[str, Any],
    thread_target: dict[str, Any] | None,
    semantic_index: Any | None,
) -> dict[str, Any]:
    previous = [
        _semantic_clean(value) for value in previous_texts if str(value).strip()
    ]
    target = dict(thread_target or {})
    lexical = candidate_thread_bleu_diagnostics(
        text=text,
        previous_texts=previous,
        calibration=lexical_calibration,
        thread_target=target,
    )
    semantic = semantic_thread_diagnostics(
        text=text,
        previous_texts=previous,
        thread_target=target,
        semantic_index=semantic_index,
    )
    components: list[float] = []
    if lexical.get("available"):
        components.append(_normalized_target_distance(lexical))
    if semantic.get("available"):
        components.append(_normalized_target_distance(semantic))
    return {
        "self_bleu": lexical,
        "semantic_cosine": semantic,
        "joint_target_distance": (
            sum(components) / len(components) if components else 0.0
        ),
    }


def semantic_thread_diagnostics(
    *,
    text: str,
    previous_texts: Iterable[str],
    thread_target: dict[str, Any] | None,
    semantic_index: Any | None,
) -> dict[str, Any]:
    previous = [
        _semantic_clean(value) for value in previous_texts if str(value).strip()
    ]
    target = dict(thread_target or {})
    target_value = _safe_float(target.get("semantic_mean_cosine"), math.nan)
    if semantic_index is None or not previous or not math.isfinite(target_value):
        return {"available": False}

    texts = [*previous, _semantic_clean(text)]
    vectors = semantic_index.encode_texts(texts)
    if len(vectors) != len(texts):
        return {"available": False}

    current_sum = 0.0
    current_pairs = 0
    for left in range(len(previous)):
        for right in range(left + 1, len(previous)):
            current_sum += float(vectors[left] @ vectors[right])
            current_pairs += 1
    candidate_scores = [
        float(vectors[-1] @ vectors[index]) for index in range(len(previous))
    ]
    candidate_pair_mean = sum(candidate_scores) / len(candidate_scores)
    proposed_pairs = current_pairs + len(candidate_scores)
    proposed_mean = (
        (current_sum + sum(candidate_scores)) / proposed_pairs
        if proposed_pairs
        else 0.0
    )
    current_mean = current_sum / current_pairs if current_pairs else target_value

    band = _metric_band(target, "semantic_mean_cosine", target_value)
    total = max(2, _safe_int(target.get("comment_count"), len(texts)))
    progress = min(1.0, len(texts) / total)
    # Prefixes with only a few comments are intrinsically noisy. The band
    # contracts to the held-out real-thread band as the thread fills.
    expansion = 1.0 + 1.5 * (1.0 - progress)
    half_width = max(0.025, (band[1] - band[0]) / 2.0) * expansion
    lower = max(-1.0, target_value - half_width)
    upper = min(1.0, target_value + half_width)
    nearest_indices = sorted(
        range(len(candidate_scores)),
        key=lambda index: candidate_scores[index],
        reverse=True,
    )[:3]
    nearest = [
        {
            "similarity": candidate_scores[index],
            "text": compact_snippet(previous[index]),
        }
        for index in nearest_indices
    ]
    return {
        "available": True,
        "current_mean": current_mean,
        "candidate_pair_mean": candidate_pair_mean,
        "proposed_mean": proposed_mean,
        "target": target_value,
        "lower": lower,
        "upper": upper,
        "target_distance": abs(proposed_mean - target_value),
        "nearest": nearest,
        "progress": progress,
    }


def semantic_distribution_problem(diagnostics: dict[str, Any]) -> str:
    if not diagnostics.get("available"):
        return ""
    proposed = float(diagnostics["proposed_mean"])
    current = float(diagnostics["current_mean"])
    candidate_mean = float(diagnostics["candidate_pair_mean"])
    target = float(diagnostics["target"])
    lower = float(diagnostics["lower"])
    upper = float(diagnostics["upper"])
    if lower <= proposed <= upper:
        return ""
    if (
        current > upper
        and candidate_mean <= target
        and abs(proposed - target) < abs(current - target)
    ):
        return ""
    if (
        current < lower
        and candidate_mean >= target
        and abs(proposed - target) < abs(current - target)
    ):
        return ""
    direction = "high" if proposed > upper else "low"
    nearest = " || ".join(
        str(item.get("text") or "")
        for item in diagnostics.get("nearest") or []
        if str(item.get("text") or "")
    )
    return (
        f"semantic_overlap_{direction}:thread_mean_cosine={proposed:.4f};"
        f"target={target:.4f};band={lower:.4f}-{upper:.4f};"
        f"candidate_pair_mean={candidate_mean:.4f};nearest={nearest}"
    )


def compact_snippet(text: str, *, limit: int = 150) -> str:
    value = SNIPPET_SPACE_RE.sub(" ", str(text or "")).strip()
    value = value.replace(";", ",").replace("|", "/")
    return value if len(value) <= limit else value[: limit - 3].rstrip() + "..."


def _semantic_clean(text: Any) -> str:
    # Mirrors score_thread_disagreement.clean_text, which is used by the
    # semantic-uniformity evaluator before embedding.
    return SNIPPET_SPACE_RE.sub(" ", str(text or "")).strip()


def _metric_band(
    target: dict[str, Any], metric: str, target_value: float
) -> tuple[float, float]:
    row = (target.get("metric_bands") or {}).get(metric) or {}
    lower = _safe_float(row.get("q10"), target_value)
    upper = _safe_float(row.get("q90"), target_value)
    return min(lower, upper), max(lower, upper)


def _normalized_target_distance(diagnostics: dict[str, Any]) -> float:
    target = float(diagnostics.get("target") or 0.0)
    lower = float(diagnostics.get("lower", target))
    upper = float(diagnostics.get("upper", target))
    scale = max(1e-6, target - lower, upper - target)
    return float(diagnostics.get("target_distance") or 0.0) / scale


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _finite_float(value: Any) -> float | None:
    number = _safe_float(value, math.nan)
    return number if math.isfinite(number) else None
