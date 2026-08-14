from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


THREAD_SIZE_BUCKETS = ("tiny", "small", "medium", "large", "very_large")
EVALUATION_METRICS = (
    "self_bleu_4",
    "self_bertscore_mean_f1",
    "semantic_mean_cosine",
    "hard_disagree_rate",
    "polite_rate",
    "impolite_rate",
    "neutral_rate",
    "length_cv",
    "avg_depth",
    "structural_virality",
    "mean_story_probability",
    "emotion_entropy",
)


def build_reference_metric_calibration(
    raw_discussions_dir: Path,
    *,
    reference_thread_ids: Iterable[str],
    excluded_seed_ids: Iterable[str],
) -> dict[str, Any]:
    """Load aggregate metric rows for evaluation-excluded real threads.

    The resulting profile contains counts and metric values only. It never
    stores comment text, authors, titles, or raw thread identifiers.
    """

    reference_ids = _normalized_ids(reference_thread_ids)
    seed_ids = _normalized_ids(excluded_seed_ids)
    overlap = reference_ids & seed_ids
    if overlap:
        raise ValueError(
            "Reference metric calibration overlaps evaluation seeds: "
            f"{len(overlap)} thread(s)"
        )

    metric_rows = _load_thread_rows(
        raw_discussions_dir,
        filename="thread_metrics_summary.json",
        reference_ids=reference_ids,
    )
    emotion_rows = _load_thread_rows(
        raw_discussions_dir,
        filename="go_emotions_results.json",
        reference_ids=reference_ids,
        row_key="threads",
    )
    story_rows = _load_thread_rows(
        raw_discussions_dir,
        filename="storyseeker_results.json",
        reference_ids=reference_ids,
        row_key="threads",
    )

    templates: dict[str, list[dict[str, Any]]] = {
        bucket: [] for bucket in THREAD_SIZE_BUCKETS
    }
    for thread_id in sorted(reference_ids):
        metric = metric_rows.get(thread_id)
        if not metric:
            continue
        comment_count = _safe_int(metric.get("comment_count"))
        if comment_count <= 0:
            continue
        emotion = emotion_rows.get(thread_id, {})
        story = story_rows.get(thread_id, {})
        emotion_counts = {
            str(label): _safe_int(count)
            for label, count in (emotion.get("dominant_emotion_counts") or {}).items()
            if _safe_int(count) > 0
        }
        if not emotion_counts:
            dominant = str(metric.get("dominant_emotion") or "neutral")
            emotion_counts = {dominant: comment_count}
        templates[thread_size_bucket(comment_count)].append(
            {
                "comment_count": comment_count,
                "story_count": _safe_int(metric.get("story_count")),
                "story_rate": _safe_float(metric.get("story_rate")),
                **{name: _safe_float(metric.get(name)) for name in EVALUATION_METRICS},
                "story_probability_tier_counts": _story_probability_tier_counts(
                    story
                ),
                "emotion_entropy": _safe_float(metric.get("emotion_entropy")),
                "dominant_emotion_counts": dict(sorted(emotion_counts.items())),
                "self_bleu_4": _safe_float(metric.get("self_bleu_4")),
                "semantic_mean_cosine": _safe_float(
                    metric.get("semantic_mean_cosine")
                ),
                "polite_rate": _safe_float(metric.get("polite_rate")),
                "impolite_rate": _safe_float(metric.get("impolite_rate")),
                "neutral_rate": _safe_float(metric.get("neutral_rate")),
                # The evaluated tone metrics are polite/impolite/neutral, but the
                # classifier emits a fourth class.  Carrying its measured rate
                # keeps the planned contract a complete partition instead of
                # renormalizing three rates and inflating each of them.
                "somewhat_polite_rate": _somewhat_polite_rate(metric),
            }
        )

    flattened = [row for bucket in THREAD_SIZE_BUCKETS for row in templates[bucket]]
    if not flattened:
        return {
            "available": False,
            "method": "evaluation-excluded aggregate metric templates",
            "reference_thread_count": 0,
            "reference_id_hash": _id_hash(reference_ids),
            "excluded_seed_id_hash": _id_hash(seed_ids),
            "seed_reference_overlap_count": 0,
            "templates_by_size": templates,
        }

    emotion_totals: Counter[str] = Counter()
    for row in flattened:
        emotion_totals.update(row["dominant_emotion_counts"])
    metric_bands_by_size = {
        bucket: {
            metric: _distribution_summary(
                row[metric]
                for row in templates[bucket]
                if int(row.get("comment_count") or 0) >= 2
            )
            for metric in EVALUATION_METRICS
        }
        for bucket in THREAD_SIZE_BUCKETS
    }
    return {
        "available": True,
        "method": "evaluation-excluded aggregate metric templates",
        "reference_thread_count": len(flattened),
        "reference_id_hash": _id_hash(reference_ids),
        "excluded_seed_id_hash": _id_hash(seed_ids),
        "seed_reference_overlap_count": 0,
        "raw_text_included": False,
        "templates_by_size": templates,
        "metric_bands_by_size": metric_bands_by_size,
        "summary": {
            "mean_story_rate": _mean(row["story_rate"] for row in flattened),
            **{f"mean_{name}": _mean(row[name] for row in flattened) for name in EVALUATION_METRICS},
            "dominant_emotion_counts": dict(sorted(emotion_totals.items())),
        },
    }


def select_reference_template(
    calibration: dict[str, Any],
    *,
    comment_count: int,
    selector: int,
) -> dict[str, Any] | None:
    """Select one same-size template drawn from the evaluation-excluded pool.

    A single drawn row is a high-variance estimate of any one thread's targets:
    in the camera corpus the ``very_large`` bucket spans 0.122 to 0.619 in
    `polite_rate` around a 0.255 median, and one run drew the maximum against a
    matched real thread at 0.249. Shrinking the draw toward the bucket median was
    measured and rejected. Leave-one-out on excluded threads confirms shrinkage
    lowers per-thread target error, to 0.71 of the raw draw at full shrinkage, but
    the evaluation compares *distributions*: a set of shrunk targets matched a set
    of real threads on only 0.48 of trials at N=10 against 0.89 for raw draws,
    because shrinkage removes the between-thread variance the tests measure.

    The draw is therefore deliberately unshrunk. When a generated metric misses
    its matched real thread, look for insensitivity to the target in realization
    before suspecting the target.
    """

    if not calibration.get("available"):
        return None
    by_size = calibration.get("templates_by_size") or {}
    preferred = list(by_size.get(thread_size_bucket(comment_count)) or [])
    candidates = preferred or [
        row
        for bucket in THREAD_SIZE_BUCKETS
        for row in (by_size.get(bucket) or [])
        if isinstance(row, dict)
    ]
    if not candidates:
        return None
    nearest_distance = min(
        abs(_safe_int(row.get("comment_count")) - comment_count) for row in candidates
    )
    nearest = [
        row
        for row in candidates
        if abs(_safe_int(row.get("comment_count")) - comment_count) == nearest_distance
    ]
    return dict(nearest[int(selector) % len(nearest)])


def thread_size_bucket(comment_count: int) -> str:
    if comment_count <= 5:
        return "tiny"
    if comment_count <= 15:
        return "small"
    if comment_count <= 40:
        return "medium"
    if comment_count <= 100:
        return "large"
    return "very_large"


def _load_thread_rows(
    root: Path,
    *,
    filename: str,
    reference_ids: set[str],
    row_key: str | None = None,
) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for path in sorted(root.rglob(filename)):
        payload = _load_json(path)
        candidates = payload.get(row_key) if row_key and isinstance(payload, dict) else payload
        if not isinstance(candidates, list):
            continue
        for row in candidates:
            if not isinstance(row, dict):
                continue
            thread_id = str(row.get("thread_id") or "").strip()
            if thread_id not in reference_ids:
                continue
            old = rows.get(thread_id)
            if old is None or _safe_int(row.get("comment_count")) > _safe_int(
                old.get("comment_count")
            ):
                rows[thread_id] = row
    return rows


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _story_probability_tier_counts(row: dict[str, Any]) -> dict[str, int]:
    """Reduce comment-level StorySeeker scores to text-free count bands."""

    counts: Counter[str] = Counter()
    for comment in row.get("comments") or []:
        if not isinstance(comment, dict):
            continue
        probability = _safe_float(comment.get("story_probability"))
        if probability < 0.05:
            tier = "very_low"
        elif probability < 0.20:
            tier = "low"
        elif probability < 0.50:
            tier = "ambiguous"
        elif probability < 0.75:
            tier = "story_mid"
        else:
            tier = "story_high"
        counts[tier] += 1
    return dict(sorted(counts.items()))


def _somewhat_polite_rate(metric: dict[str, Any]) -> float:
    """Return the measured fourth-class rate, or the residual when absent."""

    if "somewhat_polite_rate" in metric:
        return _safe_float(metric.get("somewhat_polite_rate"))
    residual = 1.0 - sum(
        _safe_float(metric.get(name))
        for name in ("polite_rate", "impolite_rate", "neutral_rate")
    )
    return round(max(0.0, residual), 8)


def _normalized_ids(values: Iterable[str]) -> set[str]:
    return {str(value).strip() for value in values if str(value).strip()}


def _id_hash(values: Iterable[str]) -> str:
    payload = "\n".join(sorted(_normalized_ids(values)))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _mean(values: Iterable[float]) -> float:
    rows = [float(value) for value in values]
    return round(sum(rows) / len(rows), 8) if rows else 0.0


def _distribution_summary(values: Iterable[float]) -> dict[str, Any]:
    rows = sorted(float(value) for value in values)
    return {
        "q10": round(_quantile(rows, 0.10), 8),
        "median": round(_quantile(rows, 0.50), 8),
        "q90": round(_quantile(rows, 0.90), 8),
        "sample_count": len(rows),
    }


def _quantile(values: list[float], probability: float) -> float:
    if not values:
        return 0.0
    position = max(0.0, min(1.0, probability)) * (len(values) - 1)
    lower = int(position)
    upper = min(len(values) - 1, lower + 1)
    weight = position - lower
    return values[lower] * (1.0 - weight) + values[upper] * weight
