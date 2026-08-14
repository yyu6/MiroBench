from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable

from .reference_metric_calibration import THREAD_SIZE_BUCKETS, thread_size_bucket


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)?|[^\w\s]", re.UNICODE)
DEFAULT_GUARD_QUANTILE = 0.90
LENGTH_BUCKETS = ("micro", "short", "medium", "long", "very_long")


def build_lexical_calibration(
    threads: Iterable[dict[str, Any]],
    *,
    quantile: float = DEFAULT_GUARD_QUANTILE,
    max_comments_per_thread: int = 100,
) -> dict[str, Any]:
    """Calibrate Writer retry thresholds on excluded real discussions.

    The primary statistic is the mean symmetric BLEU-4 over every unordered
    comment pair in each real-thread prefix. This exactly matches the paper
    evaluator while remaining available to the online Writer guard.
    """

    values: dict[str, list[float]] = {bucket: [] for bucket in LENGTH_BUCKETS}
    prefix_values: dict[str, list[float]] = {
        bucket: [] for bucket in THREAD_SIZE_BUCKETS
    }
    thread_values: dict[str, list[float]] = {
        bucket: [] for bucket in THREAD_SIZE_BUCKETS
    }
    all_values: list[float] = []
    thread_count = 0
    for thread in threads:
        comments = [
            str(row.get("body") or row.get("content") or "").strip()
            for row in (thread.get("comments") or [])[:max_comments_per_thread]
        ]
        comments = [text for text in comments if text]
        if len(comments) < 2:
            continue
        thread_count += 1
        previous: list[PreparedComment] = []
        pair_sum = 0.0
        pair_count = 0
        for text in comments:
            current = prepare(text)
            if previous:
                pair_scores = [
                    prepared_symmetric_pair_bleu(current, prior, 4)
                    for prior in previous
                ]
                score = max(pair_scores)
                bucket = length_bucket(current.length)
                values[bucket].append(score)
                all_values.append(score)
                pair_sum += sum(pair_scores)
                pair_count += len(pair_scores)
            previous.append(current)
            if len(previous) >= 2:
                prefix_values[thread_size_bucket(len(previous))].append(
                    pair_sum / pair_count
                )
        thread_values[thread_size_bucket(len(previous))].append(
            pair_sum / pair_count
        )

    fallback = _quantile(all_values, quantile) if all_values else 1.0
    thresholds = {
        bucket: _quantile(bucket_values, quantile) if len(bucket_values) >= 20 else fallback
        for bucket, bucket_values in values.items()
    }
    prefix_fallback = [value for rows in prefix_values.values() for value in rows]
    prefix_upper = {
        bucket: _quantile(rows or prefix_fallback, quantile)
        for bucket, rows in prefix_values.items()
    }
    prefix_median = {
        bucket: _quantile(rows or prefix_fallback, 0.50)
        for bucket, rows in prefix_values.items()
    }
    prefix_lower = {
        bucket: _quantile(rows or prefix_fallback, 0.10)
        for bucket, rows in prefix_values.items()
    }
    thread_median = {
        bucket: _quantile(rows or prefix_fallback, 0.50)
        for bucket, rows in thread_values.items()
    }
    return {
        "metric": "mean symmetric pairwise BLEU-4 over all unordered thread-prefix pairs",
        "tokenizer": "score_thread_self_bleu.TOKEN_PATTERN",
        "smoothing": "add-one modified n-gram precision",
        "guard_quantile": float(quantile),
        "thresholds": {key: round(value, 8) for key, value in thresholds.items()},
        "prefix_mean_upper": {
            key: round(value, 8) for key, value in prefix_upper.items()
        },
        "prefix_mean_lower": {
            key: round(value, 8) for key, value in prefix_lower.items()
        },
        "prefix_mean_median": {
            key: round(value, 8) for key, value in prefix_median.items()
        },
        "thread_mean_median": {
            key: round(value, 8) for key, value in thread_median.items()
        },
        "prefix_sample_counts": {
            key: len(value) for key, value in prefix_values.items()
        },
        "thread_sample_counts": {
            key: len(value) for key, value in thread_values.items()
        },
        "sample_counts": {key: len(value) for key, value in values.items()},
        "overall_sample_count": len(all_values),
        "reference_thread_count": thread_count,
        "max_comments_per_thread": int(max_comments_per_thread),
    }


def lexical_overlap_problem(
    *,
    text: str,
    previous_texts: Iterable[str],
    calibration: dict[str, Any],
    thread_target: dict[str, Any] | None = None,
) -> str:
    current = prepare(text)
    previous = [(prior, prepare(prior)) for prior in previous_texts if str(prior).strip()]
    if not current.tokens or not previous:
        return ""
    prefix_upper = calibration.get("prefix_mean_upper") or {}
    if prefix_upper:
        diagnostics = candidate_thread_bleu_diagnostics(
            text=text,
            previous_texts=[prior for prior, _ in previous],
            calibration=calibration,
            thread_target=thread_target,
        )
        threshold = diagnostics["upper"]
        proposed = diagnostics["proposed_mean"]
        current_mean = diagnostics["current_mean"]
        candidate_pair_mean = diagnostics["candidate_pair_mean"]
        target = diagnostics["target"]
        improves_existing = (
            len(previous) >= 2
            and proposed < current_mean - 1e-9
            and candidate_pair_mean <= target
            and abs(proposed - target) < abs(current_mean - target)
        )
        # Add-one BLEU assigns a large score to unrelated micro comments because
        # their missing 3/4-grams receive precision 1.0. Do not force an early
        # real-shaped short slot to satisfy the final-thread band immediately.
        # It is safe only while the observed pair mass can still be diluted into
        # the held-out full-thread band by the remaining planned slots.
        short_prefix_feasible = (
            current.length < 9
            and diagnostics["completion_feasible"]
            and not diagnostics.get("shared_ngrams")
        )
        if proposed <= threshold or improves_existing or short_prefix_feasible:
            return ""
        shared = " | ".join(diagnostics.get("shared_ngrams") or [])
        nearest = " || ".join(diagnostics.get("nearest_snippets") or [])
        return (
            f"lexical_overlap_high:thread_mean_bleu4={proposed:.4f};"
            f"target={target:.4f};threshold={threshold:.4f};"
            f"candidate_pair_mean={candidate_pair_mean:.4f};"
            f"current={current_mean:.4f};thread_size={len(previous) + 1};"
            f"shared={shared or 'none'};nearest={nearest}"
        )

    scored = [
        (prepared_symmetric_pair_bleu(current, tokens, 4), prior)
        for prior, tokens in previous
        if tokens.tokens
    ]
    if not scored:
        return ""
    score, prior = max(scored, key=lambda item: item[0])
    bucket = length_bucket(current.length)
    thresholds = calibration.get("thresholds") or {}
    try:
        threshold = float(thresholds[bucket])
    except (KeyError, TypeError, ValueError):
        return ""
    if score <= threshold:
        return ""
    shared = _shared_ngrams(list(current.tokens), tokenize(prior), limit=3)
    detail = " | ".join(" ".join(item) for item in shared) or "same short sentence path"
    return (
        f"lexical_overlap_high:bleu4={score:.4f};threshold={threshold:.4f};"
        f"bucket={bucket};shared={detail}"
    )


def candidate_thread_bleu_diagnostics(
    *,
    text: str,
    previous_texts: Iterable[str],
    calibration: dict[str, Any],
    thread_target: dict[str, Any] | None = None,
) -> dict[str, Any]:
    raw_previous = [str(prior).strip() for prior in previous_texts if str(prior).strip()]
    previous = [prepare(prior) for prior in raw_previous]
    current = prepare(text)
    current_mean = prepared_thread_mean_pair_bleu(previous)
    proposed_mean = prepared_thread_mean_pair_bleu([*previous, current])
    bucket = thread_size_bucket(len(previous) + 1)
    upper = _calibration_value(calibration, "prefix_mean_upper", bucket, 1.0)
    target = _calibration_value(
        calibration,
        "prefix_mean_median",
        bucket,
        upper,
    )
    lower = _calibration_value(calibration, "prefix_mean_lower", bucket, target)
    selected_target = _safe_metric_target(thread_target, "self_bleu_4")
    planned_total = max(
        len(previous) + 1,
        int((thread_target or {}).get("comment_count") or len(previous) + 1),
    )
    processed_slots = max(
        len(previous) + 1,
        min(
            planned_total,
            int(
                (thread_target or {}).get("processed_slot_count")
                or len(previous) + 1
            ),
        ),
    )
    total = len(previous) + 1 + max(0, planned_total - processed_slots)
    final_upper = upper
    if selected_target is not None:
        progress = min(1.0, (len(previous) + 1) / total)
        blend = max(0.0, min(1.0, (progress - 0.25) / 0.75))
        target = target * (1.0 - blend) + selected_target * blend
        band = ((thread_target or {}).get("metric_bands") or {}).get("self_bleu_4") or {}
        selected_lower = _safe_float_or_none(band.get("q10"))
        selected_upper = _safe_float_or_none(band.get("q90"))
        if selected_lower is not None and selected_upper is not None:
            half_width = max(0.002, (selected_upper - selected_lower) / 2.0)
            selected_lower = max(0.0, selected_target - half_width)
            selected_upper = selected_target + half_width
            final_upper = selected_upper
            lower = lower * (1.0 - blend) + selected_lower * blend
            upper = upper * (1.0 - blend) + selected_upper * blend
    pair_scores = [
        (prepared_symmetric_pair_bleu(current, prior, 4), index)
        for index, prior in enumerate(previous)
    ]
    candidate_pair_mean = (
        sum(score for score, _ in pair_scores) / len(pair_scores)
        if pair_scores
        else 0.0
    )
    nearest = sorted(pair_scores, reverse=True)[:3]
    observed_pairs = (len(previous) + 1) * len(previous) // 2
    final_pairs = total * (total - 1) // 2
    observed_pair_mass = proposed_mean * observed_pairs
    minimum_final_mean = (
        observed_pair_mass / final_pairs if final_pairs else proposed_mean
    )
    completion_feasible = (
        len(previous) + 1 < total and minimum_final_mean <= final_upper + 1e-12
    )
    shared_ngrams: list[str] = []
    nearest_snippets: list[str] = []
    for _, index in nearest:
        shared_ngrams.extend(
            " ".join(ngram)
            for ngram in _shared_ngrams(
                list(current.tokens), list(previous[index].tokens), limit=3
            )
        )
        nearest_snippets.append(_compact_snippet(raw_previous[index]))
    return {
        "available": bool(previous),
        "current_mean": current_mean,
        "proposed_mean": proposed_mean,
        "upper": upper,
        "lower": lower,
        "target": target,
        "final_upper": final_upper,
        "planned_comment_count": total,
        "configured_comment_count": planned_total,
        "processed_slot_count": processed_slots,
        "minimum_final_mean": minimum_final_mean,
        "completion_feasible": completion_feasible,
        "target_distance": abs(proposed_mean - target),
        "candidate_pair_mean": candidate_pair_mean,
        "shared_ngrams": list(dict.fromkeys(shared_ngrams))[:6],
        "nearest_snippets": nearest_snippets,
    }


def _safe_metric_target(target: dict[str, Any] | None, field: str) -> float | None:
    if not target:
        return None
    return _safe_float_or_none(target.get(field))


def _safe_float_or_none(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _compact_snippet(text: str, limit: int = 150) -> str:
    value = re.sub(r"\s+", " ", str(text or "")).strip()
    value = value.replace(";", ",").replace("|", "/")
    return value if len(value) <= limit else value[: limit - 3].rstrip() + "..."


def thread_mean_pair_bleu(texts: Iterable[str]) -> float:
    return prepared_thread_mean_pair_bleu(
        [prepare(text) for text in texts if str(text).strip()]
    )


def prepared_thread_mean_pair_bleu(comments: list[PreparedComment]) -> float:
    if len(comments) < 2:
        return 0.0
    total = 0.0
    pairs = 0
    for left_index in range(len(comments)):
        for right_index in range(left_index + 1, len(comments)):
            total += prepared_symmetric_pair_bleu(
                comments[left_index], comments[right_index], 4
            )
            pairs += 1
    return total / pairs if pairs else 0.0


def _calibration_value(
    calibration: dict[str, Any],
    field: str,
    bucket: str,
    default: float,
) -> float:
    try:
        return float((calibration.get(field) or {})[bucket])
    except (KeyError, TypeError, ValueError):
        return float(default)


def length_bucket(token_count: int) -> str:
    if token_count <= 5:
        return "micro"
    if token_count <= 10:
        return "short"
    if token_count <= 45:
        return "medium"
    if token_count <= 100:
        return "long"
    return "very_long"


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_PATTERN.findall(str(text)) if token.strip()]


@dataclass(frozen=True)
class PreparedComment:
    tokens: tuple[str, ...]
    ngrams: tuple[Counter[tuple[str, ...]], ...]

    @property
    def length(self) -> int:
        return len(self.tokens)


def prepare(text: str) -> PreparedComment:
    tokens = tuple(tokenize(text))
    return PreparedComment(
        tokens=tokens,
        ngrams=tuple(Counter() if order == 0 else _ngram_counts(list(tokens), order) for order in range(5)),
    )


def symmetric_pair_bleu(tokens_a: list[str], tokens_b: list[str], max_order: int) -> float:
    return prepared_symmetric_pair_bleu(
        prepare(" ".join(tokens_a)),
        prepare(" ".join(tokens_b)),
        max_order,
    )


def prepared_symmetric_pair_bleu(
    left: PreparedComment,
    right: PreparedComment,
    max_order: int,
) -> float:
    return (
        _prepared_sentence_bleu(left, right, max_order)
        + _prepared_sentence_bleu(right, left, max_order)
    ) / 2.0


def _prepared_sentence_bleu(
    hypothesis: PreparedComment,
    reference: PreparedComment,
    max_order: int,
) -> float:
    if not hypothesis.tokens or not reference.tokens:
        return 0.0
    log_precision_sum = 0.0
    for order in range(1, max_order + 1):
        hypothesis_counts = hypothesis.ngrams[order]
        reference_counts = reference.ngrams[order]
        total = sum(hypothesis_counts.values())
        overlap = sum(
            min(count, reference_counts[ngram])
            for ngram, count in hypothesis_counts.items()
        )
        precision = (overlap + 1.0) / (total + 1.0)
        log_precision_sum += math.log(max(precision, 1e-12))
    brevity_penalty = (
        1.0
        if hypothesis.length > reference.length
        else math.exp(1.0 - reference.length / max(1, hypothesis.length))
    )
    return float(brevity_penalty * math.exp(log_precision_sum / max_order))


def _sentence_bleu(hypothesis: list[str], references: list[list[str]], max_order: int) -> float:
    if not hypothesis or not references:
        return 0.0
    log_precision_sum = 0.0
    for order in range(1, max_order + 1):
        overlap, total = _clipped_ngram_overlap(hypothesis, references, order)
        precision = (overlap + 1.0) / (total + 1.0)
        log_precision_sum += math.log(max(precision, 1e-12))
    closest = min(
        (len(reference) for reference in references),
        key=lambda length: (abs(length - len(hypothesis)), length),
    )
    brevity_penalty = 1.0 if len(hypothesis) > closest else math.exp(1.0 - closest / max(1, len(hypothesis)))
    return float(brevity_penalty * math.exp(log_precision_sum / max_order))


def _clipped_ngram_overlap(
    hypothesis: list[str],
    references: list[list[str]],
    order: int,
) -> tuple[int, int]:
    hypothesis_counts = _ngram_counts(hypothesis, order)
    if not hypothesis_counts:
        return 0, 0
    maximum_reference_counts: Counter[tuple[str, ...]] = Counter()
    for reference in references:
        for ngram, count in _ngram_counts(reference, order).items():
            maximum_reference_counts[ngram] = max(maximum_reference_counts[ngram], count)
    overlap = sum(
        min(count, maximum_reference_counts[ngram])
        for ngram, count in hypothesis_counts.items()
    )
    return overlap, sum(hypothesis_counts.values())


def _ngram_counts(tokens: list[str], order: int) -> Counter[tuple[str, ...]]:
    if len(tokens) < order:
        return Counter()
    return Counter(tuple(tokens[index : index + order]) for index in range(len(tokens) - order + 1))


def _shared_ngrams(
    left: list[str],
    right: list[str],
    *,
    limit: int,
) -> list[tuple[str, ...]]:
    for order in (4, 3, 2):
        shared = list((_ngram_counts(left, order) & _ngram_counts(right, order)).keys())
        if shared:
            return shared[:limit]
    return []


def _quantile(values: list[float], probability: float) -> float:
    if not values:
        return 1.0
    ordered = sorted(values)
    probability = max(0.0, min(1.0, float(probability)))
    position = (len(ordered) - 1) * probability
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)
