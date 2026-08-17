#!/usr/bin/env python3
"""Score thread-level surface-form uniformity with Self-BLEU.

This evaluator measures whether comments inside one Reddit thread reuse the
same wording or template. For each thread, every unordered comment pair is
scored with symmetric pairwise BLEU, then the thread-level score is the mean
over all pairs. The script reports Self-BLEU-2, Self-BLEU-3, and Self-BLEU-4.

Higher Self-BLEU means the thread is more uniform at the surface wording level.
Lower Self-BLEU means the comments are more lexically diverse.

`diversity-eval-master` provides the same response-set framing and n-gram
diversity utilities, but it does not include a direct Self-BLEU metric or a
GEO Reddit-thread loader. This script keeps the metric local to GEO while
following that response-set evaluation style.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from score_thread_disagreement import detect_target_kind  # noqa: E402
from score_thread_semantic_uniformity import (  # noqa: E402
    ThreadComment,
    load_generated_comments,
    load_real_comments,
    median,
    weighted_average,
)


BLEU_ORDERS = (2, 3, 4)
TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)?|[^\w\s]", re.UNICODE)


def main() -> None:
    """CLI entrypoint."""

    parser = argparse.ArgumentParser(
        description="Score within-thread Self-BLEU-2/3/4 for Reddit comments."
    )
    parser.add_argument(
        "input", help="Real discussion folder/file or generated run folder/file."
    )
    parser.add_argument(
        "--target-kind",
        choices=["auto", "real", "generated"],
        default="auto",
        help="Input schema. auto detects generated discussion.json vs real Reddit bundle.",
    )
    parser.add_argument(
        "--output-file",
        default="",
        help="Where to write JSON results. Defaults next to input.",
    )
    parser.add_argument("--max-threads", type=int, default=0)
    parser.add_argument("--max-comments-per-thread", type=int, default=0)
    parser.add_argument(
        "--include-comment-scores",
        action="store_true",
        help="Include per-comment Self-BLEU scores in the output JSON.",
    )
    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    target_kind = args.target_kind
    if target_kind == "auto":
        target_kind = detect_target_kind(input_path)

    if target_kind == "generated":
        comments_by_thread, source_meta = load_generated_comments(input_path)
    else:
        comments_by_thread, source_meta = load_real_comments(input_path)

    if args.max_threads and args.max_threads > 0:
        comments_by_thread = dict(list(comments_by_thread.items())[: args.max_threads])
    if args.max_comments_per_thread and args.max_comments_per_thread > 0:
        comments_by_thread = {
            thread_id: comments[: args.max_comments_per_thread]
            for thread_id, comments in comments_by_thread.items()
        }

    thread_results = [
        score_thread(
            thread_id, comments, include_comment_scores=args.include_comment_scores
        )
        for thread_id, comments in comments_by_thread.items()
    ]
    thread_results.sort(
        key=lambda row: (
            row["self_bleu_4"],
            row["self_bleu_3"],
            row["comment_count"],
        ),
        reverse=True,
    )

    result = {
        "meta": {
            "input": str(input_path),
            "target_kind": target_kind,
            "metric": "Self-BLEU",
            "orders": list(BLEU_ORDERS),
            "aggregation": "unordered comment pairs within each thread; pair score is mean(BLEU(a,b), BLEU(b,a))",
            "smoothing": "add-one modified n-gram precision",
            "interpretation": "higher Self-BLEU means more surface-form uniformity",
            "thread_count": len(thread_results),
            "comment_count": sum(int(row["comment_count"]) for row in thread_results),
            "include_comment_scores": args.include_comment_scores,
            "source": source_meta,
        },
        "overall": aggregate_overall(thread_results),
        "threads": thread_results,
    }

    output_file = (
        Path(args.output_file).expanduser()
        if args.output_file
        else default_output_path(input_path)
    )
    if not output_file.is_absolute():
        output_file = (Path.cwd() / output_file).resolve()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    overall = result["overall"]
    print(f"Wrote {output_file}")
    print(
        "overall: "
        f"comments={overall['comment_count']} threads={overall['thread_count']} "
        f"self_bleu_2={overall['weighted_self_bleu_2']:.4f} "
        f"self_bleu_3={overall['weighted_self_bleu_3']:.4f} "
        f"self_bleu_4={overall['weighted_self_bleu_4']:.4f}"
    )


def score_thread(
    thread_id: str,
    comments: list[ThreadComment],
    include_comment_scores: bool,
) -> dict[str, Any]:
    """Compute Self-BLEU-2/3/4 for one thread."""

    tokenized = [tokenize(comment.text) for comment in comments]
    per_order_scores = {
        order: pairwise_self_bleu_for_order(tokenized, order) for order in BLEU_ORDERS
    }
    comment_rows = []
    if include_comment_scores:
        per_comment = per_comment_pairwise_self_bleu(tokenized)
        for idx, comment in enumerate(comments):
            comment_rows.append(
                {
                    "thread_id": comment.thread_id,
                    "comment_id": comment.comment_id,
                    "parent_id": comment.parent_id,
                    "author": comment.author,
                    "depth": comment.depth,
                    "text": comment.text,
                    "self_bleu_2": per_comment[idx][2],
                    "self_bleu_3": per_comment[idx][3],
                    "self_bleu_4": per_comment[idx][4],
                }
            )

    row: dict[str, Any] = {
        "thread_id": thread_id,
        "thread_title": comments[0].thread_title if comments else "",
        "comment_count": len(comments),
        "pair_count": math.comb(len(comments), 2) if len(comments) >= 2 else 0,
        "self_bleu_2": per_order_scores[2],
        "self_bleu_3": per_order_scores[3],
        "self_bleu_4": per_order_scores[4],
        "mean_comment_tokens": mean([len(tokens) for tokens in tokenized]),
        "median_comment_tokens": median([float(len(tokens)) for tokens in tokenized]),
        "note": ""
        if len(comments) >= 2
        else "Self-BLEU requires at least two usable comments.",
    }
    if include_comment_scores:
        row["comments"] = comment_rows
    return row


def per_comment_self_bleu(
    tokenized_comments: list[list[str]],
) -> list[dict[int, float]]:
    """Return each comment's multi-reference Self-BLEU scores.

    This legacy helper is kept for audit/debug use. The main thread score uses
    `per_comment_pairwise_self_bleu` and `pairwise_self_bleu_for_order`.
    """

    scores = []
    for idx, hypothesis in enumerate(tokenized_comments):
        references = [
            reference
            for ref_idx, reference in enumerate(tokenized_comments)
            if ref_idx != idx
        ]
        scores.append(
            {
                order: sentence_bleu(hypothesis, references, order)
                for order in BLEU_ORDERS
            }
        )
    return scores


def per_comment_pairwise_self_bleu(
    tokenized_comments: list[list[str]],
) -> list[dict[int, float]]:
    """Return each comment's mean pairwise BLEU against other comments."""

    scores = []
    for idx, hypothesis in enumerate(tokenized_comments):
        per_order = {}
        for order in BLEU_ORDERS:
            pair_scores = []
            for other_idx, other in enumerate(tokenized_comments):
                if other_idx == idx:
                    continue
                pair_scores.append(symmetric_pair_bleu(hypothesis, other, order))
            per_order[order] = mean(pair_scores)
        scores.append(per_order)
    return scores


def pairwise_self_bleu_for_order(
    tokenized_comments: list[list[str]], order: int
) -> float:
    """Average symmetric BLEU-n over all unordered comment pairs in the thread."""

    if len(tokenized_comments) < 2:
        return 0.0
    scores = []
    for idx in range(len(tokenized_comments)):
        for other_idx in range(idx + 1, len(tokenized_comments)):
            scores.append(
                symmetric_pair_bleu(
                    tokenized_comments[idx],
                    tokenized_comments[other_idx],
                    order,
                )
            )
    return mean(scores)


def symmetric_pair_bleu(
    tokens_a: list[str], tokens_b: list[str], max_order: int
) -> float:
    """Compute symmetric pairwise BLEU for two comments.

    BLEU is directional because it applies hypothesis-side precision and a
    brevity penalty. For a pairwise uniformity metric, we average both
    directions so long-vs-short comment order does not bias the thread score.
    """

    return mean(
        [
            sentence_bleu(tokens_a, [tokens_b], max_order),
            sentence_bleu(tokens_b, [tokens_a], max_order),
        ]
    )


def sentence_bleu(
    hypothesis: list[str], references: list[list[str]], max_order: int
) -> float:
    """Compute smoothed BLEU-n for one hypothesis against multiple references."""

    if not hypothesis or not references:
        return 0.0

    log_precision_sum = 0.0
    for order in range(1, max_order + 1):
        overlap, total = clipped_ngram_overlap(hypothesis, references, order)
        precision = (overlap + 1.0) / (total + 1.0)
        log_precision_sum += math.log(max(precision, 1e-12))

    closest_ref_len = closest_reference_length(
        len(hypothesis), [len(ref) for ref in references]
    )
    if len(hypothesis) > closest_ref_len:
        brevity_penalty = 1.0
    else:
        brevity_penalty = math.exp(1.0 - closest_ref_len / max(1, len(hypothesis)))

    return float(brevity_penalty * math.exp(log_precision_sum / max_order))


def clipped_ngram_overlap(
    hypothesis: list[str],
    references: list[list[str]],
    order: int,
) -> tuple[int, int]:
    """Return clipped BLEU n-gram overlap and hypothesis n-gram count."""

    hyp_ngrams = ngram_counts(hypothesis, order)
    if not hyp_ngrams:
        return 0, 0

    max_ref_counts: Counter[tuple[str, ...]] = Counter()
    for reference in references:
        ref_counts = ngram_counts(reference, order)
        for ngram, count in ref_counts.items():
            if count > max_ref_counts[ngram]:
                max_ref_counts[ngram] = count

    overlap = sum(
        min(count, max_ref_counts[ngram]) for ngram, count in hyp_ngrams.items()
    )
    return overlap, sum(hyp_ngrams.values())


def ngram_counts(tokens: list[str], order: int) -> Counter[tuple[str, ...]]:
    """Count n-grams for one token list."""

    if len(tokens) < order:
        return Counter()
    return Counter(
        tuple(tokens[idx : idx + order]) for idx in range(len(tokens) - order + 1)
    )


def closest_reference_length(hypothesis_len: int, reference_lengths: list[int]) -> int:
    """Return the BLEU closest reference length."""

    return min(
        reference_lengths, key=lambda ref_len: (abs(ref_len - hypothesis_len), ref_len)
    )


def tokenize(text: str) -> list[str]:
    """Tokenize Reddit text for surface-form n-gram scoring."""

    return [token.lower() for token in TOKEN_PATTERN.findall(text) if token.strip()]


def aggregate_overall(thread_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-thread Self-BLEU scores."""

    total_comments = sum(int(row["comment_count"]) for row in thread_results)
    total_pairs = sum(int(row["pair_count"]) for row in thread_results)
    if not thread_results:
        return {
            "comment_count": 0,
            "thread_count": 0,
            "pair_count": 0,
            "weighted_self_bleu_2": 0.0,
            "weighted_self_bleu_3": 0.0,
            "weighted_self_bleu_4": 0.0,
            "median_thread_self_bleu_2": 0.0,
            "median_thread_self_bleu_3": 0.0,
            "median_thread_self_bleu_4": 0.0,
        }

    valid_rows = [row for row in thread_results if int(row["comment_count"]) >= 2]
    return {
        "comment_count": total_comments,
        "thread_count": len(thread_results),
        "pair_count": total_pairs,
        "weighted_self_bleu_2": weighted_average(
            [
                (float(row["self_bleu_2"]), int(row["comment_count"]))
                for row in valid_rows
            ]
        ),
        "weighted_self_bleu_3": weighted_average(
            [
                (float(row["self_bleu_3"]), int(row["comment_count"]))
                for row in valid_rows
            ]
        ),
        "weighted_self_bleu_4": weighted_average(
            [
                (float(row["self_bleu_4"]), int(row["comment_count"]))
                for row in valid_rows
            ]
        ),
        "mean_thread_self_bleu_2": mean(
            [float(row["self_bleu_2"]) for row in valid_rows]
        ),
        "mean_thread_self_bleu_3": mean(
            [float(row["self_bleu_3"]) for row in valid_rows]
        ),
        "mean_thread_self_bleu_4": mean(
            [float(row["self_bleu_4"]) for row in valid_rows]
        ),
        "median_thread_self_bleu_2": median(
            [float(row["self_bleu_2"]) for row in valid_rows]
        ),
        "median_thread_self_bleu_3": median(
            [float(row["self_bleu_3"]) for row in valid_rows]
        ),
        "median_thread_self_bleu_4": median(
            [float(row["self_bleu_4"]) for row in valid_rows]
        ),
        "max_thread_self_bleu_4": max(
            (float(row["self_bleu_4"]) for row in valid_rows), default=0.0
        ),
    }


def mean(values: Iterable[float]) -> float:
    """Mean with empty-list protection."""

    values = list(values)
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def default_output_path(input_path: Path) -> Path:
    """Choose a default output path next to the evaluated input."""

    if input_path.is_dir():
        return input_path / "self_bleu_results.json"
    return input_path.with_suffix(input_path.suffix + ".self_bleu_results.json")


if __name__ == "__main__":
    main()
