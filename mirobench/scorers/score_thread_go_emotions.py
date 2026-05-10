#!/usr/bin/env python3
"""Score Reddit comments with GoEmotions and aggregate by thread.

This evaluator uses `SamLowe/roberta-base-go_emotions`, a RoBERTa
multi-label classifier for the 27 GoEmotions labels plus `neutral`.

For each comment, the script outputs all 28 sigmoid probabilities, labels above
the threshold, and the dominant emotion. For each thread, it reports:

- emotion entropy over dominant emotions
- average labels per comment
- emotion shift rate over adjacent comments
- dominant emotion share

Higher entropy and higher shift rate indicate more emotional variety. Higher
dominant emotion share indicates one emotion dominates the thread.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable


from .score_thread_disagreement import detect_target_kind
from .score_thread_semantic_uniformity import (
    ThreadComment,
    load_generated_comments,
    load_real_comments,
    median,
    weighted_average,
)


DEFAULT_MODEL = "SamLowe/roberta-base-go_emotions"


def main() -> None:
    """CLI entrypoint."""

    parser = argparse.ArgumentParser(
        description="Score comments with SamLowe/roberta-base-go_emotions and aggregate by thread."
    )
    parser.add_argument("input", help="Real discussion folder/file or generated run folder/file.")
    parser.add_argument(
        "--target-kind",
        choices=["auto", "real", "generated"],
        default="auto",
        help="Input schema. auto detects generated discussion.json vs real Reddit bundle.",
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL,
        help="Hugging Face model name or local path.",
    )
    parser.add_argument(
        "--output-file",
        default="",
        help="Where to write JSON results. Defaults next to input.",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max-threads", type=int, default=0)
    parser.add_argument("--max-comments-per-thread", type=int, default=0)
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Torch device. auto uses cuda, then mps, then cpu.",
    )
    parser.add_argument(
        "--no-comment-text",
        action="store_true",
        help="Omit raw comment text from per-comment output rows.",
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

    comments = [
        comment
        for thread_comments in comments_by_thread.values()
        for comment in thread_comments
    ]
    scorer = GoEmotionsScorer(
        model_name=args.model_name,
        device=args.device,
        max_length=max(8, args.max_length),
    )
    scored_comments = scorer.score_comments(
        comments=comments,
        batch_size=max(1, args.batch_size),
        threshold=args.threshold,
        include_text=not args.no_comment_text,
    )
    scores_by_comment_id = {
        row["comment_id"]: row
        for row in scored_comments
    }

    thread_results = []
    for thread_id, thread_comments in comments_by_thread.items():
        thread_rows = [
            scores_by_comment_id[comment.comment_id]
            for comment in thread_comments
            if comment.comment_id in scores_by_comment_id
        ]
        thread_results.append(
            aggregate_thread(
                thread_id=thread_id,
                comments=thread_comments,
                scored_comments=thread_rows,
                labels=scorer.labels,
            )
        )

    thread_results.sort(
        key=lambda row: (
            row["emotion_entropy"],
            row["avg_labels_per_comment"],
            row["comment_count"],
        ),
        reverse=True,
    )
    result = {
        "meta": {
            "input": str(input_path),
            "target_kind": target_kind,
            "metric": "GoEmotions multi-label emotion complexity",
            "model_name": args.model_name,
            "model_source": "https://huggingface.co/SamLowe/roberta-base-go_emotions",
            "device": scorer.device,
            "max_length": args.max_length,
            "threshold": args.threshold,
            "labels": scorer.labels,
            "comment_count": len(scored_comments),
            "thread_count": len(thread_results),
            "include_comment_text": not args.no_comment_text,
            "source": source_meta,
        },
        "overall": aggregate_overall(thread_results, scorer.labels),
        "threads": thread_results,
    }

    output_file = Path(args.output_file).expanduser() if args.output_file else default_output_path(input_path)
    if not output_file.is_absolute():
        output_file = (Path.cwd() / output_file).resolve()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    overall = result["overall"]
    print(f"Wrote {output_file}")
    print(
        "overall: "
        f"comments={overall['comment_count']} threads={overall['thread_count']} "
        f"entropy={overall['weighted_emotion_entropy']:.4f} "
        f"avg_labels={overall['weighted_avg_labels_per_comment']:.4f} "
        f"shift_rate={overall['weighted_emotion_shift_rate']:.4f} "
        f"dominant_share={overall['weighted_dominant_emotion_share']:.4f}"
    )


class GoEmotionsScorer:
    """Batch classifier wrapper for SamLowe/roberta-base-go_emotions."""

    def __init__(self, model_name: str, device: str, max_length: int) -> None:
        try:
            import torch
            from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - environment guard
            raise SystemExit(
                "GoEmotions scoring requires torch and transformers. Use system python3 "
                "in this workspace, or install those packages in your venv."
            ) from exc

        self.torch = torch
        self.model_name = model_name
        self.max_length = max_length
        self.device = self._resolve_device(device)
        self.config = AutoConfig.from_pretrained(model_name)
        self.labels = [
            str(self.config.id2label[idx])
            for idx in range(int(self.config.num_labels))
        ]
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def score_comments(
        self,
        comments: list[ThreadComment],
        batch_size: int,
        threshold: float,
        include_text: bool,
    ) -> list[dict[str, Any]]:
        """Return per-comment emotion probabilities and threshold labels."""

        rows: list[dict[str, Any]] = []
        for start in range(0, len(comments), batch_size):
            batch = comments[start : start + batch_size]
            encoded = self.tokenizer(
                [comment.text for comment in batch],
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            with self.torch.inference_mode():
                logits = self.model(**encoded).logits
                probs = self.torch.sigmoid(logits).detach().cpu().tolist()

            for comment, prob in zip(batch, probs):
                probabilities = {
                    label: float(prob[idx])
                    for idx, label in enumerate(self.labels)
                }
                predicted_labels = [
                    label
                    for label in self.labels
                    if probabilities[label] >= threshold
                ]
                dominant_emotion = max(probabilities, key=probabilities.get)
                row = {
                    "thread_id": comment.thread_id,
                    "thread_title": comment.thread_title,
                    "comment_id": comment.comment_id,
                    "parent_id": comment.parent_id,
                    "author": comment.author,
                    "depth": comment.depth,
                    "emotion_probabilities": probabilities,
                    "predicted_labels": predicted_labels,
                    "label_count": len(predicted_labels),
                    "dominant_emotion": dominant_emotion,
                    "dominant_emotion_probability": probabilities[dominant_emotion],
                }
                if include_text:
                    row["text"] = comment.text
                rows.append(row)
        return rows

    def _resolve_device(self, requested: str) -> str:
        """Resolve a torch device string."""

        if requested != "auto":
            return requested
        if self.torch.cuda.is_available():
            return "cuda"
        if getattr(self.torch.backends, "mps", None) and self.torch.backends.mps.is_available():
            return "mps"
        return "cpu"


def aggregate_thread(
    thread_id: str,
    comments: list[ThreadComment],
    scored_comments: list[dict[str, Any]],
    labels: list[str],
) -> dict[str, Any]:
    """Aggregate per-comment emotion scores for one thread."""

    comment_count = len(scored_comments)
    dominant_counts = Counter(str(row["dominant_emotion"]) for row in scored_comments)
    label_counts = Counter()
    for row in scored_comments:
        label_counts.update(str(label) for label in row["predicted_labels"])
    probability_sums = {
        label: sum(float(row["emotion_probabilities"][label]) for row in scored_comments)
        for label in labels
    }
    mean_probabilities = {
        label: probability_sums[label] / comment_count if comment_count else 0.0
        for label in labels
    }
    dominant_emotion, dominant_count = dominant_counts.most_common(1)[0] if dominant_counts else ("", 0)
    entropy = shannon_entropy(dominant_counts.values())
    label_counts_per_comment = [int(row["label_count"]) for row in scored_comments]
    dominant_sequence = [str(row["dominant_emotion"]) for row in scored_comments]
    return {
        "thread_id": thread_id,
        "thread_title": comments[0].thread_title if comments else "",
        "comment_count": comment_count,
        "emotion_entropy": entropy,
        "emotion_entropy_normalized": entropy / math.log(len(labels)) if labels and entropy else 0.0,
        "avg_labels_per_comment": mean(label_counts_per_comment),
        "median_labels_per_comment": median([float(value) for value in label_counts_per_comment]),
        "emotion_shift_rate": emotion_shift_rate(dominant_sequence),
        "dominant_emotion": dominant_emotion,
        "dominant_emotion_count": dominant_count,
        "dominant_emotion_share": dominant_count / comment_count if comment_count else 0.0,
        "dominant_emotion_counts": {label: int(dominant_counts.get(label, 0)) for label in labels},
        "threshold_label_counts": {label: int(label_counts.get(label, 0)) for label in labels},
        "mean_emotion_probabilities": mean_probabilities,
        "comments": scored_comments,
    }


def aggregate_overall(thread_results: list[dict[str, Any]], labels: list[str]) -> dict[str, Any]:
    """Aggregate thread-level emotion metrics."""

    comment_count = sum(int(row["comment_count"]) for row in thread_results)
    valid_rows = [row for row in thread_results if int(row["comment_count"]) > 0]
    dominant_counts = Counter()
    threshold_label_counts = Counter()
    probability_sums = {label: 0.0 for label in labels}
    for row in valid_rows:
        dominant_counts.update(row["dominant_emotion_counts"])
        threshold_label_counts.update(row["threshold_label_counts"])
        for label in labels:
            probability_sums[label] += (
                float(row["mean_emotion_probabilities"][label]) * int(row["comment_count"])
            )

    dominant_emotion, dominant_count = dominant_counts.most_common(1)[0] if dominant_counts else ("", 0)
    mean_probabilities = {
        label: probability_sums[label] / comment_count if comment_count else 0.0
        for label in labels
    }
    return {
        "comment_count": comment_count,
        "thread_count": len(thread_results),
        "weighted_emotion_entropy": weighted_average(
            [(float(row["emotion_entropy"]), int(row["comment_count"])) for row in valid_rows]
        ),
        "weighted_emotion_entropy_normalized": weighted_average(
            [(float(row["emotion_entropy_normalized"]), int(row["comment_count"])) for row in valid_rows]
        ),
        "weighted_avg_labels_per_comment": weighted_average(
            [(float(row["avg_labels_per_comment"]), int(row["comment_count"])) for row in valid_rows]
        ),
        "weighted_emotion_shift_rate": weighted_average(
            [(float(row["emotion_shift_rate"]), max(0, int(row["comment_count"]) - 1)) for row in valid_rows]
        ),
        "weighted_dominant_emotion_share": weighted_average(
            [(float(row["dominant_emotion_share"]), int(row["comment_count"])) for row in valid_rows]
        ),
        "mean_thread_emotion_entropy": mean([float(row["emotion_entropy"]) for row in valid_rows]),
        "median_thread_emotion_entropy": median([float(row["emotion_entropy"]) for row in valid_rows]),
        "mean_thread_avg_labels_per_comment": mean(
            [float(row["avg_labels_per_comment"]) for row in valid_rows]
        ),
        "median_thread_avg_labels_per_comment": median(
            [float(row["avg_labels_per_comment"]) for row in valid_rows]
        ),
        "dominant_emotion": dominant_emotion,
        "dominant_emotion_count": int(dominant_count),
        "dominant_emotion_share": dominant_count / comment_count if comment_count else 0.0,
        "dominant_emotion_counts": {label: int(dominant_counts.get(label, 0)) for label in labels},
        "threshold_label_counts": {label: int(threshold_label_counts.get(label, 0)) for label in labels},
        "mean_emotion_probabilities": mean_probabilities,
    }


def emotion_shift_rate(dominant_sequence: list[str]) -> float:
    """Fraction of adjacent comments whose dominant emotion changes."""

    if len(dominant_sequence) < 2:
        return 0.0
    shifts = sum(
        1
        for left, right in zip(dominant_sequence, dominant_sequence[1:])
        if left != right
    )
    return shifts / (len(dominant_sequence) - 1)


def shannon_entropy(counts: Iterable[int]) -> float:
    """Shannon entropy over a count distribution."""

    counts = [count for count in counts if count > 0]
    total = sum(counts)
    if total <= 0:
        return 0.0
    return float(
        -sum((count / total) * math.log(count / total) for count in counts)
    )


def mean(values: Iterable[float]) -> float:
    """Mean with empty-list protection."""

    values = list(values)
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def default_output_path(input_path: Path) -> Path:
    """Choose a default output path next to the evaluated input."""

    if input_path.is_dir():
        return input_path / "go_emotions_results.json"
    return input_path.with_suffix(input_path.suffix + ".go_emotions_results.json")


if __name__ == "__main__":
    main()
