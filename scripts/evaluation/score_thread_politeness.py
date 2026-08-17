#!/usr/bin/env python3
"""Score Reddit comments with Intel/polite-guard and aggregate by thread.

This evaluator treats politeness as a four-way single-label classification task.
For each individual comment/reply, it outputs class probabilities and the
predicted label. For each thread, it reports the share of comments predicted as:

- `polite`
- `somewhat_polite`
- `neutral`
- `impolite`
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from score_thread_disagreement import detect_target_kind  # noqa: E402
from score_thread_semantic_uniformity import (  # noqa: E402
    ThreadComment,
    load_generated_comments,
    load_real_comments,
    weighted_average,
)


DEFAULT_MODEL = "Intel/polite-guard"
CANONICAL_LABELS = (
    "polite",
    "somewhat_polite",
    "neutral",
    "impolite",
)


def main() -> None:
    """CLI entrypoint."""

    parser = argparse.ArgumentParser(
        description="Score comments with Intel/polite-guard and aggregate politeness by thread."
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
        "--model-name",
        default=DEFAULT_MODEL,
        help="Hugging Face model name or local path.",
    )
    parser.add_argument(
        "--output-file",
        default="",
        help="Where to write JSON results. Defaults next to input.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=256)
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
    scorer = PolitenessScorer(
        model_name=args.model_name,
        device=args.device,
        max_length=max(8, args.max_length),
    )
    scored_comments = scorer.score_comments(
        comments=comments,
        batch_size=max(1, args.batch_size),
        include_text=not args.no_comment_text,
    )
    scores_by_comment_id = {row["comment_id"]: row for row in scored_comments}

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
            )
        )

    thread_results.sort(
        key=lambda row: (
            row["impolite_rate"],
            row["polite_rate"],
            row["comment_count"],
        ),
        reverse=True,
    )
    result = {
        "meta": {
            "input": str(input_path),
            "target_kind": target_kind,
            "metric": "Polite-Guard politeness classification",
            "model_name": args.model_name,
            "model_source": "https://huggingface.co/Intel/polite-guard",
            "device": scorer.device,
            "max_length": args.max_length,
            "canonical_labels": list(CANONICAL_LABELS),
            "model_label_map": scorer.model_label_map,
            "comment_count": len(scored_comments),
            "thread_count": len(thread_results),
            "include_comment_text": not args.no_comment_text,
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
        f"polite_rate={overall['polite_rate']:.4f} "
        f"somewhat_polite_rate={overall['somewhat_polite_rate']:.4f} "
        f"neutral_rate={overall['neutral_rate']:.4f} "
        f"impolite_rate={overall['impolite_rate']:.4f}"
    )


class PolitenessScorer:
    """Batch classifier wrapper for Intel/polite-guard."""

    def __init__(self, model_name: str, device: str, max_length: int) -> None:
        try:
            import torch
            from transformers import (
                AutoConfig,
                AutoModelForSequenceClassification,
                AutoTokenizer,
            )
        except ImportError as exc:  # pragma: no cover - environment guard
            raise SystemExit(
                "Politeness scoring requires torch and transformers. Use system python3 "
                "in this workspace, or install those packages in your venv."
            ) from exc

        self.torch = torch
        self.model_name = model_name
        self.max_length = max_length
        self.device = self._resolve_device(device)
        self.config = AutoConfig.from_pretrained(model_name)
        self.model_label_map = {
            str(idx): str(label) for idx, label in sorted(self.config.id2label.items())
        }
        self.label_index_to_name = {
            int(idx): canonicalize_label(label)
            for idx, label in self.config.id2label.items()
        }
        observed = set(self.label_index_to_name.values())
        expected = set(CANONICAL_LABELS)
        if observed != expected:
            raise SystemExit(
                "Unexpected polite-guard labels. "
                f"Observed={sorted(observed)} expected={sorted(expected)} "
                f"raw_id2label={self.config.id2label}"
            )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def score_comments(
        self,
        comments: list[ThreadComment],
        batch_size: int,
        include_text: bool,
    ) -> list[dict[str, Any]]:
        """Return per-comment politeness probabilities and predicted labels."""

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
                probs = self.torch.softmax(logits, dim=-1).detach().cpu().tolist()

            for comment, prob in zip(batch, probs):
                probabilities = {
                    self.label_index_to_name[idx]: float(prob[idx])
                    for idx in range(len(prob))
                }
                pred_label = max(probabilities, key=probabilities.get)
                row = {
                    "thread_id": comment.thread_id,
                    "thread_title": comment.thread_title,
                    "comment_id": comment.comment_id,
                    "parent_id": comment.parent_id,
                    "author": comment.author,
                    "depth": comment.depth,
                    "pred_label": pred_label,
                    "polite_probability": probabilities["polite"],
                    "somewhat_polite_probability": probabilities["somewhat_polite"],
                    "neutral_probability": probabilities["neutral"],
                    "impolite_probability": probabilities["impolite"],
                    "class_probabilities": probabilities,
                }
                if include_text:
                    row["text"] = comment.text
                rows.append(row)
        return rows

    def _resolve_device(self, requested: str) -> str:
        """Resolve torch device string."""

        if requested != "auto":
            return requested
        if self.torch.cuda.is_available():
            return "cuda"
        if (
            getattr(self.torch.backends, "mps", None)
            and self.torch.backends.mps.is_available()
        ):
            return "mps"
        return "cpu"


def canonicalize_label(label: str) -> str:
    """Normalize model labels to the public politeness schema."""

    normalized = re.sub(r"[^a-z0-9]+", "_", str(label).lower()).strip("_")
    if normalized == "somewhatpolite":
        return "somewhat_polite"
    return normalized


def aggregate_thread(
    thread_id: str,
    comments: list[ThreadComment],
    scored_comments: list[dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate politeness label counts/rates for one thread."""

    comment_count = len(scored_comments)
    counts = {
        label: sum(1 for row in scored_comments if row["pred_label"] == label)
        for label in CANONICAL_LABELS
    }
    row = {
        "thread_id": thread_id,
        "thread_title": comments[0].thread_title if comments else "",
        "comment_count": comment_count,
        "comments": scored_comments,
    }
    for label in CANONICAL_LABELS:
        row[f"{label}_count"] = counts[label]
        row[f"{label}_rate"] = (
            float(counts[label] / comment_count) if comment_count else 0.0
        )
    return row


def aggregate_overall(thread_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate thread-level politeness rates across the dataset."""

    comment_count = sum(int(row["comment_count"]) for row in thread_results)
    valid_rows = [row for row in thread_results if int(row["comment_count"]) > 0]
    result: dict[str, Any] = {
        "comment_count": comment_count,
        "thread_count": len(thread_results),
    }
    for label in CANONICAL_LABELS:
        count_key = f"{label}_count"
        rate_key = f"{label}_rate"
        result[count_key] = sum(int(row[count_key]) for row in valid_rows)
        result[rate_key] = weighted_average(
            [(float(row[rate_key]), int(row["comment_count"])) for row in valid_rows]
        )
    return result


def default_output_path(input_path: Path) -> Path:
    """Choose a default output path next to the evaluated input."""

    if input_path.is_dir():
        return input_path / "politeness_results.json"
    return input_path.with_suffix(input_path.suffix + ".politeness_results.json")


if __name__ == "__main__":
    main()
