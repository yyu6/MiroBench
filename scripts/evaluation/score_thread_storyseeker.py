#!/usr/bin/env python3
"""Score each Reddit comment for personal-story likelihood with StorySeeker.

This evaluator uses `mariaantoniak/storyseeker`, a RoBERTa text-classification
model trained to detect whether an online post/comment contains a story. It
scores each individual comment and aggregates story/not-story counts by thread.

The Hugging Face model config exposes generic labels (`LABEL_0`, `LABEL_1`).
Sanity checks with clear story and non-story examples show:

- LABEL_0 = not_story
- LABEL_1 = story

The output includes per-comment probabilities plus per-thread story counts,
not-story counts, and story rates.
"""

from __future__ import annotations

import argparse
import json
import sys
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


DEFAULT_MODEL = "mariaantoniak/storyseeker"
LABEL_MAP = {
    0: "not_story",
    1: "story",
}


def main() -> None:
    """CLI entrypoint."""

    parser = argparse.ArgumentParser(
        description="Score comments with mariaantoniak/storyseeker and aggregate by thread."
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
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=512)
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
    scorer = StorySeekerScorer(
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
            row["story_rate"],
            row["story_count"],
            row["comment_count"],
        ),
        reverse=True,
    )
    result = {
        "meta": {
            "input": str(input_path),
            "target_kind": target_kind,
            "metric": "StorySeeker personal-story likelihood",
            "model_name": args.model_name,
            "model_source": "https://huggingface.co/mariaantoniak/storyseeker",
            "device": scorer.device,
            "max_length": args.max_length,
            "threshold": args.threshold,
            "label_map": LABEL_MAP,
            "label_note": (
                "Model config uses LABEL_0/LABEL_1. Sanity examples indicate "
                "LABEL_0=not_story and LABEL_1=story."
            ),
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
        f"story={overall['story_count']} not_story={overall['not_story_count']} "
        f"story_rate={overall['story_rate']:.4f} "
        f"mean_story_probability={overall['weighted_mean_story_probability']:.4f}"
    )


class StorySeekerScorer:
    """Batch classifier wrapper for mariaantoniak/storyseeker."""

    def __init__(self, model_name: str, device: str, max_length: int) -> None:
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - environment guard
            raise SystemExit(
                "StorySeeker scoring requires torch and transformers. Use system python3 "
                "in this workspace, or install those packages in your venv."
            ) from exc

        self.torch = torch
        self.model_name = model_name
        self.max_length = max_length
        self.device = self._resolve_device(device)
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
        """Return per-comment story probabilities and labels."""

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
                not_story_probability = float(prob[0])
                story_probability = float(prob[1])
                pred_label = "story" if story_probability >= threshold else "not_story"
                row = {
                    "thread_id": comment.thread_id,
                    "thread_title": comment.thread_title,
                    "comment_id": comment.comment_id,
                    "parent_id": comment.parent_id,
                    "author": comment.author,
                    "depth": comment.depth,
                    "not_story_probability": not_story_probability,
                    "story_probability": story_probability,
                    "pred_label": pred_label,
                    "raw_label_probabilities": {
                        "LABEL_0": not_story_probability,
                        "LABEL_1": story_probability,
                    },
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
        if (
            getattr(self.torch.backends, "mps", None)
            and self.torch.backends.mps.is_available()
        ):
            return "mps"
        return "cpu"


def aggregate_thread(
    thread_id: str,
    comments: list[ThreadComment],
    scored_comments: list[dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate per-comment StorySeeker scores for one thread."""

    story_count = sum(1 for row in scored_comments if row["pred_label"] == "story")
    not_story_count = sum(
        1 for row in scored_comments if row["pred_label"] == "not_story"
    )
    story_probabilities = [float(row["story_probability"]) for row in scored_comments]
    return {
        "thread_id": thread_id,
        "thread_title": comments[0].thread_title if comments else "",
        "comment_count": len(scored_comments),
        "story_count": story_count,
        "not_story_count": not_story_count,
        "story_rate": story_count / len(scored_comments) if scored_comments else 0.0,
        "mean_story_probability": mean(story_probabilities),
        "median_story_probability": median(story_probabilities),
        "max_story_probability": max(story_probabilities)
        if story_probabilities
        else 0.0,
        "min_story_probability": min(story_probabilities)
        if story_probabilities
        else 0.0,
        "comments": scored_comments,
    }


def aggregate_overall(thread_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate thread-level StorySeeker scores."""

    comment_count = sum(int(row["comment_count"]) for row in thread_results)
    story_count = sum(int(row["story_count"]) for row in thread_results)
    not_story_count = sum(int(row["not_story_count"]) for row in thread_results)
    valid_rows = [row for row in thread_results if int(row["comment_count"]) > 0]
    return {
        "comment_count": comment_count,
        "thread_count": len(thread_results),
        "story_count": story_count,
        "not_story_count": not_story_count,
        "story_rate": story_count / comment_count if comment_count else 0.0,
        "weighted_mean_story_probability": weighted_average(
            [
                (float(row["mean_story_probability"]), int(row["comment_count"]))
                for row in valid_rows
            ]
        ),
        "mean_thread_story_rate": mean(
            [float(row["story_rate"]) for row in valid_rows]
        ),
        "median_thread_story_rate": median(
            [float(row["story_rate"]) for row in valid_rows]
        ),
        "max_thread_story_rate": max(
            (float(row["story_rate"]) for row in valid_rows), default=0.0
        ),
        "mean_thread_story_probability": mean(
            [float(row["mean_story_probability"]) for row in valid_rows]
        ),
        "median_thread_story_probability": median(
            [float(row["mean_story_probability"]) for row in valid_rows]
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
        return input_path / "storyseeker_results.json"
    return input_path.with_suffix(input_path.suffix + ".storyseeker_results.json")


if __name__ == "__main__":
    main()
