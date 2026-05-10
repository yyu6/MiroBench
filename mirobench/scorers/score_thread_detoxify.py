#!/usr/bin/env python3
"""Score Reddit comments with Detoxify and aggregate toxicity by thread.

This evaluator uses the local `detoxify` checkout and the `unbiased` model,
which returns the six Jigsaw unintended-bias toxicity dimensions relevant for
aggressive discussion analysis:

- toxicity
- severe_toxicity
- obscene
- threat
- insult
- identity_attack

For each comment, the script outputs one 0-1 score per dimension plus a simple
composite `aggression_score`, defined as the unweighted mean of those six
dimensions.

For each thread, it reports mean / max / p90 for every dimension and for the
composite aggression score.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable

DETOXIFY_DIMS = (
    "toxicity",
    "severe_toxicity",
    "obscene",
    "threat",
    "insult",
    "identity_attack",
)

from .score_thread_disagreement import detect_target_kind
from .score_thread_semantic_uniformity import (
    ThreadComment,
    load_generated_comments,
    load_real_comments,
    median,
    weighted_average,
)


def main() -> None:
    """CLI entrypoint."""

    parser = argparse.ArgumentParser(
        description="Score comments with Detoxify and aggregate thread-level toxicity/aggression."
    )
    parser.add_argument("input", help="Real discussion folder/file or generated run folder/file.")
    parser.add_argument(
        "--target-kind",
        choices=["auto", "real", "generated"],
        default="auto",
        help="Input schema. auto detects generated discussion.json vs real Reddit bundle.",
    )
    parser.add_argument(
        "--detoxify-repo",
        default=str(DETOXIFY_REPO),
        help="Path to local detoxify checkout.",
    )
    parser.add_argument(
        "--model-name",
        default="unbiased",
        choices=["unbiased", "original", "multilingual", "original-small", "unbiased-small"],
        help="Detoxify model name. `unbiased` exposes the requested six dimensions.",
    )
    parser.add_argument(
        "--output-file",
        default="",
        help="Where to write JSON results. Defaults next to input.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
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
    scorer = DetoxifyScorer(
        detoxify_repo=Path(args.detoxify_repo).expanduser().resolve(),
        model_name=args.model_name,
        device=args.device,
    )
    scored_comments = scorer.score_comments(
        comments=comments,
        batch_size=max(1, args.batch_size),
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
            )
        )

    thread_results.sort(
        key=lambda row: (
            row["aggression_score_mean"],
            row["toxicity_mean"],
            row["comment_count"],
        ),
        reverse=True,
    )
    result = {
        "meta": {
            "input": str(input_path),
            "target_kind": target_kind,
            "metric": "Detoxify toxicity/aggression",
            "detoxify_repo": str(Path(args.detoxify_repo).expanduser().resolve()),
            "model_name": args.model_name,
            "device": scorer.device,
            "dimensions": list(DETOXIFY_DIMS),
            "aggression_score_note": (
                "Per-comment aggression_score is the unweighted mean of "
                "toxicity, severe_toxicity, obscene, threat, insult, and identity_attack."
            ),
            "comment_count": len(scored_comments),
            "thread_count": len(thread_results),
            "include_comment_text": not args.no_comment_text,
            "source": source_meta,
        },
        "overall": aggregate_overall(thread_results),
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
        f"toxicity_mean={overall['toxicity_mean']:.4f} "
        f"insult_mean={overall['insult_mean']:.4f} "
        f"aggression_score_mean={overall['aggression_score_mean']:.4f}"
    )


class DetoxifyScorer:
    """Batch scorer using the detoxify library."""

    def __init__(self, detoxify_repo: Path, model_name: str, device: str) -> None:
        try:
            import torch
        except ImportError as exc:
            raise SystemExit(
                "Detoxify scoring requires torch. Install: pip install torch"
            ) from exc

        self.torch = torch
        self.device = self._resolve_device(device)
        # Try pip-installed detoxify first, then local repo path
        try:
            from detoxify import Detoxify
        except ImportError:
            if str(detoxify_repo) not in sys.path:
                sys.path.insert(0, str(detoxify_repo))
            try:
                from detoxify import Detoxify
            except ImportError as exc:
                raise SystemExit(
                    "Could not import detoxify. Install: pip install detoxify"
                ) from exc
        self.model = Detoxify(model_name, device=self.device)

    def score_comments(
        self,
        comments: list[ThreadComment],
        batch_size: int,
        include_text: bool,
    ) -> list[dict[str, Any]]:
        """Return per-comment Detoxify scores."""

        rows: list[dict[str, Any]] = []
        for start in range(0, len(comments), batch_size):
            batch = comments[start : start + batch_size]
            predictions = self.model.predict([comment.text for comment in batch])
            # Detoxify returns a scalar (not a list) when batch size == 1
            for dim in DETOXIFY_DIMS:
                v = predictions[dim]
                if not hasattr(v, "__len__"):
                    predictions[dim] = [v]
            for idx, comment in enumerate(batch):
                scores = {
                    dim: float(predictions[dim][idx])
                    for dim in DETOXIFY_DIMS
                }
                aggression_score = mean(scores.values())
                row = {
                    "thread_id": comment.thread_id,
                    "thread_title": comment.thread_title,
                    "comment_id": comment.comment_id,
                    "parent_id": comment.parent_id,
                    "author": comment.author,
                    "depth": comment.depth,
                    **scores,
                    "aggression_score": aggression_score,
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
        if getattr(self.torch.backends, "mps", None) and self.torch.backends.mps.is_available():
            return "mps"
        return "cpu"


def aggregate_thread(
    thread_id: str,
    comments: list[ThreadComment],
    scored_comments: list[dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate Detoxify scores for one thread."""

    row: dict[str, Any] = {
        "thread_id": thread_id,
        "thread_title": comments[0].thread_title if comments else "",
        "comment_count": len(scored_comments),
        "comments": scored_comments,
    }
    for dim in DETOXIFY_DIMS + ("aggression_score",):
        values = [float(comment[dim]) for comment in scored_comments]
        row[f"{dim}_mean"] = mean(values)
        row[f"{dim}_max"] = max(values) if values else 0.0
        row[f"{dim}_p90"] = percentile(values, 90)
    return row


def aggregate_overall(thread_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate thread-level Detoxify metrics across the dataset."""

    comment_count = sum(int(row["comment_count"]) for row in thread_results)
    valid_rows = [row for row in thread_results if int(row["comment_count"]) > 0]
    result: dict[str, Any] = {
        "comment_count": comment_count,
        "thread_count": len(thread_results),
    }
    for dim in DETOXIFY_DIMS + ("aggression_score",):
        result[f"{dim}_mean"] = weighted_average(
            [(float(row[f"{dim}_mean"]), int(row["comment_count"])) for row in valid_rows]
        )
        result[f"{dim}_max"] = max((float(row[f"{dim}_max"]) for row in valid_rows), default=0.0)
        result[f"{dim}_p90"] = weighted_average(
            [(float(row[f"{dim}_p90"]), int(row["comment_count"])) for row in valid_rows]
        )
    return result


def percentile(values: list[float], pct: float) -> float:
    """Return percentile with linear interpolation."""

    if not values:
        return 0.0
    values = sorted(values)
    if len(values) == 1:
        return float(values[0])
    rank = (len(values) - 1) * pct / 100.0
    lower = int(rank)
    upper = min(lower + 1, len(values) - 1)
    weight = rank - lower
    return float(values[lower] * (1.0 - weight) + values[upper] * weight)


def mean(values: Iterable[float]) -> float:
    """Mean with empty-list protection."""

    values = list(values)
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def default_output_path(input_path: Path) -> Path:
    """Choose a default output path next to the evaluated input."""

    if input_path.is_dir():
        return input_path / "detoxify_results.json"
    return input_path.with_suffix(input_path.suffix + ".detoxify_results.json")


if __name__ == "__main__":
    main()
