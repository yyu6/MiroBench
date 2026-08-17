#!/usr/bin/env python3
"""Score thread-level semantic self-similarity with pairwise BERTScore.

This evaluator measures whether comments inside one Reddit thread repeat the
same deeper meaning. For each thread, every unordered comment pair is scored
with BERTScore precision/recall/F1, and the thread-level score is the mean over
all pair F1 values.

Higher Self-BERTScore means the thread is more semantically uniform. Lower
Self-BERTScore means the comments are more semantically diverse.

The script uses the local `bert_score-master` checkout by default instead of
requiring the `bert-score` package to be installed globally.

Default model policy:

- preferred: `microsoft/deberta-xlarge-mnli`
- fallback: `roberta-large`

This matches the BERTScore README guidance that DeBERTa variants correlate
better with human judgment than the original RoBERTa default, while keeping a
practical fallback when the larger model is unavailable on the current machine.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_BERT_SCORE_PATH = REPO_ROOT / "bert_score-master"
DEFAULT_MODEL = "microsoft/deberta-xlarge-mnli"
FALLBACK_MODEL = "roberta-large"
DEFAULT_NUM_LAYERS: int | None = None
_TOKENIZER_MAX_LENGTH_SENTINEL = 1_000_000
_DEFAULT_SAFE_MODEL_MAX_LENGTH = 512

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


def main() -> None:
    """CLI entrypoint."""

    parser = argparse.ArgumentParser(
        description="Score within-thread pairwise Self-BERTScore for Reddit comments."
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
        "--bert-score-path",
        default=str(DEFAULT_BERT_SCORE_PATH),
        help="Path to local bert_score-master checkout.",
    )
    parser.add_argument(
        "--model-type",
        default=DEFAULT_MODEL,
        help=(
            "Hugging Face model name or local path passed to BERTScorer. "
            "Default prefers microsoft/deberta-xlarge-mnli and falls back to roberta-large."
        ),
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=DEFAULT_NUM_LAYERS,
        help="Optional BERTScore layer count. By default, use the package's tuned layer for the model.",
    )
    parser.add_argument(
        "--output-file",
        default="",
        help="Where to write JSON results. Defaults next to input.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--top-k",
        type=int,
        default=1,
        help=(
            "For backward-compatible CSV exports only. Default 1 means "
            "`top_k_mean_bert_f1` is equivalent to per-thread max BERTScore F1."
        ),
    )
    parser.add_argument("--max-threads", type=int, default=0)
    parser.add_argument("--max-comments-per-thread", type=int, default=0)
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Torch device. auto uses cuda, then mps, then cpu.",
    )
    parser.add_argument(
        "--idf",
        action="store_true",
        help="Use IDF weighting over all comments in the evaluated input.",
    )
    parser.add_argument(
        "--rescale-with-baseline",
        action="store_true",
        help="Apply BERTScore baseline rescaling when a baseline exists for the model.",
    )
    parser.add_argument(
        "--include-pairs",
        action="store_true",
        help="Include every pair score in the output JSON. This can make real outputs large.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Force bert-score/transformers to load only from local Hugging Face cache.",
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

    pair_specs = build_pair_specs(comments_by_thread)
    (
        scorer,
        bert_hash,
        resolved_device,
        resolved_model_type,
        resolved_num_layers,
        fallback_used,
    ) = load_bert_scorer(
        bert_score_path=Path(args.bert_score_path).expanduser().resolve(),
        model_type=args.model_type,
        num_layers=args.num_layers,
        batch_size=max(1, args.batch_size),
        device=args.device,
        idf=args.idf,
        idf_sents=[
            comment.text
            for comments in comments_by_thread.values()
            for comment in comments
        ],
        rescale_with_baseline=args.rescale_with_baseline,
        local_files_only=args.local_files_only,
    )

    (
        pair_scores,
        scorer,
        bert_hash,
        resolved_device,
        resolved_model_type,
        resolved_num_layers,
        fallback_used,
    ) = score_pairs_with_device_fallback(
        scorer=scorer,
        pair_specs=pair_specs,
        batch_size=max(1, args.batch_size),
        bert_score_path=Path(args.bert_score_path).expanduser().resolve(),
        model_type=resolved_model_type,
        num_layers=resolved_num_layers,
        requested_device=args.device,
        idf=args.idf,
        idf_sents=[
            comment.text
            for comments in comments_by_thread.values()
            for comment in comments
        ],
        rescale_with_baseline=args.rescale_with_baseline,
        local_files_only=args.local_files_only,
        fallback_used=fallback_used,
    )
    thread_results = aggregate_threads(
        comments_by_thread=comments_by_thread,
        pair_scores=pair_scores,
        top_k=max(1, args.top_k),
        include_pairs=args.include_pairs,
    )
    thread_results.sort(
        key=lambda row: (
            row["mean_bert_f1"],
            row["comment_count"],
        ),
        reverse=True,
    )

    result = {
        "meta": {
            "input": str(input_path),
            "target_kind": target_kind,
            "metric": "Self-BERTScore",
            "aggregation": (
                "unordered comment pairs within each thread; thread summary reports "
                "mean / median / max pair F1"
            ),
            "interpretation": "higher Self-BERTScore F1 means more semantic uniformity",
            "bert_score_path": str(Path(args.bert_score_path).expanduser().resolve()),
            "requested_model_type": args.model_type,
            "requested_num_layers": args.num_layers,
            "model_type": resolved_model_type,
            "num_layers": resolved_num_layers,
            "fallback_used": fallback_used,
            "fallback_model_type": FALLBACK_MODEL if fallback_used else "",
            "bert_hash": bert_hash,
            "device": resolved_device,
            "batch_size": args.batch_size,
            "idf": args.idf,
            "rescale_with_baseline": args.rescale_with_baseline,
            "include_pairs": args.include_pairs,
            "local_files_only": args.local_files_only,
            "thread_count": len(thread_results),
            "comment_count": sum(int(row["comment_count"]) for row in thread_results),
            "pair_count": len(pair_scores),
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
        f"pairs={overall['pair_count']} mean_f1={overall['weighted_mean_bert_f1']:.4f} "
        f"median_thread_f1={overall['median_thread_bert_f1']:.4f} "
        f"max_thread_max_f1={overall['max_thread_max_bert_f1']:.4f}"
    )


def load_bert_scorer(
    bert_score_path: Path,
    model_type: str,
    num_layers: int | None,
    batch_size: int,
    device: str,
    idf: bool,
    idf_sents: list[str],
    rescale_with_baseline: bool,
    local_files_only: bool,
) -> tuple[Any, str, str, str, int, bool]:
    """Import local BERTScore and return a configured scorer."""

    if not bert_score_path.exists():
        raise FileNotFoundError(f"BERTScore checkout not found: {bert_score_path}")
    sys.path.insert(0, str(bert_score_path))
    if local_files_only:
        os.environ["BERT_SCORE_LOCAL_FILES_ONLY"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"

    try:
        import torch
        from bert_score import BERTScorer
    except ImportError as exc:  # pragma: no cover - environment guard
        raise SystemExit(
            "Self-BERTScore requires torch, transformers, and the local bert_score package. "
            "Use system python3 in this workspace, or install those packages in your venv."
        ) from exc

    resolved_device = resolve_device(torch, device)
    scorer_kwargs = {
        "batch_size": batch_size,
        "idf": idf,
        "idf_sents": idf_sents if idf else None,
        "device": resolved_device,
        "lang": "en",
        "rescale_with_baseline": rescale_with_baseline,
    }
    if num_layers is not None:
        scorer_kwargs["num_layers"] = num_layers

    try:
        scorer = BERTScorer(
            model_type=model_type,
            **scorer_kwargs,
        )
        normalize_tokenizer_max_length(scorer)
        return (
            scorer,
            scorer.hash,
            resolved_device,
            scorer.model_type,
            scorer.num_layers,
            False,
        )
    except Exception as exc:
        if model_type != DEFAULT_MODEL:
            raise
        fallback_kwargs = dict(scorer_kwargs)
        fallback_kwargs.pop("num_layers", None)
        print(
            "Preferred BERTScore model failed to load; falling back to roberta-large. "
            f"Original error: {type(exc).__name__}: {exc}"
        )
        scorer = BERTScorer(
            model_type=FALLBACK_MODEL,
            **fallback_kwargs,
        )
        normalize_tokenizer_max_length(scorer)
        return (
            scorer,
            scorer.hash,
            resolved_device,
            scorer.model_type,
            scorer.num_layers,
            True,
        )


def normalize_tokenizer_max_length(scorer: Any) -> None:
    """Clamp Hugging Face tokenizer max lengths to a backend-safe finite value.

    Some newer tokenizer builds expose a huge sentinel in `model_max_length`
    when the upstream config does not set a practical limit. The local
    `bert_score-master` code forwards that value into `tokenizer.encode(...)`,
    which can overflow fast-tokenizer truncation in recent transformers
    releases. We normalize the tokenizer in-place using the encoder config when
    available, falling back to 512.
    """

    tokenizer = getattr(scorer, "_tokenizer", None)
    model = getattr(scorer, "_model", None)
    if tokenizer is None:
        return

    current = getattr(tokenizer, "model_max_length", None)
    if isinstance(current, int) and 0 < current < _TOKENIZER_MAX_LENGTH_SENTINEL:
        return

    config = getattr(model, "config", None)
    safe_max_length = getattr(config, "max_position_embeddings", None)
    if not isinstance(safe_max_length, int) or safe_max_length <= 0:
        safe_max_length = _DEFAULT_SAFE_MODEL_MAX_LENGTH

    safe_max_length = int(safe_max_length)
    tokenizer.model_max_length = safe_max_length

    # Legacy tokenizer attributes are still used by some helper paths.
    for attr_name, reserve in (
        ("max_len_single_sentence", 2),
        ("max_len_sentences_pair", 4),
    ):
        current_value = getattr(tokenizer, attr_name, None)
        if (
            isinstance(current_value, int)
            and 0 < current_value < _TOKENIZER_MAX_LENGTH_SENTINEL
        ):
            continue
        try:
            setattr(tokenizer, attr_name, max(1, safe_max_length - reserve))
        except Exception:
            pass


def resolve_device(torch_module: Any, requested: str) -> str:
    """Resolve a torch device string."""

    if requested == "cpu":
        return "cpu"
    if requested == "cuda":
        return "cuda" if torch_module.cuda.is_available() else "cpu"
    if requested == "mps":
        mps_backend = getattr(torch_module.backends, "mps", None)
        if mps_backend and mps_backend.is_available():
            return "mps"
        return "cpu"
    if requested != "auto":
        return requested
    if torch_module.cuda.is_available():
        return "cuda"
    if (
        getattr(torch_module.backends, "mps", None)
        and torch_module.backends.mps.is_available()
    ):
        return "mps"
    return "cpu"


def build_pair_specs(
    comments_by_thread: dict[str, list[ThreadComment]],
) -> list[dict[str, Any]]:
    """Build unordered within-thread comment pair specifications."""

    pair_specs = []
    for thread_id, comments in comments_by_thread.items():
        for left_idx in range(len(comments)):
            for right_idx in range(left_idx + 1, len(comments)):
                left = comments[left_idx]
                right = comments[right_idx]
                pair_specs.append(
                    {
                        "thread_id": thread_id,
                        "left": left,
                        "right": right,
                    }
                )
    return pair_specs


def score_pairs(
    scorer: Any,
    pair_specs: list[dict[str, Any]],
    batch_size: int,
) -> list[dict[str, Any]]:
    """Run BERTScore for every comment pair."""

    if not pair_specs:
        return []
    cands = [pair["left"].text for pair in pair_specs]
    refs = [pair["right"].text for pair in pair_specs]
    precision, recall, f1 = scorer.score(cands, refs, batch_size=batch_size)

    pair_scores = []
    for idx, pair in enumerate(pair_specs):
        left: ThreadComment = pair["left"]
        right: ThreadComment = pair["right"]
        pair_scores.append(
            {
                "thread_id": pair["thread_id"],
                "left_comment_id": left.comment_id,
                "right_comment_id": right.comment_id,
                "left_author": left.author,
                "right_author": right.author,
                "left_depth": left.depth,
                "right_depth": right.depth,
                "bert_precision": float(precision[idx]),
                "bert_recall": float(recall[idx]),
                "bert_f1": float(f1[idx]),
            }
        )
    return pair_scores


def score_pairs_with_device_fallback(
    *,
    scorer: Any,
    pair_specs: list[dict[str, Any]],
    batch_size: int,
    bert_score_path: Path,
    model_type: str,
    num_layers: int | None,
    requested_device: str,
    idf: bool,
    idf_sents: list[str],
    rescale_with_baseline: bool,
    local_files_only: bool,
    fallback_used: bool,
) -> tuple[list[dict[str, Any]], Any, str, str, str, int, bool]:
    """Score pairs, retrying on CPU when MPS runs out of memory."""

    try:
        return (
            score_pairs(scorer=scorer, pair_specs=pair_specs, batch_size=batch_size),
            scorer,
            scorer.hash,
            getattr(scorer, "device", requested_device),
            scorer.model_type,
            scorer.num_layers,
            fallback_used,
        )
    except RuntimeError as exc:
        if not _is_mps_oom_error(exc, requested_device):
            raise
        print(
            "MPS ran out of memory during Self-BERTScore; retrying this metric on CPU.",
            flush=True,
        )
        (
            cpu_scorer,
            cpu_hash,
            cpu_device,
            cpu_model_type,
            cpu_num_layers,
            cpu_fallback_used,
        ) = load_bert_scorer(
            bert_score_path=bert_score_path,
            model_type=model_type,
            num_layers=num_layers,
            batch_size=batch_size,
            device="cpu",
            idf=idf,
            idf_sents=idf_sents,
            rescale_with_baseline=rescale_with_baseline,
            local_files_only=local_files_only,
        )
        return (
            score_pairs(
                scorer=cpu_scorer, pair_specs=pair_specs, batch_size=batch_size
            ),
            cpu_scorer,
            cpu_hash,
            cpu_device,
            cpu_model_type,
            cpu_num_layers,
            cpu_fallback_used,
        )


def _is_mps_oom_error(exc: RuntimeError, requested_device: str) -> bool:
    """Return whether *exc* is an MPS out-of-memory failure."""

    if requested_device not in {"auto", "mps"}:
        return False
    text = str(exc)
    return "MPS backend out of memory" in text


def aggregate_threads(
    comments_by_thread: dict[str, list[ThreadComment]],
    pair_scores: list[dict[str, Any]],
    top_k: int,
    include_pairs: bool,
) -> list[dict[str, Any]]:
    """Aggregate pair scores to thread-level rows."""

    scores_by_thread: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for pair in pair_scores:
        scores_by_thread[str(pair["thread_id"])].append(pair)

    thread_results = []
    for thread_id, comments in comments_by_thread.items():
        pairs = scores_by_thread.get(thread_id, [])
        f1_values = [float(pair["bert_f1"]) for pair in pairs]
        precision_values = [float(pair["bert_precision"]) for pair in pairs]
        recall_values = [float(pair["bert_recall"]) for pair in pairs]
        top_values = sorted(f1_values, reverse=True)[: min(top_k, len(f1_values))]
        max_f1 = max(f1_values) if f1_values else 0.0
        row: dict[str, Any] = {
            "thread_id": thread_id,
            "thread_title": comments[0].thread_title if comments else "",
            "comment_count": len(comments),
            "pair_count": len(pairs),
            "mean_bert_precision": mean(precision_values),
            "mean_bert_recall": mean(recall_values),
            "mean_bert_f1": mean(f1_values),
            "median_bert_f1": median(f1_values),
            "top_k": top_k,
            "top_k_mean_bert_f1": mean(top_values),
            "p90_bert_f1": percentile(f1_values, 90),
            "min_bert_f1": min(f1_values) if f1_values else 0.0,
            "max_bert_f1": max_f1,
            "note": ""
            if len(comments) >= 2
            else "Self-BERTScore requires at least two usable comments.",
        }
        if include_pairs:
            row["pairs"] = pairs
            row["comments"] = [asdict(comment) for comment in comments]
        thread_results.append(row)
    return thread_results


def aggregate_overall(thread_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-thread BERTScore metrics into one summary."""

    total_comments = sum(int(row["comment_count"]) for row in thread_results)
    total_pairs = sum(int(row["pair_count"]) for row in thread_results)
    valid_rows = [row for row in thread_results if int(row["pair_count"]) > 0]
    return {
        "comment_count": total_comments,
        "thread_count": len(thread_results),
        "pair_count": total_pairs,
        "weighted_mean_bert_precision": weighted_average(
            [
                (float(row["mean_bert_precision"]), int(row["pair_count"]))
                for row in valid_rows
            ]
        ),
        "weighted_mean_bert_recall": weighted_average(
            [
                (float(row["mean_bert_recall"]), int(row["pair_count"]))
                for row in valid_rows
            ]
        ),
        "weighted_mean_bert_f1": weighted_average(
            [(float(row["mean_bert_f1"]), int(row["pair_count"])) for row in valid_rows]
        ),
        "mean_thread_bert_f1": mean([float(row["mean_bert_f1"]) for row in valid_rows]),
        "median_thread_bert_f1": median(
            [float(row["mean_bert_f1"]) for row in valid_rows]
        ),
        "weighted_top_k_mean_bert_f1": weighted_average(
            [
                (float(row["top_k_mean_bert_f1"]), int(row["pair_count"]))
                for row in valid_rows
            ]
        ),
        "max_thread_mean_bert_f1": max(
            (float(row["mean_bert_f1"]) for row in valid_rows),
            default=0.0,
        ),
        "max_thread_max_bert_f1": max(
            (float(row["max_bert_f1"]) for row in valid_rows),
            default=0.0,
        ),
    }


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
        return input_path / "self_bertscore_results.json"
    return input_path.with_suffix(input_path.suffix + ".self_bertscore_results.json")


if __name__ == "__main__":
    main()
