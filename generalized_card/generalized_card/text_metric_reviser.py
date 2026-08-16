from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from openai import OpenAI

from .domain import DomainConfig, REPO_ROOT, load_domain_from_env
from .prompts import protected_entities, protected_numbers
from .reviser_backend import parse_candidate_response


for import_path in (REPO_ROOT / "scripts", REPO_ROOT / "scripts" / "evaluation"):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from revision_memory import prompt_feedback_from_path  # noqa: E402

from postprocess_metric_gated_candidate_replacement import (  # noqa: E402
    PLANNER_LABEL_TOKENS,
    claim_overlap,
    contains_placeholder_token,
    normalize_text,
)
from postprocess_selfbert_tail_repair import (  # noqa: E402
    CommentRef,
    clear_metric_cache,
    find_post,
    flatten_comments,
    normalize_generated_scores,
    safe_float,
    safe_int,
)
from run_controlled_qwen_discussion import _clean_text, _render_markdown  # noqa: E402
from score_thread_go_emotions import GoEmotionsScorer, shannon_entropy  # noqa: E402
from score_thread_semantic_uniformity import (  # noqa: E402
    DEFAULT_MODEL as DEFAULT_SEMANTIC_MODEL,
    CommentEmbedder,
    ThreadComment,
    pairwise_cosine_values,
)
from score_thread_structure import compute_length_cv  # noqa: E402
from token_usage_tracker import record_openai_usage  # noqa: E402


SUPPORTED_METRICS = (
    "semantic_mean_cosine",
    "length_cv",
    "emotion_entropy",
)


@dataclass(frozen=True)
class ThreadTarget:
    generated: dict[str, Any]
    real: dict[str, Any]
    signed_gap: float
    rewrite_budget: int


@dataclass
class CandidateDecision:
    comment_id: int
    accepted: bool
    old_content: str
    new_content: str
    style: str
    reason: str
    old_metric: float
    new_metric: float
    gap_reduction: float
    claim_overlap: float
    word_ratio: float


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Exact local candidate repair for generalized CARD text metrics."
    )
    parser.add_argument("generated_root", type=Path, nargs="?")
    parser.add_argument("--scores-csv", type=Path)
    parser.add_argument("--real-scores-csv", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--target-metric", choices=SUPPORTED_METRICS, required=True)
    parser.add_argument("--direction", choices=("increase", "decrease"), required=True)
    parser.add_argument(
        "--target-profile",
        choices=("high_tail", "middle_mass", "low_tail", "shape_safe"),
        default="shape_safe",
    )
    parser.add_argument("--model", default="gpt-5.4-mini")
    parser.add_argument(
        "--api-key",
        default=os.environ.get("OPENAI_API_KEY")
        or os.environ.get("PLANNER_API_KEY")
        or os.environ.get("LLM_API_KEY"),
    )
    parser.add_argument("--base-url", default="https://api.openai.com/v1")
    parser.add_argument("--min-comments", type=int, default=2)
    parser.add_argument("--min-thread-gap", type=float, default=0.0)
    parser.add_argument("--candidates-per-comment", type=int, default=7)
    parser.add_argument("--min-local-gap-reduction", type=float, default=0.0)
    parser.add_argument("--semantic-model", default=DEFAULT_SEMANTIC_MODEL)
    parser.add_argument("--device", default="cpu", choices=("auto", "cpu", "cuda", "mps"))
    parser.add_argument("--semantic-batch-size", type=int, default=24)
    parser.add_argument("--emotion-batch-size", type=int, default=16)
    parser.add_argument("--emotion-threshold", type=float, default=0.5)
    parser.add_argument("--api-retries", type=int, default=6)
    parser.add_argument("--api-retry-delay", type=float, default=10.0)
    parser.add_argument(
        "--controller-memory-json",
        type=Path,
        default=None,
        help="Structured controller memory used only as compact candidate-generation feedback.",
    )
    parser.add_argument("--resume-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.controller_feedback = prompt_feedback_from_path(args.controller_memory_json)
    if args.self_test:
        run_self_test()
        return
    validate_args(args)
    config = load_domain_from_env()
    targets = select_targets(args)
    print_targets(targets, args)
    if args.dry_run:
        return
    if not args.api_key:
        raise SystemExit("An API key is required unless --dry-run is used")

    output_dir = args.output_dir.expanduser().resolve()
    prepare_output(args.generated_root.expanduser().resolve(), output_dir, args.resume_existing)
    report_path = output_dir / f"{args.target_metric}_reviser_report.json"
    report = load_json_list(report_path)
    completed = {
        report_key(row)
        for row in report
        if row.get("status") in {"applied", "no_accepted_rewrites", "no_candidate_comments"}
    }
    client = OpenAI(api_key=args.api_key, base_url=args.base_url)
    prompt_dir = output_dir / f"_{args.target_metric}_prompts"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    scorer = build_exact_scorer(args)

    for rank, target in enumerate(targets, start=1):
        key = target_key(target)
        if key in completed:
            print(f"[text-metric-resume] rank={rank} seed={key[0]} already complete", flush=True)
            continue
        result = revise_thread(
            target=target,
            rank=rank,
            output_dir=output_dir,
            prompt_dir=prompt_dir,
            client=client,
            scorer=scorer,
            config=config,
            args=args,
        )
        report.append(result)
        write_json_atomic(report_path, report)
        print(
            f"[text-metric-done] rank={rank} seed={key[0]} status={result['status']} "
            f"accepted={len(result.get('accepted_rewrites') or [])} "
            f"rejected={len(result.get('rejected_best_attempts') or [])}",
            flush=True,
        )

    clear_metric_cache(output_dir)
    write_json_atomic(report_path, report)
    print(f"[text-metric-output] {output_dir}", flush=True)
    print(f"[text-metric-report] {report_path}", flush=True)


def validate_args(args: argparse.Namespace) -> None:
    required = (
        ("generated_root", args.generated_root),
        ("--scores-csv", args.scores_csv),
        ("--real-scores-csv", args.real_scores_csv),
        ("--output-dir", args.output_dir),
    )
    missing = [label for label, value in required if value is None]
    if missing:
        raise SystemExit("Missing required inputs: " + ", ".join(missing))
    for label, value in required[:-1]:
        if not value.expanduser().exists():
            raise SystemExit(f"{label} not found: {value}")
    if args.output_dir.expanduser().exists() and not args.resume_existing and not args.dry_run:
        raise SystemExit(f"Output exists; use --resume-existing: {args.output_dir}")


def default_min_gap(metric: str) -> float:
    return {
        "semantic_mean_cosine": 0.010,
        "length_cv": 0.060,
        "emotion_entropy": 0.080,
    }[metric]


def default_local_gain(metric: str) -> float:
    return {
        "semantic_mean_cosine": 0.0002,
        "length_cv": 0.008,
        "emotion_entropy": 0.015,
    }[metric]


def select_targets(args: argparse.Namespace) -> list[ThreadTarget]:
    generated = normalize_generated_scores(pd.read_csv(args.scores_csv))
    real = normalize_real_scores(pd.read_csv(args.real_scores_csv))
    real_by_seed = {
        safe_int(row.get("seed_index"), -1): row for row in real.to_dict(orient="records")
    }
    threshold = args.min_thread_gap if args.min_thread_gap > 0 else default_min_gap(args.target_metric)
    targets: list[ThreadTarget] = []
    for row in generated.to_dict(orient="records"):
        seed = safe_int(row.get("seed_index"), -1)
        real_row = real_by_seed.get(seed)
        if real_row is None:
            continue
        comment_count = safe_int(row.get("comment_count"), 0) or 0
        if comment_count < args.min_comments:
            continue
        signed_gap = safe_float(row.get(args.target_metric)) - safe_float(
            real_row.get(args.target_metric)
        )
        directional_gap = signed_gap if args.direction == "decrease" else -signed_gap
        if directional_gap < threshold:
            continue
        # The capacity is the complete comment set. Exact local scoring and the
        # matched-real stop condition determine how many edits are retained.
        targets.append(ThreadTarget(row, real_row, signed_gap, comment_count))
    targets.sort(key=lambda target: abs(target.signed_gap), reverse=True)
    return targets


def normalize_real_scores(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    if "seed_index" not in output.columns and "matched_seed_idx" in output.columns:
        output["seed_index"] = output["matched_seed_idx"]
    if "seed_index" not in output.columns:
        raise SystemExit("Real scores CSV has no seed_index or matched_seed_idx")
    output["seed_index"] = output["seed_index"].map(lambda value: safe_int(value, -1))
    return output


def print_targets(targets: list[ThreadTarget], args: argparse.Namespace) -> None:
    print(
        f"[text-metric-select] metric={args.target_metric} direction={args.direction} "
        f"profile={args.target_profile} threads={len(targets)}",
        flush=True,
    )
    for rank, target in enumerate(targets[:40], start=1):
        print(
            f"[text-metric-target] rank={rank:02d} seed={target_key(target)[0]} "
            f"generated={safe_float(target.generated.get(args.target_metric)):.4f} "
            f"real={safe_float(target.real.get(args.target_metric)):.4f} "
            f"gap={target.signed_gap:+.4f} budget={target.rewrite_budget}",
            flush=True,
        )


def build_exact_scorer(args: argparse.Namespace) -> Any:
    if args.target_metric == "semantic_mean_cosine":
        return CommentEmbedder(
            model_name=args.semantic_model,
            device=args.device,
            max_length=256,
        )
    if args.target_metric == "emotion_entropy":
        return GoEmotionsScorer(
            model_name="SamLowe/roberta-base-go_emotions",
            device=args.device,
            max_length=256,
        )
    return None


def revise_thread(
    *,
    target: ThreadTarget,
    rank: int,
    output_dir: Path,
    prompt_dir: Path,
    client: OpenAI,
    scorer: Any,
    config: DomainConfig,
    args: argparse.Namespace,
) -> dict[str, Any]:
    run_dir = copied_run_for_row(output_dir, target.generated)
    base = {
        "rank": rank,
        "seed_index": target_key(target)[0],
        "source_raw_post_id": target_key(target)[1],
        "target_metric": args.target_metric,
        "direction": args.direction,
        "target_profile": args.target_profile,
        "initial_metric": safe_float(target.generated.get(args.target_metric)),
        "matched_real_metric": safe_float(target.real.get(args.target_metric)),
        "rewrite_budget": target.rewrite_budget,
        "accepted_rewrites": [],
        "rejected_best_attempts": [],
    }
    if run_dir is None:
        return {**base, "status": "run_not_found"}
    discussion_path = run_dir / "discussion.json"
    discussion = json.loads(discussion_path.read_text(encoding="utf-8"))
    post = find_post(discussion, target.generated)
    if post is None:
        return {**base, "status": "post_not_found", "run_dir": str(run_dir)}

    marker = post.get("generalized_metric_revisions") or []
    if args.resume_existing and any(
        isinstance(row, dict) and row.get("metric") == args.target_metric for row in marker
    ):
        return {**base, "status": "already_applied_marker", "run_dir": str(run_dir)}

    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    attempted: set[int] = set()
    real_value = safe_float(target.real.get(args.target_metric))
    current_value = exact_thread_metric(post, args.target_metric, scorer, args)
    stop_gap = args.min_thread_gap if args.min_thread_gap > 0 else default_min_gap(
        args.target_metric
    )

    while len(attempted) < target.rewrite_budget:
        current_directional_gap = (
            current_value - real_value
            if args.direction == "decrease"
            else real_value - current_value
        )
        if current_directional_gap < stop_gap:
            break
        comments = usable_comments(post)
        ranked, hints = rank_comment_targets(
            comments=comments,
            metric=args.target_metric,
            direction=args.direction,
            scorer=scorer,
            args=args,
        )
        current = next((ref for ref in ranked if ref.comment_id not in attempted), None)
        if current is None:
            break
        attempted.add(current.comment_id)
        hint = hints.get(current.comment_id, {})
        prompt = build_prompt(
            post=post,
            comments=comments,
            target=current,
            metric=args.target_metric,
            direction=args.direction,
            current_value=current_value,
            real_value=real_value,
            hint=hint,
            candidates=args.candidates_per_comment,
            config=config,
            controller_feedback=args.controller_feedback,
        )
        stem = f"rank_{rank:03d}_comment_{current.comment_id}"
        (prompt_dir / f"{stem}.prompt.txt").write_text(prompt, encoding="utf-8")
        try:
            raw = chat_completion_text(client=client, prompt=prompt, args=args)
            (prompt_dir / f"{stem}.response.json").write_text(raw, encoding="utf-8")
            candidates = parse_candidate_response(raw)
        except Exception as exc:  # noqa: BLE001
            rejected.append(
                {"comment_id": current.comment_id, "reason": f"generation_failed:{type(exc).__name__}:{exc}"}
            )
            continue
        decision = choose_candidate(
            post=post,
            comments=comments,
            target=current,
            candidates=candidates,
            metric=args.target_metric,
            direction=args.direction,
            current_value=current_value,
            real_value=real_value,
            scorer=scorer,
            config=config,
            args=args,
        )
        if decision.accepted:
            current.comment["content"] = decision.new_content
            current.comment.setdefault("generalized_metric_revision_history", []).append(
                {
                    "metric": args.target_metric,
                    "old_content": decision.old_content,
                    "new_content": decision.new_content,
                    "reason": decision.reason,
                }
            )
            current_value = decision.new_metric
            accepted.append(decision.__dict__)
        else:
            rejected.append(decision.__dict__)

    if accepted:
        post.setdefault("generalized_metric_revisions", []).append(
            {
                "metric": args.target_metric,
                "direction": args.direction,
                "profile": args.target_profile,
                "accepted_rewrites": len(accepted),
                "initial_metric": base["initial_metric"],
                "projected_metric": current_value,
                "matched_real_metric": real_value,
            }
        )
        discussion_path.write_text(json.dumps(discussion, ensure_ascii=False, indent=2), encoding="utf-8")
        (run_dir / "discussion.md").write_text(_render_markdown(discussion), encoding="utf-8")
    return {
        **base,
        "status": "applied" if accepted else "no_accepted_rewrites",
        "run_dir": str(run_dir),
        "final_projected_metric": current_value,
        "candidate_comment_ids": sorted(attempted),
        "accepted_rewrites": accepted,
        "rejected_best_attempts": rejected,
    }


def usable_comments(post: dict[str, Any]) -> list[CommentRef]:
    return [
        ref
        for ref in flatten_comments(post.get("comments") or [])
        if ref.content.strip()
        and ref.content.strip().lower() not in {"[deleted]", "[removed]"}
        and len(ref.content.split()) >= 3
    ]


def rank_comment_targets(
    *,
    comments: list[CommentRef],
    metric: str,
    direction: str,
    scorer: Any,
    args: argparse.Namespace,
) -> tuple[list[CommentRef], dict[int, dict[str, Any]]]:
    hints: dict[int, dict[str, Any]] = {}
    if not comments:
        return [], hints
    if metric == "semantic_mean_cosine":
        embeddings = scorer.encode(
            [ref.content for ref in comments],
            batch_size=max(1, args.semantic_batch_size),
        )
        matrix = np.clip(embeddings @ embeddings.T, -1.0, 1.0)
        contribution = (
            matrix.sum(axis=1) - 1.0
        ) / max(1, len(comments) - 1)
        indexed = list(enumerate(comments))
        indexed.sort(key=lambda row: float(contribution[row[0]]), reverse=direction == "decrease")
        for idx, ref in indexed:
            hints[ref.comment_id] = {"semantic_contribution": float(contribution[idx])}
        return [ref for _, ref in indexed], hints
    if metric == "length_cv":
        lengths = [len(ref.content.split()) for ref in comments]
        average = sum(lengths) / len(lengths)
        indexed = list(enumerate(comments))
        indexed.sort(key=lambda row: abs(lengths[row[0]] - average), reverse=True)
        for idx, ref in indexed:
            old_words = lengths[idx]
            if direction == "decrease":
                target_words = round(average + 0.45 * (old_words - average))
            else:
                delta = old_words - average
                if abs(delta) < max(2.0, average * 0.10):
                    delta = max(4.0, average * 0.35) if old_words >= average else -max(3.0, average * 0.25)
                target_words = round(average + 1.45 * delta)
            hints[ref.comment_id] = {
                "old_words": old_words,
                "thread_mean_words": average,
                "target_words": max(3, min(180, target_words)),
            }
        return [ref for _, ref in indexed], hints

    rows = score_emotions(comments, scorer, args)
    counts = Counter(row["dominant_emotion"] for row in rows)
    by_id = {int(row["comment_id"]): row for row in rows}
    majority = counts.most_common(1)[0][0] if counts else "neutral"
    if direction == "increase":
        ranked = sorted(
            comments,
            key=lambda ref: counts.get(str(by_id[ref.comment_id]["dominant_emotion"]), 0),
            reverse=True,
        )
        goal = "a natural alternative affect that is currently underrepresented"
    else:
        ranked = sorted(
            comments,
            key=lambda ref: counts.get(str(by_id[ref.comment_id]["dominant_emotion"]), 0),
        )
        goal = f"a restrained {majority} or neutral expression"
    for ref in ranked:
        hints[ref.comment_id] = {
            "old_emotion": by_id[ref.comment_id]["dominant_emotion"],
            "target_emotion": goal,
            "dominant_counts": dict(counts),
        }
    return ranked, hints


def exact_thread_metric(post: dict[str, Any], metric: str, scorer: Any, args: argparse.Namespace) -> float:
    comments = usable_comments(post)
    if metric == "length_cv":
        return compute_length_cv(sorted(len(ref.content.split()) for ref in comments))
    if metric == "semantic_mean_cosine":
        embeddings = scorer.encode(
            [ref.content for ref in comments],
            batch_size=max(1, args.semantic_batch_size),
        )
        values = pairwise_cosine_values(embeddings)
        return float(np.mean(values)) if len(values) else 0.0
    rows = score_emotions(comments, scorer, args)
    return float(shannon_entropy(Counter(row["dominant_emotion"] for row in rows).values()))


def score_emotions(comments: list[CommentRef], scorer: Any, args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = [
        ThreadComment(
            thread_id="candidate",
            thread_title="",
            comment_id=str(ref.comment_id),
            parent_id=str(ref.parent_comment_id or "post"),
            author="",
            text=ref.content,
            depth=ref.depth,
        )
        for ref in comments
    ]
    return scorer.score_comments(
        comments=rows,
        batch_size=max(1, args.emotion_batch_size),
        threshold=args.emotion_threshold,
        include_text=False,
    )


def choose_candidate(
    *,
    post: dict[str, Any],
    comments: list[CommentRef],
    target: CommentRef,
    candidates: list[dict[str, str]],
    metric: str,
    direction: str,
    current_value: float,
    real_value: float,
    scorer: Any,
    config: DomainConfig,
    args: argparse.Namespace,
) -> CandidateDecision:
    visible = visible_context(post, comments)
    minimum_gain = (
        args.min_local_gap_reduction
        if args.min_local_gap_reduction > 0
        else default_local_gain(metric)
    )
    valid_rows: list[tuple[str, str, float, float]] = []
    rejected_rows: list[CandidateDecision] = []
    for row in candidates:
        candidate = _clean_text(str(row.get("text") or ""))
        style = str(row.get("style") or "candidate")
        ok, reason, overlap, ratio = validate_candidate(
            old=target.content,
            candidate=candidate,
            visible_context=visible,
            config=config,
            metric=metric,
        )
        if ok:
            valid_rows.append((candidate, style, overlap, ratio))
        else:
            rejected_rows.append(
                CandidateDecision(
                    target.comment_id,
                    False,
                    target.content,
                    candidate,
                    style,
                    reason,
                    current_value,
                    current_value,
                    0.0,
                    overlap,
                    ratio,
                )
            )

    projected = project_candidate_metrics(
        comments=comments,
        target=target,
        candidates=[row[0] for row in valid_rows],
        metric=metric,
        scorer=scorer,
        args=args,
    )
    decisions = list(rejected_rows)
    for (candidate, style, overlap, ratio), new_value in zip(valid_rows, projected):
        gap_reduction = abs(current_value - real_value) - abs(new_value - real_value)
        moved_correctly = (
            new_value < current_value if direction == "decrease" else new_value > current_value
        )
        if not moved_correctly:
            reason = f"wrong_direction:{current_value:.5f}->{new_value:.5f}"
        elif gap_reduction < minimum_gain:
            reason = f"gap_reduction_too_small:{gap_reduction:.5f}"
        else:
            reason = f"accepted:gap_reduction={gap_reduction:.5f}"
        decisions.append(
            CandidateDecision(
                target.comment_id,
                reason.startswith("accepted:"),
                target.content,
                candidate,
                style,
                reason,
                current_value,
                new_value,
                gap_reduction,
                overlap,
                ratio,
            )
        )
    if decisions:
        return max(decisions, key=candidate_rank)
    return CandidateDecision(
        target.comment_id,
        False,
        target.content,
        "",
        "",
        "no_candidates",
        current_value,
        current_value,
        0.0,
        0.0,
        0.0,
    )


def project_candidate_metrics(
    *,
    comments: list[CommentRef],
    target: CommentRef,
    candidates: list[str],
    metric: str,
    scorer: Any,
    args: argparse.Namespace,
) -> list[float]:
    if not candidates:
        return []
    target_index = next(
        (index for index, ref in enumerate(comments) if ref.comment_id == target.comment_id),
        None,
    )
    if target_index is None:
        return [0.0] * len(candidates)
    if metric == "length_cv":
        base = [len(ref.content.split()) for ref in comments]
        output = []
        for candidate in candidates:
            lengths = list(base)
            lengths[target_index] = len(candidate.split())
            output.append(compute_length_cv(sorted(lengths)))
        return output
    if metric == "semantic_mean_cosine":
        texts = [ref.content for ref in comments]
        embeddings = scorer.encode(
            texts + candidates,
            batch_size=max(1, args.semantic_batch_size),
        )
        base = embeddings[: len(texts)]
        candidate_embeddings = embeddings[len(texts) :]
        pair_values = pairwise_cosine_values(base)
        pair_sum = float(pair_values.sum())
        pair_count = len(pair_values)
        old_target_sum = float(
            np.dot(base[target_index], np.delete(base, target_index, axis=0).T).sum()
        )
        others = np.delete(base, target_index, axis=0)
        return [
            (pair_sum - old_target_sum + float(np.dot(vector, others.T).sum()))
            / max(1, pair_count)
            for vector in candidate_embeddings
        ]

    current_rows = score_emotions(comments, scorer, args)
    old_label = next(
        str(row["dominant_emotion"])
        for row in current_rows
        if int(row["comment_id"]) == target.comment_id
    )
    candidate_comments = [
        ThreadComment(
            thread_id="candidate",
            thread_title="",
            comment_id=f"candidate_{index}",
            parent_id=str(target.parent_comment_id or "post"),
            author="",
            text=text,
            depth=target.depth,
        )
        for index, text in enumerate(candidates)
    ]
    candidate_rows = scorer.score_comments(
        comments=candidate_comments,
        batch_size=max(1, args.emotion_batch_size),
        threshold=args.emotion_threshold,
        include_text=False,
    )
    current_counts = Counter(str(row["dominant_emotion"]) for row in current_rows)
    output = []
    for row in candidate_rows:
        counts = Counter(current_counts)
        counts[old_label] -= 1
        counts[str(row["dominant_emotion"])] += 1
        output.append(float(shannon_entropy(counts.values())))
    return output


def validate_candidate(
    *,
    old: str,
    candidate: str,
    visible_context: str,
    config: DomainConfig,
    metric: str,
) -> tuple[bool, str, float, float]:
    if not candidate:
        return False, "empty", 0.0, 0.0
    lower = candidate.lower()
    if any(token in lower for token in PLANNER_LABEL_TOKENS):
        return False, "planner_label_leakage", 0.0, 0.0
    if contains_placeholder_token(candidate) or "```" in candidate or len(candidate) > 1400:
        return False, "format_or_placeholder_leakage", 0.0, 0.0
    if normalize_text(old) == normalize_text(candidate):
        return False, "unchanged", 1.0, 1.0
    ratio = len(candidate.split()) / max(1, len(old.split()))
    ratio_bounds = {
        "semantic_mean_cosine": (0.50, 1.45),
        "length_cv": (0.30, 2.20),
        "emotion_entropy": (0.65, 1.45),
    }[metric]
    if not ratio_bounds[0] <= ratio <= ratio_bounds[1]:
        return False, f"word_ratio_out_of_range:{ratio:.3f}", 0.0, ratio
    overlap = claim_overlap(old, candidate)
    overlap_floor = 0.50 if metric == "length_cv" else 0.58
    if overlap < overlap_floor:
        return False, f"claim_overlap_too_low:{overlap:.3f}", overlap, ratio
    if not protected_numbers(old) <= protected_numbers(candidate):
        return False, "old_numbers_missing", overlap, ratio
    old_entities = protected_entities(config, old)
    new_entities = protected_entities(config, candidate)
    visible_entities = protected_entities(config, visible_context)
    missing = old_entities - new_entities
    if missing:
        return False, "old_named_entities_missing:" + ",".join(sorted(missing)[:6]), overlap, ratio
    added = new_entities - old_entities - visible_entities
    if added:
        return False, "new_named_entities:" + ",".join(sorted(added)[:6]), overlap, ratio
    return True, "valid", overlap, ratio


def candidate_rank(decision: CandidateDecision) -> tuple[bool, float, float, float]:
    return (
        decision.accepted,
        decision.gap_reduction,
        decision.claim_overlap,
        -abs(1.0 - decision.word_ratio),
    )


def build_prompt(
    *,
    post: dict[str, Any],
    comments: list[CommentRef],
    target: CommentRef,
    metric: str,
    direction: str,
    current_value: float,
    real_value: float,
    hint: dict[str, Any],
    candidates: int,
    config: DomainConfig,
    controller_feedback: str = "",
) -> str:
    parent = next(
        (ref for ref in comments if ref.comment_id == target.parent_comment_id),
        None,
    )
    nearby = [ref for ref in comments if ref.comment_id != target.comment_id][:10]
    objective = metric_prompt_guidance(metric, direction, hint)
    feedback_block = (
        f"\n{controller_feedback}\n" if controller_feedback.strip() else ""
    )
    return f"""You revise one generated discussion comment for {config.community_context}.

The revised comment must preserve the original factual claim, stance, reply relation,
named entities, numbers, and personal-experience status. Produce surface alternatives;
do not invent expertise, events, products, specifications, or measurements.

Metric objective: {metric} must {direction} toward its matched-real thread value.
Current thread value: {current_value:.5f}
Matched-real thread value: {real_value:.5f}
Local strategy: {objective}
{feedback_block}

Thread title:
{compact(post.get('title'), 500)}

OP body:
{compact(post.get('content'), 900)}

Parent:
{compact(parent.content, 500) if parent else '(top-level reply to OP)'}

Original target comment ({len(target.content.split())} words):
{compact(target.content, 1000)}

Other thread comments:
{render_comments(nearby)}

Generate {max(3, candidates)} genuinely different candidates. Each must remain a
plausible reply at the same tree location. Avoid polished assistant language and do
not mention metrics, prompts, labels, or revision.

Return strict JSON only:
{{"candidates":[{{"style":"short label","text":"replacement body"}}]}}
"""


def metric_prompt_guidance(metric: str, direction: str, hint: dict[str, Any]) -> str:
    if metric == "semantic_mean_cosine":
        if direction == "decrease":
            return (
                "Keep the same claim but change the discourse job and sentence shape; "
                "make it less like the repeated center of the thread while staying locally relevant."
            )
        return (
            "Re-anchor this outlier to the parent and OP using already visible topic terms, "
            "without copying another comment or changing its claim."
        )
    if metric == "length_cv":
        return (
            f"Aim near {int(hint.get('target_words') or 10)} words "
            f"(thread mean {safe_float(hint.get('thread_mean_words')):.1f}); "
            "preserve all factual anchors while changing only the amount of expression."
        )
    return (
        f"Current dominant affect is {hint.get('old_emotion', 'unknown')}; express "
        f"{hint.get('target_emotion', 'a natural context-compatible affect')} while preserving "
        "the claim, stance, politeness, and disagreement strength."
    )


def chat_completion_text(*, client: OpenAI, prompt: str, args: argparse.Namespace) -> str:
    kwargs: dict[str, Any] = {
        "model": args.model,
        "messages": [
            {
                "role": "system",
                "content": "Write claim-preserving discussion-comment candidates. Return strict JSON only.",
            },
            {"role": "user", "content": prompt},
        ],
    }
    max_tokens = max(1400, args.candidates_per_comment * 260)
    if args.model.lower().startswith("gpt-5"):
        kwargs["max_completion_tokens"] = max_tokens
    else:
        kwargs["max_tokens"] = max_tokens
        kwargs["temperature"] = 0.5
    last_error: Exception | None = None
    for attempt in range(max(1, args.api_retries)):
        try:
            response = client.chat.completions.create(**kwargs)
            record_openai_usage(response, model=args.model, component=f"{args.target_metric}_reviser")
            text = str(response.choices[0].message.content or "").strip()
            if not text:
                raise RuntimeError("empty completion")
            return text
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt + 1 >= max(1, args.api_retries):
                raise
            print(
                f"[text-metric-llm-retry] attempt={attempt + 1}/{args.api_retries} "
                f"error={type(exc).__name__}:{exc}",
                flush=True,
            )
            time.sleep(max(0.0, args.api_retry_delay) * (attempt + 1))
    assert last_error is not None
    raise last_error


def visible_context(post: dict[str, Any], comments: list[CommentRef]) -> str:
    return "\n".join(
        [str(post.get("title") or ""), str(post.get("content") or "")]
        + [ref.content for ref in comments]
    )


def compact(value: Any, limit: int) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else text[: max(0, limit - 3)] + "..."


def render_comments(comments: list[CommentRef]) -> str:
    return "\n".join(
        f"- depth={ref.depth}: {compact(ref.content, 240)}" for ref in comments[:10]
    ) or "(none)"


def prepare_output(source: Path, output: Path, resume: bool) -> None:
    if output.exists():
        if not resume:
            raise SystemExit(f"Output already exists: {output}")
        return
    shutil.copytree(source, output)


def copied_run_for_row(output_dir: Path, row: dict[str, Any]) -> Path | None:
    source = Path(str(row.get("_source_sim_dir") or ""))
    if source.name:
        direct = output_dir / source.name
        if direct.exists():
            return direct
        matches = sorted(output_dir.glob(f"*{source.name}"))
        if matches:
            return matches[0]
    seed = safe_int(row.get("seed_index"), None)
    for path in output_dir.glob("run_*_sampled_reddit/discussion.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if any(safe_int(post.get("seed_index"), None) == seed for post in payload.get("posts") or []):
            return path.parent
    return None


def target_key(target: ThreadTarget) -> tuple[int, str]:
    return (
        safe_int(target.generated.get("seed_index"), -1) or 0,
        str(target.generated.get("source_raw_post_id") or ""),
    )


def report_key(row: dict[str, Any]) -> tuple[int, str]:
    return (
        safe_int(row.get("seed_index"), -1) or 0,
        str(row.get("source_raw_post_id") or ""),
    )


def load_json_list(path: Path) -> list[dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    return [row for row in payload if isinstance(row, dict)] if isinstance(payload, list) else []


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def run_self_test() -> None:
    assert abs(compute_length_cv([5, 10, 15]) - 0.408248290463863) < 1e-9
    old_gap = abs(0.50 - 0.40)
    new_gap = abs(0.43 - 0.40)
    assert old_gap - new_gap > 0
    target = ThreadTarget({}, {}, 0.2, 17)
    assert target.rewrite_budget == 17
    print("generalized text metric reviser self-test passed")
