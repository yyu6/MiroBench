#!/usr/bin/env python3
"""CARD-style full-collection controller for exact text-distribution metrics."""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import pandas as pd


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_ROOT.parent
for import_path in (PACKAGE_ROOT, REPO_ROOT / "scripts"):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

import run_metric_revision_controller as card_controller  # noqa: E402
from distribution_diagnostics import diagnose_distribution  # noqa: E402
from revision_memory import (  # noqa: E402
    build_memory,
    choose_history_aware_strategy,
    history_path as revision_history_path,
    load_json_list,
    memory_sha256,
    merge_strategy_history,
    memory_path as revision_memory_path,
    persist_memory,
    restore_controller_state,
    round_memory_path,
    summarize_reviser_output,
    write_json_atomic,
)


SUPPORTED_METRICS = (
    "semantic_mean_cosine",
    "length_cv",
    "emotion_entropy",
)
ALL_METRICS = (
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


@dataclass(frozen=True)
class SignedShape:
    q10_gap: float
    q25_gap: float
    q50_gap: float
    q75_gap: float
    q90_gap: float
    min_gap: float
    max_gap: float
    direction: str
    profile: str
    region: str = "shape_only"
    tolerance: float = 0.0
    active_regions: tuple[str, ...] = ()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("generated_root", type=Path)
    parser.add_argument("--scores-csv", type=Path, required=True)
    parser.add_argument("--matched-eval-dir", type=Path, required=True)
    parser.add_argument("--seed-post-pool-json", type=Path, required=True)
    parser.add_argument("--real-scores-csv", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--target-metric", choices=SUPPORTED_METRICS, required=True)
    parser.add_argument("--max-rounds", type=int, default=7)
    parser.add_argument("--pass-threshold", type=float, default=0.05)
    parser.add_argument("--min-round-improvement", type=float, default=0.0005)
    parser.add_argument("--model", default="gpt-5.4-mini")
    parser.add_argument(
        "--api-key",
        default=os.environ.get("OPENAI_API_KEY")
        or os.environ.get("PLANNER_API_KEY")
        or os.environ.get("LLM_API_KEY"),
    )
    parser.add_argument("--base-url", default="https://api.openai.com/v1")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--metric-parallel", type=int, default=5)
    parser.add_argument("--expected-seeds", type=int, default=150)
    parser.add_argument("--posts-per-run", type=int, default=5)
    parser.add_argument(
        "--reviser-script",
        type=Path,
        default=PACKAGE_ROOT / "scripts" / "run_text_metric_reviser.py",
    )
    parser.add_argument(
        "--cleanup-script",
        type=Path,
        default=REPO_ROOT / "scripts" / "postprocess_generated_discussions_gpt_cleanup.py",
    )
    parser.add_argument(
        "--score-script",
        type=Path,
        default=REPO_ROOT / "scripts" / "evaluation" / "score_sampled_generated_runs.py",
    )
    parser.add_argument(
        "--match-script",
        type=Path,
        default=REPO_ROOT / "scripts" / "evaluate_matched_seed_group.py",
    )
    parser.add_argument("--continue-after-reject", action="store_true")
    parser.add_argument("--strategy-history-json", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.self_test:
        run_self_test()
        return
    if args.max_rounds <= 0:
        raise SystemExit("--max-rounds must be positive")
    card_controller.validate_start(
        args,
        args.generated_root,
        args.scores_csv,
        args.matched_eval_dir,
    )
    if not args.api_key and not args.dry_run:
        raise SystemExit("An API key is required unless --dry-run is used")

    history_file = revision_history_path(args.output_prefix)
    memory_file = revision_memory_path(args.output_prefix)
    history = load_json_list(history_file)
    current_root, current_scores, current_matched, next_round = restore_controller_state(
        history,
        initial_root=args.generated_root,
        initial_scores=args.scores_csv,
        initial_matched=args.matched_eval_dir,
    )
    card_controller.validate_start(args, current_root, current_scores, current_matched)
    if history:
        print(
            f"[text-controller-resume] completed_rounds={next_round - 1} "
            f"next_round={next_round} current_root={current_root}",
            flush=True,
        )
    final_root = current_root
    final_matched = current_matched

    for round_idx in range(next_round, args.max_rounds + 1):
        before_eval = card_controller.load_eval(current_matched)
        before = card_controller.snapshot(before_eval, args.target_metric)
        shape = signed_distribution_shape(
            generated_scores=current_matched / "matched_generated_thread_scores.csv",
            real_scores=current_matched / "matched_real_thread_scores.csv",
            metric=args.target_metric,
        )
        print_round_header(round_idx, args.target_metric, before, shape)
        if card_controller.metric_passes(before, args.pass_threshold):
            print(f"[text-controller-stop] target passes at round {round_idx}", flush=True)
            break

        memory = build_memory(
            merge_strategy_history(args.strategy_history_json, history),
            controller="text_metric",
            target_metrics=(args.target_metric,),
            current_input_root=current_root,
        )
        round_memory_file = round_memory_path(args.output_prefix, round_idx)
        if not args.dry_run:
            write_json_atomic(memory_file, memory)
            write_json_atomic(round_memory_file, memory)
        base_profile = shape.profile
        selected_profile = choose_history_aware_strategy(
            base_profile,
            text_profile_candidates(base_profile),
            memory,
        )
        shape = replace(shape, profile=selected_profile)

        output_root = Path(
            f"{args.output_prefix}_round{round_idx:02d}_{shape.direction}_{shape.profile}"
        )
        command = build_reviser_command(
            args=args,
            input_root=current_root,
            scores_csv=current_scores,
            matched_eval_dir=current_matched,
            output_root=output_root,
            shape=shape,
            controller_memory_json=round_memory_file,
        )
        memo: dict[str, Any] = {
            "round": round_idx,
            "target_metric": args.target_metric,
            "before": before.__dict__,
            "shape": asdict(shape),
            "base_profile": base_profile,
            "profile_params": profile_params(args.target_metric, shape.profile),
            "input_root": str(current_root),
            "input_scores_csv": str(current_scores),
            "input_matched_eval_dir": str(current_matched),
            "output_root": str(output_root),
            "command": card_controller.redact_command(command),
            "controller_memory_snapshot": str(round_memory_file),
            "controller_memory_sha256": memory_sha256(memory),
        }
        print(
            f"[text-controller-decision] direction={shape.direction} "
            f"profile={shape.profile} output={output_root}",
            flush=True,
        )
        print(
            "[text-controller-command] "
            + " ".join(card_controller.redact_command(command)),
            flush=True,
        )
        if args.dry_run:
            history.append({**memo, "decision": "dry_run"})
            continue

        card_controller.run(command)
        clean_root, eval_dir, matched_dir = card_controller.cleanup_score_match(output_root, args)
        after_eval = card_controller.load_eval(matched_dir)
        after = card_controller.snapshot(after_eval, args.target_metric)
        after_shape = signed_distribution_shape(
            generated_scores=matched_dir / "matched_generated_thread_scores.csv",
            real_scores=matched_dir / "matched_real_thread_scores.csv",
            metric=args.target_metric,
        )
        protected = card_controller.protected_report(
            before_eval,
            after_eval,
            ",".join(metric for metric in ALL_METRICS if metric != args.target_metric),
        )
        improved = card_controller.metric_improved(
            before,
            after,
            args.min_round_improvement,
        )
        protected_ok = not protected["hard_failure"]
        accepted = improved and protected_ok
        rejection_reasons = []
        if not improved:
            rejection_reasons.append("target_improvement_below_threshold")
        if not protected_ok:
            rejection_reasons.append("protected_metric_regression")
        card_controller.print_round_result(
            round_idx,
            before,
            after,
            protected,
            improved,
            protected_ok,
        )
        next_root = clean_root if accepted else current_root
        next_scores = (
            eval_dir / "revised_generated_thread_scores.csv" if accepted else current_scores
        )
        next_matched = matched_dir if accepted else current_matched
        memo.update(
            {
                "after": after.__dict__,
                "after_gaps": asdict(after_shape),
                "protected_report": protected,
                "improved": improved,
                "protected_ok": protected_ok,
                "accepted_round": accepted,
                "decision": "accepted" if accepted else "rejected",
                "rejection_reasons": rejection_reasons,
                "candidate_summary": summarize_reviser_output(output_root),
                "clean_root": str(clean_root),
                "eval_dir": str(eval_dir),
                "matched_eval_dir": str(matched_dir),
                "next_input_root": str(next_root),
                "next_scores_csv": str(next_scores),
                "next_matched_eval_dir": str(next_matched),
                "rollback_to_root": None if accepted else str(current_root),
            }
        )
        history.append(memo)
        card_controller.write_history(args.output_prefix, history)
        persist_memory(
            memory_file,
            merge_strategy_history(args.strategy_history_json, history),
            controller="text_metric",
            target_metrics=(args.target_metric,),
            current_input_root=next_root,
        )

        if accepted:
            current_root = clean_root
            current_scores = next_scores
            current_matched = matched_dir
            final_root = clean_root
            final_matched = matched_dir
            print(
                f"[text-controller-accept] round={round_idx} next_input={current_root}",
                flush=True,
            )
        else:
            print(
                f"[text-controller-reject] round={round_idx} rollback_to={current_root}",
                flush=True,
            )
            if args.continue_after_reject and round_idx < args.max_rounds:
                continue
            break
        if card_controller.metric_passes(after, args.pass_threshold):
            print("[text-controller-stop] target passes after accepted round", flush=True)
            break

    if not args.dry_run:
        card_controller.write_history(args.output_prefix, history)
        persist_memory(
            memory_file,
            merge_strategy_history(args.strategy_history_json, history),
            controller="text_metric",
            target_metrics=(args.target_metric,),
            current_input_root=current_root,
        )
    print(f"[text-controller-done] final_root={final_root}", flush=True)
    print(f"[text-controller-done] final_matched_eval={final_matched}", flush=True)


def signed_distribution_shape(*, generated_scores: Path, real_scores: Path, metric: str) -> SignedShape:
    generated = pd.to_numeric(pd.read_csv(generated_scores)[metric], errors="coerce").dropna()
    real = pd.to_numeric(pd.read_csv(real_scores)[metric], errors="coerce").dropna()
    if generated.empty or real.empty:
        raise SystemExit(f"Cannot compute distribution shape for {metric}")
    diagnosis = diagnose_distribution(generated, real, minimum_tolerance=0.0005)
    profile = diagnosis.recommended_profile
    if diagnosis.region == "low_tail":
        profile = "low_tail"
    return SignedShape(
        q10_gap=diagnosis.q10_gap,
        q25_gap=diagnosis.q25_gap,
        q50_gap=diagnosis.q50_gap,
        q75_gap=diagnosis.q75_gap,
        q90_gap=diagnosis.q90_gap,
        min_gap=diagnosis.min_gap,
        max_gap=diagnosis.max_gap,
        direction=diagnosis.direction if diagnosis.direction in {"increase", "decrease"} else (
            "decrease" if float(generated.mean()) > float(real.mean()) else "increase"
        ),
        profile=profile,
        region=diagnosis.region,
        tolerance=diagnosis.tolerance,
        active_regions=diagnosis.active_regions,
    )


def build_reviser_command(
    *,
    args: argparse.Namespace,
    input_root: Path,
    scores_csv: Path,
    matched_eval_dir: Path,
    output_root: Path,
    shape: SignedShape,
    controller_memory_json: Path | None = None,
) -> list[str]:
    params = profile_params(args.target_metric, shape.profile)
    command = [
        sys.executable,
        str(args.reviser_script),
        str(input_root),
        "--scores-csv",
        str(scores_csv),
        "--real-scores-csv",
        str(matched_eval_dir / "matched_real_thread_scores.csv"),
        "--output-dir",
        str(output_root),
        "--target-metric",
        args.target_metric,
        "--direction",
        shape.direction,
        "--target-profile",
        shape.profile,
        "--model",
        args.model,
        "--api-key",
        args.api_key or "",
        "--base-url",
        args.base_url,
        "--device",
        args.device,
        "--resume-existing",
    ]
    if controller_memory_json is not None:
        command.extend(["--controller-memory-json", str(controller_memory_json)])
    for key, value in params.items():
        command.extend([f"--{key}", str(value)])
    return command


def profile_params(metric: str, profile: str) -> dict[str, int | float]:
    gap = {
        "semantic_mean_cosine": 0.006,
        "length_cv": 0.045,
        "emotion_entropy": 0.060,
    }[metric]
    local = {
        "semantic_mean_cosine": 0.0002,
        "length_cv": 0.008,
        "emotion_entropy": 0.015,
    }[metric]
    if profile == "middle_mass":
        return {
            "min-thread-gap": gap * 0.75,
            "candidates-per-comment": 9,
            "min-local-gap-reduction": local,
        }
    if profile in {"high_tail", "low_tail"}:
        return {
            "min-thread-gap": gap,
            "candidates-per-comment": 8,
            "min-local-gap-reduction": local,
        }
    return {
        "min-thread-gap": gap * 0.60,
        "candidates-per-comment": 8,
        "min-local-gap-reduction": local * 0.75,
    }


def text_profile_candidates(base_profile: str) -> list[str]:
    orders = {
        "high_tail": ["high_tail", "middle_mass", "shape_safe", "low_tail"],
        "low_tail": ["low_tail", "middle_mass", "shape_safe", "high_tail"],
        "middle_mass": ["middle_mass", "shape_safe", "high_tail", "low_tail"],
        "shape_safe": ["shape_safe", "middle_mass", "high_tail", "low_tail"],
    }
    return orders.get(base_profile, [base_profile, "shape_safe", "middle_mass"])


def print_round_header(
    round_idx: int,
    metric: str,
    before: Any,
    shape: SignedShape,
) -> None:
    print(
        f"[text-controller-round] round={round_idx} metric={metric} status={before.status} "
        f"MWU={before.mwu:.5g} KS={before.ks:.5g} Cliff={before.cliff:.5g} "
        f"W={before.wasserstein:.5g} gen_mean={before.generated_mean:.5g} "
        f"real_mean={before.real_mean:.5g}",
        flush=True,
    )
    print(
        f"[text-controller-shape] direction={shape.direction} profile={shape.profile} "
        f"q10={shape.q10_gap:+.5g} q25={shape.q25_gap:+.5g} "
        f"q50={shape.q50_gap:+.5g} q75={shape.q75_gap:+.5g} "
        f"q90={shape.q90_gap:+.5g} min={shape.min_gap:+.5g} max={shape.max_gap:+.5g}",
        flush=True,
    )


def run_self_test() -> None:
    frame_generated = pd.DataFrame({"metric": [0.1, 0.2, 0.4, 0.9]})
    frame_real = pd.DataFrame({"metric": [0.1, 0.2, 0.3, 0.4]})
    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        generated_path = Path(directory) / "generated.csv"
        real_path = Path(directory) / "real.csv"
        frame_generated.to_csv(generated_path, index=False)
        frame_real.to_csv(real_path, index=False)
        shape = signed_distribution_shape(
            generated_scores=generated_path,
            real_scores=real_path,
            metric="metric",
        )
    assert shape.direction == "decrease"
    assert shape.region == "broad_high"
    assert shape.profile == "middle_mass"
    print("generalized text metric controller self-test passed")


if __name__ == "__main__":
    main()
