#!/usr/bin/env python3
"""CARD-style matched-real controller for domain-neutral Self-BERT revision."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_ROOT.parent
for import_path in (PACKAGE_ROOT, REPO_ROOT / "scripts"):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

import run_metric_revision_controller as card_controller  # noqa: E402
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


TARGET_METRIC = "self_bertscore_mean_f1"
PROTECTED_METRICS = (
    "self_bleu_4",
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("generated_root", type=Path)
    parser.add_argument("--scores-csv", type=Path, required=True)
    parser.add_argument("--matched-eval-dir", type=Path, required=True)
    parser.add_argument("--seed-post-pool-json", type=Path, required=True)
    parser.add_argument("--real-scores-csv", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
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
    parser.add_argument(
        "--base-url",
        default=os.environ.get("OPENAI_BASE_URL")
        or os.environ.get("LLM_BASE_URL")
        or "https://api.openai.com/v1",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--metric-parallel", type=int, default=5)
    parser.add_argument("--expected-seeds", type=int, default=150)
    parser.add_argument("--posts-per-run", type=int, default=5)
    parser.add_argument(
        "--reviser-script",
        type=Path,
        default=PACKAGE_ROOT / "scripts" / "run_selfbert_reviser_backend.py",
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
            f"[selfbert-controller-resume] completed_rounds={next_round - 1} "
            f"next_round={next_round} current_root={current_root}",
            flush=True,
        )
    final_root = current_root
    final_matched = current_matched

    for round_idx in range(next_round, args.max_rounds + 1):
        before_eval = card_controller.load_eval(current_matched)
        before = card_controller.snapshot(before_eval, TARGET_METRIC)
        shape = card_controller.distribution_shape(
            generated_scores=current_matched / "matched_generated_thread_scores.csv",
            real_scores=current_matched / "matched_real_thread_scores.csv",
            metric=TARGET_METRIC,
        )
        direction = (
            shape.direction
            if shape.direction in {"increase", "decrease"}
            else "decrease" if before.generated_mean > before.real_mean else "increase"
        )
        card_controller.print_round_header(round_idx, before, shape)
        if card_controller.metric_passes(before, args.pass_threshold):
            print(f"[selfbert-controller-stop] target already passes at round {round_idx}", flush=True)
            break
        memory = build_memory(
            merge_strategy_history(args.strategy_history_json, history),
            controller="selfbert",
            target_metrics=(TARGET_METRIC,),
            current_input_root=current_root,
        )
        round_memory_file = round_memory_path(args.output_prefix, round_idx)
        if not args.dry_run:
            write_json_atomic(memory_file, memory)
            write_json_atomic(round_memory_file, memory)
        base_profile = card_controller.choose_profile(before, shape)
        profile = choose_history_aware_strategy(
            base_profile,
            card_controller.profile_candidates(base_profile),
            memory,
        )
        output_root = Path(f"{args.output_prefix}_round{round_idx:02d}_{profile}")
        command = build_reviser_command(
            args=args,
            profile=profile,
            input_root=current_root,
            scores_csv=current_scores,
            matched_eval_dir=current_matched,
            output_root=output_root,
            direction=direction,
            controller_memory_json=round_memory_file,
        )
        memo: dict[str, Any] = {
            "round": round_idx,
            "target_metric": TARGET_METRIC,
            "before": before.__dict__,
            "shape": shape.__dict__,
            "direction": direction,
            "selected_profile": profile,
            "base_profile": base_profile,
            "profile_params": profile_params(profile),
            "input_root": str(current_root),
            "input_scores_csv": str(current_scores),
            "input_matched_eval_dir": str(current_matched),
            "output_root": str(output_root),
            "command": card_controller.redact_command(command),
            "controller_memory_snapshot": str(round_memory_file),
            "controller_memory_sha256": memory_sha256(memory),
        }
        print(f"[selfbert-controller-decision] profile={profile} output={output_root}", flush=True)
        print(
            "[selfbert-controller-command] "
            + " ".join(card_controller.redact_command(command)),
            flush=True,
        )
        if args.dry_run:
            history.append({**memo, "decision": "dry_run"})
            continue

        card_controller.run(command)
        clean_root, eval_dir, matched_dir = card_controller.cleanup_score_match(output_root, args)
        after_eval = card_controller.load_eval(matched_dir)
        after = card_controller.snapshot(after_eval, TARGET_METRIC)
        after_shape = card_controller.distribution_shape(
            generated_scores=matched_dir / "matched_generated_thread_scores.csv",
            real_scores=matched_dir / "matched_real_thread_scores.csv",
            metric=TARGET_METRIC,
        )
        protected = card_controller.protected_report(
            before_eval,
            after_eval,
            ",".join(PROTECTED_METRICS),
        )
        improved = card_controller.metric_improved(before, after, args.min_round_improvement)
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
        memo.update(
            {
                "after": after.__dict__,
                "after_gaps": after_shape.__dict__,
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
                "next_input_root": str(clean_root if accepted else current_root),
                "next_scores_csv": str(
                    eval_dir / "revised_generated_thread_scores.csv"
                    if accepted
                    else current_scores
                ),
                "next_matched_eval_dir": str(matched_dir if accepted else current_matched),
                "rollback_to_root": None if accepted else str(current_root),
            }
        )
        history.append(memo)
        card_controller.write_history(args.output_prefix, history)
        persist_memory(
            memory_file,
            merge_strategy_history(args.strategy_history_json, history),
            controller="selfbert",
            target_metrics=(TARGET_METRIC,),
            current_input_root=clean_root if accepted else current_root,
        )

        if accepted:
            current_root = clean_root
            current_scores = eval_dir / "revised_generated_thread_scores.csv"
            current_matched = matched_dir
            final_root = clean_root
            final_matched = matched_dir
            print(
                f"[selfbert-controller-accept] round={round_idx} next_input={current_root}",
                flush=True,
            )
        else:
            print(
                f"[selfbert-controller-reject] round={round_idx} rollback_to={current_root}",
                flush=True,
            )
            if args.continue_after_reject and round_idx < args.max_rounds:
                continue
            break
        if card_controller.metric_passes(after, args.pass_threshold):
            print("[selfbert-controller-stop] target passes after accepted round", flush=True)
            break

    if not args.dry_run:
        card_controller.write_history(args.output_prefix, history)
        persist_memory(
            memory_file,
            merge_strategy_history(args.strategy_history_json, history),
            controller="selfbert",
            target_metrics=(TARGET_METRIC,),
            current_input_root=current_root,
        )
    print(f"[selfbert-controller-done] final_root={final_root}", flush=True)
    print(f"[selfbert-controller-done] final_matched_eval={final_matched}", flush=True)


def build_reviser_command(
    *,
    args: argparse.Namespace,
    profile: str,
    input_root: Path,
    scores_csv: Path,
    matched_eval_dir: Path,
    output_root: Path,
    direction: str = "decrease",
    controller_memory_json: Path | None = None,
) -> list[str]:
    params = profile_params(profile)
    command = [
        sys.executable,
        str(args.reviser_script),
        str(input_root),
        "--scores-csv",
        str(scores_csv),
        "--real-scores-csv",
        str(matched_eval_dir / "matched_real_thread_scores.csv"),
        "--seed-post-pool-json",
        str(args.seed_post_pool_json),
        "--output-dir",
        str(output_root),
        "--model",
        args.model,
        "--api-key",
        args.api_key or "",
        "--base-url",
        args.base_url,
        "--exact-selfbert-gate",
        "--direction",
        direction,
        "--exact-target-selection",
        "--selfbert-device",
        args.device,
        "--preserve-comment-metadata",
        "--deviation-driven-coverage",
        "--resume-existing",
    ]
    if controller_memory_json is not None:
        command.extend(["--controller-memory-json", str(controller_memory_json)])
    for key, value in params.items():
        command.extend([f"--{key}", str(value)])
    return command


def profile_params(profile: str) -> dict[str, Any]:
    if profile == "high_tail":
        return {
            "min-selfbert-excess": 0.018,
            "candidates-per-comment": 7,
            "metric-min-thread-selfbert-gain": 0.00045,
            "metric-max-pair-selfbert-drop": 0.12,
            "metric-max-real-undershoot": 0.018,
            "target-selfbert-excess": 0.012,
        }
    if profile == "middle_mass":
        return {
            "min-selfbert-excess": 0.009,
            "candidates-per-comment": 8,
            "metric-min-thread-selfbert-gain": 0.00030,
            "metric-max-pair-selfbert-drop": 0.10,
            "metric-max-real-undershoot": 0.014,
            "target-selfbert-excess": 0.006,
        }
    return {
        "min-selfbert-excess": 0.004,
        "candidates-per-comment": 9,
        "metric-min-thread-selfbert-gain": 0.00020,
        "metric-max-pair-selfbert-drop": 0.08,
        "metric-max-real-undershoot": 0.010,
        "target-selfbert-excess": 0.003,
    }


def run_self_test() -> None:
    middle = profile_params("middle_mass")
    high = profile_params("high_tail")
    safe = profile_params("shape_safe")
    assert safe["min-selfbert-excess"] < middle["min-selfbert-excess"]
    assert middle["candidates-per-comment"] > high["candidates-per-comment"]
    print("selfbert controller self-test passed")


if __name__ == "__main__":
    main()
