#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from generalized_card.domain import REPO_ROOT  # noqa: E402
from generalized_card.core_contract import (  # noqa: E402
    CORE_POLICY_VERSION,
    upgrade_revision_policy_config,
    verify_core_contract,
    verify_revision_policy,
)

SCRIPTS_ROOT = REPO_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from revision_memory import (  # noqa: E402
    completed_history,
    merge_strategy_history,
    write_json_atomic,
)


DEFAULT_PRICES = {
    "gpt-5.4-mini": (0.75, 0.075, 4.50),
    "gpt-4o-mini": (0.15, 0.075, 0.60),
}

SELF_BLEU_PROTECTED_METRICS = (
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one generalized CARD self-loop revision stage.")
    parser.add_argument("--tag", required=True)
    parser.add_argument(
        "--stage",
        choices=[
            "diversity",
            "selfbert",
            "semantic",
            "tone",
            "emotion",
            "length",
            "story",
            "structure",
            "story-structure",
        ],
        required=True,
    )
    parser.add_argument("--model", default="")
    parser.add_argument("--base-url", default="")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--max-rounds", type=int, default=7)
    parser.add_argument(
        "--rounds-this-call",
        type=int,
        default=0,
        help="Limit new attempted rounds in this invocation; 0 uses the remaining stage budget.",
    )
    parser.add_argument("--story-rounds", type=int, default=4)
    parser.add_argument("--structure-rounds", type=int, default=3)
    parser.add_argument("--metric-parallel", type=int, default=5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--price-input-per-1m", type=float)
    parser.add_argument("--price-cached-input-per-1m", type=float)
    parser.add_argument("--price-output-per-1m", type=float)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--upgrade-revision-policy",
        action="store_true",
        help="Upgrade only reviser lineage; generator lineage remains unchanged.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.max_rounds <= 0:
        raise SystemExit("--max-rounds must be positive")
    if args.stage == "story":
        args.structure_rounds = 0
    elif args.stage == "structure":
        args.story_rounds = 0
    if _is_story_structure_stage(args.stage) and args.story_rounds + args.structure_rounds > args.max_rounds:
        raise SystemExit("--story-rounds + --structure-rounds cannot exceed --max-rounds")

    run_root = REPO_ROOT / "artifacts" / "generalized_card" / "runs" / args.tag
    config_path = run_root / "run_config.json"
    config = _load_json(config_path)
    if not config:
        raise SystemExit(f"Run config not found: {config_path}")
    if args.upgrade_revision_policy:
        config = upgrade_revision_policy_config(config)
        _write_json(config_path, config)
    revision_policy = verify_revision_policy(
        config,
        operation=f"run {args.stage} revision",
    )
    artifact_path = run_root / "current_artifact.json"
    current = _load_json(artifact_path)
    generated_root = Path(current.get("root") or run_root / "cleaned")
    scores = Path(current.get("scores") or run_root / "evaluation" / "revised_generated_thread_scores.csv")
    matched = Path(current.get("matched") or run_root / "matched_evaluation")
    for path in (generated_root, scores, matched / "matched_seed_group_eval.json"):
        if not path.exists():
            raise SystemExit(f"Evaluation input missing; run run_evaluate.py first: {path}")

    model = args.model or str(config["model"])
    base_url = args.base_url or str(config["base_url"])
    api_key = os.environ.get(args.api_key_env, "").strip()
    if not api_key and not args.dry_run:
        raise SystemExit(f"API key is missing: environment variable {args.api_key_env}")
    core_names = {
        "diversity": (
            "selfbleu_controller",
            "generalized_selfbleu_controller",
            "selfbleu_reviser",
            "selfbleu_backend",
            "revision_memory",
        ),
        "selfbert": (
            "selfbert_controller",
            "selfbert_reviser",
            "selfbert_backend",
            "revision_memory",
        ),
        "semantic": ("text_metric_controller", "text_metric_reviser", "revision_memory"),
        "emotion": ("text_metric_controller", "text_metric_reviser", "revision_memory"),
        "length": ("text_metric_controller", "text_metric_reviser", "revision_memory"),
        "tone": ("tone_controller", "tone_reviser", "tone_backend", "revision_memory"),
        "story": (
            "story_structure_controller",
            "story_reviser",
            "story_backend",
            "structure_reviser",
            "structure_backend",
            "revision_memory",
        ),
        "structure": (
            "story_structure_controller",
            "story_reviser",
            "story_backend",
            "structure_reviser",
            "structure_backend",
            "revision_memory",
        ),
        "story-structure": (
            "story_structure_controller",
            "story_reviser",
            "story_backend",
            "structure_reviser",
            "structure_backend",
            "revision_memory",
        ),
    }.get(args.stage, ())
    core_names = (
        "revision_stage_runner",
        "reviser_adapter",
        "domain_prompt_adapter",
        "distribution_diagnostics",
        "cleanup",
        "score_runner",
        "matched_evaluator",
        *core_names,
    )
    verify_core_contract(core_names)
    attempts_by_stage = dict(current.get("revision_attempts") or {})
    prior_attempts = int(attempts_by_stage.get(args.stage) or 0)
    controller_args = copy.copy(args)
    remaining = args.max_rounds - prior_attempts
    if remaining <= 0:
        raise SystemExit(
            f"{args.stage} already used {prior_attempts}/{args.max_rounds} rounds; "
            "the accepted artifact remains in current_artifact.json"
        )
    call_budget = (
        min(remaining, args.rounds_this_call)
        if args.rounds_this_call > 0
        else remaining
    )
    if not _is_story_structure_stage(args.stage):
        controller_args.max_rounds = (
            min(remaining, args.rounds_this_call)
            if args.rounds_this_call > 0
            else remaining
        )
    else:
        controller_args.max_rounds = call_budget
        if args.stage == "story":
            controller_args.story_rounds = min(args.story_rounds, call_budget)
            controller_args.structure_rounds = 0
        elif args.stage == "structure":
            controller_args.story_rounds = 0
            controller_args.structure_rounds = min(args.structure_rounds, call_budget)
        else:
            controller_args.story_rounds = min(args.story_rounds, call_budget)
            controller_args.structure_rounds = min(
                args.structure_rounds,
                call_budget - controller_args.story_rounds,
            )
    lineage = _artifact_lineage(generated_root, scores, matched, revision_policy)
    if _is_story_structure_stage(args.stage) and args.rounds_this_call > 0:
        if args.stage == "structure":
            controller_args.story_rounds = 0
            controller_args.structure_rounds = 1
        elif args.story_rounds > 0:
            controller_args.story_rounds = 1
            controller_args.structure_rounds = 0
        elif args.structure_rounds > 0:
            controller_args.story_rounds = 0
            controller_args.structure_rounds = 1
        controller_args.max_rounds = 1
    prefix_suffix = f"{lineage}_a{prior_attempts:02d}"
    prefix = run_root / "revisions" / f"{args.stage.replace('-', '_')}_{prefix_suffix}"
    strategy_history = (
        run_root
        / "revisions"
        / f"{args.stage.replace('-', '_')}_strategy_history.json"
    )
    command = _controller_command(
        args=controller_args,
        config=config,
        generated_root=generated_root,
        scores=scores,
        matched=matched,
        model=model,
        base_url=base_url,
        prefix=prefix,
        strategy_history=strategy_history,
    )
    print("[revision-command] " + " ".join(_redact(command)), flush=True)
    if args.dry_run:
        return

    env = os.environ.copy()
    env["GENERALIZED_CARD_DOMAIN"] = str(config.get("domain_config") or config["domain"]["domain_id"])
    env["OPENAI_API_KEY"] = api_key
    env["PLANNER_API_KEY"] = api_key
    env["LLM_API_KEY"] = api_key
    env["TOKEN_USAGE_LOG_JSONL"] = str(run_root / "logs" / "token_usage.jsonl")
    env["TOKEN_USAGE_RUN_TAG"] = str(config["tag"])
    _set_prices(env, args, model)
    state_path = run_root / "run_state.json"
    state = _load_json(state_path)
    prior_elapsed = float(state.get("elapsed_seconds") or 0.0)
    started = time.monotonic()
    return_code = 1
    status = f"{args.stage}_failed"
    try:
        completed = subprocess.run(command, cwd=REPO_ROOT, env=env, check=False)
        return_code = int(completed.returncode)
        status = f"{args.stage}_complete" if return_code == 0 else status
    except KeyboardInterrupt:
        return_code = 130
        status = f"{args.stage}_interrupted"
        print("[interrupted] resuming later will start from the latest accepted round", flush=True)
    finally:
        fallback = {"root": str(generated_root), "scores": str(scores), "matched": str(matched)}
        history_rows = _controller_history(args.stage, prefix)
        completed_rows = completed_history(history_rows)
        attempts_used = len(completed_rows)
        if return_code == 0:
            write_json_atomic(
                strategy_history,
                merge_strategy_history(strategy_history, completed_rows),
            )
            final = _resolve_final_artifact(
                stage=args.stage,
                prefix=prefix,
                fallback=fallback,
            )
            attempts_by_stage[args.stage] = prior_attempts + attempts_used
            final.update(
                {
                    "stage": args.stage,
                    "previous": fallback,
                    "revision_attempts": attempts_by_stage,
                    "controller_prefix": str(prefix),
                    "strategy_history": str(strategy_history),
                    "updated_at_epoch": time.time(),
                }
            )
            _write_json(artifact_path, final)
        else:
            # Keep the exact starting artifact and prefix after interruption.
            # The reviser report and derived outputs under this prefix are
            # resumable, so advancing current_artifact here would abandon the
            # partially completed round and create a different prefix.
            print(
                f"[revision-resume-boundary] current_artifact unchanged; "
                f"reuse_prefix={prefix} completed_rounds={attempts_used}",
                flush=True,
            )
        elapsed = prior_elapsed + (time.monotonic() - started)
        state.update(
            {
                "status": status,
                "return_code": return_code,
                "elapsed_seconds": elapsed,
                "active_revision_prefix": str(prefix) if return_code else None,
                "updated_at_epoch": time.time(),
            }
        )
        _write_json(state_path, state)
        _summarize(run_root, elapsed, env)
    if return_code:
        raise SystemExit(return_code)


def _controller_command(
    *,
    args: argparse.Namespace,
    config: dict[str, Any],
    generated_root: Path,
    scores: Path,
    matched: Path,
    model: str,
    base_url: str,
    prefix: Path,
    strategy_history: Path | None = None,
) -> list[str]:
    common = [
        str(generated_root),
        "--scores-csv", str(scores),
        "--matched-eval-dir", str(matched),
        "--seed-post-pool-json", str(config["seed_pool"]),
        "--real-scores-csv", str(config["domain"]["real_scores_csv"]),
        "--output-prefix", str(prefix),
        "--model", model,
        "--base-url", base_url,
        "--device", args.device,
        "--metric-parallel", str(args.metric_parallel),
        "--expected-seeds", str(config["max_posts"]),
        "--posts-per-run", str(config["posts_per_run"]),
    ]
    if strategy_history is not None:
        common.extend(["--strategy-history-json", str(strategy_history)])
    if args.stage == "diversity":
        command = [
            sys.executable,
            str(PACKAGE_ROOT / "scripts" / "run_selfbleu_revision_controller.py"),
            *common,
            "--max-rounds", str(args.max_rounds),
            "--target-metric", "self_bleu_4",
            "--protected-metrics", ",".join(SELF_BLEU_PROTECTED_METRICS),
            "--min-round-improvement", "0.001",
            "--playbook",
            str(PACKAGE_ROOT / "REVISION_PLAYBOOK.md"),
            "--reviser-script", str(PACKAGE_ROOT / "scripts" / "run_selfbleu_reviser_backend.py"),
            "--continue-after-reject",
            "--unbounded-coverage",
            "--verbose-candidates",
        ]
        return command
    if args.stage == "selfbert":
        command = [
            sys.executable,
            str(PACKAGE_ROOT / "scripts" / "run_selfbert_revision_controller.py"),
            *common,
            "--max-rounds", str(args.max_rounds),
            "--reviser-script", str(PACKAGE_ROOT / "scripts" / "run_selfbert_reviser_backend.py"),
            "--continue-after-reject",
        ]
        return command
    if args.stage == "tone":
        command = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "run_tone_revision_controller.py"),
            *common,
            "--max-rounds", str(args.max_rounds),
            "--tone-reviser-script", str(PACKAGE_ROOT / "scripts" / "run_tone_reviser_backend.py"),
            "--protected-quality-drop-tolerance", "0.01",
            "--min-round-improvement", "0.005",
            "--continue-after-reject",
            "--unbounded-coverage",
        ]
        return command
    text_metric = {
        "semantic": "semantic_mean_cosine",
        "emotion": "emotion_entropy",
        "length": "length_cv",
    }.get(args.stage)
    if text_metric:
        command = [
            sys.executable,
            str(PACKAGE_ROOT / "scripts" / "run_text_metric_revision_controller.py"),
            *common,
            "--max-rounds", str(args.max_rounds),
            "--target-metric", text_metric,
            "--reviser-script", str(PACKAGE_ROOT / "scripts" / "run_text_metric_reviser.py"),
            "--continue-after-reject",
        ]
        return command

    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "run_story_structure_revision_controller.py"),
        *common,
        "--max-story-rounds", str(args.story_rounds),
        "--max-structure-rounds", str(args.structure_rounds),
        "--story-reviser-script", str(PACKAGE_ROOT / "scripts" / "run_story_reviser_backend.py"),
        "--structure-reviser-script", str(PACKAGE_ROOT / "scripts" / "run_structure_reviser_backend.py"),
        "--protected-quality-drop-tolerance", "0.01",
        "--deviation-driven-coverage",
    ]
    if args.resume:
        command.append("--resume-existing")
    return command


def _set_prices(env: dict[str, str], args: argparse.Namespace, model: str) -> None:
    defaults = DEFAULT_PRICES.get(model.lower())
    values = (
        args.price_input_per_1m if args.price_input_per_1m is not None else defaults[0] if defaults else None,
        args.price_cached_input_per_1m if args.price_cached_input_per_1m is not None else defaults[1] if defaults else None,
        args.price_output_per_1m if args.price_output_per_1m is not None else defaults[2] if defaults else None,
    )
    for key, value in zip(
        ("TOKEN_PRICE_INPUT_PER_1M", "TOKEN_PRICE_CACHED_INPUT_PER_1M", "TOKEN_PRICE_OUTPUT_PER_1M"),
        values,
    ):
        if value is not None:
            env[key] = str(value)


def _summarize(run_root: Path, elapsed: float, env: dict[str, str]) -> None:
    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "summarize_token_usage.py"),
            str(run_root / "logs" / "token_usage.jsonl"),
            "--output", str(run_root / "logs" / "token_usage_summary.json"),
            "--elapsed-seconds", str(elapsed),
        ],
        cwd=REPO_ROOT,
        env=env,
        check=False,
    )


def _resolve_final_artifact(
    *,
    stage: str,
    prefix: Path,
    fallback: dict[str, str],
) -> dict[str, str]:
    if _is_story_structure_stage(stage):
        state = _load_json(Path(f"{prefix}_controller_state.json"))
        if state.get("current_root") and state.get("current_scores") and state.get("current_matched"):
            return {
                "root": str(state["current_root"]),
                "scores": str(state["current_scores"]),
                "matched": str(state["current_matched"]),
                "controller_state": str(Path(f"{prefix}_controller_state.json")),
                "controller_history": str(Path(f"{prefix}_controller_history.json")),
                "controller_memory": str(Path(f"{prefix}_controller_memory.json")),
            }
        return fallback

    history_path = (
        Path(f"{prefix}_tone_controller_history.json")
        if stage == "tone"
        else Path(f"{prefix}_controller_history.json")
    )
    history = _load_json_list(history_path)
    memory_path = (
        Path(f"{prefix}_tone_controller_memory.json")
        if stage == "tone"
        else Path(f"{prefix}_controller_memory.json")
    )
    accepted = [
        row for row in history
        if bool(row.get("accepted_round"))
        or (bool(row.get("improved")) and bool(row.get("protected_ok")))
    ]
    if not accepted:
        return {
            **fallback,
            "controller_history": str(history_path),
            "controller_memory": str(memory_path),
        }
    row = accepted[-1]
    root = str(row.get("next_input_root") or row.get("clean_root") or fallback["root"])
    score_value = row.get("next_scores_csv")
    if not score_value and row.get("eval_dir"):
        score_value = Path(str(row["eval_dir"])) / "revised_generated_thread_scores.csv"
    scores = str(score_value or fallback["scores"])
    matched = str(row.get("next_matched_eval_dir") or row.get("matched_eval_dir") or fallback["matched"])
    return {
        "root": root,
        "scores": scores,
        "matched": matched,
        "controller_history": str(history_path),
        "controller_memory": str(memory_path),
    }


def _artifact_lineage(
    root: Path,
    scores: Path,
    matched: Path,
    policy_version: str = CORE_POLICY_VERSION,
) -> str:
    value = "\n".join(
        [policy_version, *(str(path.resolve()) for path in (root, scores, matched))]
    )
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:10]


def _is_story_structure_stage(stage: str) -> bool:
    return stage in {"story", "structure", "story-structure"}


def _controller_history(stage: str, prefix: Path) -> list[dict[str, Any]]:
    path = (
        Path(f"{prefix}_tone_controller_history.json")
        if stage == "tone"
        else Path(f"{prefix}_controller_history.json")
    )
    return _load_json_list(path)


def _redact(command: list[str]) -> list[str]:
    output = list(command)
    for index, token in enumerate(output[:-1]):
        if token == "--api-key":
            output[index + 1] = "[REDACTED]"
    return output


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _load_json_list(path: Path) -> list[dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    return [row for row in payload if isinstance(row, dict)] if isinstance(payload, list) else []


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


if __name__ == "__main__":
    main()
