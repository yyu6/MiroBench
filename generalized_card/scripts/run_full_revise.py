#!/usr/bin/env python3
"""Run generalized CARD revisers in the same accepted-artifact chain as CARD."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from generalized_card.domain import REPO_ROOT  # noqa: E402
from generalized_card.core_contract import (  # noqa: E402
    upgrade_revision_policy_config,
    verify_revision_policy,
)


TONE_METRICS = (
    "hard_disagree_rate",
    "polite_rate",
    "impolite_rate",
    "neutral_rate",
)
STRUCTURE_METRICS = ("avg_depth", "structural_virality")
ALL_METRICS = (
    "self_bleu_4",
    "self_bertscore_mean_f1",
    "semantic_mean_cosine",
    *TONE_METRICS,
    "length_cv",
    *STRUCTURE_METRICS,
    "mean_story_probability",
    "emotion_entropy",
)
DIRECT_TARGETS = {
    "self_bleu_4",
    "self_bertscore_mean_f1",
    *TONE_METRICS,
    *STRUCTURE_METRICS,
    "mean_story_probability",
    "semantic_mean_cosine",
    "length_cv",
    "emotion_entropy",
}
CARD_CORE_STAGES = ("diversity", "tone")
EXTENDED_STAGES = (
    "diversity",
    "selfbert",
    "semantic",
    "tone",
    "emotion",
    "length",
    "story",
    "structure",
)
REQUIRED_EXTENDED_METRICS = (
    "self_bleu_4",
    "self_bertscore_mean_f1",
    "semantic_mean_cosine",
    "mean_story_probability",
    "emotion_entropy",
)


@dataclass(frozen=True)
class StageDecision:
    stage: str
    required: bool
    reason: str
    max_rounds: int
    story_rounds: int = 0
    structure_rounds: int = 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--model", default="")
    parser.add_argument("--base-url", default="")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument(
        "--revision-profile",
        choices=("card-core", "extended"),
        default="card-core",
        help=(
            "card-core runs the CARD Self-BLEU -> Tone metric order; extended "
            "adds generalized exact-metric stages."
        ),
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=7,
        help="Global attempted-round budget shared by all revision stages.",
    )
    parser.add_argument("--selfbleu-rounds", type=int, default=7)
    parser.add_argument("--selfbert-rounds", type=int, default=7)
    parser.add_argument("--semantic-rounds", type=int, default=7)
    parser.add_argument("--tone-rounds", type=int, default=7)
    parser.add_argument("--emotion-rounds", type=int, default=7)
    parser.add_argument("--length-rounds", type=int, default=7)
    parser.add_argument("--story-rounds", type=int, default=7)
    parser.add_argument("--structure-rounds", type=int, default=7)
    parser.add_argument("--pass-threshold", type=float, default=0.05)
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
    validate_rounds(args)
    run_root = REPO_ROOT / "artifacts" / "generalized_card" / "runs" / args.tag
    config_path = run_root / "run_config.json"
    config = load_json(config_path)
    if not config:
        raise SystemExit(f"Run config not found: {config_path}")
    if args.upgrade_revision_policy:
        config = upgrade_revision_policy_config(config)
        write_json(config_path, config)
    verify_revision_policy(config, operation="run the full revision chain")
    artifact_path = run_root / "current_artifact.json"
    if not artifact_path.exists():
        raise SystemExit("Run evaluation before revision: current_artifact.json is missing")

    model = args.model or str(config["model"])
    base_url = args.base_url or str(config["base_url"])
    api_key = os.environ.get(args.api_key_env, "").strip()
    if not api_key and not args.dry_run:
        raise SystemExit(f"API key is missing: {args.api_key_env}")

    history_path = run_root / (
        "full_revision_dry_run.json" if args.dry_run else "full_revision_history.json"
    )
    history = [] if args.dry_run else load_json_list(history_path)
    env = os.environ.copy()
    if api_key:
        env["OPENAI_API_KEY"] = api_key
        env["PLANNER_API_KEY"] = api_key
        env["LLM_API_KEY"] = api_key

    stages = revision_stages(args.revision_profile)
    stalled_stages: set[str] = set()
    while True:
        artifact = load_json(artifact_path)
        global_attempts = sum(
            int(value or 0)
            for value in (artifact.get("revision_attempts") or {}).values()
        )
        if global_attempts >= args.max_rounds:
            print(
                f"[full-controller-budget] global attempted rounds exhausted: "
                f"{global_attempts}/{args.max_rounds}",
                flush=True,
            )
            break
        executed_stage = False
        attempts_by_stage = {
            str(key): int(value or 0)
            for key, value in (artifact.get("revision_attempts") or {}).items()
        }
        # Preserve CARD's metric order for the first proposal, then give every
        # still-required stage one proposal before any stage receives another.
        # This is the generalized cross-metric scheduler; each local reviser
        # and its round-level acceptance gate remain unchanged.
        ordered_stages = coverage_order(stages, attempts_by_stage)
        for stage_name in ordered_stages:
            artifact = load_json(artifact_path)
            evaluation = load_current_evaluation(artifact)
            before_summary = metric_summary(evaluation, args.pass_threshold)
            decision = stage_decision(stage_name, evaluation, args)
            attempts = int((artifact.get("revision_attempts") or {}).get(stage_name) or 0)
            record: dict[str, Any] = {
                "stage": stage_name,
                "revision_profile": args.revision_profile,
                "checked_at_epoch": time.time(),
                "input_root": artifact.get("root"),
                "input_matched": artifact.get("matched"),
                "attempts_before": attempts,
                "global_round": global_attempts + 1,
                "global_attempts_before": global_attempts,
                "global_max_rounds": args.max_rounds,
                "required": decision.required,
                "reason": decision.reason,
                "before": before_summary,
            }
            print(
                f"[full-controller-stage] stage={stage_name} required={decision.required} "
                f"attempts={attempts}/{decision.max_rounds} reason={decision.reason}",
                flush=True,
            )
            if stage_name in stalled_stages:
                record["decision"] = "skipped_stalled_no_attempt"
                history.append(record)
                write_json(history_path, history)
                continue
            if not decision.required:
                record["decision"] = (
                    "skipped_monitor_only"
                    if "monitor_only" in decision.reason
                    else "skipped_pass"
                )
                history.append(record)
                write_json(history_path, history)
                continue
            if attempts >= decision.max_rounds:
                record["decision"] = "skipped_budget_exhausted"
                history.append(record)
                write_json(history_path, history)
                print(
                    f"[full-controller-budget] stage={stage_name} exhausted; "
                    "keeping the latest accepted artifact",
                    flush=True,
                )
                continue

            command = build_stage_command(
                args=args,
                stage=decision,
                model=model,
                base_url=base_url,
            )
            record["command"] = command
            print("[full-controller-command] " + " ".join(command), flush=True)
            if args.dry_run:
                record["decision"] = "dry_run"
                history.append(record)
                write_json(history_path, history)
                continue
            try:
                completed = subprocess.run(command, cwd=REPO_ROOT, env=env, check=False)
            except KeyboardInterrupt:
                record["decision"] = "interrupted"
                history.append(record)
                write_json(history_path, history)
                print(
                    "[full-controller-interrupted] latest accepted artifact and partial round remain resumable",
                    flush=True,
                )
                raise SystemExit(130) from None
            record["return_code"] = int(completed.returncode)
            if completed.returncode:
                record["decision"] = "failed_or_interrupted"
                history.append(record)
                write_json(history_path, history)
                raise SystemExit(completed.returncode)

            updated_artifact = load_json(artifact_path)
            updated_eval = load_current_evaluation(updated_artifact)
            after_summary = metric_summary(updated_eval, args.pass_threshold)
            observed_pass_regressions = passing_metric_regressions(
                before_summary,
                after_summary,
            )
            attempts_after = int(
                (updated_artifact.get("revision_attempts") or {}).get(stage_name) or 0
            )
            if attempts_after <= attempts:
                stalled_stages.add(stage_name)
                record.update(
                    {
                        "decision": "stalled_no_attempt",
                        "output_root": updated_artifact.get("root"),
                        "output_matched": updated_artifact.get("matched"),
                        "attempts_after": attempts_after,
                        "after": after_summary,
                    }
                )
                history.append(record)
                write_json(history_path, history)
                print(
                    f"[full-controller-stalled] stage={stage_name} attempts={attempts_after}",
                    flush=True,
                )
                executed_stage = True
                break
            # The stage controller is the sole acceptance authority. A second,
            # stricter outer gate would change CARD's accepted-round chain.
            record.update(
                {
                    "decision": "completed",
                    "output_root": updated_artifact.get("root"),
                    "output_matched": updated_artifact.get("matched"),
                    "attempts_after": attempts_after,
                    "after": after_summary,
                    "observed_pass_regressions": observed_pass_regressions,
                }
            )
            history.append(record)
            write_json(history_path, history)
            executed_stage = True
            # Every proposal changes history or the accepted artifact. Restart
            # from the first metric group and make decisions from fresh scores.
            break

        if args.dry_run or not executed_stage:
            break

    final_artifact = load_json(artifact_path)
    final_eval = load_current_evaluation(final_artifact)
    summary = metric_summary(final_eval, args.pass_threshold)
    pass_count = sum(1 for row in summary.values() if row["status"] == "PASS")
    unsupported = monitor_only_metrics(summary)
    print(
        f"[full-controller-done] profile={args.revision_profile} "
        f"passes={pass_count}/{len(ALL_METRICS)} "
        f"final_root={final_artifact.get('root')}",
        flush=True,
    )
    if unsupported:
        print(
            "[full-controller-monitor-only] nonpassing metrics without a claim-safe direct rewriter: "
            + ",".join(unsupported),
            flush=True,
        )
    print(f"[full-controller-history] {history_path}", flush=True)
    required_nonpass = required_nonpassing_metrics(summary)
    if args.revision_profile == "extended" and required_nonpass:
        print(
            "[full-controller-no-fail] FAILED required metrics="
            + ",".join(required_nonpass),
            flush=True,
        )
        raise SystemExit(2)
    if args.revision_profile == "extended":
        print(
            "[full-controller-no-fail] PASS required metrics="
            + ",".join(REQUIRED_EXTENDED_METRICS),
            flush=True,
        )


def revision_stages(profile: str) -> tuple[str, ...]:
    if profile == "card-core":
        return CARD_CORE_STAGES
    if profile == "extended":
        return EXTENDED_STAGES
    raise ValueError(f"Unknown revision profile: {profile}")


def coverage_order(
    stages: tuple[str, ...],
    attempts_by_stage: dict[str, int],
) -> tuple[str, ...]:
    """Order stages by prior proposals, retaining CARD order for ties."""

    order = {stage: index for index, stage in enumerate(stages)}
    return tuple(
        sorted(
            stages,
            key=lambda stage: (int(attempts_by_stage.get(stage) or 0), order[stage]),
        )
    )


def validate_rounds(args: argparse.Namespace) -> None:
    if args.max_rounds <= 0:
        raise SystemExit("--max-rounds must be positive")
    for name in (
        "selfbleu_rounds",
        "selfbert_rounds",
        "semantic_rounds",
        "tone_rounds",
        "emotion_rounds",
        "length_rounds",
        "story_rounds",
        "structure_rounds",
    ):
        if int(getattr(args, name)) < 0:
            raise SystemExit(f"--{name.replace('_', '-')} cannot be negative")


def stage_decision(
    stage: str,
    evaluation: dict[str, Any],
    args: argparse.Namespace,
) -> StageDecision:
    if stage == "diversity":
        failed = not metric_passes(evaluation, "self_bleu_4", args.pass_threshold)
        row = evaluation.get("self_bleu_4") or {}
        direction_supported = number(row.get("generated_mean")) > number(row.get("real_mean"))
        required = failed and direction_supported
        reason = metric_reason(evaluation, ("self_bleu_4",), args.pass_threshold)
        if failed and not direction_supported:
            reason += "; generated_below_real_is_monitor_only"
        return StageDecision(stage, required, reason, args.selfbleu_rounds)
    if stage == "selfbert":
        failed = not metric_passes(evaluation, "self_bertscore_mean_f1", args.pass_threshold)
        reason = metric_reason(evaluation, ("self_bertscore_mean_f1",), args.pass_threshold)
        return StageDecision(stage, failed, reason, args.selfbert_rounds)
    if stage == "semantic":
        failed = not metric_passes(evaluation, "semantic_mean_cosine", args.pass_threshold)
        return StageDecision(
            stage,
            failed,
            metric_reason(evaluation, ("semantic_mean_cosine",), args.pass_threshold),
            args.semantic_rounds,
        )
    if stage == "tone":
        failed = any(not metric_passes(evaluation, metric, args.pass_threshold) for metric in TONE_METRICS)
        return StageDecision(stage, failed, metric_reason(evaluation, TONE_METRICS, args.pass_threshold), args.tone_rounds)
    if stage == "emotion":
        failed = not metric_passes(evaluation, "emotion_entropy", args.pass_threshold)
        return StageDecision(
            stage,
            failed,
            metric_reason(evaluation, ("emotion_entropy",), args.pass_threshold),
            args.emotion_rounds,
        )
    if stage == "length":
        failed = not metric_passes(evaluation, "length_cv", args.pass_threshold)
        return StageDecision(
            stage,
            failed,
            metric_reason(evaluation, ("length_cv",), args.pass_threshold),
            args.length_rounds,
        )
    if stage == "story":
        failed = not metric_passes(
            evaluation,
            "mean_story_probability",
            args.pass_threshold,
        )
        reason = metric_reason(
            evaluation,
            ("mean_story_probability",),
            args.pass_threshold,
        )
        return StageDecision(
            stage,
            failed,
            reason,
            args.story_rounds,
            story_rounds=args.story_rounds if failed else 0,
        )
    if stage == "structure":
        supported = [
            metric
            for metric in STRUCTURE_METRICS
            if not metric_passes(evaluation, metric, args.pass_threshold)
            and number((evaluation.get(metric) or {}).get("generated_mean"))
            < number((evaluation.get(metric) or {}).get("real_mean"))
        ]
        unsupported = [
            metric
            for metric in STRUCTURE_METRICS
            if not metric_passes(evaluation, metric, args.pass_threshold)
            and metric not in supported
        ]
        reason = metric_reason(evaluation, STRUCTURE_METRICS, args.pass_threshold)
        if unsupported:
            reason += "; structure_generated_above_real_is_monitor_only=" + "+".join(
                unsupported
            )
        return StageDecision(
            stage,
            bool(supported),
            reason,
            args.structure_rounds,
            structure_rounds=args.structure_rounds if supported else 0,
        )
    if stage == "story-structure":
        # Compatibility path for callers that explicitly request the older
        # combined stage. The full generalized scheduler uses separate stages.
        story = stage_decision("story", evaluation, args)
        structure = stage_decision("structure", evaluation, args)
        return StageDecision(
            stage,
            story.required or structure.required,
            story.reason + "; " + structure.reason,
            story.max_rounds + structure.max_rounds,
            story_rounds=story.story_rounds,
            structure_rounds=structure.structure_rounds,
        )
    raise ValueError(f"Unknown revision stage: {stage}")


def build_stage_command(
    *,
    args: argparse.Namespace,
    stage: StageDecision,
    model: str,
    base_url: str,
) -> list[str]:
    command = [
        sys.executable,
        str(PACKAGE_ROOT / "scripts" / "run_revise.py"),
        "--tag",
        args.tag,
        "--stage",
        stage.stage,
        "--model",
        model,
        "--base-url",
        base_url,
        "--api-key-env",
        args.api_key_env,
        "--max-rounds",
        str(stage.max_rounds),
        "--rounds-this-call",
        "1",
        "--metric-parallel",
        str(args.metric_parallel),
        "--device",
        args.device,
        "--resume",
    ]
    if stage.stage in {"story", "structure", "story-structure"}:
        command.extend(
            [
                "--story-rounds",
                str(stage.story_rounds),
                "--structure-rounds",
                str(stage.structure_rounds),
            ]
        )
    for flag, value in (
        ("--price-input-per-1m", args.price_input_per_1m),
        ("--price-cached-input-per-1m", args.price_cached_input_per_1m),
        ("--price-output-per-1m", args.price_output_per_1m),
    ):
        if value is not None:
            command.extend([flag, str(value)])
    if args.dry_run:
        command.append("--dry-run")
    return command


def load_current_evaluation(artifact: dict[str, Any]) -> dict[str, Any]:
    matched = Path(str(artifact.get("matched") or ""))
    path = matched / "matched_seed_group_eval.json"
    if not path.exists():
        raise SystemExit(f"Current matched evaluation is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"Invalid matched evaluation: {path}")
    return payload


def metric_passes(evaluation: dict[str, Any], metric: str, threshold: float) -> bool:
    row = evaluation.get(metric) or {}
    return number(row.get("mwu_p_value")) > threshold and number(row.get("ks_p_value")) > threshold


def metric_reason(
    evaluation: dict[str, Any],
    metrics: tuple[str, ...],
    threshold: float,
) -> str:
    parts = []
    for metric in metrics:
        row = evaluation.get(metric) or {}
        status = "PASS" if metric_passes(evaluation, metric, threshold) else "NONPASS"
        parts.append(
            f"{metric}:{status}:mwu={number(row.get('mwu_p_value')):.5g}:"
            f"ks={number(row.get('ks_p_value')):.5g}:"
            f"mean_gap={number(row.get('generated_mean')) - number(row.get('real_mean')):+.5g}"
        )
    return ",".join(parts)


def metric_summary(evaluation: dict[str, Any], threshold: float) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for metric in ALL_METRICS:
        row = evaluation.get(metric) or {}
        mwu = number(row.get("mwu_p_value"))
        ks = number(row.get("ks_p_value"))
        status = "PASS" if mwu > threshold and ks > threshold else "PARTIAL" if mwu > threshold or ks > threshold else "FAIL"
        output[metric] = {
            "status": status,
            "mwu": mwu,
            "ks": ks,
            "cliff": number(row.get("cliffs_delta")),
            "wasserstein": number(row.get("wasserstein_distance")),
            "real_mean": number(row.get("real_mean")),
            "generated_mean": number(row.get("generated_mean")),
        }
    return output


def required_nonpassing_metrics(
    summary: dict[str, dict[str, Any]],
) -> list[str]:
    """Return required extended metrics that have not passed both tests."""

    return [
        metric
        for metric in REQUIRED_EXTENDED_METRICS
        if (summary.get(metric) or {}).get("status") != "PASS"
    ]


def passing_metric_regressions(
    before: dict[str, dict[str, Any]],
    after: dict[str, dict[str, Any]],
) -> list[str]:
    """Return metrics that changed from full PASS to PARTIAL/FAIL."""

    return [
        metric
        for metric in ALL_METRICS
        if (before.get(metric) or {}).get("status") == "PASS"
        and (after.get(metric) or {}).get("status") != "PASS"
    ]


def monitor_only_metrics(summary: dict[str, dict[str, Any]]) -> list[str]:
    """Return nonpassing directions without a claim-safe CARD rewriter."""

    output: list[str] = []
    downward_only = (
        "self_bleu_4",
    )
    upward_only = ("avg_depth", "structural_virality")
    for metric in downward_only:
        row = summary.get(metric) or {}
        if (
            row.get("status") != "PASS"
            and number(row.get("generated_mean")) <= number(row.get("real_mean"))
        ):
            output.append(metric)
    for metric in upward_only:
        row = summary.get(metric) or {}
        if (
            row.get("status") != "PASS"
            and number(row.get("generated_mean")) >= number(row.get("real_mean"))
        ):
            output.append(metric)
    return output


def number(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def load_json_list(path: Path) -> list[dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    return [row for row in payload if isinstance(row, dict)] if isinstance(payload, list) else []


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


if __name__ == "__main__":
    main()
