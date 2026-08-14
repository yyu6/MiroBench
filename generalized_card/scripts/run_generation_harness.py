#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

from generalized_card.backend import configure_generator_backend, load_generator_backend
from generalized_card.domain import REPO_ROOT, load_domain_config
from generalized_card.domain_profile import build_domain_profile


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Offline preflight for generalized CARD generation."
    )
    parser.add_argument("--domain", required=True)
    parser.add_argument("--seed-pool", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--artifact",
        type=Path,
        help="Optional prior generated root for Writer-failure regression diagnostics.",
    )
    args = parser.parse_args()

    config = load_domain_config(args.domain)
    seed_pool = args.seed_pool.expanduser().resolve()
    with tempfile.TemporaryDirectory(prefix="generalized-card-harness-") as directory:
        profile_path = Path(directory) / "domain_profile.json"
        profile = build_domain_profile(
            config,
            seed_pool_path=seed_pool,
            output_path=profile_path,
        )
        source = profile["source"]
        metrics = profile["reference_metric_calibration"]
        lexical = profile["lexical_quality"]

        checks = {
            "seed_reference_overlap_zero": source["seed_reference_overlap_count"] == 0,
            "metric_seed_overlap_zero": metrics["seed_reference_overlap_count"] == 0,
            "test_content_hidden": source["test_content_visible"] is False,
            "metric_templates_have_no_raw_text": metrics.get("raw_text_included") is False,
            "reference_bank_sufficient": source["reference_thread_count"] >= 20,
            "metric_calibration_available": bool(metrics.get("available")),
            "lexical_prefix_calibration_available": bool(
                lexical.get("prefix_mean_upper")
            ),
            "semantic_metric_calibration_available": all(
                "semantic_mean_cosine" in (row or {})
                for row in (metrics.get("metric_bands_by_size") or {}).values()
            ),
        }
        failures = [name for name, passed in checks.items() if not passed]
        if failures:
            raise RuntimeError("Generation harness failed: " + ", ".join(failures))

        previous_profile = os.environ.get("GENERALIZED_CARD_DOMAIN_PROFILE")
        os.environ["GENERALIZED_CARD_DOMAIN_PROFILE"] = str(profile_path)
        try:
            module = configure_generator_backend(load_generator_backend(), config)
            module.run_self_test()
        finally:
            if previous_profile is None:
                os.environ.pop("GENERALIZED_CARD_DOMAIN_PROFILE", None)
            else:
                os.environ["GENERALIZED_CARD_DOMAIN_PROFILE"] = previous_profile

        report = {
            "status": "pass",
            "domain": config.domain_id,
            "seed_pool": str(seed_pool),
            "checks": checks,
            "reference_thread_count": source["reference_thread_count"],
            "reference_metric_template_count": metrics["reference_thread_count"],
            "reference_summary": metrics.get("summary") or {},
            "reference_metric_bands_by_size": (
                metrics.get("metric_bands_by_size") or {}
            ),
            "behavior_targets": profile["behavior_targets"],
            "lexical_prefix_upper": lexical["prefix_mean_upper"],
            "lexical_prefix_median": lexical["prefix_mean_median"],
            "lexical_prefix_sample_counts": lexical["prefix_sample_counts"],
            "core_parity": module.GENERALIZED_CARD_PARITY,
        }
        if args.artifact:
            report["artifact_regression"] = _audit_generation_records(
                args.artifact.expanduser().resolve()
            )

    output = (args.output or (REPO_ROOT / "artifacts" / "generalized_card" / "harness" / f"{config.domain_id}.json")).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[generation-harness] PASS output={output}", flush=True)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


def _audit_generation_records(generated_root: Path) -> dict[str, object]:
    records: list[dict[str, object]] = []
    for path in sorted(generated_root.glob("run_*_sampled_reddit/generation_records.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, list):
            records.extend(row for row in payload if isinstance(row, dict))

    problem_counts: Counter[str] = Counter()
    recoverable_repetition_only = 0
    contradictory_length_attempts = 0
    skipped = 0
    surface_contract_conflicts: list[dict[str, object]] = []
    length_pairs: list[tuple[int, int]] = []
    semantic_rows: defaultdict[str, list[tuple[int, str]]] = defaultdict(list)
    opening_counts: Counter[str] = Counter()
    for record in records:
        attempts = [
            row for row in (record.get("attempts") or []) if isinstance(row, dict)
        ]
        if not record.get("comment"):
            skipped += 1
        task = record.get("task") if isinstance(record.get("task"), dict) else {}
        comment = record.get("comment") if isinstance(record.get("comment"), dict) else {}
        real_words = _safe_int(task.get("real_word_count"), 0)
        generated_words = _safe_int(comment.get("word_count"), 0)
        if not generated_words:
            generated_words = len(str(comment.get("content") or "").split())
        if real_words > 0 and generated_words > 0:
            length_pairs.append((real_words, generated_words))
        if _surface_contract_conflict(task):
            surface_contract_conflicts.append(
                {
                    "post_id": str(record.get("post_id") or ""),
                    "task_id": _safe_int(task.get("local_task_id"), 0),
                    "real_word_count": real_words,
                    "real_surface_shape": str(task.get("real_surface_shape") or ""),
                    "payload_type": str(task.get("payload_type") or ""),
                    "utterance_mode": str(task.get("utterance_mode") or ""),
                }
            )
        semantic_move = " ".join(str(task.get("semantic_move") or "").split())
        if semantic_move:
            semantic_rows[str(record.get("post_id") or "")].append(
                (_safe_int(task.get("local_task_id"), 0), semantic_move)
            )
        opening = _opening_signature(str(comment.get("content") or ""))
        if opening:
            opening_counts[opening] += 1
        recoverable = False
        for attempt in attempts:
            problems = [str(item) for item in (attempt.get("problems") or [])]
            problem_counts.update(problem.split(":", 1)[0] for problem in problems)
            if "length_too_long" in problems and "real_slot_too_short" in problems:
                contradictory_length_attempts += 1
            if problems and all(
                problem in {
                    "opening_reused",
                    "opener_family_reused",
                    "template_phrase_reused",
                }
                or problem.startswith("lexical_overlap_high:")
                or problem.startswith("semantic_overlap_high:")
                or problem.startswith("semantic_overlap_low:")
                for problem in problems
            ):
                recoverable = True
        if not record.get("comment") and recoverable:
            recoverable_repetition_only += 1
    long_pairs = [pair for pair in length_pairs if pair[0] >= 100]
    plan_collisions = _semantic_plan_collisions(semantic_rows)
    return {
        "generated_root": str(generated_root),
        "record_count": len(records),
        "skipped_record_count": skipped,
        "recoverable_repetition_only_skips": recoverable_repetition_only,
        "contradictory_length_attempts": contradictory_length_attempts,
        "surface_contract_conflict_count": len(surface_contract_conflicts),
        "surface_contract_conflict_examples": surface_contract_conflicts[:20],
        "length_real_to_generated_ratio": _mean_ratio(length_pairs),
        "long_slot_real_to_generated_ratio": _mean_ratio(long_pairs),
        "long_slot_count": len(long_pairs),
        "semantic_plan_collision_count": len(plan_collisions),
        "semantic_plan_collision_examples": plan_collisions[:20],
        "repeated_opening_signatures": dict(
            sorted(
                (
                    (opening, count)
                    for opening, count in opening_counts.items()
                    if count >= 2
                ),
                key=lambda item: (-item[1], item[0]),
            )[:20]
        ),
        "writer_problem_counts": dict(sorted(problem_counts.items())),
    }


def _surface_contract_conflict(task: dict[str, object]) -> bool:
    words = _safe_int(task.get("real_word_count"), 0)
    shape = str(task.get("real_surface_shape") or "")
    mode = str(task.get("utterance_mode") or "")
    hard_short_shapes = {
        "deleted_removed",
        "template_notice",
        "link_reference",
        "quote_link_reference",
        "micro_reaction",
        "short_direct_answer",
        "short_question",
        "thanks_ack",
        "joke_reaction",
    }
    return (
        words >= 35
        and shape not in hard_short_shapes
        and mode in {"fragment_only", "question_only", "joke_only", "template_notice"}
    )


def _semantic_plan_collisions(
    rows_by_post: dict[str, list[tuple[int, str]]],
    *,
    threshold: float = 0.72,
) -> list[dict[str, object]]:
    collisions: list[dict[str, object]] = []
    for post_id, rows in rows_by_post.items():
        token_rows = [(task_id, text, _tokens(text)) for task_id, text in rows]
        for index, (left_id, left_text, left_tokens) in enumerate(token_rows):
            for right_id, right_text, right_tokens in token_rows[index + 1 :]:
                union = left_tokens | right_tokens
                score = len(left_tokens & right_tokens) / max(1, len(union))
                if score < threshold:
                    continue
                collisions.append(
                    {
                        "post_id": post_id,
                        "left_task_id": left_id,
                        "right_task_id": right_id,
                        "token_jaccard": round(score, 6),
                        "left_move": left_text[:180],
                        "right_move": right_text[:180],
                    }
                )
    return sorted(collisions, key=lambda row: -float(row["token_jaccard"]))


def _tokens(text: str) -> set[str]:
    return {
        token.lower()
        for token in re.findall(r"[A-Za-z0-9]+(?:'[A-Za-z]+)?", text)
        if len(token) >= 3
    }


def _opening_signature(text: str) -> str:
    words = list(_tokens_in_order(text))
    return " ".join(words[:3])


def _tokens_in_order(text: str) -> list[str]:
    return [
        token.lower()
        for token in re.findall(r"[A-Za-z0-9]+(?:'[A-Za-z]+)?", text)
    ]


def _mean_ratio(rows: list[tuple[int, int]]) -> float | None:
    if not rows:
        return None
    return round(sum(generated / real for real, generated in rows) / len(rows), 6)


def _safe_int(value: object, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


if __name__ == "__main__":
    main()
