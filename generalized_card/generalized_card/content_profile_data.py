from __future__ import annotations

import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

from .data import find_matched_real_thread, load_real_thread_bank
from .distribution_stats import distribution_stats


EVALUATION_METRICS = (
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


def generation_records(run_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(run_dir.glob("generated/run_*/generation_records.json")):
        payload = load_json(path)
        if isinstance(payload, list):
            rows.extend(row for row in payload if isinstance(row, dict))
    return rows


def generated_threads(records: list[dict[str, Any]]) -> dict[str, list[str]]:
    threads: dict[str, list[str]] = defaultdict(list)
    for record in records:
        comment = record.get("comment") or {}
        text = clean_text(comment.get("content"))
        post_id = str(record.get("post_id") or "").strip()
        if post_id and text:
            threads[post_id].append(text)
    return dict(threads)


def matched_threads(
    run_dir: Path,
    config: Any,
    records: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    pool = load_json(seed_pool_path(run_dir))
    seeds = {
        int(row["seed_index"]): row
        for row in (pool.get("seed_posts") or [])
        if isinstance(row, dict) and "seed_index" in row
    }
    bank = load_real_thread_bank(config.raw_discussions_dir)
    seed_by_post: dict[str, int] = {}
    for record in records:
        post_id = str(record.get("post_id") or "").strip()
        index = record.get("seed_index")
        if post_id and isinstance(index, int):
            seed_by_post.setdefault(post_id, index)

    matches: dict[str, dict[str, Any]] = {}
    missing: list[dict[str, Any]] = []
    for post_id, seed_index in sorted(seed_by_post.items()):
        seed = seeds.get(seed_index)
        if not seed:
            missing.append(
                {"post_id": post_id, "seed_index": seed_index, "reason": "missing_seed"}
            )
            continue
        matched = find_matched_real_thread(
            bank,
            SimpleNamespace(
                source_raw_post_id=str(seed.get("source_raw_post_id") or ""),
                metadata=seed,
            ),
        )
        if not matched:
            missing.append(
                {"post_id": post_id, "seed_index": seed_index, "reason": "missing_real"}
            )
            continue
        matches[post_id] = {
            "seed_index": seed_index,
            "real_thread_id": str(matched.get("post_id") or ""),
            "source_dir": str(matched.get("source_dir") or ""),
            "texts": [
                clean_text(row.get("body"))
                for row in (matched.get("comments") or [])
                if clean_text(row.get("body"))
            ],
        }
    return matches, missing


def matched_model_rows(
    run_dir: Path,
    matches: dict[str, dict[str, Any]],
    paired_keys: list[str],
) -> dict[str, dict[str, dict[str, dict[str, Any]]]]:
    filenames = {
        "emotion": "go_emotions_results.json",
        "story": "storyseeker_results.json",
        "tone": "politeness_results.json",
    }
    output: dict[str, dict[str, dict[str, dict[str, Any]]]] = {
        "generated": {name: {} for name in filenames},
        "real": {name: {} for name in filenames},
    }
    for name, filename in filenames.items():
        generated_rows = thread_rows(run_dir.glob(f"cleaned/*/{filename}"))
        for key in paired_keys:
            thread = generated_rows.get(key)
            if thread:
                output["generated"][name][key] = thread
            match = matches[key]
            real_rows = thread_rows([Path(match["source_dir"]) / filename])
            real_id = match["real_thread_id"]
            if real_id in real_rows:
                output["real"][name][key] = real_rows[real_id]
    return output


def planner_reference_templates(
    run_dir: Path,
    records: list[dict[str, Any]],
    paired_keys: list[str],
) -> tuple[list[dict[str, Any]], str]:
    """Load the exact excluded-real template selected for each generated post."""

    templates: dict[str, dict[str, Any]] = {}
    for path in sorted(run_dir.glob("generated/run_*/discussion.json")):
        discussion = load_json(path)
        if not isinstance(discussion, dict):
            continue
        for post in discussion.get("posts") or []:
            if not isinstance(post, dict):
                continue
            post_id = str(post.get("post_id") or "").strip()
            thread_plan = post.get("thread_plan") or {}
            template = (
                thread_plan.get("reference_metric_template")
                if isinstance(thread_plan, dict)
                else None
            )
            if post_id and isinstance(template, dict) and template:
                templates[post_id] = template
    if templates:
        return _complete_templates(templates, paired_keys), "atomic_thread_plan"

    reports = read_jsonl(run_dir / "logs" / "story_affect_distribution.jsonl")
    reports = [
        row
        for row in reports
        if isinstance(row.get("reference_template"), dict) and row["reference_template"]
    ]
    if not reports:
        raise ValueError("no persisted Planner reference metric templates found")

    post_seed = {
        str(row.get("post_id") or ""): int(row["seed_index"])
        for row in records
        if str(row.get("post_id") or "").strip()
        and isinstance(row.get("seed_index"), int)
    }
    pool = load_json(seed_pool_path(run_dir))
    raw_by_seed = {
        int(row["seed_index"]): str(row.get("source_raw_post_id") or "")
        for row in (pool.get("seed_posts") or [])
        if isinstance(row, dict) and isinstance(row.get("seed_index"), int)
    }
    post_by_seed_key = {
        raw_by_seed[seed_index]: post_id
        for post_id, seed_index in post_seed.items()
        if raw_by_seed.get(seed_index)
    }
    keyed_reports = [row for row in reports if str(row.get("seed_key") or "").strip()]
    if keyed_reports:
        for row in keyed_reports:
            post_id = post_by_seed_key.get(str(row["seed_key"]).strip())
            if post_id:
                templates[post_id] = dict(row["reference_template"])
        return _complete_templates(templates, paired_keys), "audit_log_seed_key"

    ordered_posts = list(dict.fromkeys(post_seed))
    sequence_rows: dict[int, dict[str, Any]] = {}
    for row in reports:
        index = row.get("sequence_index")
        if not isinstance(index, int) or index in sequence_rows:
            raise ValueError(
                "legacy Planner template log is ambiguous after resume; "
                "duplicate or invalid sequence_index"
            )
        sequence_rows[index] = row
    if set(sequence_rows) != set(range(len(ordered_posts))):
        raise ValueError(
            "legacy Planner template log cannot be aligned exactly to generated posts; "
            f"posts={len(ordered_posts)} sequence_indices={sorted(sequence_rows)}"
        )
    templates = {
        ordered_posts[index]: dict(sequence_rows[index]["reference_template"])
        for index in range(len(ordered_posts))
    }
    return _complete_templates(templates, paired_keys), "legacy_audit_log_sequence"


def metric_report(
    run_dir: Path,
    planner_templates: list[dict[str, Any]],
) -> dict[str, Any]:
    root = run_dir / "matched_evaluation"
    generated = read_csv(root / "matched_generated_thread_scores.csv")
    real = read_csv(root / "matched_real_thread_scores.csv")
    saved = load_json(root / "matched_seed_group_eval.json")
    if not real or len(real) != len(generated):
        raise ValueError(
            "matched real/generated score cohorts differ or are empty; "
            f"real={len(real)} generated={len(generated)}"
        )
    count = len(real)
    if len(planner_templates) != count:
        raise ValueError(
            "Planner template cohort differs from matched evaluation; "
            f"templates={len(planner_templates)} matched={count}"
        )
    rows = []
    for metric in EVALUATION_METRICS:
        real_values = numeric_values(real, metric)
        generated_values = numeric_values(generated, metric)
        planner_values = [
            value
            for template in planner_templates
            if (value := optional_float(template.get(metric))) is not None
        ]
        observed = {
            "real": len(real_values),
            "planner": len(planner_values),
            "generated": len(generated_values),
        }
        if any(value != count for value in observed.values()):
            raise ValueError(
                f"metric {metric} has incomplete real/Planner/generated coverage: "
                f"expected={count} observed={observed}"
            )
        real_mean = statistics.mean(real_values)
        planner_mean = statistics.mean(planner_values)
        generated_mean = statistics.mean(generated_values)
        target_stats = distribution_stats(real_values, planner_values)
        gap = generated_mean - real_mean
        saved_row = saved.get(metric) if isinstance(saved, dict) else {}
        saved_row = saved_row if isinstance(saved_row, dict) else {}
        rows.append(
            {
                "metric": metric,
                "real_mean": real_mean,
                "planner_target_mean": planner_mean,
                "generated_mean": generated_mean,
                "gap": gap,
                "absolute_gap": abs(gap),
                "planner_target_minus_real": planner_mean - real_mean,
                "generated_minus_planner_target": generated_mean - planner_mean,
                "planner_target_absolute_gap": abs(planner_mean - real_mean),
                "writer_realization_absolute_gap": abs(generated_mean - planner_mean),
                "generated_over_real": (
                    generated_mean / real_mean if real_mean != 0.0 else None
                ),
                "planner_target_mwu_p_value": target_stats["mwu_p_value"],
                "planner_target_ks_p_value": target_stats["ks_p_value"],
                "planner_target_cliffs_delta": target_stats["cliffs_delta"],
                "planner_target_wasserstein_distance": target_stats[
                    "wasserstein_distance"
                ],
                "mwu_p_value": optional_float(saved_row.get("mwu_p_value")),
                "ks_p_value": optional_float(saved_row.get("ks_p_value")),
                "cliffs_delta": optional_float(saved_row.get("cliffs_delta")),
                "wasserstein_distance": optional_float(
                    saved_row.get("wasserstein_distance")
                ),
                "inferential_status": (
                    "descriptive_only_n1" if count <= 1 else saved_status(saved_row)
                ),
                "planner_target_inferential_status": (
                    "descriptive_only_n1" if count <= 1 else saved_status(target_stats)
                ),
            }
        )
    interpretation = (
        "descriptive only: one matched thread cannot validate MWU/KS p-values"
        if count <= 1
        else "group inference uses the saved matched MWU/KS results"
    )
    return {"matched_rows": count, "rows": rows, "interpretation": interpretation}


def _complete_templates(
    templates: dict[str, dict[str, Any]],
    paired_keys: list[str],
) -> list[dict[str, Any]]:
    missing = [key for key in paired_keys if key not in templates]
    extra = sorted(set(templates) - set(paired_keys))
    if missing or extra:
        raise ValueError(
            "Planner reference templates do not match the exact content cohort; "
            f"missing={missing} extra={extra}"
        )
    return [templates[key] for key in paired_keys]


def thread_rows(paths: Iterable[Path]) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for path in paths:
        payload = load_json(path)
        if not isinstance(payload, dict):
            continue
        for row in payload.get("threads") or []:
            if isinstance(row, dict) and row.get("thread_id") is not None:
                rows[str(row["thread_id"])] = row
    return rows


def seed_pool_path(run_dir: Path) -> Path:
    config = load_json(run_dir / "run_config.json")
    if not isinstance(config, dict):
        return Path()
    return Path(str(config.get("seed_post_pool_json") or config.get("seed_pool") or ""))


def read_csv(path: Path) -> list[dict[str, str]]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))
    except OSError:
        return []


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    rows: list[dict[str, Any]] = []
    for line in lines:
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def numeric_values(rows: list[dict[str, str]], key: str) -> list[float]:
    return [
        value for row in rows if (value := optional_float(row.get(key))) is not None
    ]


def saved_status(row: dict[str, Any]) -> str:
    mwu = optional_float(row.get("mwu_p_value"))
    ks = optional_float(row.get("ks_p_value"))
    if mwu is None or ks is None:
        return "unavailable"
    if mwu > 0.05 and ks > 0.05:
        return "PASS"
    if mwu > 0.05 or ks > 0.05:
        return "PARTIAL"
    return "FAIL"


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def clean_text(value: object) -> str:
    return " ".join(str(value or "").split())


def optional_float(value: object) -> float | None:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None
