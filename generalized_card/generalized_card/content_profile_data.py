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
            missing.append({"post_id": post_id, "seed_index": seed_index, "reason": "missing_seed"})
            continue
        matched = find_matched_real_thread(
            bank,
            SimpleNamespace(
                source_raw_post_id=str(seed.get("source_raw_post_id") or ""),
                metadata=seed,
            ),
        )
        if not matched:
            missing.append({"post_id": post_id, "seed_index": seed_index, "reason": "missing_real"})
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


def metric_report(run_dir: Path) -> dict[str, Any]:
    root = run_dir / "matched_evaluation"
    generated = read_csv(root / "matched_generated_thread_scores.csv")
    real = read_csv(root / "matched_real_thread_scores.csv")
    saved = load_json(root / "matched_seed_group_eval.json")
    count = min(len(real), len(generated))
    rows = []
    for metric in EVALUATION_METRICS:
        real_values = numeric_values(real, metric)
        generated_values = numeric_values(generated, metric)
        real_mean = statistics.mean(real_values) if real_values else None
        generated_mean = statistics.mean(generated_values) if generated_values else None
        gap = (
            generated_mean - real_mean
            if real_mean is not None and generated_mean is not None
            else None
        )
        saved_row = saved.get(metric) if isinstance(saved, dict) else {}
        saved_row = saved_row if isinstance(saved_row, dict) else {}
        rows.append(
            {
                "metric": metric,
                "real_mean": real_mean,
                "generated_mean": generated_mean,
                "gap": gap,
                "absolute_gap": abs(gap) if gap is not None else None,
                "generated_over_real": (
                    generated_mean / real_mean
                    if generated_mean is not None and real_mean not in (None, 0.0)
                    else None
                ),
                "mwu_p_value": optional_float(saved_row.get("mwu_p_value")),
                "ks_p_value": optional_float(saved_row.get("ks_p_value")),
                "cliffs_delta": optional_float(saved_row.get("cliffs_delta")),
                "wasserstein_distance": optional_float(saved_row.get("wasserstein_distance")),
                "inferential_status": (
                    "descriptive_only_n1" if count <= 1 else saved_status(saved_row)
                ),
            }
        )
    interpretation = (
        "descriptive only: one matched thread cannot validate MWU/KS p-values"
        if count <= 1
        else "group inference uses the saved matched MWU/KS results"
    )
    return {"matched_rows": count, "rows": rows, "interpretation": interpretation}


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


def numeric_values(rows: list[dict[str, str]], key: str) -> list[float]:
    return [value for row in rows if (value := optional_float(row.get(key))) is not None]


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
