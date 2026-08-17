#!/usr/bin/env python3
"""Merge per-thread evaluation metrics into one table.

Each evaluator writes a JSON file with a top-level `threads` list. This script
joins those thread-level rows by `thread_id` and writes one CSV/JSON record per
thread so downstream analysis can compare real and simulated discussions
directly.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


METRIC_FILES = {
    "stance_disagreement": "stance_disagreement_results.json",
    "self_bleu": "self_bleu_results.json",
    "self_bertscore": "self_bertscore_results.json",
    "semantic_uniformity": "semantic_uniformity_results.json",
    "storyseeker": "storyseeker_results.json",
    "go_emotions": "go_emotions_results.json",
    "politeness": "politeness_results.json",
    "detoxify": "detoxify_results.json",
    "thread_structure": "thread_structure_results.json",
}

SUMMARY_MEAN_THREAD_ID = "__summary_mean__"
SUMMARY_MEAN_THREAD_TITLE = "__summary_mean__"


CSV_COLUMNS = [
    "thread_id",
    "thread_title",
    "comment_count",
    "pair_count",
    "mean_disagree_probability",
    "hard_disagree_rate",
    "self_bleu_2",
    "self_bleu_3",
    "self_bleu_4",
    "self_bertscore_mean_precision",
    "self_bertscore_mean_recall",
    "self_bertscore_mean_f1",
    "self_bertscore_median_f1",
    "self_bertscore_top_k_mean_f1",
    "semantic_mean_cosine",
    "semantic_median_cosine",
    "semantic_top_k_mean_cosine",
    "semantic_p90_cosine",
    "story_count",
    "not_story_count",
    "story_rate",
    "mean_story_probability",
    "emotion_entropy",
    "emotion_entropy_normalized",
    "avg_labels_per_comment",
    "emotion_shift_rate",
    "dominant_emotion",
    "dominant_emotion_share",
    "polite_rate",
    "somewhat_polite_rate",
    "neutral_rate",
    "impolite_rate",
    "length_std",
    "length_iqr",
    "length_cv",
    "max_depth",
    "avg_depth",
    "avg_branching_factor",
    "structural_virality",
    "toxicity_mean",
    "toxicity_max",
    "toxicity_p90",
    "severe_toxicity_mean",
    "severe_toxicity_max",
    "severe_toxicity_p90",
    "obscene_mean",
    "obscene_max",
    "obscene_p90",
    "threat_mean",
    "threat_max",
    "threat_p90",
    "insult_mean",
    "insult_max",
    "insult_p90",
    "identity_attack_mean",
    "identity_attack_max",
    "identity_attack_p90",
    "aggression_score_mean",
    "aggression_score_max",
    "aggression_score_p90",
]


def main() -> None:
    """CLI entrypoint."""

    parser = argparse.ArgumentParser(
        description="Merge per-thread metric JSON outputs into CSV/JSON summaries."
    )
    parser.add_argument("results_dir", help="Directory containing metric result JSON files.")
    parser.add_argument(
        "--output-prefix",
        default="thread_metrics_summary",
        help="Output basename without extension.",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir).expanduser().resolve()
    thread_rows = merge_thread_metrics(results_dir)
    rows = append_mean_summary_row(thread_rows)

    csv_path = results_dir / f"{args.output_prefix}.csv"
    json_path = results_dir / f"{args.output_prefix}.json"
    write_csv(csv_path, rows)
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")
    print(f"threads={len(thread_rows)}")


def merge_thread_metrics(results_dir: Path) -> list[dict[str, Any]]:
    """Read available metric JSON files and return merged per-thread rows."""

    merged: dict[str, dict[str, Any]] = {}

    def row_for(thread: dict[str, Any]) -> dict[str, Any]:
        thread_id = str(thread.get("thread_id", ""))
        if thread_id not in merged:
            merged[thread_id] = {
                "thread_id": thread_id,
                "thread_title": thread.get("thread_title", ""),
            }
        if not merged[thread_id].get("thread_title") and thread.get("thread_title"):
            merged[thread_id]["thread_title"] = thread.get("thread_title", "")
        return merged[thread_id]

    stance_disagreement = load_metric(results_dir, "stance_disagreement")
    for thread in stance_disagreement:
        row = row_for(thread)
        row.update(
            {
                "pair_count": thread.get("pair_count", row.get("pair_count", 0)),
                "mean_disagree_probability": thread.get("mean_disagree_probability", 0.0),
                "hard_disagree_rate": thread.get("hard_disagree_rate", 0.0),
            }
        )

    self_bleu = load_metric(results_dir, "self_bleu")
    for thread in self_bleu:
        row = row_for(thread)
        row.update(
            {
                "comment_count": thread.get("comment_count", row.get("comment_count", 0)),
                "pair_count": thread.get("pair_count", row.get("pair_count", 0)),
                "self_bleu_2": thread.get("self_bleu_2", 0.0),
                "self_bleu_3": thread.get("self_bleu_3", 0.0),
                "self_bleu_4": thread.get("self_bleu_4", 0.0),
            }
        )

    self_bertscore = load_metric(results_dir, "self_bertscore")
    for thread in self_bertscore:
        row = row_for(thread)
        row.update(
            {
                "comment_count": thread.get("comment_count", row.get("comment_count", 0)),
                "pair_count": thread.get("pair_count", row.get("pair_count", 0)),
                "self_bertscore_mean_precision": thread.get("mean_bert_precision", 0.0),
                "self_bertscore_mean_recall": thread.get("mean_bert_recall", 0.0),
                "self_bertscore_mean_f1": thread.get("mean_bert_f1", 0.0),
                "self_bertscore_median_f1": thread.get("median_bert_f1", 0.0),
                "self_bertscore_top_k_mean_f1": thread.get("top_k_mean_bert_f1", 0.0),
            }
        )

    semantic = load_metric(results_dir, "semantic_uniformity")
    for thread in semantic:
        row = row_for(thread)
        row.update(
            {
                "comment_count": thread.get("comment_count", row.get("comment_count", 0)),
                "pair_count": thread.get("pair_count", row.get("pair_count", 0)),
                "semantic_mean_cosine": thread.get("mean_cosine_similarity", 0.0),
                "semantic_median_cosine": thread.get("median_cosine_similarity", 0.0),
                "semantic_top_k_mean_cosine": thread.get("top_k_mean_cosine_similarity", 0.0),
                "semantic_p90_cosine": thread.get("p90_cosine_similarity", 0.0),
            }
        )

    storyseeker = load_metric(results_dir, "storyseeker")
    for thread in storyseeker:
        row = row_for(thread)
        row.update(
            {
                "comment_count": thread.get("comment_count", row.get("comment_count", 0)),
                "story_count": thread.get("story_count", 0),
                "not_story_count": thread.get("not_story_count", 0),
                "story_rate": thread.get("story_rate", 0.0),
                "mean_story_probability": thread.get("mean_story_probability", 0.0),
            }
        )

    go_emotions = load_metric(results_dir, "go_emotions")
    for thread in go_emotions:
        row = row_for(thread)
        row.update(
            {
                "comment_count": thread.get("comment_count", row.get("comment_count", 0)),
                "emotion_entropy": thread.get("emotion_entropy", 0.0),
                "emotion_entropy_normalized": thread.get("emotion_entropy_normalized", 0.0),
                "avg_labels_per_comment": thread.get("avg_labels_per_comment", 0.0),
                "emotion_shift_rate": thread.get("emotion_shift_rate", 0.0),
                "dominant_emotion": thread.get("dominant_emotion", ""),
                "dominant_emotion_share": thread.get("dominant_emotion_share", 0.0),
            }
        )

    politeness = load_metric(results_dir, "politeness")
    for thread in politeness:
        row = row_for(thread)
        row.update(
            {
                "comment_count": thread.get("comment_count", row.get("comment_count", 0)),
                "polite_rate": thread.get("polite_rate", 0.0),
                "somewhat_polite_rate": thread.get("somewhat_polite_rate", 0.0),
                "neutral_rate": thread.get("neutral_rate", 0.0),
                "impolite_rate": thread.get("impolite_rate", 0.0),
            }
        )

    thread_structure = load_metric(results_dir, "thread_structure")
    for thread in thread_structure:
        row = row_for(thread)
        row.update(
            {
                "comment_count": thread.get("comment_count", row.get("comment_count", 0)),
                "length_std": thread.get("length_std", 0.0),
                "length_iqr": thread.get("length_iqr", 0.0),
                "length_cv": thread.get("length_cv", 0.0),
                "max_depth": thread.get("max_depth", 0.0),
                "avg_depth": thread.get("avg_depth", 0.0),
                "avg_branching_factor": thread.get("avg_branching_factor", 0.0),
                "structural_virality": thread.get("structural_virality", 0.0),
            }
        )

    detoxify = load_metric(results_dir, "detoxify")
    for thread in detoxify:
        row = row_for(thread)
        row.update(
            {
                "comment_count": thread.get("comment_count", row.get("comment_count", 0)),
                "toxicity_mean": thread.get("toxicity_mean", 0.0),
                "toxicity_max": thread.get("toxicity_max", 0.0),
                "toxicity_p90": thread.get("toxicity_p90", 0.0),
                "severe_toxicity_mean": thread.get("severe_toxicity_mean", 0.0),
                "severe_toxicity_max": thread.get("severe_toxicity_max", 0.0),
                "severe_toxicity_p90": thread.get("severe_toxicity_p90", 0.0),
                "obscene_mean": thread.get("obscene_mean", 0.0),
                "obscene_max": thread.get("obscene_max", 0.0),
                "obscene_p90": thread.get("obscene_p90", 0.0),
                "threat_mean": thread.get("threat_mean", 0.0),
                "threat_max": thread.get("threat_max", 0.0),
                "threat_p90": thread.get("threat_p90", 0.0),
                "insult_mean": thread.get("insult_mean", 0.0),
                "insult_max": thread.get("insult_max", 0.0),
                "insult_p90": thread.get("insult_p90", 0.0),
                "identity_attack_mean": thread.get("identity_attack_mean", 0.0),
                "identity_attack_max": thread.get("identity_attack_max", 0.0),
                "identity_attack_p90": thread.get("identity_attack_p90", 0.0),
                "aggression_score_mean": thread.get("aggression_score_mean", 0.0),
                "aggression_score_max": thread.get("aggression_score_max", 0.0),
                "aggression_score_p90": thread.get("aggression_score_p90", 0.0),
            }
        )

    rows = list(merged.values())
    for row in rows:
        for column in CSV_COLUMNS:
            row.setdefault(column, "")
    rows.sort(key=lambda item: str(item["thread_id"]))
    return rows


def append_mean_summary_row(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Append one synthetic row containing mean values across numeric columns."""
    if not rows:
        return rows

    summary_row: dict[str, Any] = {
        "thread_id": SUMMARY_MEAN_THREAD_ID,
        "thread_title": SUMMARY_MEAN_THREAD_TITLE,
    }

    for column in CSV_COLUMNS:
        if column in ("thread_id", "thread_title", "dominant_emotion"):
            summary_row.setdefault(column, "")
            continue

        numeric_values: list[float] = []
        for row in rows:
            value = row.get(column, "")
            if value in ("", None):
                continue
            try:
                numeric_values.append(float(value))
            except (TypeError, ValueError):
                continue

        summary_row[column] = (
            sum(numeric_values) / len(numeric_values)
            if numeric_values else ""
        )

    return [*rows, summary_row]


def load_metric(results_dir: Path, metric_key: str) -> list[dict[str, Any]]:
    """Load one metric's thread rows if the JSON exists."""

    path = results_dir / METRIC_FILES[metric_key]
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return list(data.get("threads", []))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write rows to CSV with a stable column order."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


if __name__ == "__main__":
    main()
