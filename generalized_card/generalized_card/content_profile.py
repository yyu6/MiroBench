from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .content_profile_analysis import (
    content_properties,
    discourse_properties,
    examples,
    model_properties,
    realization,
    repetition_diagnostics,
    surface_diagnostics,
)
from .content_profile_data import (
    generated_threads,
    generation_records,
    matched_model_rows,
    matched_threads,
    metric_report,
    optional_float,
)


SCHEMA_VERSION = "matched-content-profile-v2"


def build_content_profile(run_dir: Path, config: Any) -> dict[str, Any]:
    """Build a read-only matched content and plan-realization report."""

    records = generation_records(run_dir)
    generated = generated_threads(records)
    if not generated:
        raise ValueError(f"no accepted generated comments under {run_dir}")
    matches, missing_matches = matched_threads(run_dir, config, records)
    real = {key: row["texts"] for key, row in matches.items()}
    paired_keys = sorted(set(generated) & set(real))
    unmatched_generated = sorted(set(generated) - set(real))
    if not paired_keys or unmatched_generated:
        raise ValueError(
            "content profile requires an exact real match for every generated thread; "
            f"paired={len(paired_keys)} unmatched_generated={unmatched_generated} "
            f"details={missing_matches}"
        )
    generated_all = [text for key in paired_keys for text in generated[key]]
    real_all = [text for key in paired_keys for text in real[key]]

    models = matched_model_rows(run_dir, matches, paired_keys)
    metrics = metric_report(run_dir)
    if metrics["matched_rows"] != len(paired_keys):
        raise ValueError(
            "content profile must use the same cohort as matched evaluation; "
            f"paired_threads={len(paired_keys)} metric_rows={metrics['matched_rows']}"
        )
    model_coverage = {
        side: {name: len(rows) for name, rows in kinds.items()}
        for side, kinds in models.items()
    }
    incomplete_models = {
        f"{side}.{name}": count
        for side, kinds in model_coverage.items()
        for name, count in kinds.items()
        if count != len(paired_keys)
    }
    if incomplete_models:
        raise ValueError(
            "content profile requires exact matched per-comment scorer coverage; "
            f"expected={len(paired_keys)} observed={incomplete_models}"
        )
    interpretation = metrics.pop("interpretation")
    return {
        "schema_version": SCHEMA_VERSION,
        "run_dir": str(run_dir),
        "scope": {
            "generated_threads": len(generated),
            "matched_real_threads": len(real),
            "paired_threads": len(paired_keys),
            "generated_comments": len(generated_all),
            "matched_real_usable_comments": len(real_all),
            "model_thread_coverage": model_coverage,
            "missing_matches": missing_matches,
            "note": (
                "Lexical rows include every usable raw comment. Model rows use only "
                "comments present in saved scorer outputs; their common loader "
                "excludes comments shorter than two whitespace tokens."
            ),
        },
        "statistical_interpretation": interpretation,
        "metrics": metrics,
        "content_properties": content_properties(real_all, generated_all, config),
        "discourse_properties": discourse_properties(real, generated, paired_keys),
        "model_scored_properties": model_properties(models["real"], models["generated"]),
        "planner_writer_realization": realization(records, models["generated"]),
        "surface_diagnostics": surface_diagnostics(real_all, generated_all),
        "repetition": {
            "real": repetition_diagnostics(real_all),
            "generated": repetition_diagnostics(generated_all),
        },
        "examples": examples(records, real_all, models["generated"]),
        "evidence_boundaries": {
            "formal_metrics": "Saved matched scorer CSVs; authoritative for the 12 metrics.",
            "model_realization": (
                "Saved per-comment StorySeeker, GoEmotions, and politeness outputs, "
                "joined to the exact matched thread."
            ),
            "planner_controls": "Persisted generation_records task fields.",
            "weak_surface_probes": (
                "Regex/lexical diagnostics only; never semantic ground truth for advice, "
                "emotion, story, or Reddit naturalness."
            ),
        },
    }


def render_markdown(report: dict[str, Any]) -> str:
    scope = report["scope"]
    lines = [
        "# Matched content profile",
        "",
        f"- schema: `{report['schema_version']}`",
        f"- paired threads: {scope['paired_threads']}",
        (
            "- comments: generated "
            f"{scope['generated_comments']}, matched real usable {scope['matched_real_usable_comments']}"
        ),
        f"- inference: {report['statistical_interpretation']}",
        "",
        "## Twelve formal metrics",
        "",
        "| metric | real | generated | gap | ratio | status |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in report["metrics"]["rows"]:
        lines.append(
            f"| {row['metric']} | {fmt(row['real_mean'])} | {fmt(row['generated_mean'])} | "
            f"{fmt(row['gap'], signed=True)} | {fmt(row['generated_over_real'])} | "
            f"{row['inferential_status']} |"
        )

    lines.extend(["", "## Direct matched content properties", ""])
    lines.extend(comparison_table(report["content_properties"]))
    lines.extend(["", "## Exact matched model properties", ""])
    lines.extend(comparison_table(report["model_scored_properties"]))
    lines.extend(realization_lines(report["planner_writer_realization"]))
    lines.extend(["", "## Weak surface probes (not semantic labels)", ""])
    lines.extend(comparison_table(report["surface_diagnostics"]["rows"]))
    lines.extend(repetition_lines(report["repetition"]))
    lines.extend(example_lines(report["examples"]))
    return "\n".join(lines) + "\n"


def realization_lines(realized: dict[str, Any]) -> list[str]:
    return [
        "",
        "## Planner → Writer realization",
        "",
        f"- accepted records: {realized['accepted_records']}",
        (
            "- tone exact realization: "
            f"{fmt(realized['tone']['exact_rate'])} "
            f"({realized['tone']['aligned']}/{realized['tone']['covered']})"
        ),
        (
            "- affect exact dominant-emotion realization: "
            f"{fmt(realized['affect']['exact_rate'])} "
            f"({realized['affect']['aligned']}/{realized['affect']['covered']})"
        ),
        (
            "- planned story mean P(story): "
            f"{fmt(realized['story']['planned_story']['mean_probability'])}; "
            "planned no-story mean P(story): "
            f"{fmt(realized['story']['planned_no_story']['mean_probability'])}"
        ),
        (
            "- planned advice/function share: "
            f"{fmt(realized['planned_surface']['recommendation_advice_share'])}; "
            "planned soft-helpful payload share: "
            f"{fmt(realized['planned_surface']['soft_helpful_payload_share'])}"
        ),
    ]


def repetition_lines(repetition: dict[str, Any]) -> list[str]:
    real, generated = repetition["real"], repetition["generated"]
    lines = [
        "",
        "## Repetition",
        "",
        (
            "- repeated 4-gram share: real "
            f"{fmt(real['repeated_4gram_share'])}, generated "
            f"{fmt(generated['repeated_4gram_share'])}"
        ),
        (
            "- repeated 5-gram share: real "
            f"{fmt(real['repeated_5gram_share'])}, generated "
            f"{fmt(generated['repeated_5gram_share'])}"
        ),
        "",
        "Top generated repeated phrases:",
        "",
    ]
    lines.extend(
        f"- `{row['phrase']}` — {row['comment_count']} comments, {row['occurrences']} occurrences"
        for row in generated["top_repeated_ngrams"][:10]
    )
    return lines


def example_lines(examples_by_kind: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    sections = (
        ("Strongest generated story mismatches", "story_mismatches"),
        ("Generated tone mismatches", "tone_mismatches"),
        ("Planned helpful/advice examples", "planned_helpful"),
        ("Generated colloquial/profane examples", "generated_colloquial"),
        ("Matched-real colloquial/profane examples", "real_colloquial"),
    )
    for title, key in sections:
        rows = examples_by_kind.get(key) or []
        if not rows:
            continue
        lines.extend(["", f"## {title}", ""])
        for row in rows:
            details = ", ".join(
                f"{name}={value}" for name, value in row.items() if name != "text"
            )
            prefix = f"{details}: " if details else ""
            lines.append(f"- {prefix}{row['text']}")
    return lines


def comparison_table(rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| property | real | generated | gap | note |",
        "|---|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['property']} | {fmt(row['real'])} | {fmt(row['generated'])} | "
            f"{fmt(row['gap'], signed=True)} | {row['note']} |"
        )
    return lines


def write_report(report: dict[str, Any], *, json_path: Path, markdown_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(render_markdown(report), encoding="utf-8")


def fmt(value: object, *, signed: bool = False) -> str:
    number = optional_float(value)
    if number is None:
        return "n/a"
    return f"{number:+.4f}" if signed else f"{number:.4f}"
