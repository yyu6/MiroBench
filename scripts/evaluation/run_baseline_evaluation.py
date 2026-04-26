#!/usr/bin/env python3
"""Baseline evaluation: score real threads, generate vanilla simulations, compare.

Produces a complete pre-calibration (or post-calibration) baseline that
quantifies how close OASIS simulations are to real Reddit discussions using
the full metric suite plus Mann-Whitney U, Kolmogorov-Smirnov, and Cliff's
delta statistical tests.

Usage — full pipeline (score real + generate sims + compare):

    python scripts/evaluation/run_baseline_evaluation.py \
        --category-dir data/raw/discussions/credit_cards \
        --products-json data/processed/splits/credit_cards/product_descriptions_train.json \
        --sim-runs 25 --seed-posts 4 \
        --agents 50 --hours 24 --rounds 24 --start-seed 44

Usage — skip simulation (real scoring + stats only on existing sims):

    python scripts/evaluation/run_baseline_evaluation.py \
        --category-dir data/raw/discussions/credit_cards \
        --sim-scores-csv artifacts/baselines/.../vanilla/thread_scores.csv

Output structure:

    artifacts/baselines/<category>_<timestamp>/
        baseline_config.json
        real/
            thread_scores.csv                  All real threads, all metrics
        vanilla/
            runs/<sim_dirs>/                   Individual simulation run outputs
            thread_scores.csv                  All simulated threads merged
        statistical_comparison.csv             MWU, KS, Cliff's delta per metric
        combined_thread_scores.csv             real + simulated with dataset column
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
RUN_DISCUSSION = REPO_ROOT / "run_discussion.py"
SUMMARIZE_SCRIPT = SCRIPT_DIR / "summarize_thread_metrics.py"
DEFAULT_SELF_BERTSCORE_MODEL_TYPE = "microsoft/deberta-xlarge-mnli"

# Numeric metrics eligible for statistical tests.  Superset of the list in
# compare_thread_metric_summaries.py — includes all numeric columns from
# summarize_thread_metrics.py.
NUMERIC_METRICS = [
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

# Column ordering for the combined output CSV (mirrors the real_simulated
# thread_scores_table schema).
OUTPUT_COLUMNS = [
    "dataset",
    "product",
    "thread_id",
    "comment_count",
    "pair_count",
] + NUMERIC_METRICS + [
    "dominant_emotion",
]

METRIC_OUTPUT_FILES = {
    "stance_disagreement": "stance_disagreement_results.json",
    "self_bleu": "self_bleu_results.json",
    "self_bertscore": "self_bertscore_results.json",
    "semantic_uniformity": "semantic_uniformity_results.json",
    "storyseeker": "storyseeker_results.json",
    "go_emotions": "go_emotions_results.json",
    "politeness": "politeness_results.json",
    "thread_structure": "thread_structure_results.json",
    "detoxify": "detoxify_results.json",
}


# ── CLI ──────────────────────────────────────────────────────────────────────


def main() -> None:
    args = parse_args()

    python = str(Path(args.python).expanduser())
    category_dir = Path(args.category_dir).expanduser().resolve()
    category_name = category_dir.name

    # Resolve output directory.
    if args.output_dir:
        output_root = Path(args.output_dir).expanduser().resolve()
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_root = (REPO_ROOT / "artifacts" / "baselines" / f"{category_name}_{ts}").resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    real_out_dir = output_root / "real"
    real_out_dir.mkdir(parents=True, exist_ok=True)

    # ── Phase 1: Score all real threads ──────────────────────────────────────
    print(f"\n{'='*70}")
    print("PHASE 1: Score real discussion threads")
    print(f"{'='*70}")

    real_csv = score_all_real_threads(
        category_dir=category_dir,
        output_csv=real_out_dir / "thread_scores.csv",
        python=python,
        device=args.device,
        self_bertscore_model_type=args.self_bertscore_model_type,
        self_bertscore_batch_size=args.self_bertscore_batch_size,
        force=args.force,
    )
    real_rows = read_csv(real_csv)
    print(f"\n→ {len(real_rows)} real threads scored across all products.")

    # ── Phase 1b: Train / Val / Test split ──────────────────────────────────
    print(f"\n{'='*70}")
    print("PHASE 1b: Train / Validation / Test split (by thread)")
    print(f"{'='*70}")

    ratios = [float(r) for r in args.split_ratios.split(",")]
    if len(ratios) != 3 or abs(sum(ratios) - 1.0) > 0.01:
        print(f"ERROR: --split-ratios must be three floats summing to 1.0, got {ratios}")
        sys.exit(1)

    train_rows, val_rows, test_rows = split_threads(
        real_rows, ratios=ratios, seed=args.split_seed,
    )
    train_csv = real_out_dir / "thread_scores_train.csv"
    val_csv = real_out_dir / "thread_scores_val.csv"
    test_csv = real_out_dir / "thread_scores_test.csv"
    write_scored_csv(train_csv, train_rows)
    write_scored_csv(val_csv, val_rows)
    write_scored_csv(test_csv, test_rows)
    print(f"  train: {len(train_rows)} threads → {train_csv.name}")
    print(f"  val  : {len(val_rows)} threads → {val_csv.name}")
    print(f"  test : {len(test_rows)} threads → {test_csv.name}")

    # ── Phase 2: Generate vanilla simulations ────────────────────────────────
    if args.sim_scores_csv:
        print(f"\n{'='*70}")
        print("PHASE 2: Using pre-existing simulated scores")
        print(f"{'='*70}")
        sim_csv = Path(args.sim_scores_csv).expanduser().resolve()
        sim_rows = read_csv(sim_csv)
        print(f"→ {len(sim_rows)} simulated threads loaded from {sim_csv}")
    elif args.products_json:
        print(f"\n{'='*70}")
        print("PHASE 2: Generate and score vanilla OASIS simulations")
        print(f"{'='*70}")

        vanilla_dir = output_root / "vanilla"
        runs_dir = vanilla_dir / "runs"
        runs_dir.mkdir(parents=True, exist_ok=True)

        sim_csv = generate_and_score_simulations(
            products_json=Path(args.products_json).expanduser().resolve(),
            runs_dir=runs_dir,
            output_csv=vanilla_dir / "thread_scores.csv",
            sim_runs=args.sim_runs,
            seed_posts=args.seed_posts,
            agents=args.agents,
            hours=args.hours,
            rounds=args.rounds,
            start_seed=args.start_seed,
            python=python,
            device=args.device,
            self_bertscore_model_type=args.self_bertscore_model_type,
            self_bertscore_batch_size=args.self_bertscore_batch_size,
        )
        sim_rows = read_csv(sim_csv)
        print(f"\n→ {len(sim_rows)} simulated threads scored.")
    else:
        print("\n[skip] No --products-json or --sim-scores-csv provided. Skipping simulation phase.")
        sim_csv = None
        sim_rows = []

    # ── Phase 3: Statistical comparison ──────────────────────────────────────
    if real_rows and sim_rows:
        print(f"\n{'='*70}")
        print("PHASE 3: Statistical comparison (MWU, KS, Cliff's delta)")
        print(f"{'='*70}")

        comparison_csv = output_root / "statistical_comparison.csv"
        compute_statistical_comparison(real_rows, sim_rows, comparison_csv)
        print(f"→ Wrote {comparison_csv}")

        # Combined table with dataset column.
        combined_csv = output_root / "combined_thread_scores.csv"
        build_combined_table(real_rows, sim_rows, combined_csv)
        print(f"→ Wrote {combined_csv}")

    # ── Save config ──────────────────────────────────────────────────────────
    config = {
        "created_at": datetime.now().isoformat(),
        "category_dir": str(category_dir),
        "category_name": category_name,
        "products_json": args.products_json or "",
        "sim_runs": args.sim_runs,
        "seed_posts": args.seed_posts,
        "agents": args.agents,
        "hours": args.hours,
        "rounds": args.rounds,
        "start_seed": args.start_seed,
        "device": args.device,
        "self_bertscore_model_type": args.self_bertscore_model_type,
        "real_thread_count": len(real_rows),
        "real_train_count": len(train_rows),
        "real_val_count": len(val_rows),
        "real_test_count": len(test_rows),
        "split_ratios": ratios,
        "split_seed": args.split_seed,
        "simulated_thread_count": len(sim_rows),
        "output_root": str(output_root),
    }
    config_path = output_root / "baseline_config.json"
    config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n{'='*70}")
    print("BASELINE EVALUATION COMPLETE")
    print(f"{'='*70}")
    print(f"Output: {output_root}")
    print(f"  baseline_config.json")
    print(f"  real/thread_scores.csv              ({len(real_rows)} threads)")
    print(f"  real/thread_scores_train.csv        ({len(train_rows)} threads)")
    print(f"  real/thread_scores_val.csv          ({len(val_rows)} threads)")
    print(f"  real/thread_scores_test.csv         ({len(test_rows)} threads)")
    if sim_rows:
        print(f"  vanilla/thread_scores.csv           ({len(sim_rows)} threads)")
        print(f"  statistical_comparison.csv")
        print(f"  combined_thread_scores.csv")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Baseline evaluation pipeline: score real threads, generate vanilla "
            "OASIS simulations, and compute statistical comparison."
        )
    )
    p.add_argument(
        "--category-dir",
        required=True,
        help="Category directory containing product subdirectories (e.g., data/raw/discussions/credit_cards).",
    )
    p.add_argument(
        "--products-json",
        default="",
        help="Product descriptions JSON for simulation (e.g., data/processed/splits/credit_cards/product_descriptions_train.json).",
    )
    p.add_argument(
        "--sim-scores-csv",
        default="",
        help="Pre-existing simulated thread scores CSV. Skips simulation generation if provided.",
    )
    p.add_argument("--output-dir", default="", help="Output directory. Defaults to artifacts/baselines/<category>_<timestamp>.")
    p.add_argument("--sim-runs", type=int, default=12, help="Number of simulation runs (default: 12, ~50 threads).")
    p.add_argument("--seed-posts", type=int, default=4, help="Seed posts per simulation run (default: 4).")
    p.add_argument("--agents", type=int, default=50, help="Agents per simulation (default: 50).")
    p.add_argument("--hours", type=int, default=24, help="Simulated hours (default: 24).")
    p.add_argument("--rounds", type=int, default=24, help="Max simulation rounds (default: 24).")
    p.add_argument("--start-seed", type=int, default=44, help="Starting random seed (default: 44).")
    p.add_argument("--python", default=sys.executable, help="Python executable.")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps", "auto"], help="Torch device.")
    p.add_argument(
        "--self-bertscore-model-type",
        default=DEFAULT_SELF_BERTSCORE_MODEL_TYPE,
        help="Self-BERTScore backbone (default: microsoft/deberta-xlarge-mnli).",
    )
    p.add_argument("--self-bertscore-batch-size", type=int, default=16, help="Batch size for Self-BERTScore.")
    p.add_argument("--force", action="store_true", help="Recompute metrics even if output JSON already exists.")
    p.add_argument(
        "--split-ratios",
        default="0.70,0.15,0.15",
        help="Train/val/test split ratios for real threads (default: 0.70,0.15,0.15).",
    )
    p.add_argument("--split-seed", type=int, default=42, help="Random seed for train/val/test split (default: 42).")
    return p.parse_args()


# ── Phase 1: Real thread scoring ─────────────────────────────────────────────


def score_all_real_threads(
    category_dir: Path,
    output_csv: Path,
    python: str,
    device: str,
    self_bertscore_model_type: str,
    self_bertscore_batch_size: int,
    force: bool,
) -> Path:
    """Score every product directory under category_dir and merge into one CSV."""

    product_dirs = sorted(
        p for p in category_dir.iterdir()
        if p.is_dir() and any(p.glob("*.comments.jsonl"))
    )
    if not product_dirs:
        raise RuntimeError(f"No product directories with *.comments.jsonl found in {category_dir}")

    print(f"Found {len(product_dirs)} product directories to score.\n")
    all_rows: list[dict[str, str]] = []
    failed: list[str] = []

    for i, product_dir in enumerate(product_dirs, 1):
        product_name = product_dir.name
        summary_csv = product_dir / "thread_metrics_summary.csv"

        print(f"[{i}/{len(product_dirs)}] {product_name}")

        # Score individual metrics if needed.
        try:
            score_single_directory(
                target_dir=product_dir,
                target_kind="real",
                python=python,
                device=device,
                self_bertscore_model_type=self_bertscore_model_type,
                self_bertscore_batch_size=self_bertscore_batch_size,
                force=force,
            )
        except subprocess.CalledProcessError as exc:
            print(f"  [FAIL] Metric scoring failed for {product_name}: {exc}")
            failed.append(product_name)
            continue

        if not summary_csv.exists():
            print(f"  [WARN] No thread_metrics_summary.csv produced for {product_name}")
            failed.append(product_name)
            continue

        rows = read_csv(summary_csv)
        for row in rows:
            row["product"] = product_name
        all_rows.extend(rows)
        print(f"  → {len(rows)} threads")

    if failed:
        print(f"\n[WARN] {len(failed)} products failed scoring: {', '.join(failed[:10])}")

    # Write merged real scores.
    write_scored_csv(output_csv, all_rows)
    print(f"\nMerged real scores → {output_csv} ({len(all_rows)} threads)")
    return output_csv


def score_single_directory(
    target_dir: Path,
    target_kind: str,
    python: str,
    device: str,
    self_bertscore_model_type: str,
    self_bertscore_batch_size: int,
    force: bool,
) -> None:
    """Run all 9 metric scripts on a single directory, then summarize."""

    commands = build_single_dir_metric_commands(
        python=python,
        target_dir=target_dir,
        target_kind=target_kind,
        device=device,
        self_bertscore_model_type=self_bertscore_model_type,
        self_bertscore_batch_size=self_bertscore_batch_size,
    )

    for output_name, cmd in commands:
        output_path = target_dir / output_name
        if not force and output_path.exists():
            # For self-BERTScore, also check model mismatch.
            if output_name == METRIC_OUTPUT_FILES["self_bertscore"]:
                if not self_bertscore_output_mismatch(output_path, self_bertscore_model_type):
                    continue
            else:
                continue
        run_cmd(cmd)

    # Summarize all metric JSONs into thread_metrics_summary.csv.
    run_cmd([python, str(SUMMARIZE_SCRIPT), str(target_dir)])


def build_single_dir_metric_commands(
    python: str,
    target_dir: Path,
    target_kind: str,
    device: str,
    self_bertscore_model_type: str,
    self_bertscore_batch_size: int,
) -> list[tuple[str, list[str]]]:
    """Return (output_filename, command) for each metric script."""

    td = str(target_dir)
    tk = target_kind
    return [
        (
            "stance_disagreement_results.json",
            [python, str(SCRIPT_DIR / "score_thread_disagreement.py"), td, "--target-kind", tk, "--device", device],
        ),
        (
            "self_bleu_results.json",
            [python, str(SCRIPT_DIR / "score_thread_self_bleu.py"), td, "--target-kind", tk],
        ),
        (
            "self_bertscore_results.json",
            [
                python, str(SCRIPT_DIR / "score_thread_self_bertscore.py"), td,
                "--target-kind", tk,
                "--device", device,
                "--batch-size", str(self_bertscore_batch_size),
                "--model-type", self_bertscore_model_type,
            ],
        ),
        (
            "semantic_uniformity_results.json",
            [python, str(SCRIPT_DIR / "score_thread_semantic_uniformity.py"), td, "--target-kind", tk, "--device", device],
        ),
        (
            "storyseeker_results.json",
            [python, str(SCRIPT_DIR / "score_thread_storyseeker.py"), td, "--target-kind", tk, "--device", device],
        ),
        (
            "go_emotions_results.json",
            [python, str(SCRIPT_DIR / "score_thread_go_emotions.py"), td, "--target-kind", tk, "--device", device],
        ),
        (
            "politeness_results.json",
            [python, str(SCRIPT_DIR / "score_thread_politeness.py"), td, "--target-kind", tk, "--device", device],
        ),
        (
            "thread_structure_results.json",
            [python, str(SCRIPT_DIR / "score_thread_structure.py"), td, "--target-kind", tk],
        ),
        (
            "detoxify_results.json",
            [python, str(SCRIPT_DIR / "score_thread_detoxify.py"), td, "--target-kind", tk, "--device", device],
        ),
    ]


# ── Phase 2: Vanilla simulation generation and scoring ───────────────────────


def generate_and_score_simulations(
    products_json: Path,
    runs_dir: Path,
    output_csv: Path,
    sim_runs: int,
    seed_posts: int,
    agents: int,
    hours: int,
    rounds: int,
    start_seed: int,
    python: str,
    device: str,
    self_bertscore_model_type: str,
    self_bertscore_batch_size: int,
) -> Path:
    """Generate N simulation runs, score each, and merge into one CSV."""

    all_rows: list[dict[str, str]] = []

    for i in range(sim_runs):
        seed = start_seed + i
        run_index = i + 1
        print(f"\n[sim {run_index}/{sim_runs}] seed={seed} seed_posts={seed_posts}")

        # Track existing dirs to detect newly created output.
        existing_dirs = {p.name for p in runs_dir.iterdir() if p.is_dir()}

        # Launch simulation.
        cmd = [
            python, str(RUN_DISCUSSION), str(products_json),
            "--agents", str(agents),
            "--hours", str(hours),
            "--rounds", str(rounds),
            "--seed-posts", str(seed_posts),
            "--seed", str(seed),
            "--output-dir", str(runs_dir),
        ]
        run_cmd(cmd)

        # Find the newly created directory.
        new_dirs = [
            p for p in runs_dir.iterdir()
            if p.is_dir() and p.name not in existing_dirs
        ]
        if not new_dirs:
            print(f"  [WARN] No new run directory created for seed={seed}. Skipping.")
            continue
        new_dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        sim_dir = new_dirs[0]
        print(f"  → Created: {sim_dir.name}")

        # Score the simulation.
        try:
            score_single_directory(
                target_dir=sim_dir,
                target_kind="generated",
                python=python,
                device=device,
                self_bertscore_model_type=self_bertscore_model_type,
                self_bertscore_batch_size=self_bertscore_batch_size,
                force=False,
            )
        except subprocess.CalledProcessError as exc:
            print(f"  [FAIL] Metric scoring failed for {sim_dir.name}: {exc}")
            continue

        summary_csv = sim_dir / "thread_metrics_summary.csv"
        if not summary_csv.exists():
            print(f"  [WARN] No thread_metrics_summary.csv for {sim_dir.name}")
            continue

        rows = read_csv(summary_csv)
        for row in rows:
            row["product"] = sim_dir.name
        all_rows.extend(rows)
        print(f"  → {len(rows)} threads scored")

    write_scored_csv(output_csv, all_rows)
    print(f"\nMerged simulated scores → {output_csv} ({len(all_rows)} threads)")
    return output_csv


# ── Phase 3: Statistical comparison ──────────────────────────────────────────


def compute_statistical_comparison(
    real_rows: list[dict[str, str]],
    sim_rows: list[dict[str, str]],
    output_csv: Path,
) -> None:
    """Compute MWU, KS, Cliff's delta for every numeric metric."""

    from scipy.stats import ks_2samp, mannwhitneyu

    comparison_rows: list[dict[str, Any]] = []

    for metric in NUMERIC_METRICS:
        real_vals = extract_numeric(real_rows, metric)
        sim_vals = extract_numeric(sim_rows, metric)

        row: dict[str, Any] = {
            "metric": metric,
            "real_n": len(real_vals),
            "sim_n": len(sim_vals),
            "real_mean": safe_mean(real_vals),
            "real_median": safe_median(real_vals),
            "sim_mean": safe_mean(sim_vals),
            "sim_median": safe_median(sim_vals),
        }

        # Mann-Whitney U test (two-sided).
        if len(real_vals) >= 2 and len(sim_vals) >= 2:
            try:
                stat, p = mannwhitneyu(real_vals, sim_vals, alternative="two-sided")
                row["mann_whitney_u_statistic"] = stat
                row["mann_whitney_u_pvalue"] = p
            except ValueError:
                row["mann_whitney_u_statistic"] = ""
                row["mann_whitney_u_pvalue"] = ""
        else:
            row["mann_whitney_u_statistic"] = ""
            row["mann_whitney_u_pvalue"] = ""

        # Kolmogorov-Smirnov test (two-sided).
        if len(real_vals) >= 2 and len(sim_vals) >= 2:
            try:
                stat, p = ks_2samp(real_vals, sim_vals)
                row["ks_test_statistic"] = stat
                row["ks_test_pvalue"] = p
            except ValueError:
                row["ks_test_statistic"] = ""
                row["ks_test_pvalue"] = ""
        else:
            row["ks_test_statistic"] = ""
            row["ks_test_pvalue"] = ""

        # Cliff's delta.
        if real_vals and sim_vals:
            row["cliffs_delta"] = cliffs_delta(real_vals, sim_vals)
            row["cliffs_delta_interpretation"] = interpret_cliffs_delta(row["cliffs_delta"])
        else:
            row["cliffs_delta"] = ""
            row["cliffs_delta_interpretation"] = ""

        comparison_rows.append(row)

    # Write comparison CSV.
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if comparison_rows:
        fieldnames = list(comparison_rows[0].keys())
        with output_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in comparison_rows:
                writer.writerow(row)

    # Also write JSON for programmatic consumption.
    json_path = output_csv.with_suffix(".json")
    json_path.write_text(
        json.dumps(comparison_rows, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


def cliffs_delta(group1: list[float], group2: list[float]) -> float:
    """Compute Cliff's delta effect size between two groups.

    delta = (# concordant - # discordant) / (n1 * n2)
    where concordant = pairs where group1[i] > group2[j]
          discordant = pairs where group1[i] < group2[j]

    Returns a value in [-1, 1].
    """

    n1, n2 = len(group1), len(group2)
    if n1 == 0 or n2 == 0:
        return 0.0

    concordant = 0
    discordant = 0
    for x in group1:
        for y in group2:
            if x > y:
                concordant += 1
            elif x < y:
                discordant += 1

    return (concordant - discordant) / (n1 * n2)


def interpret_cliffs_delta(d: float) -> str:
    """Interpret Cliff's delta magnitude (Romano et al. 2006 thresholds)."""

    abs_d = abs(d)
    if abs_d < 0.147:
        return "negligible"
    elif abs_d < 0.33:
        return "small"
    elif abs_d < 0.474:
        return "medium"
    else:
        return "large"


# ── Phase 4: Combined output ─────────────────────────────────────────────────


def build_combined_table(
    real_rows: list[dict[str, str]],
    sim_rows: list[dict[str, str]],
    output_csv: Path,
) -> None:
    """Build a combined CSV with a 'dataset' column (real vs simulated)."""

    combined: list[dict[str, str]] = []

    for row in real_rows:
        out = {"dataset": "real"}
        for col in OUTPUT_COLUMNS:
            if col == "dataset":
                continue
            out[col] = row.get(col, "")
        combined.append(out)

    for row in sim_rows:
        out = {"dataset": "simulated"}
        for col in OUTPUT_COLUMNS:
            if col == "dataset":
                continue
            out[col] = row.get(col, "")
        combined.append(out)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in combined:
            writer.writerow(row)


# ── Thread-level split ────────────────────────────────────────────────────────


def split_threads(
    rows: list[dict[str, str]],
    ratios: list[float],
    seed: int = 42,
) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    """Randomly split thread rows into train / val / test by thread count.

    Parameters
    ----------
    rows : list of dicts (one per thread)
    ratios : [train_ratio, val_ratio, test_ratio] summing to ~1.0
    seed : random seed for reproducibility

    Returns
    -------
    (train_rows, val_rows, test_rows)
    """
    rng = random.Random(seed)
    shuffled = list(rows)
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    # test gets the remainder to avoid rounding loss
    train = shuffled[:n_train]
    val = shuffled[n_train:n_train + n_val]
    test = shuffled[n_train + n_val:]
    return train, val, test


# ── Shared helpers ────────────────────────────────────────────────────────────


def self_bertscore_output_mismatch(output_path: Path, requested_model_type: str) -> bool:
    """Return True when an existing BERTScore JSON used a different model."""

    if not output_path.exists():
        return False
    try:
        data = json.loads(output_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return True

    meta = data.get("meta") or {}
    recorded = str(meta.get("requested_model_type") or meta.get("model_type") or "")
    return recorded.strip() != requested_model_type.strip()


def extract_numeric(rows: list[dict[str, str]], key: str) -> list[float]:
    """Extract numeric values from a column, skipping non-parseable entries."""

    values = []
    for row in rows:
        raw = row.get(key, "")
        if raw == "" or raw is None:
            continue
        try:
            values.append(float(raw))
        except (ValueError, TypeError):
            continue
    return values


def safe_mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def safe_median(values: list[float]) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    n = len(s)
    mid = n // 2
    if n % 2 == 0:
        return (s[mid - 1] + s[mid]) / 2
    return s[mid]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_scored_csv(path: Path, rows: list[dict[str, str]]) -> None:
    """Write thread scores CSV with a stable column order."""

    if not rows:
        # Write header-only CSV so downstream readers see the schema.
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
            writer.writeheader()
        return

    # Use the union of all keys, with OUTPUT_COLUMNS first.
    all_keys = set()
    for row in rows:
        all_keys.update(row.keys())

    fieldnames = []
    for col in OUTPUT_COLUMNS:
        if col in all_keys:
            fieldnames.append(col)
            all_keys.discard(col)
    fieldnames.extend(sorted(all_keys))

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def run_cmd(cmd: list[str]) -> None:
    """Run a subprocess and fail on non-zero exit."""

    print(f"  $ {' '.join(cmd[:6])}{'...' if len(cmd) > 6 else ''}")
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


if __name__ == "__main__":
    main()
