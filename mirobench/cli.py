"""MiroBench CLI: score generated threads and compare against real references."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from . import __version__

DATA_DIR = Path(__file__).parent / "data"

DOMAINS = {
    "credit_cards": DATA_DIR / "credit_cards",
    "cameras": DATA_DIR / "cameras",
    "cell_phones": DATA_DIR / "cell_phones",
    "headphones": DATA_DIR / "headphones",
    "laptops": DATA_DIR / "laptops",
}

SCORERS = [
    ("score_thread_disagreement", "stance_disagreement_results.json"),
    ("score_thread_self_bleu", "self_bleu_results.json"),
    ("score_thread_self_bertscore", "self_bertscore_results.json"),
    ("score_thread_semantic_uniformity", "semantic_uniformity_results.json"),
    ("score_thread_storyseeker", "storyseeker_results.json"),
    ("score_thread_go_emotions", "go_emotions_results.json"),
    ("score_thread_politeness", "politeness_results.json"),
    ("score_thread_structure", "thread_structure_results.json"),
    ("score_thread_detoxify", "detoxify_results.json"),
]


def cmd_score(args: argparse.Namespace) -> None:
    """Score generated discussion threads."""
    input_dir = Path(args.input).resolve()
    device = args.device

    if not input_dir.exists():
        print(f"Error: input path not found: {input_dir}")
        sys.exit(1)

    # Find all thread directories (contain discussion.json)
    thread_dirs = []
    disc = input_dir / "discussion.json"
    if disc.exists():
        thread_dirs = [input_dir]
    else:
        for sub in sorted(input_dir.iterdir()):
            if sub.is_dir() and (sub / "discussion.json").exists():
                thread_dirs.append(sub)

    if not thread_dirs:
        print(f"Error: no discussion.json found in {input_dir} or its subdirectories.")
        print("Each thread should be a directory containing a discussion.json file.")
        print(f"See the format example: {DATA_DIR / 'example_thread_format.json'}")
        sys.exit(1)

    print(f"Found {len(thread_dirs)} thread(s) to score")
    scorer_dir = Path(__file__).parent / "scorers"

    total = len(thread_dirs)
    for i, td in enumerate(thread_dirs, 1):
        print(f"\n[{i}/{total}] Scoring {td.name}")
        for scorer_name, result_file in SCORERS:
            if (td / result_file).exists() and not args.force:
                continue

            module_name = f"mirobench.scorers.{scorer_name}"
            cmd = [sys.executable, "-m", module_name, str(td), "--target-kind", "generated"]

            if scorer_name == "score_thread_self_bertscore":
                cmd += ["--model-type", "microsoft/deberta-xlarge-mnli",
                        "--batch-size", "32", "--device", device]
            elif scorer_name in ("score_thread_semantic_uniformity",
                                  "score_thread_storyseeker",
                                  "score_thread_go_emotions",
                                  "score_thread_detoxify"):
                cmd += ["--device", device]

            try:
                subprocess.run(cmd, capture_output=True, text=True, check=True)
                print(f"  {scorer_name.replace('score_thread_', '')}: OK")
            except subprocess.CalledProcessError as e:
                print(f"  {scorer_name.replace('score_thread_', '')}: FAILED")
                if args.verbose:
                    print(f"    stderr: {e.stderr[:200]}")

    # Merge per-thread result JSONs into top-level merged files,
    # then run the summarizer. Each per-thread file has a "threads"
    # array with one element; we concatenate across all thread dirs,
    # tagging each entry with its directory name as thread_id.
    print(f"\nAggregating scores...")
    metric_files = [result_file for _, result_file in SCORERS]
    for mf in metric_files:
        all_threads = []
        meta = None
        for td in thread_dirs:
            f = td / mf
            if not f.exists():
                continue
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
            except Exception:
                continue
            if meta is None:
                meta = data.get("meta", {})
            for t in data.get("threads", []):
                # Override thread_id with the directory name so each
                # thread's row keeps its original identity.
                t["thread_id"] = td.name
                all_threads.append(t)
        if all_threads:
            (input_dir / mf).write_text(
                json.dumps({"meta": meta or {}, "threads": all_threads},
                           ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    output_prefix = args.output_prefix or "thread_scores"
    cmd = [sys.executable, "-m", "mirobench.scorers.summarize_thread_metrics",
           str(input_dir), "--output-prefix", output_prefix]
    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        out_csv = input_dir / f"{output_prefix}.csv"
        if out_csv.exists():
            import pandas as pd
            df = pd.read_csv(out_csv)
            df = df[df["thread_id"].astype(str) != "__summary_mean__"]
            print(f"Done! Scores written to: {out_csv} ({len(df)} threads)")
        else:
            print(f"Done! Check {input_dir} for output files.")
    except subprocess.CalledProcessError as e:
        print(f"Summarization failed: {e.stderr[:300]}")
        sys.exit(1)


# The 16 selected MiroBench metrics: 5 families × (3+4+3+2+4) = 16.
CORE_METRICS = {
    # Diversity (3)
    "self_bleu_4", "semantic_mean_cosine", "self_bertscore_mean_f1",
    # Tone (4)
    "hard_disagree_rate", "polite_rate", "impolite_rate", "neutral_rate",
    # Structure (3)
    "length_cv", "avg_depth", "structural_virality",
    # Content (2)
    "mean_story_probability", "emotion_entropy",
    # Toxicity (4)
    "toxicity_mean", "severe_toxicity_mean", "obscene_mean", "threat_mean",
}


def cmd_compare(args: argparse.Namespace) -> None:
    """Compare scored threads against real reference data."""
    from .compare import compare_against_reference, write_comparison_csv

    sim_csv = Path(args.scores_csv).resolve()
    if not sim_csv.exists():
        print(f"Error: scores CSV not found: {sim_csv}")
        sys.exit(1)

    domains_to_compare = args.domains or list(DOMAINS.keys())
    all_rows = []

    if args.core_only:
        print(f"Filtering to the {len(CORE_METRICS)} core MiroBench metrics.")

    for domain in domains_to_compare:
        if domain not in DOMAINS:
            print(f"Warning: unknown domain '{domain}', skipping.")
            continue

        ref_dir = DOMAINS[domain] / "reference_scores"
        ref_csv = ref_dir / "thread_scores.csv"
        if not ref_csv.exists():
            print(f"Warning: reference data not found for '{domain}', skipping.")
            continue

        rows = compare_against_reference(
            sim_csv=sim_csv,
            ref_csv=ref_csv,
            domain=domain,
            model=args.model_name or sim_csv.stem,
        )
        if args.core_only:
            rows = [r for r in rows if r.get("metric") in CORE_METRICS]
        all_rows.extend(rows)
        print(f"  {domain}: {len(rows)} metrics compared")

    if not all_rows:
        print("Error: no valid comparisons produced.")
        sys.exit(1)

    output = Path(args.output or f"mirobench_comparison.csv")
    write_comparison_csv(all_rows, output)
    print(f"\nComparison written to: {output} ({len(all_rows)} rows)")

    # Print summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    for domain in domains_to_compare:
        domain_rows = [r for r in all_rows if r.get("domain") == domain]
        if not domain_rows:
            continue
        negligible = sum(1 for r in domain_rows
                         if r.get("cliffs_delta_interpretation") == "negligible")
        small = sum(1 for r in domain_rows
                    if r.get("cliffs_delta_interpretation") == "small")
        medium = sum(1 for r in domain_rows
                     if r.get("cliffs_delta_interpretation") == "medium")
        large = sum(1 for r in domain_rows
                    if r.get("cliffs_delta_interpretation") == "large")
        avg_wd = sum(r.get("wasserstein", 0) for r in domain_rows) / len(domain_rows)
        avg_fail = sum(r.get("empirical_fail_rate", 0) for r in domain_rows) / len(domain_rows)
        print(f"\n  {domain} ({len(domain_rows)} metrics):")
        print(f"    Cliff's delta: {negligible} negligible, {small} small, "
              f"{medium} medium, {large} large")
        print(f"    Avg Wasserstein distance: {avg_wd:.4f}")
        print(f"    Avg empirical fail rate:  {avg_fail:.2%}")


def cmd_domains(args: argparse.Namespace) -> None:
    """List available benchmark domains."""
    print("Available domains:")
    for name, path in DOMAINS.items():
        ref = path / "reference_scores" / "thread_scores.csv"
        if ref.exists():
            import pandas as pd
            df = pd.read_csv(ref)
            df = df[df["thread_id"].astype(str) != "__summary_mean__"]
            print(f"  {name}: {len(df)} real threads")
        else:
            print(f"  {name}: (reference data not found)")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="mirobench",
        description="MiroBench: Benchmark for Evaluating Synthetic Online Product Discussions",
    )
    parser.add_argument("--version", action="version", version=f"mirobench {__version__}")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # score
    p_score = subparsers.add_parser("score", help="Score generated discussion threads")
    p_score.add_argument("input", help="Directory with discussion.json files (one per thread)")
    p_score.add_argument("--device", default="cpu",
                         help="Device for model inference (cpu, cuda, mps)")
    p_score.add_argument("--output-prefix", default="thread_scores",
                         help="Prefix for output CSV/JSON files")
    p_score.add_argument("--force", action="store_true",
                         help="Re-score even if results exist")
    p_score.add_argument("--verbose", "-v", action="store_true")
    p_score.set_defaults(func=cmd_score)

    # compare
    p_compare = subparsers.add_parser("compare",
                                       help="Compare scored threads against real references")
    p_compare.add_argument("scores_csv", help="Path to thread_scores.csv from 'mirobench score'")
    p_compare.add_argument("--domains", nargs="+",
                           help="Domains to compare against (default: all)")
    p_compare.add_argument("--model-name", default="",
                           help="Model name label for the output CSV")
    p_compare.add_argument("--output", "-o", default="",
                           help="Output CSV path (default: mirobench_comparison.csv)")
    p_compare.add_argument("--core-only", action="store_true",
                           help="Only report the 16 core metrics (5 families: "
                                "Diversity / Tone / Structure / Content / Toxicity)")
    p_compare.set_defaults(func=cmd_compare)

    # domains
    p_domains = subparsers.add_parser("domains", help="List available benchmark domains")
    p_domains.set_defaults(func=cmd_domains)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(0)
    args.func(args)


if __name__ == "__main__":
    main()
