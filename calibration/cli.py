from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
from dotenv import load_dotenv


def _force_vanilla_backbone(
    reference_run_config: dict | None,
) -> dict | None:
    """Return a copy of ``reference_run_config`` forced onto vanilla OASIS.

    Calibration and post-calibration evaluation should default to GEO's patched
    runtime, but a fresh Phase 0 baseline must stay vanilla when explicitly
    requested. This helper keeps that branch isolated.
    """

    if reference_run_config is None:
        return None
    forced = dict(reference_run_config)
    forced["discussion_backbone"] = "vanilla_oasis"
    return forced


def main() -> None:
    # Load .env from the repo root so LLM_API_KEY / OPENAI_API_KEY etc. are
    # available to the subprocess `mirobench generate` calls.
    repo_root = Path(__file__).resolve().parent.parent
    load_dotenv(repo_root / ".env")
    parser = argparse.ArgumentParser(
        description="Iterative LLM-driven calibration for Reddit discussion simulation."
    )

    # ── Real data (train/val/test CSVs from baseline evaluation) ──────────────
    parser.add_argument(
        "--real-train-csv",
        required=True,
        help="Real thread scores CSV for the TRAIN split (used for qualitative context and sample real threads, not candidate ranking).",
    )
    parser.add_argument(
        "--real-val-csv",
        required=True,
        help="Real thread scores CSV for the VALIDATION split (used for scoring/selecting candidates).",
    )
    parser.add_argument(
        "--real-test-csv",
        required=True,
        help="Real thread scores CSV for the TEST split (used only for final post-calibration evaluation).",
    )

    # ── Vanilla baseline (for before/after comparison) ───────────────────────
    parser.add_argument(
        "--vanilla-scores-csv",
        default="",
        help=(
            "Pre-existing vanilla simulation scores CSV (from run_baseline_evaluation.py). "
            "If provided, enables before/after improvement analysis against real_test."
        ),
    )
    parser.add_argument(
        "--rerun-phase0-vanilla",
        action="store_true",
        help=(
            "Run a fresh vanilla baseline simulation batch for Phase 0 using the current "
            "simulation config, instead of only reusing --vanilla-scores-csv."
        ),
    )

    # ── Few-shot source (real discussion directory) ───────────────────────────
    parser.add_argument(
        "--few-shot-dir",
        required=True,
        help=(
            "Category directory containing real discussion subdirs with "
            ".comments.jsonl files (e.g. data/raw/discussions/credit_cards). "
            "Used as the few-shot source for all candidate simulations."
        ),
    )
    parser.add_argument(
        "--few-shot-count",
        type=int,
        default=5,
        help="Number of real thread examples as few-shot style anchors (default: 5).",
    )

    # ── Simulation parameters ─────────────────────────────────────────────────
    parser.add_argument("--products-json", required=True, help="Product JSON file.")
    parser.add_argument("--agents", type=int, default=50, help="Number of agents (default: 50).")
    parser.add_argument("--hours", type=int, default=24, help="Simulated hours (default: 24).")
    parser.add_argument("--rounds", type=int, default=24, help="Max simulation rounds (default: 24).")
    parser.add_argument("--seed-posts", type=int, default=4, help="Seed posts per run (default: 4).")
    parser.add_argument("--hint", type=str, default=None, help="Optional hint for persona/topic generation.")
    parser.add_argument(
        "--discussion-backbone",
        type=str,
        default="geo_patched",
        choices=["geo_patched", "vanilla_oasis"],
        help=(
            "Discussion-generation backbone for candidate/final simulations. "
            "Use geo_patched to enable GEO's visible-comment snapshot, reply-first guards, "
            "and anti-template runtime logic. (default: geo_patched)"
        ),
    )

    # ── Calibration control ───────────────────────────────────────────────────
    parser.add_argument("--iterations", type=int, default=12, help="Calibration iterations (default: 12).")
    parser.add_argument(
        "--combination-start-iteration",
        type=int,
        default=None,
        help=(
            "Iteration index at which Phase 1 should shift into combination-heavy candidate "
            "generation. Defaults to half of --iterations."
        ),
    )
    parser.add_argument("--candidates", type=int, default=5, help="Candidates per iteration (default: 5).")
    parser.add_argument("--parallel", type=int, default=1, help="Max concurrent simulations (default: 1).")
    parser.add_argument("--calibration-model", default="gpt-4o-mini", help="LLM for calibration reasoning.")
    parser.add_argument(
        "--calibration-reasoning-effort",
        default="",
        help="Optional reasoning effort for the calibration LLM (for example: minimal).",
    )
    parser.add_argument(
        "--simulation-reasoning-effort",
        default="",
        help=(
            "Optional reasoning effort for all simulation-side LLM calls "
            "(phase 0/1/2 sims). Leave empty to omit the parameter entirely."
        ),
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    parser.add_argument("--output-dir", default="artifacts/calibration_runs", help="Output directory.")
    parser.add_argument("--resume", action="store_true", help="Resume a previous run.")
    parser.add_argument(
        "--evaluate-overlay-json",
        default="",
        help=(
            "Skip Phase 1 calibration and run only the post-calibration evaluation "
            "pipeline for the given overlay JSON."
        ),
    )
    parser.add_argument(
        "--before-group-eval-json",
        default="",
        help=(
            "Optional existing before_calibration_group_eval.json to reuse during "
            "overlay-only evaluation, so Phase 0 statistics are not recomputed."
        ),
    )
    parser.add_argument(
        "--stop-after-phase1",
        action="store_true",
        help="Stop after Phase 1 completes and skip Phase 2/3.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda", "mps", "auto"],
        help="Device for torch-based metrics.",
    )
    parser.add_argument("--python", default=sys.executable, help="Python executable.")
    parser.add_argument(
        "--metric-parallel",
        type=int,
        default=3,
        help="Max concurrent metric scoring scripts per simulation (default: 3).",
    )
    parser.add_argument(
        "--calibration-rounds",
        type=int,
        default=None,
        help=(
            "Override --rounds during Phase 1 calibration iterations for faster "
            "candidate ranking. Full --rounds are used for Phase 0 baseline and "
            "Phase 2 final evaluation. (default: same as --rounds)"
        ),
    )

    # ── Final evaluation ──────────────────────────────────────────────────────
    parser.add_argument(
        "--final-sim-runs",
        type=int,
        default=12,
        help="Max simulation runs for after-calibration evaluation (default: 12).",
    )
    parser.add_argument(
        "--min-sim-threads",
        type=int,
        default=50,
        help="Stop after-calibration runs early once this many threads are collected (default: 50, 0=no limit).",
    )

    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    output_dir = Path(args.output_dir).resolve()

    real_train_csv = Path(args.real_train_csv).resolve()
    real_val_csv = Path(args.real_val_csv).resolve()
    real_test_csv = Path(args.real_test_csv).resolve()
    few_shot_dir = Path(args.few_shot_dir).resolve()

    vanilla_scores_csv = None
    if args.vanilla_scores_csv:
        vanilla_scores_csv = Path(args.vanilla_scores_csv).resolve()
        if not vanilla_scores_csv.exists():
            print(f"ERROR: Vanilla scores CSV not found: {vanilla_scores_csv}")
            sys.exit(1)

    for csv_path in [real_train_csv, real_val_csv, real_test_csv]:
        if not csv_path.exists():
            print(f"ERROR: CSV not found: {csv_path}")
            sys.exit(1)
    if not few_shot_dir.exists():
        print(f"ERROR: Few-shot directory not found: {few_shot_dir}")
        sys.exit(1)

    # Build reference_run_config from CLI args
    reference_run_config: dict = {
        "input_file": str(Path(args.products_json).resolve()),
        "agents": args.agents,
        "hours": args.hours,
        "rounds": args.rounds,
        "seed_posts": args.seed_posts,
        "seed": args.seed,
        "discussion_backbone": args.discussion_backbone,
        "few_shot_source": str(few_shot_dir),
        "few_shot_count": args.few_shot_count,
    }
    if args.hint:
        reference_run_config["hint"] = args.hint

    evaluate_overlay_json = None
    if args.evaluate_overlay_json:
        evaluate_overlay_json = Path(args.evaluate_overlay_json).resolve()
        if not evaluate_overlay_json.exists():
            print(f"ERROR: Overlay JSON not found: {evaluate_overlay_json}")
            sys.exit(1)
    before_group_eval_json = None
    if args.before_group_eval_json:
        before_group_eval_json = Path(args.before_group_eval_json).resolve()
        if not before_group_eval_json.exists():
            print(f"ERROR: before_group_eval JSON not found: {before_group_eval_json}")
            sys.exit(1)

    # Create timestamped output dir if not resuming or doing overlay-only eval
    if not args.resume and evaluate_overlay_json is None:
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = output_dir / f"calibration_{ts}"

    api_key = os.environ.get("LLM_API_KEY", "")
    base_url = os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1")
    simulation_reasoning_effort = (args.simulation_reasoning_effort or "").strip().lower()
    if simulation_reasoning_effort == "none":
        simulation_reasoning_effort = ""

    # Load metric definitions
    metric_defs_path = repo_root / "docs" / "thread_metric_score_reference.md"
    metric_definitions = ""
    if metric_defs_path.exists():
        metric_definitions = metric_defs_path.read_text(encoding="utf-8")

    from .orchestrator import (
        _print_group_eval_summary,
        _print_improvement_summary,
        _run_after_calibration_evaluation,
        _run_before_calibration_evaluation,
        run_calibration_loop,
    )
    from .overlay import save_overlay
    from .scorer import DEFAULT_METRICS
    from .stats import compare_before_after

    if evaluate_overlay_json is not None:
        import pandas as pd

        output_dir.mkdir(parents=True, exist_ok=True)
        overlay = json.loads(evaluate_overlay_json.read_text(encoding="utf-8"))
        save_overlay(overlay, output_dir / "best_overlay.json")

        metrics = DEFAULT_METRICS
        before_generated_df = pd.read_csv(vanilla_scores_csv) if vanilla_scores_csv is not None else None
        if before_group_eval_json is not None:
            before_eval = json.loads(before_group_eval_json.read_text(encoding="utf-8"))
            (output_dir / "before_calibration_group_eval.json").write_text(
                json.dumps(before_eval, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            print(f"\n{'='*60}")
            print("PHASE 0: Before-calibration group evaluation (reused)")
            print(f"{'='*60}")
            if before_generated_df is not None:
                print(f"  vanilla threads:   {len(before_generated_df)}")
            print(f"  → Reused {before_group_eval_json}")
        else:
            before_reference_run_config = (
                _force_vanilla_backbone(reference_run_config)
                if args.rerun_phase0_vanilla
                else None
            )
            before_eval, before_generated_df, _ = _run_before_calibration_evaluation(
                output_dir=output_dir,
                real_test_csv=real_test_csv,
                metrics=metrics,
                vanilla_scores_csv=vanilla_scores_csv,
                reference_run_config=before_reference_run_config,
                sim_runs=args.final_sim_runs,
                python=args.python,
                repo_root=repo_root,
                device=args.device,
                min_sim_threads=args.min_sim_threads,
                metric_parallel=args.metric_parallel,
                simulation_reasoning_effort=simulation_reasoning_effort or None,
            )
        after_eval = _run_after_calibration_evaluation(
            output_dir=output_dir,
            best_overlay=overlay,
            real_test_csv=real_test_csv,
            reference_run_config=reference_run_config,
            sim_runs=args.final_sim_runs,
            metrics=metrics,
            python=args.python,
            repo_root=repo_root,
            device=args.device,
            min_sim_threads=args.min_sim_threads,
            metric_parallel=args.metric_parallel,
            simulation_reasoning_effort=simulation_reasoning_effort or None,
        )
        if after_eval.get("group_eval"):
            print("  Calibrated vs real_test (key metrics):")
            _print_group_eval_summary(after_eval["group_eval"])

        print(f"\n{'='*60}")
        print("PHASE 3: Improvement analysis")
        print(f"{'='*60}")
        improvement = None
        if before_eval is not None and after_eval.get("group_eval") is not None:
            improvement = compare_before_after(
                before_eval,
                after_eval["group_eval"],
                real_df=pd.read_csv(real_test_csv),
                before_df=before_generated_df,
                after_df=after_eval.get("_all_sim_df"),
                metrics=metrics,
            )
            (output_dir / "before_after_improvement_summary.json").write_text(
                json.dumps(improvement, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            _print_improvement_summary(improvement)

        summary = {
            "best_overlay": overlay,
            "best_score": {},
            "completed_iterations": 0,
            "output_dir": str(output_dir),
            "after_calibration_evaluation": {
                k: v for k, v in after_eval.items() if k not in {"group_eval", "_all_sim_df"}
            },
            "improvement": improvement,
            "evaluate_overlay_json": str(evaluate_overlay_json),
        }
        (output_dir / "calibration_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"\n{'='*60}")
        print("OVERLAY EVALUATION COMPLETE")
        print(f"{'='*60}")
        print(f"Output: {output_dir}")
        return

    summary = run_calibration_loop(
        output_dir=output_dir,
        real_train_csv=real_train_csv,
        real_val_csv=real_val_csv,
        real_test_csv=real_test_csv,
        reference_run_config=reference_run_config,
        max_iterations=args.iterations,
        candidates_per_iter=args.candidates,
        parallel=args.parallel,
        calibration_model=args.calibration_model,
        api_key=api_key,
        base_url=base_url,
        seed=args.seed,
        python=args.python,
        repo_root=repo_root,
        metric_definitions=metric_definitions,
        device=args.device,
        final_sim_runs=args.final_sim_runs,
        vanilla_scores_csv=vanilla_scores_csv,
        rerun_phase0_vanilla=args.rerun_phase0_vanilla,
        min_sim_threads=args.min_sim_threads,
        metric_parallel=args.metric_parallel,
        calibration_reasoning_effort=args.calibration_reasoning_effort or None,
        simulation_reasoning_effort=simulation_reasoning_effort or None,
        stop_after_phase1=args.stop_after_phase1,
        calibration_rounds=args.calibration_rounds,
        combination_start_iteration=args.combination_start_iteration,
    )

    print(f"\n{'='*60}")
    print("CALIBRATION COMPLETE")
    print(f"{'='*60}")
    print(f"Output: {output_dir}")
    best_score = summary.get("best_score") or {}
    if best_score.get("quantile_fail_rate") is not None:
        print(f"Best quantile fail (val): {best_score.get('quantile_fail_rate', 'N/A')}")
        print(f"Best pct dist (val):      {best_score.get('mean_percentile_distance', 'N/A')}")
        print(f"Best robust z (val):      {best_score.get('mean_abs_robust_z', 'N/A')}")
        print(f"Legacy fail rate (val):   {best_score.get('fail_rate', 'N/A')}")
        print(f"Legacy mean |delta| (val): {best_score.get('mean_abs_delta', 'N/A')}")
    else:
        print(f"Best fail rate (val):    {best_score.get('fail_rate', 'N/A')}")
        print(f"Best mean |delta| (val): {best_score.get('mean_abs_delta', 'N/A')}")
    print(f"Iterations run:          {summary.get('completed_iterations', 'N/A')}")
    print(f"Best overlay:            {output_dir / 'best_overlay.json'}")
    if summary.get("stopped_after_phase1"):
        print("Stopped after phase 1:  YES")
    after = summary.get("after_calibration_evaluation") or {}
    if after.get("fail_rate") is not None:
        print(f"After-cal run fail rate: {after['fail_rate']:.4f}")
        print(f"After-cal run |delta|:   {after['mean_abs_delta']:.4f}")
    if summary.get("improvement"):
        s = summary["improvement"].get("summary", {})
        print(
            "Improvement: metric-avg fail rate "
            f"{s.get('overall_fail_rate_before', 0):.4f} → {s.get('overall_fail_rate_after', 0):.4f}"
        )
        print(
            "Improvement: avg Wasserstein "
            f"{s.get('avg_wasserstein_distance_before', float('nan')):.4f} → "
            f"{s.get('avg_wasserstein_distance_after', float('nan')):.4f}"
        )


if __name__ == "__main__":
    main()
