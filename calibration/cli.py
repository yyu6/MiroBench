from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
from dotenv import load_dotenv


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Iterative LLM-driven calibration for Reddit discussion simulation."
    )

    # ── Real data ─────────────────────────────────────────────────────────────
    parser.add_argument(
        "--real-dir",
        required=True,
        help=(
            "Category dir containing real discussion subdirs "
            "(e.g., data/raw/discussions/credit_cards), or a single product "
            "dir with a thread_metrics_summary.csv.  All found CSVs are "
            "aggregated into the real baseline."
        ),
    )

    # ── Simulation parameters (replaces --reference-run-dir) ─────────────────
    parser.add_argument("--products-json", required=True, help="Product JSON file.")
    parser.add_argument("--agents", type=int, default=50, help="Number of agents (default: 50).")
    parser.add_argument("--hours", type=int, default=24, help="Simulated hours (default: 24).")
    parser.add_argument("--rounds", type=int, default=24, help="Max simulation rounds (default: 24).")
    parser.add_argument("--seed-posts", type=int, default=4, help="Seed posts per run (default: 4).")
    parser.add_argument("--hint", type=str, default=None, help="Optional hint for persona/topic generation.")

    # ── Calibration control ───────────────────────────────────────────────────
    parser.add_argument("--iterations", type=int, default=10, help="Calibration iterations (default: 10).")
    parser.add_argument("--candidates", type=int, default=5, help="Candidates per iteration (default: 5).")
    parser.add_argument("--parallel", type=int, default=1, help="Max concurrent simulations (default: 1).")
    parser.add_argument("--calibration-model", default="gpt-4o-mini", help="LLM for calibration reasoning.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    parser.add_argument("--output-dir", default="artifacts/calibration_runs", help="Output directory.")
    parser.add_argument("--resume", action="store_true", help="Resume a previous run.")
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda", "mps", "auto"],
        help="Device for torch-based metrics.",
    )
    parser.add_argument("--python", default=sys.executable, help="Python executable.")

    # ── Few-shot ──────────────────────────────────────────────────────────────
    parser.add_argument(
        "--baseline-sim-dir",
        default=None,
        help=(
            "Simulation output directory from a prior baseline evaluation run "
            "used as the few-shot source for calibration iteration 0.  "
            "Subsequent iterations use the winning candidate sim dir automatically."
        ),
    )
    parser.add_argument(
        "--few-shot-count",
        type=int,
        default=3,
        help="Few-shot examples injected per candidate simulation (default: 3).",
    )

    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    real_dir = Path(args.real_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    # Build reference_run_config directly from CLI args (no run_config.json needed)
    reference_run_config: dict = {
        "input_file": str(Path(args.products_json).resolve()),
        "agents": args.agents,
        "hours": args.hours,
        "rounds": args.rounds,
        "seed_posts": args.seed_posts,
        "seed": args.seed,
        "discussion_backbone": "vanilla_oasis",
    }
    if args.hint:
        reference_run_config["hint"] = args.hint

    # Create timestamped output dir if not resuming
    if not args.resume:
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = output_dir / f"calibration_{ts}"

    api_key = os.environ.get("LLM_API_KEY", "")
    base_url = os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1")

    # Load metric definitions
    metric_defs_path = repo_root / "docs" / "thread_metric_score_reference.md"
    metric_definitions = ""
    if metric_defs_path.exists():
        metric_definitions = metric_defs_path.read_text(encoding="utf-8")

    from .orchestrator import run_calibration_loop

    baseline_sim_dir = (
        Path(args.baseline_sim_dir).resolve()
        if args.baseline_sim_dir
        else None
    )

    summary = run_calibration_loop(
        output_dir=output_dir,
        real_dir=real_dir,
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
        baseline_sim_dir=baseline_sim_dir,
        few_shot_count=args.few_shot_count,
    )

    print(f"\n{'='*60}")
    print("CALIBRATION COMPLETE")
    print(f"{'='*60}")
    print(f"Output: {output_dir}")
    best_score = summary.get("best_score") or {}
    print(f"Best fail rate:     {best_score.get('fail_rate', 'N/A')}")
    print(f"Best mean |delta|:  {best_score.get('mean_abs_delta', 'N/A')}")
    print(f"Iterations run:     {summary.get('completed_iterations', 'N/A')}")
    print(f"Best overlay:       {output_dir / 'best_overlay.json'}")
