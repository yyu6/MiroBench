from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
from dotenv import load_dotenv


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Iterative LLM-driven calibration for Reddit discussion simulation."
    )
    parser.add_argument("--products-json", required=True, help="Product JSON file.")
    parser.add_argument("--real-dir", required=True, help="Real discussion directory.")
    parser.add_argument("--reference-run-dir", required=True, help="Reference simulation directory.")
    parser.add_argument("--iterations", type=int, default=10, help="Calibration iterations (default: 10).")
    parser.add_argument("--candidates", type=int, default=5, help="Candidates per iteration (default: 5).")
    parser.add_argument("--parallel", type=int, default=1, help="Max concurrent simulations (default: 1).")
    parser.add_argument("--calibration-model", default="gpt-4o-mini", help="LLM for calibration reasoning.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    parser.add_argument("--output-dir", default="artifacts/calibration_runs", help="Output directory.")
    parser.add_argument("--resume", action="store_true", help="Resume a previous run.")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps", "auto"], help="Device for torch-based metrics.")
    parser.add_argument("--python", default=sys.executable, help="Python executable.")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    real_dir = Path(args.real_dir).resolve()
    reference_run_dir = Path(args.reference_run_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    # Load reference run config
    run_config_path = reference_run_dir / "run_config.json"
    if not run_config_path.exists():
        print(f"ERROR: run_config.json not found in {reference_run_dir}")
        sys.exit(1)
    reference_run_config = json.loads(run_config_path.read_text(encoding="utf-8"))

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
    )

    print(f"\n{'='*60}")
    print("CALIBRATION COMPLETE")
    print(f"{'='*60}")
    print(f"Output: {output_dir}")
    print(f"Best fail rate: {summary.get('best_fail_rate', 'N/A')}")
    print(f"Successful strategies: {summary.get('successful_strategies', [])}")
    print(f"Failed strategies: {summary.get('failed_strategies', [])}")
