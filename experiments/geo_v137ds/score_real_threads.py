#!/usr/bin/env python3
"""Score every real thread of a domain into the thread_scores.csv GEO evaluates against.

run_baseline_evaluation.py has no --real-only: its phase 1 scores real threads,
but phase 2 runs simulations and spends API money. This calls phase 1's function
directly, so building a domain's real reference costs nothing but local compute.

  python3 experiments/geo_v137ds/score_real_threads.py celebrity --device mps
"""
import argparse, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "evaluation"))
from run_baseline_evaluation import score_all_real_threads  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("domain", help="the <name> in data/raw/discussions/<name>_geo")
ap.add_argument("--device", default="auto", choices=["cpu", "cuda", "mps", "auto"])
ap.add_argument("--force", action="store_true")
ap.add_argument("--batch-size", type=int, default=16)
a = ap.parse_args()

d = a.domain if a.domain.endswith("_geo") else f"{a.domain}_geo"
category = REPO / "data/raw/discussions" / d
if not category.is_dir():
    sys.exit(f"no corpus at {category} -- run enable_domain.sh {a.domain} first")
out = REPO / "artifacts/baselines" / d / "real/thread_scores.csv"
out.parent.mkdir(parents=True, exist_ok=True)

print(f"scoring real threads for {d} -> {out.relative_to(REPO)}")
csv_path = score_all_real_threads(
    category_dir=category,
    output_csv=out,
    python=sys.executable,
    device=a.device,
    self_bertscore_model_type="microsoft/deberta-xlarge-mnli",
    self_bertscore_batch_size=a.batch_size,
    force=a.force,
)
import csv as _csv
n = sum(1 for _ in _csv.DictReader(open(csv_path)))
print(f"\n{n} real threads scored -> {Path(csv_path).relative_to(REPO)}")
