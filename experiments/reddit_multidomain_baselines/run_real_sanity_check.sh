#!/usr/bin/env bash
# Score the 12 fixed real references once, then run repeated n=150 vs n=150 checks.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
python_bin="${PYTHON_BIN:-$root/.venv_reddit_baselines/bin/python}"
device="auto"
metric_parallel="2"
skip_scoring="false"
analysis_args=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --device)
      device="$2"
      shift 2
      ;;
    --metric-parallel)
      metric_parallel="$2"
      shift 2
      ;;
    --skip-scoring)
      skip_scoring="true"
      shift
      ;;
    -h|--help)
      cat <<'EOF'
Usage: run_real_sanity_check.sh [options]

Options:
  --device cpu|mps|cuda|auto   Device used only for missing metric scores.
  --metric-parallel N          Concurrent metric scorers per real-reference run.
  --skip-scoring               Require all cached real score CSVs to exist.

All other arguments are passed to real_vs_real_sanity.py. Examples:
  ./experiments/reddit_multidomain_baselines/run_real_sanity_check.sh --device mps
  ./experiments/reddit_multidomain_baselines/run_real_sanity_check.sh \
    --skip-scoring --repeats 200 --sample-size 150

The default protocol draws two independent bootstrap samples of 150 from each
domain's fixed 150-thread real reference, repeated 200 times. Metric scoring is
cached and resumed; the statistical resampling itself makes no model/API calls.
EOF
      exit 0
      ;;
    *)
      analysis_args+=("$1")
      shift
      ;;
  esac
done

if [[ ! -x "$python_bin" ]]; then
  python_bin="${PYTHON:-python3}"
fi

run_root="$root/artifacts/reddit_multidomain_baselines"
domains=(camera celebrity cellphone credit_cards game headphones health_issue laptop movies news sports tv_series)

if [[ "$skip_scoring" != "true" ]]; then
  "$python_bin" \
    "$root/experiments/reddit_multidomain_baselines/scripts/prepare_real_sanity_scores.py" \
    --run-root "$run_root" \
    --domains "${domains[@]}" \
    --device "$device" \
    --metric-parallel "$metric_parallel" \
    --expected-threads 150
fi

exec "$python_bin" \
  "$root/experiments/reddit_multidomain_baselines/scripts/real_vs_real_sanity.py" \
  --run-root "$run_root" \
  --publish-dir "$root/experiments/reddit_multidomain_baselines/results/real_vs_real_sanity" \
  "${analysis_args[@]}"
