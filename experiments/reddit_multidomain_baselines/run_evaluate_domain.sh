#!/usr/bin/env bash
# Evaluate every generated baseline/model result for one Reddit domain.
set -euo pipefail

if [[ $# -lt 1 || "$1" == "-h" || "$1" == "--help" ]]; then
  cat >&2 <<'EOF'
Usage: run_evaluate_domain.sh <domain> [evaluation options]

Examples:
  ./experiments/reddit_multidomain_baselines/run_evaluate_domain.sh laptop --device auto
  ./experiments/reddit_multidomain_baselines/run_evaluate_domain.sh news --models gpt-4o-mini --baselines oasis

The first argument is the only domain to evaluate. Remaining options are
passed to run_evaluate_all.sh; run with --help after a domain to see all such
options.
EOF
  exit 2
fi

domain="$1"
shift
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

exec "$root/experiments/reddit_multidomain_baselines/run_evaluate_all.sh" \
  --domains "$domain" "$@"
