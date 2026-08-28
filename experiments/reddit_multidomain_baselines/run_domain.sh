#!/usr/bin/env bash
# Generate every configured model and baseline for one Reddit domain.
set -euo pipefail

if [[ $# -lt 1 || "$1" == "-h" || "$1" == "--help" ]]; then
  cat >&2 <<'EOF'
Usage: run_domain.sh <domain> [generation options]

Examples:
  ./experiments/reddit_multidomain_baselines/run_domain.sh laptop
  ./experiments/reddit_multidomain_baselines/run_domain.sh camera --dry-run --max-seeds 3
  ./experiments/reddit_multidomain_baselines/run_domain.sh news --models gpt-4o-mini --baselines oasis

The first argument is the only domain to run. Remaining options are passed to
run_generate_all.sh; run with --help after a domain to see all such options.
EOF
  exit 2
fi

domain="$1"
shift
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

exec "$root/experiments/reddit_multidomain_baselines/run_generate_all.sh" \
  --domains "$domain" --continue-on-error "$@"
