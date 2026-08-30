#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
default_python="python3"
if [[ -x "$ROOT/.venv_reddit_baselines/bin/python" ]]; then
  default_python="$ROOT/.venv_reddit_baselines/bin/python"
fi
exec "${PYTHON:-$default_python}" "$ROOT/experiments/reddit_multidomain_baselines/scripts/run_generation.py" "$@"
