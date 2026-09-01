#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

if [[ -x "$REPO_ROOT/.venv_reddit_baselines/bin/python" ]]; then
  PYTHON="$REPO_ROOT/.venv_reddit_baselines/bin/python"
else
  PYTHON="${PYTHON:-python3}"
fi

exec "$PYTHON" "$SCRIPT_DIR/scripts/recalculate_generation_costs.py" "$@"
