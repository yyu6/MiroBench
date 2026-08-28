#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
exec "${PYTHON:-python3}" "$ROOT/experiments/reddit_multidomain_baselines/scripts/run_generation.py" "$@"
