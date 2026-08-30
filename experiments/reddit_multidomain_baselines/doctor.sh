#!/usr/bin/env bash
# Read-only fresh-machine readiness check.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
experiment="$root/experiments/reddit_multidomain_baselines"
python="$root/.venv_reddit_baselines/bin/python"
skip_stance=0
skip_synthpai=0

for arg in "$@"; do
  case "$arg" in
    --skip-stance) skip_stance=1 ;;
    --skip-synthpai) skip_synthpai=1 ;;
    *) echo "Unknown doctor option: $arg" >&2; exit 2 ;;
  esac
done

[[ -x "$python" ]] || { echo "Missing environment: run $experiment/setup.sh" >&2; exit 1; }
"$python" "$experiment/scripts/install_portable_inputs.py" --verify-only
bootstrap_args=(--verify-only)
if [[ "$skip_synthpai" -eq 1 ]]; then
  bootstrap_args+=(--skip-synthpai)
fi
"$python" "$experiment/scripts/bootstrap_external_repos.py" "${bootstrap_args[@]}"
if [[ "$skip_stance" -eq 0 ]]; then
  "$python" "$experiment/scripts/download_stance_checkpoint.py" --verify-only
fi
"$python" -c 'import bert_score, numpy, openai, pandas, scipy, sentence_transformers, torch, transformers'
"$python" -m pip check >/dev/null
"$python" "$experiment/scripts/run_generation.py" --help >/dev/null
"$python" "$experiment/scripts/evaluate.py" --help >/dev/null
if [[ "$skip_synthpai" -eq 0 ]]; then
  "$root/SynthPAI/.venv/bin/python" -c 'import openai, numpy, scipy, yaml, pyinputplus'
fi
echo "[doctor-ok] fresh-machine runtime is ready"
