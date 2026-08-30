#!/usr/bin/env bash
# Bootstrap a fresh clone for multi-domain generation and evaluation.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
experiment="$root/experiments/reddit_multidomain_baselines"
python_bin="${PYTHON_BIN:-}"
venv="$root/.venv_reddit_baselines"
skip_stance=0
skip_synthpai=0

for arg in "$@"; do
  case "$arg" in
    --skip-stance) skip_stance=1 ;;
    --skip-synthpai) skip_synthpai=1 ;;
    -h|--help)
      cat <<'EOF'
Usage: ./experiments/reddit_multidomain_baselines/setup.sh [options]

Options:
  --skip-stance    Do not download the ~500 MB StanceRel checkpoint.
                   Generation still works; full Behavior evaluation does not.
  --skip-synthpai  Prepare only OASIS generation and evaluation.

Environment:
  PYTHON_BIN        Python 3.11/3.12 executable (default: prefer python3.11)
EOF
      exit 0
      ;;
    *) echo "Unknown setup option: $arg" >&2; exit 2 ;;
  esac
done

if [[ -z "$python_bin" ]]; then
  if command -v python3.11 >/dev/null 2>&1; then
    python_bin="python3.11"
  elif command -v python3.12 >/dev/null 2>&1; then
    python_bin="python3.12"
  else
    python_bin="python3"
  fi
fi

"$python_bin" -c 'import sys; assert (3, 11) <= sys.version_info[:2] < (3, 13), "Python 3.11 or 3.12 is required"'

if [[ ! -x "$venv/bin/python" ]]; then
  "$python_bin" -m venv "$venv"
fi
"$venv/bin/python" -m pip install --upgrade pip setuptools wheel
"$venv/bin/python" -m pip install -e "$root"
"$venv/bin/python" -m pip install -r "$experiment/requirements-evaluation.txt"

bootstrap_command=(
  "$venv/bin/python"
  "$experiment/scripts/bootstrap_external_repos.py"
)
doctor_command=("$experiment/doctor.sh")
if [[ "$skip_synthpai" -eq 1 ]]; then
  bootstrap_command+=(--skip-synthpai)
  doctor_command+=(--skip-synthpai)
fi
"${bootstrap_command[@]}"
"$venv/bin/python" -m pip install -r "$root/third_party/MiroFish/backend/requirements.txt"

if [[ "$skip_synthpai" -eq 0 ]]; then
  synthpai_venv="$root/SynthPAI/.venv"
  if [[ ! -x "$synthpai_venv/bin/python" ]]; then
    "$python_bin" -m venv "$synthpai_venv"
  fi
  "$synthpai_venv/bin/python" -m pip install --upgrade pip setuptools wheel
  "$synthpai_venv/bin/python" -m pip install -r "$experiment/requirements-synthpai.txt"
fi

"$venv/bin/python" "$experiment/scripts/install_portable_inputs.py"
if [[ "$skip_stance" -eq 0 ]]; then
  "$venv/bin/python" "$experiment/scripts/download_stance_checkpoint.py"
else
  doctor_command+=(--skip-stance)
fi

"${doctor_command[@]}"

cat <<EOF

[setup-complete]
Activate the environment before running commands:
  source "$venv/bin/activate"

Then export the API key(s) you need and run, for example:
  ./experiments/reddit_multidomain_baselines/run_domain.sh laptop --models gpt-4o-mini --baselines oasis
  ./experiments/reddit_multidomain_baselines/run_evaluate_domain.sh laptop --models gpt-4o-mini --baselines oasis --device auto
EOF
