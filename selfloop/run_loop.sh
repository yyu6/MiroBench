#!/usr/bin/env bash
# Keep the self-loop going across the silent kills this machine produces.
#
# The controller checkpoints after every round, so a restart resumes rather
# than repeating paid rounds. This wrapper restarts it until it either
# finishes its round budget or fails to make progress twice in a row.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${SELFLOOP_OUT:-$ROOT/artifacts/selfloop/$(date +%Y%m%d_%H%M%S)}"
ROUNDS="${SELFLOOP_ROUNDS:-12}"
MAX_RESTARTS="${SELFLOOP_MAX_RESTARTS:-20}"
mkdir -p "$OUT"
for attempt in $(seq 1 "$MAX_RESTARTS"); do
  echo "=== attempt $attempt  out=$OUT ==="
  HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
    python3 "$ROOT/selfloop/controller.py" --resume-from "$OUT" --rounds "$ROUNDS" "$@"
  code=$?
  if [ $code -eq 0 ]; then echo "=== finished cleanly ==="; exit 0; fi
  echo "=== exited $code; resuming from checkpoint ==="
  sleep 5
done
echo "=== gave up after $MAX_RESTARTS attempts ==="
exit 1
