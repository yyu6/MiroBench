#!/usr/bin/env bash
# Wait until an arm has N finished threads, then score and print the verdict.
#
#   ./experiments/geo_v137ds/auto_read.sh <arm> <N>
#
# Reading early costs one scoring pass and can retire an arm outright: a
# rejection at low N has survived the lowest power we will ever give it, so more
# samples can only sharpen it. A pass at low N means nothing -- see
# early_read.sh.
#
# It also carries a bias that must be read with it. Threads finish in size
# order, because a hundred-comment thread is ten times the Writer calls of a
# ten-comment one, so the first N to complete are the SMALLEST N. On celebrity
# the first ten came in at a median of 9 comments against the fifty seeds' own
# median of 52. A small thread has far less room for within-thread similarity to
# diverge, and its avg_depth and structural_virality sit in a different range
# entirely. So an early read compares an arm against a different population than
# a full run does, and effect sizes from the two are not interchangeable.
# Early FAIL: act on it. Early PASS: it means the arm survived the easy cases.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
arm="${1:?arm}"; want="${2:-10}"
while :; do
  n=$(find "$ROOT/artifacts/generalized_card/runs/${arm}_20260902_p"*/generated \
        -name discussion.json 2>/dev/null | wc -l | tr -d ' ')
  [ "${n:-0}" -ge "$want" ] && break
  # Stop waiting if generation has died, or N will never arrive.
  live=$(ps -eo args 2>/dev/null | grep -c "[-]-tag ${arm}_20260902_p" || true)
  if [ "${live:-0}" = "0" ]; then
    echo "生成已停，当前只有 ${n:-0} 条；直接读这些"
    break
  fi
  sleep 60
done
exec "$ROOT/experiments/geo_v137ds/early_read.sh" "$arm"
