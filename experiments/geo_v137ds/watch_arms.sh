#!/usr/bin/env bash
# Keep each arm's cohort tag list current and report progress.
#
#   ./experiments/geo_v137ds/watch_arms.sh
#
# An arm is spread over its original prefix and any refill prefixes, and a shard
# that died leaves a log and no run directory. Counting directories therefore
# overstates progress and pooling by the original prefix alone silently drops the
# refill. Both are answered here from the only reliable evidence: a discussion
# file on disk.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUNS="$ROOT/artifacts/generalized_card/runs"
printf '%-10s %-6s %-6s %s\n' arm 完成 在跑 tag
for pair in "iso2 iso2b" "isopt isoptb" "obs obsb" "iso3 iso3b" "win2 win2" "raw2 raw2"; do
  set -- $pair; a="$1"; b="$2"
  pat="^(${a}|${b})_20260902_p[0-9]+$"
  tags=$(ls "$RUNS" 2>/dev/null | grep -E "$pat" | sort -u)
  done_n=0; list=""
  for t in $tags; do
    k=$(find "$RUNS/$t/generated" -name discussion.json 2>/dev/null | wc -l | tr -d ' ')
    if [ "$k" != "0" ]; then
      done_n=$((done_n + k)); list="$list$t
"
    fi
  done
  live=$(ps aux | grep "[r]un_generate.py" | grep -cE "(${a}|${b})_20260902_p")
  printf '%-10s %-6s %-6s %s\n' "$a" "$done_n/50" "$live" "$(echo "$tags" | wc -l | tr -d ' ')"
  printf '%s' "$list" > "$ROOT/experiments/geo_v137ds/cohorts/${a}_celebrity.tags"
done
