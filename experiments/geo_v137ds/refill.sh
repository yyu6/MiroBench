#!/usr/bin/env bash
# Relaunch the seeds of a cohort that have neither finished nor are still running.
#
#   ./experiments/geo_v137ds/refill.sh <tag-prefix> <total-seeds> [extra run flags...]
#
# A shard that died leaves a log and no run directory, so counting directories
# overstates progress. This asks the only two questions that matter: did the seed
# produce a discussion, and is a process still working on it.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
prefix="${1:?tag-prefix}"; total="${2:?total seeds}"; shift 2

missing=""
for i in $(seq 0 $((total - 1))); do
  d="$ROOT/artifacts/generalized_card/runs/${prefix}_p${i}"
  if [ -f "$d/generated/run_00_sampled_reddit/discussion.json" ]; then continue; fi
  if ps aux | grep "[r]un_generate.py" | grep -q "${prefix}_p${i} "; then continue; fi
  missing="$missing ${i}:1"
done
n=$(echo $missing | wc -w | tr -d ' ')
if [ "$n" = "0" ]; then echo "$prefix: 没有需要补的"; exit 0; fi
echo "$prefix: 补 $n 个 seed"
exec "$ROOT/experiments/geo_v137ds/run_geo_domain.sh" celebrity_geo \
  --writer deepseek-v4-flash --matched-profiles --reference-floor measured \
  --tag-prefix "$prefix" --pool-size 150 --seeds "$missing" "$@"
