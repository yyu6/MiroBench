#!/usr/bin/env bash
# Wait for a tag prefix's shards to stop generating, then score and summarise.
#
#   chain.sh <domain> <writer> <tag-prefix> [--cohort NAME] [--planner P]
#
# Generation and evaluation are separate stages that people otherwise have to
# join by hand, which is how a finished cohort sits unscored for hours. This
# polls for the shards to finish, evaluates every shard that produced threads,
# and merges the pooled result into the shared summary.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${GEO_PYTHON:-/Users/yaoningyu/.pyenv/versions/3.11.8/bin/python3}"

domain="${1:?domain}"; writer="${2:?writer}"; prefix="${3:?tag-prefix}"; shift 3
planner="gpt-5.4-mini"; cohort=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --cohort)  cohort="$2"; shift 2 ;;
    --planner) planner="$2"; shift 2 ;;
    *) shift ;;
  esac
done
LOG="$ROOT/artifacts/geo_v137ds_logs/chain_${prefix}.log"
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

say "waiting for shards matching ${prefix}_p* to finish generating"
while pgrep -f "run_generate.py .*--tag ${prefix}_p" >/dev/null 2>&1; do sleep 60; done
say "generation stopped"

# Every shard of this prefix that produced at least one thread.
tags=$(ls "$ROOT/artifacts/generalized_card/runs" 2>/dev/null | grep -E "^${prefix}_p[0-9]+$" || true)
scored=0
for t in $tags; do
  n=$(find "$ROOT/artifacts/generalized_card/runs/$t/generated" -name discussion.json 2>/dev/null | wc -l | tr -d ' ')
  [[ "$n" == "0" ]] && { say "  skip $t (0 threads)"; continue; }
  printf '[%s]   %-40s %s threads ... ' "$(date +%H:%M:%S)" "$t" "$n" | tee -a "$LOG"
  if HF_HUB_OFFLINE=1 "$PY" "$ROOT/generalized_card/scripts/run_evaluate.py" \
       --tag "$t" --metric-parallel 4 --device auto --resume \
       >> "$ROOT/artifacts/geo_v137ds_logs/eval_${t}.log" 2>&1
  then echo ok | tee -a "$LOG"; scored=$((scored+1))
  else echo FAILED | tee -a "$LOG"; fi
done
say "scored $scored shards"

# Pool with whatever else belongs to this cohort. A fill run is only part of a
# cohort, so the summary row must come from the whole tag set, not this prefix.
alltags="$tags"
if [[ -n "$cohort" && -f "$ROOT/experiments/geo_v137ds/cohorts/$cohort.tags" ]]; then
  alltags="$(cat "$ROOT/experiments/geo_v137ds/cohorts/$cohort.tags") $tags"
  # keep the cohort file current so the next run sees the fill shards too
  printf '%s\n' $alltags | sort -u > "$ROOT/experiments/geo_v137ds/cohorts/$cohort.tags"
fi
[[ -z "$(echo $alltags)" ]] && { say "no tags to summarise"; exit 0; }

say "merging into the shared summary"
"$PY" "$ROOT/experiments/geo_v137ds/matched_pair_table.py" \
  --cohort "$domain" "$writer" --tags $alltags 2>&1 | tee -a "$LOG"
say "done"
