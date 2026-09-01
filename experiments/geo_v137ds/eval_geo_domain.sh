#!/usr/bin/env bash
# Score every shard of one (domain, writer) cohort and print the pooled 12-metric
# table -- the same numbers the camera N=150 result was reported from.
#
#   ./experiments/geo_v137ds/eval_geo_domain.sh headphone --writer deepseek-v4-flash
#
# --dedupe drops a thread whose real counterpart already appeared in an earlier
# shard, so overlapping shards cannot double-weight a real thread.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT/experiments/geo_v137ds/geo_config.sh"
PY="${GEO_PYTHON:-/Users/yaoningyu/.pyenv/versions/3.11.8/bin/python3}"

domain="${1:-}"; shift || true
writer="deepseek-v4-flash"; planner="$GEO_V137DS_PLANNER"; date_tag=""; parallel=5; device="auto"; skip_score=0; tags_in=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --writer) writer="$2"; shift 2 ;;
    --planner) planner="$2"; shift 2 ;;
    --model)  planner="$2"; writer="$2"; shift 2 ;;
    --date)   date_tag="$2"; shift 2 ;;
    --metric-parallel) parallel="$2"; shift 2 ;;
    --device) device="$2"; shift 2 ;;
    --table-only) skip_score=1; shift ;;
    --tags) shift; while [ $# -gt 0 ] && [ "${1#--}" = "$1" ]; do tags_in="$tags_in $1"; shift; done ;;
    --cohort) tags_in="$tags_in $(tr '\n' ' ' < "$ROOT/experiments/geo_v137ds/cohorts/$2.tags")"; shift 2 ;;
    -h|--help) grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) shift ;;
  esac
done
[[ -n "$domain" ]] || { echo "usage: eval_geo_domain.sh <domain> [--writer M] [--date YYYYMMDD]" >&2; exit 2; }

ws="$(geo_model_short "$writer")"; ps="$(geo_model_short "$planner")"
if [ "$planner" = "$writer" ]; then arm="$ws"; else arm="${ps}x${ws}"; fi
pat="geo137_${domain}_${arm}_${date_tag:+${date_tag}_}"
# bash 3.2 on macOS: no mapfile, so read into a positional list.
# --tags takes an explicit list, for cohorts generated before this naming existed.
if [ -n "$tags_in" ]; then set -- $tags_in
else set -- $(ls "$ROOT/artifacts/generalized_card/runs" | grep -E "^${pat}.*_p[0-9]+$" | sort -t p -k2 -n); fi
[ $# -gt 0 ] || { echo "no shards match ${pat}*_pNNN under artifacts/generalized_card/runs" >&2; exit 1; }
tags=("$@")
echo "found ${#tags[@]} shards for domain=$domain writer=$writer"

if [[ "$skip_score" == "0" ]]; then
  echo "scoring (existing metric files are reused; safe to re-run)"
  for t in "${tags[@]}"; do
    printf '  %-46s' "$t"
    if HF_HUB_OFFLINE=1 "$PY" "$ROOT/generalized_card/scripts/run_evaluate.py" \
        --tag "$t" --metric-parallel "$parallel" --device "$device" --resume \
        > "$ROOT/artifacts/geo_v137ds_logs/eval_${t}.log" 2>&1
    then echo ok; else echo "FAILED (see artifacts/geo_v137ds_logs/eval_${t}.log)"; fi
  done
fi

echo
echo "########## GEO v137ds -- domain=$domain  writer=$writer ##########"
"$PY" "$ROOT/generalized_card/analysis/self_similarity/combined_eval.py" --dedupe --tags "${tags[@]}"

# Land the same numbers in the shared summary the baselines use, so one file
# answers "how did every generator do on every domain". Rows merge on
# (baseline, model, domain, test, metric): re-running one cohort replaces only
# its own rows.
echo
"$PY" "$ROOT/experiments/geo_v137ds/matched_pair_table.py" \
  --cohort "$domain" "$writer" --tags "${tags[@]}"
