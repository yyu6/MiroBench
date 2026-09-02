#!/usr/bin/env bash
# Generate one (domain, writer) cohort under the pinned v137ds configuration.
#
#   ./experiments/geo_v137ds/run_geo_domain.sh headphone --writer deepseek-v4-flash
#
# Preflight runs ONCE and alone -- it builds the seed pool's domain profile, and
# two processes building it at the same time race.  Only after it succeeds do the
# generation shards fan out, one process per --shard-size seeds.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT/experiments/geo_v137ds/geo_config.sh"

PY="${GEO_PYTHON:-/Users/yaoningyu/.pyenv/versions/3.11.8/bin/python3}"
domain="${1:-}"; shift || true
writer="deepseek-v4-flash"; planner="$GEO_V137DS_PLANNER"
shard=3; jobs=0; pool=""; seed=""; date_tag="$(date +%Y%m%d)"
dry=0; seeds_spec=""; tag_prefix=""; matched=0; matched_dir=""; extra=()
# Per-seed isolation share by default; --fixed-isolation holds the Planner at
# the domain-wide constant instead, which is the only difference between the
# v141iso and v142isopt arms.
iso_csv="artifacts/geo_v137ds/isolation/{domain}.csv"
adopt=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --writer)      writer="$2"; shift 2 ;;
    --planner)     planner="$2"; shift 2 ;;
    --model)       planner="$2"; writer="$2"; shift 2 ;;
    --shard-size)  shard="$2"; shift 2 ;;
    --max-parallel) jobs="$2"; shift 2 ;;
    --pool-size)   pool="$2"; shift 2 ;;
    --sampling-seed) seed="$2"; shift 2 ;;
    --date)        date_tag="$2"; shift 2 ;;
    --seeds)       seeds_spec="$2"; shift 2 ;;
    --tag-prefix)  tag_prefix="$2"; shift 2 ;;
    --matched-profiles) matched=1; shift ;;
    --fixed-isolation)  iso_csv=""; shift ;;
    --adopt-observations) adopt="--adopt-observations"; shift ;;
    --dry-run)     dry=1; shift ;;
    -h|--help)     grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *)             extra+=("$1"); shift ;;
  esac
done

[[ -n "$domain" ]] || { echo "usage: run_geo_domain.sh <domain> [--writer M] [--shard-size N] [--max-parallel J]" >&2; exit 2; }
grep -qw "$domain" <<<"$GEO_V137DS_DOMAINS" || {
  echo "ERROR: '$domain' has no GEO domain config." >&2
  echo "       v137ds can serve: $GEO_V137DS_DOMAINS" >&2
  echo "       The other reddit_multidomain_baselines domains (celebrity, credit_cards," >&2
  echo "       game, health_issue, movies, news, sports, tv_series) have no entry under" >&2
  echo "       generalized_card/configs/domains/ and no scored real threads." >&2
  exit 2; }

read -r dpool dseed <<<"$(geo_pool_for "$domain")"
pool="${pool:-$dpool}"; seed="${seed:-$dseed}"
did="$(geo_domain_id "$domain")"
poolfile="$ROOT/artifacts/generalized_card/seed_pools/${did}_${pool}_seed${seed}.json"
if [[ ! -f "$poolfile" && "$domain" == *_geo ]]; then
  echo "NOTE: no seed pool yet for $domain; preflight will build"
  echo "      $(basename "$poolfile") and that file becomes the pin for every"
  echo "      later run of this domain. Do not delete it (G165)."
fi
[[ -f "$poolfile" || "$domain" == *_geo ]] || {
  echo "ERROR: seed pool missing: $poolfile" >&2
  echo "       run_generate would REBUILD it from (${pool}, ${seed}) and a rebuild does" >&2
  echo "       not reproduce the original sample (docs/DECISIONS.md G165). Restore the" >&2
  echo "       file, or pass --pool-size/--sampling-seed naming a pool that exists:" >&2
  ls "$ROOT/artifacts/generalized_card/seed_pools/" | sed 's/^/         /' >&2
  exit 2; }

read -r wurl wkey <<<"$(geo_model_endpoint "$writer")"
[[ -n "$wurl" ]] || { echo "ERROR: unknown writer '$writer' -- add it to geo_model_endpoint()." >&2; exit 2; }
read -r purl pkey <<<"$(geo_model_endpoint "$planner")"
[[ -n "$purl" ]] || { echo "ERROR: unknown planner '$planner' -- add it to geo_model_endpoint()." >&2; exit 2; }

# Fan-out is set by the WRITER, not the slower end: the Planner is called once
# per thread while the Writer is called once per comment, so a 45-comment thread
# is ~45 Writer calls to 1 Planner call and the Writer is what meets a rate
# limit.  The shipped camera cohort ran the gpt-5.4-mini x deepseek arm 38 ways
# with no OpenAI throttling.  DeepSeek v4-flash allows 2500 concurrent.
if [[ "$jobs" == "0" ]]; then jobs="$(geo_default_parallel "$writer")"; fi
ws="$(geo_model_short "$writer")"; ps="$(geo_model_short "$planner")"
if [[ "$planner" == "$writer" ]]; then arm="$ws"; else arm="${ps}x${ws}"; fi
prefix="${tag_prefix:-geo137_${domain}_${arm}_${date_tag}}"
LOGS="$ROOT/artifacts/geo_v137ds_logs/$prefix"; mkdir -p "$LOGS"

echo "domain=$domain  planner=$planner  writer=$writer"
echo "pool=$(basename "$poolfile")  seeds=$pool  shard-size=$shard  max-parallel=$jobs"
echo "tags=${prefix}_pNNN   logs=$LOGS"

gen() {   # gen <tag> <extra args...>
  cd "$ROOT/generalized_card"
  set -a; . "$ROOT/third_party/MiroFish/.env"; set +a
  "$PY" -u scripts/run_generate.py \
    --domain "$domain" --model "$planner" --base-url "$purl" --api-key-env "$pkey" \
    --writer-model "$writer" --writer-base-url "$wurl" --writer-api-key-env "$wkey" \
    --pool-size "$pool" --sampling-seed "$seed" --posts-per-run 1 \
    "${GEO_V137DS_FLAGS[@]}" "${extra[@]+"${extra[@]}"}" "$@"
}

if [[ "$dry" == "1" ]]; then
  echo; echo "--- preflight that would run ---"
  echo "run_generate.py --domain $domain --model $planner --writer-model $writer \\"
  echo "  --pool-size $pool --sampling-seed $seed <${#GEO_V137DS_FLAGS[@]} pinned flags> --prepare-only"
  if [[ -n "$seeds_spec" ]]; then
    echo "--- then these seed ranges, $jobs at a time ---"
    for sp in $seeds_spec; do echo "      ${prefix}_p${sp%%:*}  seeds ${sp%%:*}-$(( ${sp%%:*} + ${sp##*:} - 1 ))"; done
  else
    echo "--- then $(( (pool + shard - 1) / shard )) shards of $shard seeds, $jobs at a time ---"
  fi
  exit 0
fi

echo; echo "[1/2] preflight (builds the domain profile; must not run concurrently)"
gen --tag "${prefix}_preflight" --prepare-only 2>&1 | tee "$LOGS/preflight.log" | tail -5

# Hand the preflight's profile to every shard.  run_generate defaults the path
# to <run_root>/domain_profile.json, which is PER SHARD -- so without this each
# process rebuilds the profile from scratch, loading a sentence-transformer and
# embedding every reference thread.  Fifty of those at once is tens of GB of
# resident memory and will take a 24GB machine into swap until it stops
# responding; that, not any API error, is what killed the first celebrity run.
# The profile is a frozen measurement over non-seed threads, identical for every
# shard, so sharing it is also the only correct thing to do.
PROFILE="$ROOT/artifacts/generalized_card/runs/${prefix}_preflight/domain_profile.json"
if [[ -f "$PROFILE" ]]; then
  # Under --matched-profiles each shard names its own profile instead; adding
  # the shared one here too would leave two --domain-profile flags on the same
  # command line and make correctness depend on argparse's last-wins ordering.
  if [[ "$matched" != "1" ]]; then
    extra+=(--domain-profile "$PROFILE")
  fi
  echo "  shards will reuse $(basename "$(dirname "$PROFILE")")/domain_profile.json"
else
  echo "  WARNING: preflight left no domain_profile.json; each shard will build" >&2
  echo "           its own. Drop --max-parallel to 4 or fewer or the machine" >&2
  echo "           will swap." >&2
fi

# --matched-profiles: each seed gets behaviour targets read off ITS OWN matched
# real thread instead of the domain aggregate.  The aggregate sits 0.77 sd from
# any single thread, a property of thread-to-thread variance that no amount of
# aggregate refinement removes, so runs built this way are labelled
# `matched_profile` and must never be pooled with held-out-target runs.
# Shard size is forced to 1 because a shard carries one profile.
if [[ "$matched" == "1" ]]; then
  [[ -f "$PROFILE" ]] || { echo "ERROR: --matched-profiles needs the preflight profile as its base." >&2; exit 2; }
  if [[ "$shard" != "1" ]]; then
    echo "  --matched-profiles: forcing --shard-size 1 (one profile per shard)"
    shard=1
  fi
  matched_dir="$ROOT/artifacts/generalized_card/matched_profiles/${prefix}"
  if [[ -n "$seeds_spec" ]]; then
    mseeds=""
    for sp in $seeds_spec; do
      st="${sp%%:*}"; ct="${sp##*:}"
      for k in $(seq "$st" $((st + ct - 1))); do mseeds="$mseeds $k"; done
    done
  else
    mseeds="$(seq 0 $((pool - 1)) | tr '\n' ' ')"
  fi
  echo "  building per-seed profiles -> $(basename "$matched_dir")"
  "$PY" "$ROOT/experiments/geo_v137ds/matched_profile.py" "$did" \
      --base "$PROFILE" --out-dir "$matched_dir" --seeds $mseeds \
      --isolation-csv "$iso_csv" $adopt 2>&1 | tail -4
fi

# --seeds "6:9 23:2" generates exactly those ranges instead of sweeping the pool.
# Use it to fill the gaps a partial cohort left, without regenerating what exists.
if [[ -n "$seeds_spec" ]]; then
  plan=""
  for sp in $seeds_spec; do plan="$plan ${sp%%:*}:${sp##*:}"; done
  nshard=$(echo $plan | wc -w | tr -d ' ')
else
  plan=""
  for start in $(seq 0 "$shard" $((pool - 1))); do plan="$plan ${start}:${shard}"; done
  nshard=$(( (pool + shard - 1) / shard ))
fi

echo; echo "[2/2] generating $nshard shards, $jobs at a time"
running=0
for sp in $plan; do
  start="${sp%%:*}"; count="${sp##*:}"
  tag="${prefix}_p${start}"
  perseed=()
  if [[ -n "$matched_dir" ]]; then
    if [[ -f "$matched_dir/seed_${start}.json" ]]; then
      perseed=(--domain-profile "$matched_dir/seed_${start}.json")
    else
      echo "  SKIP $tag: no matched profile for seed $start (real thread unscored)" >&2
      continue
    fi
  fi
  ( gen --tag "$tag" --max-posts "$count" --start-seed-index "$start" --resume \
      "${perseed[@]+"${perseed[@]}"}" > "$LOGS/${tag}.log" 2>&1 ) &
  echo "  $tag  seeds ${start}-$((start + count - 1))  pid $!"
  # macOS ships bash 3.2, which has no `wait -n`; poll the job table instead.
  while [ "$(jobs -pr | wc -l | tr -d ' ')" -ge "$jobs" ]; do sleep 1; done
  sleep 0.3
done
wait
echo
echo "done. evaluate with:"
echo "  ./experiments/geo_v137ds/eval_geo_domain.sh $domain --writer $writer --planner $planner --date $date_tag"
