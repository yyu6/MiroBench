#!/usr/bin/env bash
# Report each thread's plan spread the moment it lands.
#
#   ./experiments/geo_v137ds/watch_plans.sh <prefix>
#
# A run can be wrong from its first thread and stay wrong for hours. The two
# signals that show it immediately: whether the Planner's own perspective choice
# survives to the task (seed_local was overwritten to 0.0% for 1,878 slots
# before anyone looked), and whether a thread's local_topic values repeat.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
prefix="${1:?prefix}"
seen=""
while :; do
  for f in "$ROOT/artifacts/generalized_card/runs/${prefix}"_p*/generated/run_00_sampled_reddit/discussion.json; do
    [ -f "$f" ] || continue
    tag=$(basename "$(dirname "$(dirname "$(dirname "$f")")")")
    case " $seen " in *" $tag "*) continue ;; esac
    seen="$seen $tag"
    "${GEO_PYTHON:-python3}" - "$f" "$tag" <<'PY'
import json, sys, collections
f, tag = sys.argv[1], sys.argv[2]
post = json.load(open(f))["posts"][0]
recs = [r for r in (post.get("generation_records") or []) if isinstance(r.get("comment"), dict)]
topics = [str((r.get("task") or {}).get("local_topic") or "") for r in recs]
persp = collections.Counter(str((r.get("task") or {}).get("perspective_id") or "") for r in recs)
tones = collections.Counter(str((r.get("task") or {}).get("tone_target") or "") for r in recs)
n = max(1, len(recs))
dupes = len(topics) - len(set(topics))
print(f"[{tag}] {len(recs)} 槽  重复 local_topic {dupes}  "
      f"seed_local {persp.get('seed_local',0)/n*100:.0f}%  "
      f"最常用镜头 {persp.most_common(1)[0][0]} {persp.most_common(1)[0][1]/n*100:.0f}%  "
      f"语气 {'/'.join(f'{k[:4]}{v*100//n}' for k,v in tones.most_common(3))}")
PY
  done
  sleep 30
done
