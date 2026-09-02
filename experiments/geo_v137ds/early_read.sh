#!/usr/bin/env bash
# Score whatever an arm has finished so far and print the verdict table.
#
#   ./experiments/geo_v137ds/early_read.sh <arm>          # iso3, win2, raw2 ...
#
# An arm whose semantic_mean_cosine is still rejected at N around 13 does not
# need to reach N=50: small N means low power, so a rejection that survives it
# is a gap more samples can only sharpen. The converse does not hold -- a pass
# at that N is bought by the sample size, so this can retire an arm but never
# ship one.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${GEO_PYTHON:-/Users/yaoningyu/.pyenv/versions/3.11.8/bin/python3}"
arm="${1:?arm}"
"$ROOT/experiments/geo_v137ds/watch_arms.sh" >/dev/null 2>&1
tagfile="$ROOT/experiments/geo_v137ds/cohorts/${arm}_celebrity.tags"
[ -s "$tagfile" ] || { echo "$arm: 还没有完成的 thread"; exit 1; }
tags=$(cat "$tagfile")
for t in $tags; do
  find "$ROOT/artifacts/generalized_card/runs/$t/generated" -name discussion.json >/dev/null 2>&1 || continue
  printf '[%s] scoring %s\n' "$(date +%H:%M:%S)" "$t"
  HF_HUB_OFFLINE=1 "$PY" "$ROOT/generalized_card/scripts/run_evaluate.py" \
      --tag "$t" --metric-parallel 2 --device auto --resume \
      >> "$ROOT/artifacts/geo_v137ds_logs/eval_${t}.log" 2>&1
done
"$PY" "$ROOT/experiments/geo_v137ds/matched_pair_table.py" \
  --cohort celebrity_geo deepseek-v4-flash --tags $tags 2>&1 | grep -v Warning
"$PY" - <<'PY'
import csv
rows = [r for r in csv.DictReader(open("artifacts/reddit_multidomain_baselines/summary/evaluation_summary.csv"))
        if r.get("baseline") == "geo" and r.get("test") == "matched_pair"][-12:]
if not rows:
    raise SystemExit
hdr = f"\n{'metric':<26}{'gen':>9}{'real':>9}{'rel%':>8}{'mwu':>8}{'ks':>8}{'d':>7}  判定"
print(f"N = {rows[0]['generated_n']}")
print(hdr); print("-" * (len(hdr) - 1))
npass = 0
for r in rows:
    g, rl = float(r["generated_mean"]), float(r["real_mean"])
    mw, ks, d = float(r["mwu_p_value"]), float(r["ks_p_value"]), float(r["cliffs_delta"])
    ok = mw > 0.05 and ks > 0.05
    npass += ok
    print(f"{r['metric']:<26}{g:>9.4f}{rl:>9.4f}{(g - rl) / abs(rl) * 100:>7.1f}%"
          f"{mw:>8.3f}{ks:>8.3f}{d:>+7.2f}  {'PASS' if ok else 'FAIL'}")
print(f"\nPASS {npass}/12")
tgt = {r["metric"]: r for r in rows}
for m in ("semantic_mean_cosine", "self_bertscore_mean_f1"):
    r = tgt.get(m)
    if r:
        print(f"  {m:<24} d={float(r['cliffs_delta']):+.2f}  "
              f"mwu={float(r['mwu_p_value']):.3f}  ks={float(r['ks_p_value']):.3f}")
sc = tgt.get("semantic_mean_cosine")
if sc and (float(sc["mwu_p_value"]) <= 0.05 or float(sc["ks_p_value"]) <= 0.05):
    print("\n  -> semantic_mean_cosine 在低功效下仍被拒：这一支可以停，不必跑到 50")
PY
