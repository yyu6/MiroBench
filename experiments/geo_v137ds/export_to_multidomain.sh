#!/usr/bin/env bash
# Publish a finished GEO v137ds cohort into artifacts/reddit_multidomain_baselines
# as baseline "geo", so the harness's own evaluator scores it exactly like oasis
# and synthpai -- same code, same real reference, no change to that harness.
#
#   ./experiments/geo_v137ds/export_to_multidomain.sh camera --writer deepseek-v4-flash
#   ./experiments/reddit_multidomain_baselines/run_evaluate_domain.sh camera \
#       --models deepseek-v4-flash --baselines geo --device auto
#
# READ THIS BEFORE COMPARING NUMBERS.  Two different tests now exist for the same
# cohort and they are NOT interchangeable:
#
#   * GEO's own table (eval_geo_domain.sh) pairs every generated thread with
#     the real thread it was built from, under data/raw/discussions/.  That is
#     the 8/12 result.
#   * The multidomain table scores the same threads against
#     inputs/real_reference/<domain>, a DIFFERENT real corpus -- camera there is
#     r/photography from data/reddit_domain_posts, sharing ZERO posts with GEO's
#     camera_product pool.  It is the right comparison against oasis/synthpai and
#     the wrong one to quote as the matched-pair result.
#
# Both land side by side; GEO's table is kept as geo_matched_pair.txt.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT/experiments/geo_v137ds/geo_config.sh"
PY="${GEO_PYTHON:-/Users/yaoningyu/.pyenv/versions/3.11.8/bin/python3}"

domain="${1:-}"; shift || true
writer="deepseek-v4-flash"; planner="$GEO_V137DS_PLANNER"; date_tag=""; mode="symlink"; tags_in=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --writer) writer="$2"; shift 2 ;;
    --planner) planner="$2"; shift 2 ;;
    --model)  planner="$2"; writer="$2"; shift 2 ;;
    --date)   date_tag="$2"; shift 2 ;;
    --tags)   shift; while [[ $# -gt 0 && "$1" != --* ]]; do tags_in="$tags_in $1"; shift; done ;;
    --cohort) tags_in="$tags_in $(tr '\n' ' ' < "$ROOT/experiments/geo_v137ds/cohorts/$2.tags")"; shift 2 ;;
    --copy)   mode="copy"; shift ;;
    -h|--help) grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) shift ;;
  esac
done
[[ -n "$domain" ]] || { echo "usage: export_to_multidomain.sh <domain> [--writer M] [--tags T1 T2 ...] [--copy]" >&2; exit 2; }

case "$domain" in
  camera) mdd="camera" ;; cell_phone) mdd="cellphone" ;;
  headphone) mdd="headphones" ;; laptop) mdd="laptop" ;;
  # enable_domain.sh names a domain <name>_geo; the harness knows it as <name>.
  *_geo)  mdd="${domain%_geo}" ;;
  *) echo "ERROR: no multidomain name for GEO domain '$domain'" >&2; exit 2 ;;
esac
MD="$ROOT/artifacts/reddit_multidomain_baselines"
[[ -f "$MD/inputs/real_reference/$mdd/reference_manifest.json" ]] || {
  echo "ERROR: no real reference at $MD/inputs/real_reference/$mdd -- the harness" >&2
  echo "       cannot score a domain it has no reference for." >&2; exit 2; }

if [[ -z "$tags_in" ]]; then
  ws="$(geo_model_short "$writer")"; ps="$(geo_model_short "$planner")"
  if [ "$planner" = "$writer" ]; then arm="$ws"; else arm="${ps}x${ws}"; fi
  pat="^geo137_${domain}_${arm}_${date_tag:+${date_tag}_}.*_p[0-9]+$"
  tags_in=$(ls "$ROOT/artifacts/generalized_card/runs" | grep -E "$pat" | sort -t p -k2 -n | tr '\n' ' ')
fi
[[ -n "$tags_in" ]] || { echo "no shards found; pass --tags explicitly" >&2; exit 1; }

read -r wurl wkey <<<"$(geo_writer_endpoint "$writer")"
out="$MD/generation/card/$writer/$mdd"

GEO_ROOT="$ROOT" GEO_OUT="$out" GEO_MODE="$mode" GEO_WRITER="$writer" \
GEO_MDD="$mdd" GEO_URL="$wurl" "$PY" - $tags_in <<'PYX'
"""Link each shard's threads into the multidomain layout, deduped on source post."""
import json, os, pathlib, shutil, sys, datetime

root = pathlib.Path(os.environ["GEO_ROOT"])
out = pathlib.Path(os.environ["GEO_OUT"])
mode, writer, mdd = os.environ["GEO_MODE"], os.environ["GEO_WRITER"], os.environ["GEO_MDD"]
tags = sys.argv[1:]

gen = out / "generated"
gen.mkdir(parents=True, exist_ok=True)
for old in gen.glob("run_*_sampled_reddit"):
    shutil.rmtree(old)
print(f"exporting {len(tags)} shards -> {gen}  (mode: {mode})")

seen, n, threads, dup, missing = set(), 0, 0, 0, []
for tag in tags:
    src = root / "artifacts/generalized_card/runs" / tag / "cleaned"
    if not src.is_dir():
        missing.append(tag); continue
    for rd in sorted(src.glob("run_*_sampled_reddit")):
        f = rd / "discussion.json"
        if not f.is_file():
            continue
        posts = json.load(open(f))["posts"]
        pids = [str(p.get("source_raw_post_id")) for p in posts]
        fresh = [p for p in pids if p not in seen]
        dup += len(pids) - len(fresh)
        if not fresh:
            continue
        if len(fresh) != len(pids):
            print(f"  WARN  {tag}/{rd.name} mixes new and duplicate seeds -- exported whole")
        seen.update(pids)
        dst = gen / f"run_{n:03d}_sampled_reddit"
        dst.mkdir(parents=True, exist_ok=True)
        if mode == "copy":
            shutil.copy2(f, dst / "discussion.json")
        else:
            os.symlink(f, dst / "discussion.json")
        threads += len(pids); n += 1

for t in missing:
    print(f"  WARN  {t} has no cleaned/ -- skipped (generation ran, clean+score did not)")
print(f"  {n} run dirs, {threads} threads exported"
      + (f", {dup} duplicate seeds dropped" if dup else ""))

json.dump({
    "status": "success", "dry_run": False,
    "baseline": "geo", "model": writer, "domain": mdd,
    "base_url": os.environ.get("GEO_URL", ""),
    "generator": f"GEO v137ds (Planner gpt-5.4-mini + Writer {writer})",
    "generator_config": str(root / "experiments/geo_v137ds/geo_config.sh"),
    "source_tags": tags,
    "generated_root": str(gen),
    "run_count": n, "thread_count": threads,
    "exported_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "note": (f"Scored here against inputs/real_reference/{mdd}, a different real corpus "
             "from the data/raw/discussions pool GEO was matched to. See "
             "geo_matched_pair.txt for GEO's own matched-pair table."),
}, open(out / "generation_report.json", "w"), indent=2)
print(f"  wrote {out}/generation_report.json  (baseline=geo model={writer} domain={mdd})")
PYX

echo "  attaching GEO's own matched-pair table"
"$PY" "$ROOT/generalized_card/analysis/self_similarity/combined_eval.py" --dedupe --tags $tags_in \
  > "$out/geo_matched_pair.txt" 2>&1 \
  || echo "    (skipped: shards not scored yet -- run eval_geo_domain.sh first)"

cat <<EOF

next:
  ./experiments/reddit_multidomain_baselines/run_evaluate_domain.sh $mdd \\
    --models $writer --baselines geo --device auto
EOF
