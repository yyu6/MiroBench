#!/usr/bin/env bash
# Snapshot every module the v137ds result depends on, with hashes.
#
#   ./experiments/geo_v137ds/freeze.sh                # write the snapshot
#   ./experiments/geo_v137ds/freeze.sh --verify       # check it still matches
#
# The snapshot is a copy, not a substitute for the git commit: run_generate.py
# refuses to start when a core-contract source is uncommitted, so the commit is
# what actually pins a run.  This exists so the version survives a branch move,
# and so a later run can be proved byte-identical to the one that scored 8/12.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DEST="${GEO_FREEZE_DEST:-$ROOT/artifacts/geo_v137ds_frozen}"
verify=0; [[ "${1:-}" == "--verify" ]] && verify=1

PATHS="
generalized_card/generalized_card
generalized_card/scripts
generalized_card/configs
generalized_card/analysis/self_similarity
scripts/evaluation
experiments/geo_v137ds
"
POOLS="camera_product_150_seed907 camera_product_95_seed907 headphone_product_150_seed42 cell_phone_product_100_seed42 laptop_product_100_seed42"

if [[ "$verify" == "1" ]]; then
  [[ -f "$DEST/MANIFEST.sha256" ]] || { echo "no snapshot at $DEST" >&2; exit 1; }
  cd "$DEST"
  if shasum -a 256 -c MANIFEST.sha256 --quiet 2>/dev/null; then
    echo "snapshot intact: $DEST"
  else
    echo "SNAPSHOT DRIFTED -- files above no longer match MANIFEST.sha256" >&2; exit 1
  fi
  echo; echo "--- snapshot vs working tree ---"
  drift=0
  for p in $PATHS; do
    if ! diff -rq --exclude=__pycache__ --exclude='*.pyc' "$DEST/tree/$p" "$ROOT/$p" >/tmp/geo_drift.$$ 2>&1; then
      echo "DRIFT in $p:"; sed 's/^/    /' /tmp/geo_drift.$$; drift=1
    fi
  done
  rm -f /tmp/geo_drift.$$
  [[ "$drift" == "0" ]] && echo "working tree matches the frozen version" || \
    echo "the working tree has moved away from v137ds -- a run from it is NOT v137ds"
  exit 0
fi

rm -rf "$DEST"; mkdir -p "$DEST/tree" "$DEST/seed_pools"
for p in $PATHS; do
  mkdir -p "$DEST/tree/$(dirname "$p")"
  rsync -a --exclude='__pycache__' --exclude='*.pyc' "$ROOT/$p/" "$DEST/tree/$p/"
done
for pool in $POOLS; do
  f="$ROOT/artifacts/generalized_card/seed_pools/$pool.json"
  [[ -f "$f" ]] && cp "$f" "$DEST/seed_pools/" || echo "  (no pool $pool)"
done

{
  echo "# GEO v137ds -- frozen $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo
  echo "git commit : $(cd "$ROOT" && git rev-parse HEAD)"
  echo "git branch : $(cd "$ROOT" && git rev-parse --abbrev-ref HEAD)"
  echo "dirty      : $(cd "$ROOT" && git status --porcelain $PATHS | wc -l | tr -d ' ') tracked path(s) modified at freeze time"
  echo
  echo "Measured result, camera_product N=150 matched-pair (docs/DECISIONS.md G154-G175):"
  echo "  PASS 8/12"
  echo "  self_bertscore  d +0.10  mwu 0.143  PASS      <- target metric"
  echo "  self_bleu_4     d -0.03  mwu 0.610  PASS      <- target metric"
  echo "  fails: hard_disagree KS 0.011, impolite 0.000, length_cv 0.032, emotion_entropy KS 0.005"
  echo
  echo "Text-level discriminability (G174/G175):"
  echo "  bag-of-words gen-vs-real AUC 0.987, thread level 1.000"
  echo "  function words only 0.951 against a human-human band of 0.65-0.72"
  echo
  echo "Frozen paths:"; for p in $PATHS; do echo "  $p"; done
  echo "Frozen seed pools:"; ls "$DEST/seed_pools" | sed 's/^/  /'
} > "$DEST/VERSION.txt"

cd "$DEST"
find tree seed_pools -type f ! -name '*.pyc' -print0 | sort -z | xargs -0 shasum -a 256 > MANIFEST.sha256
echo "frozen -> $DEST"
echo "  $(wc -l < MANIFEST.sha256 | tr -d ' ') files, $(du -sh "$DEST" | cut -f1)"
echo "  verify later with: ./experiments/geo_v137ds/freeze.sh --verify"
