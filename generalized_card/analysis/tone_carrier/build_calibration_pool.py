"""Build a seed pool with ZERO overlap against the evaluation pool, for calibration only.

`tone_realization.POLITE_ASSIGNMENT_CAP` is pinned at the measured `agree` stance
share because P(realize polite | assign polite) has only ever been observed on
agree slots (n=261) plus a thin n=17 on uncertain. Raising it needs a calibration
run under `--tone-quota calibrate`, and that run must not sit on evaluation seeds
or the calibration is in-sample for the artifact it calibrates.

`build_seed_pool` takes no exclusion list and its distribution-preserving sample
at a different seed still overlaps: seed 7 shares 55 of 150 posts with the
evaluation pool, and there is no contiguous clean window for
`--start-seed-index`. So this writes a filtered pool directly.

`run_generate.py` reuses an existing pool file and only builds one when the path
is missing, so the file this writes is what the run will use. That is also the
hazard: deleting it and re-running would silently produce an UNFILTERED pool at
the same path. Run this script immediately before the calibration run, and check
the printed overlap is 0.

Usage:
    python3 generalized_card/analysis/tone_carrier/build_calibration_pool.py
"""
from __future__ import annotations
import json, sys
from pathlib import Path

REPO = Path("/Users/yaoningyu/Desktop/UIUC/GEO")
sys.path.insert(0, str(REPO / "generalized_card"))
from generalized_card.data import build_seed_pool  # noqa: E402
from generalized_card.domain import load_domain_config  # noqa: E402

DOMAIN = "camera_product"
EVAL_POOL = REPO / "artifacts/generalized_card/seed_pools/camera_product_150_seed42.json"
# A marker value, not a real sampling seed: the pool at this path is filtered and
# is NOT what `build_seed_pool(count=N, seed=907)` would produce on its own.
CALIBRATION_SEED = 907
SOURCE_SEED = 7

evaluation_ids = {
    str(row["source_raw_post_id"])
    for row in json.load(open(EVAL_POOL))["seed_posts"]
}
config = load_domain_config(DOMAIN)
scratch = Path("/private/tmp/claude-501/-Users-yaoningyu-Desktop-UIUC-GEO/"
               "1f41c5a0-3c0b-415c-9b23-9fb381e5c727/scratchpad/_calib_source.json")
payload = build_seed_pool(config, scratch, count=150, seed=SOURCE_SEED)

clean = [
    row for row in payload["seed_posts"]
    if str(row["source_raw_post_id"]) not in evaluation_ids
]
for index, row in enumerate(clean):
    row["seed_index"] = index

overlap = len(payload["seed_posts"]) - len(clean)
print(f"source pool seed{SOURCE_SEED}: {len(payload['seed_posts'])} posts, "
      f"{overlap} of them in the evaluation pool")
print(f"calibration pool: {len(clean)} posts, overlap with the evaluation 150 = "
      f"{len({str(r['source_raw_post_id']) for r in clean} & evaluation_ids)}")

out = (REPO / "artifacts/generalized_card/seed_pools"
       / f"{DOMAIN}_{len(clean)}_seed{CALIBRATION_SEED}.json")
payload["seed_posts"] = clean
payload["meta"] = dict(payload.get("meta") or {})
payload["meta"].update({
    "builder": "generalized_card/analysis/tone_carrier/build_calibration_pool.py",
    "count": len(clean),
    "purpose": "tone realization calibration only; never an evaluation pool",
    "source_sampling_seed": SOURCE_SEED,
    "filtered_against": str(EVAL_POOL.name),
    "evaluation_overlap": 0,
    "warning": (
        "run_generate rebuilds this path UNFILTERED if the file is deleted. "
        "Rebuild with the builder above, not by deleting."
    ),
})
out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print(f"\nwrote {out}")
print(f"\nrun it with:  --pool-size {len(clean)} --sampling-seed {CALIBRATION_SEED}")
print("first ten threads:")
for row in clean[:10]:
    print(f"   seed_index {row['seed_index']:>3}  comments {row['real_num_comments']:>4}  "
          f"{row['source_product_dir']}")
print(f"\ntotal comments in the first ten: {sum(r['real_num_comments'] for r in clean[:10])}")
