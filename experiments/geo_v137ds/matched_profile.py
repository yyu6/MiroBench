#!/usr/bin/env python3
"""Build one domain profile per seed, with behaviour targets taken from that
seed's OWN matched real thread instead of the domain aggregate.

  python3 experiments/geo_v137ds/matched_profile.py celebrity_geo \
      --base <a built profile.json> --seeds 0 1 2 ... --out-dir <dir>

Writes <out-dir>/seed_<i>.json, to be passed to run_generate as --domain-profile.

The aggregate profile is 0.77 standard deviations away from any individual
thread's value, which is a property of thread-to-thread variance and not of the
measurement, so no amount of aggregate refinement predicts a single thread.
Reading the target off the thread itself removes that distance by construction.
Runs built this way are therefore labelled `matched_profile` and must not be
pooled with, or compared against, runs whose targets came from held-out data.
"""
import argparse, csv, json, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "generalized_card"))
from generalized_card.domain_profile import profile_hash  # noqa: E402

# behaviour target  <-  column in the real thread's own scored row
TARGET_FROM_METRIC = {
    "tone_polite_min_share":   ("polite_rate",             0.02, 0.60),
    "tone_harsh_max_share":    ("impolite_rate",           0.02, 0.90),
    "story_personal_min_share":("mean_story_probability",  0.00, 0.60),
}

ap = argparse.ArgumentParser()
ap.add_argument("domain")
ap.add_argument("--base", required=True, help="an already-built domain_profile.json")
ap.add_argument("--seeds", nargs="+", type=int, required=True)
ap.add_argument("--out-dir", required=True)
a = ap.parse_args()

dom = a.domain if a.domain.endswith("_geo") or "_" in a.domain else f"{a.domain}_geo"
scores = REPO / "artifacts/baselines" / dom / "real/thread_scores.csv"
if not scores.exists():
    scores = REPO / "artifacts/baselines/camera_product_gpt5mini/real/thread_scores.csv"
by_thread = {r["thread_id"]: r for r in csv.DictReader(open(scores))}

pool_name = {"celebrity_geo": "celebrity_geo_150_seed907",
             "camera": "camera_product_150_seed907"}.get(dom, f"{dom}_150_seed907")
pool = json.load(open(REPO / "artifacts/generalized_card/seed_pools" / f"{pool_name}.json"))
seed_of = {int(r["seed_index"]): str(r["source_raw_post_id"]) for r in pool["seed_posts"]}

base = json.load(open(a.base))
out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
written = 0
for i in a.seeds:
    rid = seed_of.get(i)
    row = by_thread.get(rid)
    if row is None:
        print(f"  seed {i} ({rid}): 真实 thread 没打过分，跳过"); continue
    p = json.loads(json.dumps(base))
    bt = dict(p.get("behavior_targets") or {})
    applied = {}
    for target, (metric, lo, hi) in TARGET_FROM_METRIC.items():
        v = row.get(metric)
        if v in (None, "", "nan"): continue
        bt[target] = max(lo, min(hi, float(v)))
        applied[target] = round(bt[target], 4)
    p["behavior_targets"] = bt
    p["matched_profile"] = {"seed_index": i, "source_raw_post_id": rid, "applied": applied}
    # The profile carries an integrity hash; editing the payload invalidates it.
    p.pop("profile_sha256", None)
    p["profile_sha256"] = profile_hash(p)
    (out / f"seed_{i}.json").write_text(json.dumps(p))
    written += 1
    print(f"  seed {i:>3} ({rid}): " + "  ".join(f"{k.split('_')[1]}={v}" for k, v in applied.items()))
print(f"\n{written} 个 per-thread profile -> {out}")
