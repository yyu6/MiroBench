#!/usr/bin/env python3
"""Two arms on the SAME seed, every layer that has meaning at N=1.

  python3 experiments/geo_v137ds/compare_same_seed.py 1 v153_20260903 a5dsfit_20260902

The 12-metric table needs many threads before its p-values mean anything, but
the surface quantities are per-thread measurements and are readable on one:
within-thread content cosine, the function-word register cosine, comment length,
and identity variety. Each is reported against the seed's OWN matched real
thread, so "better" means closer to that thread rather than closer to an
aggregate (G192: never compare an arm against another arm's pooled figure).
"""
import collections
import json
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "experiments/geo_v137ds"))
from surface_vs_content import profiles, real_by_seed  # noqa: E402

MARKER = re.compile(r'persona-id="([^"]+)"')
MODEL = "sentence-transformers/all-mpnet-base-v2"


def arm(prefix, seed):
    d = REPO / "artifacts/generalized_card/runs" / f"{prefix}_p{seed}"
    f = d / "generated/run_00_sampled_reddit/generation_records.json"
    if not f.exists():
        return None
    recs = json.load(open(f))
    return {
        "texts": [str(r.get("raw") or "").strip() for r in recs if str(r.get("raw") or "").strip()],
        "tasks": [r["task"] for r in recs],
        "personas": [MARKER.search(r.get("prompt") or "").group(1)
                     for r in recs if MARKER.search(r.get("prompt") or "")],
        "speakers": {str((r.get("comment") or {}).get("author") or "") for r in recs},
    }


def offd(v):
    iu = np.triu_indices(len(v), 1)
    return (v @ v.T)[iu]


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    prefixes = sys.argv[2:] or ["v153_20260903", "a5dsfit_20260902"]
    from sentence_transformers import SentenceTransformer

    m = SentenceTransformer(MODEL, device="cpu")
    real = (real_by_seed().get(seed) or [])
    arms = [(p, arm(p, seed)) for p in prefixes]
    arms = [(p, a) for p, a in arms if a]
    if not arms or len(real) < 4:
        sys.exit("缺数据")

    n = len(arms[0][1]["texts"])
    rt = real[:n]
    rV = m.encode(rt, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
    rc = offd(rV).mean()
    rR = offd(profiles(rt)[0]).mean()
    rw = np.median([len(t.split()) for t in rt])

    print(f"\nseed {seed}   真实 thread {len(real)} 条\n")
    print(f"{'':<26}{'内容余弦':>10}{'说话方式':>10}{'词数中位':>10}{'不同身份':>10}{'lens':>7}")
    print(f"  {'真人':<24}{rc:>10.4f}{rR:>10.4f}{rw:>10.0f}{'—':>10}{'—':>7}")
    for p, a in arms:
        V = m.encode(a["texts"], normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
        c = offd(V).mean()
        R = offd(profiles(a["texts"])[0]).mean()
        w = np.median([len(t.split()) for t in a["texts"]])
        lens = len({str(t.get("perspective_id") or "") for t in a["tasks"]})
        print(f"  {p.split('_')[0]:<24}{c:>10.4f}{R:>10.4f}{w:>10.0f}"
              f"{len(set(a['personas'])):>10}{lens:>7}")
        print(f"  {'':<24}{(c-rc)/rc:>+9.0%}{(R-rR)/rR:>+10.0%}"
              f"{(w-rw)/rw:>+9.0%}   /{len(a['speakers'])} 说话人")


if __name__ == "__main__":
    main()
