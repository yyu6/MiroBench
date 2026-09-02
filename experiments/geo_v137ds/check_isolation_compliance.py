#!/usr/bin/env python3
"""Did the Planner actually deliver the isolation the quota asked for?

  python3 experiments/geo_v137ds/check_isolation_compliance.py iso2_20260902

The outsider quota's history is why this exists: G99 asked for 12% and the
Planner delivered 1.9%, and the arm was carried for two versions before anyone
measured the realized rate rather than the instruction. An instruction that is
issued is not an instruction that is obeyed.

Compares, per generated thread, the isolated-comment share the run achieved
against the share its own profile asked for, using the same embedding model and
the same threshold as measure_isolation.py.
"""
import argparse, csv, json, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MODEL = "sentence-transformers/all-mpnet-base-v2"

ap = argparse.ArgumentParser()
ap.add_argument("prefix")
ap.add_argument("--threshold", type=float, default=0.35)
ap.add_argument("--fixed-share", type=float, default=0.12,
                help="what a run without per-thread shares was asked for")
a = ap.parse_args()

runs = sorted((REPO / "artifacts/generalized_card/runs").glob(f"{a.prefix}_p*"))
prof_dir = REPO / "artifacts/generalized_card/matched_profiles" / a.prefix

rows = []
for r in runs:
    for disc in sorted(r.glob("generated/**/discussion.json")):
        d = json.loads(disc.read_text())
        # Comments hang off posts[0], carry their text in `content`, and nest
        # their own children under `replies`; a flat read of d["comments"] finds
        # nothing at all and reports the run as empty.
        def walk(nodes):
            for c in nodes or []:
                t = str(c.get("content") or c.get("body") or "").strip()
                if t:
                    yield t
                yield from walk(c.get("replies"))
        bodies = [t for post in d.get("posts") or [] for t in walk(post.get("comments"))]
        if len(bodies) < 4:
            continue
        seed = r.name.rsplit("_p", 1)[-1]
        pf = prof_dir / f"seed_{seed}.json"
        asked = a.fixed_share
        if pf.exists():
            v = json.loads(pf.read_text()).get("thread_isolation_share")
            if v is not None:
                asked = float(v)
        rows.append({"tag": r.name, "bodies": bodies, "asked": asked})

if not rows:
    sys.exit(f"{a.prefix}: 还没有生成出来的 thread")

from sentence_transformers import SentenceTransformer
import numpy as np

m = SentenceTransformer(MODEL, device="cpu")
flat, bounds, off = [], [], 0
for r in rows:
    flat.extend(r["bodies"]); bounds.append((off, off + len(r["bodies"]))); off += len(r["bodies"])
emb = m.encode(flat, normalize_embeddings=True, show_progress_bar=False,
               batch_size=128, convert_to_numpy=True)

out = []
for r, (lo, hi) in zip(rows, bounds):
    e = emb[lo:hi]
    sim = e @ e.T
    np.fill_diagonal(sim, -1.0)
    got = float((sim.max(axis=1) < a.threshold).mean())
    out.append((r["tag"], len(r["bodies"]), r["asked"], got))

asked = np.array([o[2] for o in out]); got = np.array([o[3] for o in out])
print(f"{a.prefix}: {len(out)} 个 thread\n")
print(f"  要求的孤立比例  平均 {asked.mean():.3f}")
print(f"  实际做到的      平均 {got.mean():.3f}")
if asked.mean() > 0:
    print(f"  执行率          {got.mean() / asked.mean() * 100:.0f}%")
if len(out) >= 5 and asked.std() > 1e-6:
    from scipy.stats import spearmanr
    rho, p = spearmanr(asked, got)
    print(f"\n  要求 vs 实际的相关  rho={rho:+.3f}  p={p:.4f}")
    print("  (逐 thread 支要看这个：相关高才说明 Planner 真的在跟着每条 thread 走)")
print(f"\n  {'tag':<26}{'评论数':>6}{'要求':>8}{'实际':>8}")
for tag, n, q, g in out[:20]:
    print(f"  {tag:<26}{n:>6}{q:>8.3f}{g:>8.3f}")
