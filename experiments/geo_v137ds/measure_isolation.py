#!/usr/bin/env python3
"""Measure, per real thread, the share of comments that are semantically
isolated from every sibling in the same thread.

  python3 experiments/geo_v137ds/measure_isolation.py celebrity_geo \
      --out artifacts/geo_v137ds/isolation/celebrity_geo.csv

A comment is "isolated" when its highest cosine similarity to any OTHER comment
in the same thread falls below --threshold. The share varies enormously between
real threads (celebrity: 0.04 to 1.00), which is why a single domain-wide
constant is the wrong instrument -- half the threads get an instruction that
contradicts what their own humans did. `matched_profile.py` reads this file and
gives each seed the share measured on its own matched real thread.

Embedding model is the one the semantic scorer itself uses, so the number the
Planner is asked to hit lives on the same scale as the metric being judged.
"""
import argparse, csv, json, sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MODEL = "sentence-transformers/all-mpnet-base-v2"

ap = argparse.ArgumentParser()
ap.add_argument("domain")
ap.add_argument("--threshold", type=float, default=0.35)
ap.add_argument("--min-comments", type=int, default=4)
ap.add_argument("--max-comments", type=int, default=120,
                help="cap per thread; the tail is a cost sink, not a signal")
ap.add_argument("--pool", default="",
                help="seed pool json; restricts the measurement to that pool's "
                     "own threads instead of the whole corpus")
ap.add_argument("--out", required=True)
a = ap.parse_args()

cfg = json.loads((REPO / "generalized_card/configs/domains" / f"{a.domain}.json").read_text())
raw = REPO / cfg["raw_discussions_dir"]
jl = sorted(raw.rglob("*.comments.jsonl"))
if not jl:
    sys.exit(f"没找到 {raw} 下的 *.comments.jsonl")

by_post = defaultdict(list)
for f in jl:
    for line in f.open():
        line = line.strip()
        if not line:
            continue
        c = json.loads(line)
        body = (c.get("body") or "").strip()
        if body and body not in ("[deleted]", "[removed]"):
            by_post[str(c.get("post_id"))].append(body)

threads = {p: c[: a.max_comments] for p, c in by_post.items() if len(c) >= a.min_comments}
if a.pool:
    pf = Path(a.pool)
    if not pf.is_absolute():
        pf = REPO / "artifacts/generalized_card/seed_pools" / pf
    want = {str(r["source_raw_post_id"]) for r in json.loads(pf.read_text())["seed_posts"]}
    threads = {p: c for p, c in threads.items() if p in want}
    print(f"限定在 {pf.name} 的 {len(want)} 条种子帖 -> 命中 {len(threads)} 个")
print(f"{len(threads)} 个 thread (>= {a.min_comments} 条评论)，开始编码…")

from sentence_transformers import SentenceTransformer
import numpy as np

m = SentenceTransformer(MODEL, device="cpu")

# One encode call for the whole corpus, then slice per thread. Encoding thread
# by thread instead costs one model-dispatch round trip per thread, which on CPU
# dominates the arithmetic and made a ten-minute job open-ended.
order = sorted(threads.items())
flat, bounds, off = [], [], 0
for pid, bodies in order:
    flat.extend(bodies)
    bounds.append((pid, off, off + len(bodies)))
    off += len(bodies)
print(f"共 {len(flat)} 条评论，一次性编码…")
emb = m.encode(flat, normalize_embeddings=True, show_progress_bar=False,
               batch_size=128, convert_to_numpy=True)

rows = []
for pid, lo, hi in bounds:
    e = emb[lo:hi]
    sim = e @ e.T
    np.fill_diagonal(sim, -1.0)          # a comment is never its own neighbour
    nn = sim.max(axis=1)
    rows.append({"thread_id": pid,
                 "comment_count": hi - lo,
                 "isolation_share": round(float((nn < a.threshold).mean()), 4),
                 "nn_cosine_mean": round(float(nn.mean()), 4)})

out = REPO / a.out
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=["thread_id", "comment_count",
                                       "isolation_share", "nn_cosine_mean"])
    w.writeheader()
    w.writerows(rows)

s = sorted(r["isolation_share"] for r in rows)
q = lambda p: s[min(len(s) - 1, int(len(s) * p))]
print(f"\n{len(rows)} 行 -> {out}")
print(f"孤立比例  中位 {q(0.5):.3f}   四分位 {q(0.25):.3f}~{q(0.75):.3f}   "
      f"范围 {s[0]:.3f}~{s[-1]:.3f}")
