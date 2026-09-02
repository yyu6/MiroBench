#!/usr/bin/env python3
"""Compare what each arm actually generated against the real seed threads.

  python3 experiments/geo_v137ds/compare_arms.py

Reads the surface properties the arms are meant to move -- comment length, the
share of semantically isolated comments, and mean within-thread cosine -- so an
arm can be judged before its p-values arrive, and so an arm that changed nothing
is not left running for hours.
"""
import collections, glob, json, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
MODEL = "sentence-transformers/all-mpnet-base-v2"
ARMS = [("真实", None),
        ("v139mp 基准", ("mprof_",)),
        ("v141iso", ("iso2_", "iso2b_")),
        ("v145isopt", ("iso3_", "iso3b_")),
        ("v144win2 半拿掉", ("win2_",)),
        ("v146raw2 全拿掉", ("raw2_",))]


def walk(nodes):
    for c in nodes or []:
        t = str(c.get("content") or c.get("body") or "").strip()
        if t:
            yield t
        yield from walk(c.get("replies"))


def generated(prefixes):
    out = []
    for f in glob.glob(str(REPO / "artifacts/generalized_card/runs/*/generated/run_00_sampled_reddit/discussion.json")):
        tag = f.split("/")[-4]
        # isopt_ also startswith-matches nothing else, but iso2_ vs isopt_ do not
        # overlap only because of the trailing underscore -- keep it.
        if not any(tag.startswith(p) for p in prefixes):
            continue
        for post in json.load(open(f)).get("posts") or []:
            bodies = list(walk(post.get("comments")))
            if len(bodies) >= 4:
                out.append(bodies)
    return out


def real_threads(limit=50):
    by_post = collections.defaultdict(list)
    src = REPO / "data/raw/discussions/celebrity_geo/celebrity/celebrity.comments.jsonl"
    for line in src.open():
        c = json.loads(line)
        t = (c.get("body") or "").strip()
        if t and t not in ("[deleted]", "[removed]"):
            by_post[str(c.get("post_id"))].append(t)
    pool = json.loads((REPO / "artifacts/generalized_card/seed_pools/celebrity_geo_150_seed907.json").read_text())
    out = []
    for r in pool["seed_posts"][:limit]:
        b = by_post.get(str(r["source_raw_post_id"]), [])[:120]
        if len(b) >= 4:
            out.append(b)
    return out


def main() -> None:
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(MODEL, device="cpu")
    print("目标是靠近『真实』那一行\n")
    print(f"  {'':<16}{'thread':>7}{'词数中位':>9}{'≤10词':>8}{'孤立':>8}{'thread内余弦':>13}")
    for name, prefixes in ARMS:
        threads = real_threads() if prefixes is None else generated(prefixes)
        if not threads:
            print(f"  {name:<16}{'(还没有数据)':>10}")
            continue
        flat = [t for th in threads for t in th]
        emb = m.encode(flat, normalize_embeddings=True, batch_size=128,
                       show_progress_bar=False, convert_to_numpy=True)
        off, iso, cos = 0, [], []
        for th in threads:
            k = len(th)
            v = emb[off:off + k]
            off += k
            s = v @ v.T
            np.fill_diagonal(s, -1.0)
            iso.append(float((s.max(axis=1) < 0.35).mean()))
            s = v @ v.T
            np.fill_diagonal(s, np.nan)
            cos.append(float(np.nanmean(s)))
        w = [len(t.split()) for t in flat]
        print(f"  {name:<16}{len(threads):>7}{np.median(w):>9.0f}"
              f"{np.mean([x <= 10 for x in w]) * 100:>7.1f}%"
              f"{np.mean(iso) * 100:>7.1f}%{np.mean(cos):>13.4f}")


if __name__ == "__main__":
    main()
