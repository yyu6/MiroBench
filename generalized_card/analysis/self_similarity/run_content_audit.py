#!/usr/bin/env python3
"""Guard + surface audit over real run artifacts, against matched real.

Reports the checks `validate_writer_text` enforces alongside the surface
profile, so a metric win bought by degraded text is visible rather than hidden.
"""
from __future__ import annotations
import argparse, json, re, statistics, sys
from collections import Counter
from pathlib import Path
REPO = Path("/Users/yaoningyu/Desktop/UIUC/GEO")
sys.path.insert(0, str(REPO / "scripts" / "evaluation"))
from score_thread_semantic_uniformity import load_generated_comments, load_real_comments  # noqa: E402

SKELETON = re.compile(r"\bP\d{2}\|S\d+\|B\d+\b")
LEAK = re.compile(r"\b(semantic_move|claim_family|surface_skeleton|context_aperture|payload form|opener_type|perspective_id)\b", re.I)
BANNED = {", honestly": re.compile(r",\s*honestly"), "that part": re.compile(r"\bthat part\b")}
ASSERT = re.compile(r"\b(is|are|isn't|aren't)\s+the\b|\bthat's\s+the\b|\bthe\s+(real|only|actual|whole|key|main)\b", re.I)
FIRST = re.compile(r"\b(i|i'm|i've|i'd|my|me)\b", re.I)
TICS = ["actually", "check", "matters", "tradeoff", "honestly", "whether", "personally"]
TOK = re.compile(r"[a-z0-9']+")

def overlap(texts, n):
    g = []
    for t in texts:
        w = TOK.findall(t.lower())
        g.append(set(zip(*[w[i:] for i in range(n)])) if len(w) >= n else set())
    v = []
    for i in range(len(g)):
        for j in range(i + 1, len(g)):
            u = g[i] | g[j]
            if u: v.append(len(g[i] & g[j]) / len(u))
    return statistics.mean(v) if v else float("nan")

def profile(by_thread):
    texts = [t for v in by_thread.values() for t in v]
    n = len(texts)
    words = [len(t.split()) for t in texts]
    rep = 0
    for t in texts:
        w = TOK.findall(t.lower())
        if len(w) >= 12:
            c = Counter(zip(w, w[1:]))
            if c and c.most_common(1)[0][1] >= 4: rep += 1
    out = {"comments": n, "threads": len(by_thread),
           "empty": sum(1 for t in texts if not t.strip()),
           "mean_words": statistics.mean(words), "median_words": statistics.median(words),
           "skeleton_residue": sum(bool(SKELETON.search(t)) for t in texts),
           "control_leak": sum(bool(LEAK.search(t)) for t in texts),
           "internal_repetition": rep,
           "assertion_frame": sum(bool(ASSERT.search(t)) for t in texts) / n,
           "first_person": sum(bool(FIRST.search(t)) for t in texts) / n,
           "question": sum("?" in t for t in texts) / n}
    for k, rx in BANNED.items():
        out[f"banned[{k}]"] = sum(bool(rx.search(t)) for t in texts)
    for tic in TICS:
        out[f"tic[{tic}]"] = sum(bool(re.search(rf"\b{tic}\b", t, re.I)) for t in texts) / n
    out["unigram_overlap"] = statistics.mean([overlap(v, 1) for v in by_thread.values() if len(v) > 1])
    out["bigram_overlap"] = statistics.mean([overlap(v, 2) for v in by_thread.values() if len(v) > 1])
    return out

def gen_texts(tag):
    base = REPO / "artifacts/generalized_card/runs" / tag / "cleaned"
    if not base.is_dir(): base = REPO / "artifacts/generalized_card/runs" / tag / "generated"
    out = {}
    for d in sorted(base.glob("run_*_sampled_reddit")):
        cbt, _ = load_generated_comments(d)
        for tid, cs in cbt.items():
            out[f"seed{int(tid.split('seed')[-1]):03d}"] = [c.text for c in cs]
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tags", nargs="+")
    ap.add_argument("--samples", type=int, default=0)
    a = ap.parse_args()
    sets = {t: gen_texts(t) for t in a.tags}
    pool = json.loads((REPO / "artifacts/generalized_card/seed_pools/camera_product_150_seed42.json").read_text())["seed_posts"]
    by_seed = {int(p["seed_index"]): p for p in pool}
    real, cache = {}, {}
    for tid in next(iter(sets.values())):
        sp = by_seed.get(int(tid.replace("seed", "")))
        if not sp: continue
        d = REPO / "data/raw/discussions/camera_product" / sp["source_product_dir"]
        if d not in cache: cache[d] = load_real_comments(d)[0]
        real[tid] = [c.text for c in (cache[d].get(sp["source_raw_post_id"]) or [])]
    corp = {"REAL": real, **{t.split("_")[0]: v for t, v in sets.items()}}
    profs = {k: profile(v) for k, v in corp.items()}
    keys = list(profs["REAL"])
    w = max(len(k) for k in keys) + 2
    print(f"{'metric':<{w}} " + " ".join(f"{k:>12}" for k in corp))
    print("-" * (w + 13 * len(corp)))
    for k in keys:
        row = f"{k:<{w}} "
        for c in corp:
            v = profs[c][k]
            row += f"{v:12.4f} " if isinstance(v, float) else f"{v:12d} "
        print(row)
    if a.samples:
        tid = sorted(real)[0]
        print(f"\n{'='*80}\nthread {tid}")
        for c in corp:
            print(f"\n--- {c} ---")
            for t in (corp[c].get(tid) or [])[:a.samples]:
                print(f"  {t[:280]}".replace("\n", " / "))

if __name__ == "__main__":
    main()
