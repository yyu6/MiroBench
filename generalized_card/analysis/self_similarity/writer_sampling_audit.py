#!/usr/bin/env python3
"""Audit a sampling sweep: guards, surface profile, target metrics, and samples.

A raw prompt replay bypasses `validate_writer_text`, so anything the shipped
pipeline would have rejected shows up here as if it were normal output. Every
quality check the guards enforce is therefore repeated explicitly, and the
numbers are reported beside the metrics so a metric win bought by degraded text
is visible rather than hidden.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

REPO = Path("/Users/yaoningyu/Desktop/UIUC/GEO")
sys.path.insert(0, str(REPO / "scripts" / "evaluation"))
from score_thread_semantic_uniformity import load_real_comments  # noqa: E402

SKELETON = re.compile(r"\bP\d{2}\|S\d+\|B\d+\b")
LEAK = re.compile(r"\b(semantic_move|claim_family|surface_skeleton|context_aperture|payload form|opener_type|perspective_id)\b", re.I)
BANNED = {", honestly": re.compile(r",\s*honestly"), "that part": re.compile(r"\bthat part\b")}
ASSERT = re.compile(r"\b(is|are|isn't|aren't)\s+the\b|\bthat's\s+the\b|\bthe\s+(real|only|actual|whole|key|main)\b", re.I)
FIRST = re.compile(r"\b(i|i'm|i've|i'd|my|me)\b", re.I)
TICS = ["actually", "check", "matters", "tradeoff", "honestly", "whether"]
TOK = re.compile(r"[a-z0-9']+")


def ngram_overlap(texts: list[str], n: int) -> float:
    grams = []
    for t in texts:
        w = TOK.findall(t.lower())
        grams.append(set(zip(*[w[i:] for i in range(n)])) if len(w) >= n else set())
    vals = []
    for i in range(len(grams)):
        for j in range(i + 1, len(grams)):
            u = grams[i] | grams[j]
            if u:
                vals.append(len(grams[i] & grams[j]) / len(u))
    return statistics.mean(vals) if vals else float("nan")


def profile(texts: list[str]) -> dict:
    n = len(texts)
    words = [len(t.split()) for t in texts]
    out = {
        "n": n,
        "empty": sum(1 for t in texts if not t.strip()),
        "mean_words": statistics.mean(words) if words else 0,
        "median_words": statistics.median(words) if words else 0,
        "skeleton_residue": sum(bool(SKELETON.search(t)) for t in texts),
        "control_leak": sum(bool(LEAK.search(t)) for t in texts),
        "assertion_frame": sum(bool(ASSERT.search(t)) for t in texts) / n,
        "first_person": sum(bool(FIRST.search(t)) for t in texts) / n,
        "question": sum("?" in t for t in texts) / n,
        "bigram_overlap": ngram_overlap(texts, 2),
        "unigram_overlap": ngram_overlap(texts, 1),
    }
    for label, rx in BANNED.items():
        out[f"banned[{label}]"] = sum(bool(rx.search(t)) for t in texts)
    for tic in TICS:
        out[f"tic[{tic}]"] = sum(bool(re.search(rf"\b{tic}\b", t, re.I)) for t in texts) / n
    # degenerate repetition inside one comment
    rep = 0
    for t in texts:
        w = TOK.findall(t.lower())
        if len(w) >= 12:
            c = Counter(zip(w, w[1:]))
            if c and c.most_common(1)[0][1] >= 4:
                rep += 1
    out["internal_repetition"] = rep
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("replays", nargs="+")
    ap.add_argument("--samples", type=int, default=3)
    args = ap.parse_args()

    sets: dict[str, dict[str, list[str]]] = {}
    orig_by_thread: dict[str, list[str]] = {}
    for path in args.replays:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        s = payload.get("sampling") or {}
        label = f"T={s.get('temperature')}" if s.get("temperature") is not None else payload["model"]
        by_thread: dict[str, list[str]] = {}
        for item in payload["items"]:
            by_thread.setdefault(item["thread_id"], []).append(item["replayed"])
            orig_by_thread.setdefault(item["thread_id"], []).append(item["original"])
        sets[label] = by_thread
    # original appears once per replay file; de-duplicate
    first = json.loads(Path(args.replays[0]).read_text(encoding="utf-8"))
    orig_by_thread = {}
    for item in first["items"]:
        orig_by_thread.setdefault(item["thread_id"], []).append(item["original"])

    pool = json.loads((REPO / "artifacts/generalized_card/seed_pools/camera_product_150_seed42.json").read_text())["seed_posts"]
    by_seed = {int(p["seed_index"]): p for p in pool}
    real_by_thread: dict[str, list[str]] = {}
    cache: dict = {}
    for tid in orig_by_thread:
        p = by_seed.get(int(tid.replace("seed", "")))
        if not p:
            continue
        d = REPO / "data/raw/discussions/camera_product" / p["source_product_dir"]
        if d not in cache:
            cache[d] = load_real_comments(d)[0]
        rcs = cache[d].get(p["source_raw_post_id"]) or []
        real_by_thread[tid] = [c.text for c in rcs]

    corpora = {"REAL": real_by_thread, "SHIPPED": orig_by_thread, **sets}
    flat = {k: [t for v in d.values() for t in v] for k, d in corpora.items()}

    keys = ["n", "empty", "mean_words", "median_words", "skeleton_residue", "control_leak",
            "internal_repetition", "banned[, honestly]", "banned[that part]",
            "assertion_frame", "first_person", "question", "unigram_overlap", "bigram_overlap"] + [f"tic[{t}]" for t in TICS]
    profs = {k: profile(v) for k, v in flat.items()}
    width = max(len(k) for k in keys) + 2
    print(f"{'metric':<{width}} " + " ".join(f"{k:>12}" for k in corpora))
    print("-" * (width + 13 * len(corpora)))
    for k in keys:
        row = f"{k:<{width}} "
        for c in corpora:
            v = profs[c][k]
            row += f"{v:12.4f} " if isinstance(v, float) else f"{v:12d} "
        print(row)

    print("\n" + "=" * 90)
    for tid in sorted(orig_by_thread):
        print(f"\n### thread {tid}")
        for c in corpora:
            texts = corpora[c].get(tid) or []
            print(f"\n  --- {c} ---")
            for t in texts[: args.samples]:
                print(f"    {t[:300].replace(chr(10), ' / ')}")
        break
    return 0


if __name__ == "__main__":
    sys.exit(main())
