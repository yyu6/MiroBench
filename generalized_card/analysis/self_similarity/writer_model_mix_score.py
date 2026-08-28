#!/usr/bin/env python3
"""Score a writer-model replay: original, replayed, and a simulated mixed policy.

Reuses the evaluation pipeline's own BERTScore loader so the numbers are on the
same footing as `self_bertscore_mean_f1` in the metric suite (deberta-xlarge-mnli).
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path("/Users/yaoningyu/Desktop/UIUC/GEO")
sys.path.insert(0, str(REPO / "scripts" / "evaluation"))

from score_thread_semantic_uniformity import ThreadComment, load_real_comments  # noqa: E402
from score_thread_self_bertscore import (  # noqa: E402
    DEFAULT_BERT_SCORE_PATH,
    DEFAULT_MODEL,
    aggregate_threads,
    build_pair_specs,
    load_bert_scorer,
    score_pairs,
)


def as_comments(thread_id: str, texts: list[str]) -> list[ThreadComment]:
    return [
        ThreadComment(
            thread_id=thread_id,
            thread_title="",
            comment_id=str(i),
            parent_id="",
            author="",
            text=t,
            depth=0,
        )
        for i, t in enumerate(texts)
        if t and t.strip()
    ]


def mean_f1(corpora: dict[str, list[ThreadComment]], scorer, batch_size: int) -> dict[str, float]:
    specs = build_pair_specs(corpora)
    scores = score_pairs(scorer, specs, batch_size)
    rows = aggregate_threads(corpora, scores, top_k=10, include_pairs=False)
    return {r["thread_id"]: float(r["mean_bert_f1"]) for r in rows if r["pair_count"]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seed-pool", default="artifacts/generalized_card/seed_pools/camera_product_150_seed42.json")
    args = ap.parse_args()

    payload = json.loads(Path(args.replay).read_text(encoding="utf-8"))
    items = payload["items"]
    by_thread: dict[str, list[dict]] = defaultdict(list)
    for item in items:
        by_thread[item["thread_id"]].append(item)

    orig, repl, mixed = {}, {}, {}
    for tid, rows in by_thread.items():
        o = [r["original"] for r in rows]
        n = [r["replayed"] for r in rows]
        m = [n[i] if i % 2 else o[i] for i in range(len(rows))]
        if len(o) < 4:
            continue
        orig[tid] = as_comments(tid, o)
        repl[tid] = as_comments(tid, n)
        mixed[tid] = as_comments(tid, m)

    # matched real threads for the same seeds
    pool = json.loads((REPO / args.seed_pool).read_text(encoding="utf-8"))["seed_posts"]
    by_seed = {int(p["seed_index"]): p for p in pool}
    real, cache = {}, {}
    for tid in orig:
        seed = int(tid.replace("seed", ""))
        p = by_seed.get(seed)
        if not p:
            continue
        d = REPO / "data/raw/discussions/camera_product" / p["source_product_dir"]
        if d not in cache:
            cache[d] = load_real_comments(d)[0]
        rcs = cache[d].get(p["source_raw_post_id"]) or []
        if len(rcs) >= 4:
            real[tid] = as_comments(tid, [c.text for c in rcs][: len(orig[tid])])

    scorer, *_ = load_bert_scorer(
        bert_score_path=DEFAULT_BERT_SCORE_PATH,
        model_type=DEFAULT_MODEL,
        num_layers=None,
        batch_size=args.batch_size,
        device=args.device,
        idf=False,
        idf_sents=[],
        rescale_with_baseline=False,
        local_files_only=False,
    )

    results = {}
    for label, corpora in (("real", real), ("original", orig), ("replayed", repl), ("mixed", mixed)):
        if not corpora:
            continue
        results[label] = mean_f1(corpora, scorer, args.batch_size)
        print(f"scored {label}: {len(results[label])} threads", flush=True)

    threads = sorted(set.intersection(*(set(v) for v in results.values())))
    print(f"\nmodel replayed: {payload['model']}   threads compared: {len(threads)}\n")
    print(f"{'thread':<12} " + " ".join(f"{k:>10}" for k in results))
    for t in threads:
        print(f"{t:<12} " + " ".join(f"{results[k][t]:10.4f}" for k in results))
    print()
    base = statistics.mean(results["real"][t] for t in threads)
    for k in results:
        m = statistics.mean(results[k][t] for t in threads)
        closed = ""
        if k not in ("real", "original"):
            o = statistics.mean(results["original"][t] for t in threads)
            gap0, gap = o - base, m - base
            closed = f"   closes {100 * (gap0 - gap) / gap0:5.1f}% of the gap" if gap0 else ""
            wins = sum(results[k][t] < results["original"][t] for t in threads)
            closed += f"   better in {wins}/{len(threads)} threads"
        print(f"  {k:<10} mean {m:.4f}   vs real {m - base:+.4f}{closed}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
