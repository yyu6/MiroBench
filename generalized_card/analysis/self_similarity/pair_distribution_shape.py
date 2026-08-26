#!/usr/bin/env python3
"""Is `self_bertscore`'s residual a SHIFT or a COMPRESSION?

G68 closed the authorial-voice channel: generated's same-author vs
different-author separation is now 1.16x real's, so the *relative* structure is
right. What is left is described in G3 and confirmed there as "a uniform +0.02
lift on every pair, flat under trimming". Both statements are about the mean.

A mean can move two ways and they need different mechanisms:

  SHIFT       every pair is more similar by the same amount. Something shared by
              all comments -- one register, one system prompt -- raises the floor.
  COMPRESSION the pairs that should be very DISSIMILAR are missing. Real threads
              contain comments with nothing in common; generated ones may not,
              because every slot is planned against the same thread contract.

They are distinguishable by the spread and the tails of the per-pair F1
distribution, which no measurement in this project has looked at -- every reading
so far is a thread mean. If it is compression, the mechanism is the Planner's
partition (G35: `forbidden_decision_subjects` on 532/532 slots, 10.1 subjects each
slot is forbidden from discussing), and that is a lever nobody has priced.

Usage:  python3 generalized_card/analysis/self_similarity/pair_distribution_shape.py [tag]
"""
from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "scripts" / "evaluation"))
sys.path.insert(0, str(REPO / "generalized_card"))

from score_thread_self_bertscore import load_bert_scorer  # noqa: E402
from score_thread_semantic_uniformity import (  # noqa: E402
    load_generated_comments,
    load_real_comments,
)

POOL = REPO / "artifacts/generalized_card/seed_pools/camera_product_150_seed42.json"
RAW = REPO / "data/raw/discussions/camera_product"


def pairs(scorer, texts: list[str]) -> list[float]:
    usable = [t for t in texts if len(str(t or "").split()) >= 2]
    cand, ref = [], []
    for i in range(len(usable)):
        for j in range(i + 1, len(usable)):
            cand.append(usable[i]); ref.append(usable[j])
    if not cand:
        return []
    _, _, f1 = scorer.score(cand, ref, batch_size=8)
    return [float(v) for v in f1]


def describe(name: str, values: list[float]) -> dict[str, float]:
    values = sorted(values)
    n = len(values)

    def q(p: float) -> float:
        return values[min(n - 1, int(p * n))]

    return {
        "n": n,
        "mean": st.mean(values),
        "sd": st.pstdev(values),
        "p01": q(0.01),
        "p05": q(0.05),
        "p25": q(0.25),
        "p50": q(0.50),
        "p75": q(0.75),
        "p95": q(0.95),
        "p99": q(0.99),
    }


def main() -> None:
    tag = sys.argv[1] if len(sys.argv) > 1 else "v117_calibration_20260826_v1"
    # deberta-xlarge over every pair of ten threads is ~73k forward passes on CPU
    # and does not finish. Four threads is ~20k and does. Stated, not hidden: the
    # quantile shape is what is being read, and it is stable well before ten.
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    seeds = {int(x["seed_index"]): x for x in json.loads(POOL.read_text())["seed_posts"]}
    root = REPO / "artifacts/generalized_card/runs" / tag
    source = root / "cleaned" if (root / "cleaned").exists() else root / "generated"
    scorer, _, _, _, _, _ = load_bert_scorer(
        bert_score_path=REPO / "bert_score-master",
        model_type="microsoft/deberta-xlarge-mnli", num_layers=None, batch_size=8,
        device="auto", idf=False, idf_sents=[], rescale_with_baseline=False,
        local_files_only=True,
    )
    gen_all, real_all, cache = [], [], {}
    done = 0
    for folder in sorted(source.glob("run_*_sampled_reddit")):
        if done >= limit:
            break
        cbt, _ = load_generated_comments(folder)
        for tid, comments in cbt.items():
            if done >= limit:
                break
            done += 1
            gen_all.extend(pairs(scorer, [c.text for c in comments]))
            try:
                idx = int(str(tid).split("seed")[-1])
            except ValueError:
                continue
            post = seeds.get(idx)
            if not post:
                continue
            product = RAW / post["source_product_dir"]
            if product not in cache:
                cache[product] = load_real_comments(product)[0]
            real_all.extend(
                pairs(scorer, [c.text for c in (cache[product].get(post["source_raw_post_id"]) or [])])
            )
    if not gen_all or not real_all:
        raise SystemExit("no pairs")
    g, r = describe("generated", gen_all), describe("real", real_all)
    print(f"\n=== {tag}: per-pair BERTScore F1 distribution ({limit} threads) ===")
    print(f"{'stat':>6}{'real':>11}{'generated':>12}{'gen-real':>11}")
    for key in ("n", "mean", "sd", "p01", "p05", "p25", "p50", "p75", "p95", "p99"):
        if key == "n":
            print(f"{key:>6}{r[key]:>11.0f}{g[key]:>12.0f}{'':>11}")
            continue
        print(f"{key:>6}{r[key]:>11.4f}{g[key]:>12.4f}{g[key]-r[key]:>+11.4f}")
    print(f"\nspread ratio (generated sd / real sd): {g['sd']/r['sd']:.3f}")
    lo = (g["p05"] - r["p05"]) / (g["mean"] - r["mean"]) if g["mean"] != r["mean"] else float("nan")
    hi = (g["p95"] - r["p95"]) / (g["mean"] - r["mean"]) if g["mean"] != r["mean"] else float("nan")
    print(f"the p05 tail moves {lo:.2f}x the mean; the p95 tail moves {hi:.2f}x the mean")
    print("\nReading: a pure SHIFT moves every quantile by the same amount "
          "(both ratios ~1.00\nand the sd ratio ~1.00). A COMPRESSION moves the "
          "LOW tail much more than the high\none (lo >> 1, sd ratio < 1) -- the "
          "dissimilar pairs are the ones that are missing.")


if __name__ == "__main__":
    main()
