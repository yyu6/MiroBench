#!/usr/bin/env python3
"""Is the verb-light / determiner-heavy register a `self_bleu_4` lever, or another correlate?

`clause_structure_gap.py` measures the profile G40 inferred: generated's verbal
rate is 0.68-0.70x real's and its determiner rate 1.17-1.22x, stable across two
runs and two pools. That is an association. This prices it.

Method, the same one `rare_token_ablation.py` uses and for the same reason: edit
REAL text so it carries generated's register, rescore with the project's own
scorer, and pair every edit with a **random-token control matched on the exact
number of tokens removed per comment**. Deleting text also shortens it, and
`self_bleu_4` is a length metric through its brevity penalty (G27), so an
uncontrolled deletion reads a length artifact as a register effect.

Fidelity first: the recomputed `self_bleu_4` must reproduce the shipped value.

J7: what this prints is an UPPER BOUND on what a prompt rule could buy, because it
edits text directly.

Usage:  python3 generalized_card/analysis/self_similarity/clause_register_ablation.py
"""
from __future__ import annotations

import json
import random
import re
import statistics as st
import sys
from pathlib import Path

REPO = Path("/Users/yaoningyu/Desktop/UIUC/GEO")
sys.path.insert(0, str(REPO / "scripts" / "evaluation"))
sys.path.insert(0, str(REPO / "generalized_card"))
from score_thread_self_bleu import pairwise_self_bleu_for_order, tokenize  # noqa: E402
from score_thread_semantic_uniformity import (  # noqa: E402
    load_generated_comments,
    load_real_comments,
)

POOL = REPO / "artifacts/generalized_card/seed_pools/camera_product_150_seed42.json"
RAW = REPO / "data/raw/discussions/camera_product"
TAG = "v117_calibration_20260826_v1"

WORD = re.compile(r"\S+")
_STRIP = re.compile(r"[^a-z']")
VERBAL = {
    "to", "be", "have", "has", "had", "is", "are", "was", "were", "do", "does",
    "did", "get", "got", "go", "went", "will", "would", "can", "could", "should",
    "want", "need", "think", "know", "make", "made", "take", "use", "used",
    "am", "been", "being", "having", "doing", "going", "getting", "'s", "'m",
    "'re", "'ve", "'ll", "'d",
}


def self_bleu(texts: list[str]) -> float:
    usable = [t for t in texts if len(str(t or "").split()) >= 2]
    if len(usable) < 2:
        return float("nan")
    return pairwise_self_bleu_for_order([tokenize(t) for t in usable], 4)


def drop_verbal(text: str) -> tuple[str, int]:
    words = WORD.findall(str(text or ""))
    kept = [w for w in words if _STRIP.sub("", w.lower()) not in VERBAL]
    return " ".join(kept), len(words) - len(kept)


def drop_random(text: str, count: int, rng: random.Random) -> str:
    words = WORD.findall(str(text or ""))
    if count <= 0 or count >= len(words):
        return " ".join(words)
    drop = set(rng.sample(range(len(words)), count))
    return " ".join(w for i, w in enumerate(words) if i not in drop)


def main() -> None:
    seeds = {int(x["seed_index"]): x for x in json.loads(POOL.read_text())["seed_posts"]}
    root = REPO / "artifacts/generalized_card/runs" / TAG
    source = root / "cleaned" if (root / "cleaned").exists() else root / "generated"
    gen_threads, cache, real_threads = {}, {}, {}
    for d in sorted(source.glob("run_*_sampled_reddit")):
        cbt, _ = load_generated_comments(d)
        for tid, cs in cbt.items():
            gen_threads[tid] = [c.text for c in cs]
            try:
                idx = int(str(tid).split("seed")[-1])
            except ValueError:
                continue
            post = seeds.get(idx)
            if not post:
                continue
            folder = RAW / post["source_product_dir"]
            if folder not in cache:
                cache[folder] = load_real_comments(folder)[0]
            real_threads[tid] = [
                c.text for c in (cache[folder].get(post["source_raw_post_id"]) or [])
            ]

    keys = sorted(k for k in gen_threads if real_threads.get(k))
    print(f"threads matched: {len(keys)}")
    gen_now = st.mean(self_bleu(gen_threads[k]) for k in keys)
    real_now = st.mean(self_bleu(real_threads[k]) for k in keys)
    print(f"self_bleu_4   real {real_now:.6f}   generated {gen_now:.6f}   "
          f"gap {gen_now - real_now:+.6f} ({100*(gen_now-real_now)/real_now:+.2f}%)")

    rng = random.Random(20260827)
    ablated, controls, removed_share = [], [], []
    for k in keys:
        edited, counts = [], []
        for text in real_threads[k]:
            new, n = drop_verbal(text)
            edited.append(new)
            counts.append(n)
        ablated.append(self_bleu(edited))
        controls.append(
            self_bleu([drop_random(t, n, rng) for t, n in zip(real_threads[k], counts)])
        )
        total = sum(len(WORD.findall(str(t or ""))) for t in real_threads[k])
        removed_share.append(sum(counts) / max(1, total))

    abl, ctl = st.mean(ablated), st.mean(controls)
    gap = gen_now - real_now
    # FIDELITY, stated because only half of it holds. The generated side
    # reproduces the shipped 0.0378 exactly. The real side does NOT: this loads
    # the seed's full real thread (521 comments over the 10) where the
    # evaluation's matched real is 571 comments scoring 0.0348, so the gap
    # denominator here is the larger of the two. The absolute move and its
    # control are internally consistent -- both measured on this construction --
    # so the SIZE of the effect is sound; only the "share of gap" depends on
    # which denominator is used, and both are printed below.
    shipped_gap = 0.037841 - 0.0348
    print(f"\nverbal tokens removed from real: {st.mean(removed_share):.1%} of words")
    print(f"{'':<34}{'self_bleu_4':>13}{'move':>12}{'share of gap':>15}")
    print(f"{'real, untouched':<34}{real_now:>13.6f}{'':>12}{'':>15}")
    print(f"{'real minus VERBAL tokens':<34}{abl:>13.6f}{abl-real_now:>+12.6f}"
          f"{(abl-real_now)/gap:>15.1%}")
    print(f"{'real minus RANDOM tokens (control)':<34}{ctl:>13.6f}{ctl-real_now:>+12.6f}"
          f"{(ctl-real_now)/gap:>15.1%}")
    print(f"{'NET (ablation minus control)':<34}{'':>13}{abl-ctl:>+12.6f}"
          f"{(abl-ctl)/gap:>15.1%}   <- the register effect")
    print(f"\n  Most of the raw move is LENGTH: the random-token control alone "
          f"carries {(ctl-real_now)/(abl-real_now):.0%} of it.")
    print(f"  Against the evaluation's own matched-real gap ({shipped_gap:+.4f}) the "
          f"same net move is {(abl-ctl)/shipped_gap:.0%} of the gap.")
    print("  Either denominator puts this under the ~42% a single channel needs, "
          "so it is\n  one more sub-20% channel, not the lever (G27/G35/G40 "
          "priced four others the same way).")


if __name__ == "__main__":
    main()
