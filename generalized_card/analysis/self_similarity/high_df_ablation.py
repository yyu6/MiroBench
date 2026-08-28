#!/usr/bin/env python3
"""Controlled ablation of the high-document-frequency bigram band.

G127's lesson: a band can carry a large share of the decomposed excess and be
worth almost nothing when removed, because pair-SHARE and excess-MASS are
different quantities. So this prices the band the only way that counts --
delete it, delete the same mass at random, and compare both against real.
"""
from __future__ import annotations
import json, re, statistics, sys, random
from collections import Counter
from pathlib import Path
REPO = Path("/Users/yaoningyu/Desktop/UIUC/GEO")
sys.path.insert(0, str(REPO / "scripts" / "evaluation"))
from score_thread_semantic_uniformity import load_generated_comments, load_real_comments  # noqa: E402
TOK = re.compile(r"[a-z0-9']+")
rng = random.Random(0)
# Real threads repeat their TOPIC -- `the ricoh`, `the sony`, `the price` are
# among their few high-frequency bigrams. We repeat grammar. So the band worth
# suppressing is the one where both tokens are function words; naming a product
# bigram would push us away from real, not toward it.
FUNCTION = set("""a an the and or but if so then than that this these those there here
is are was were be been being am s re ve d ll t don't doesn't isn't aren't wasn't
of to in on at for with from by about into over after before as like just only
very really pretty quite kind sort bit lot much many more most less least
i you he she it we they me him her us them my your his its our their mine yours
what which who whom whose when where why how all any both each few other some such
no nor not too own same can could would should may might must will shall do does did
have has had get got go goes went one two 1 2 up out off down again still even also
""".split())


def is_function_bigram(g) -> bool:
    return g[0] in FUNCTION and g[1] in FUNCTION


def bigrams(words):
    return list(zip(words, words[1:]))


def overlap(word_lists, cap=6000):
    sets = [set(bigrams(w)) for w in word_lists]
    idx = [(i, j) for i in range(len(sets)) for j in range(i + 1, len(sets))]
    if len(idx) > cap:
        idx = rng.sample(idx, cap)
    vals = []
    for i, j in idx:
        u = sets[i] | sets[j]
        if u:
            vals.append(len(sets[i] & sets[j]) / len(u))
    return statistics.mean(vals) if vals else float("nan")


def strip_high_df(word_lists, frac):
    """Drop the SECOND token of every occurrence of a high-DF bigram."""
    n = len(word_lists)
    df = Counter()
    for w in word_lists:
        for g in set(bigrams(w)):
            df[g] += 1
    hot = {g for g, v in df.items() if v / n > frac and is_function_bigram(g)}
    out, removed = [], []
    for w in word_lists:
        keep, drop = [], 0
        i = 0
        while i < len(w):
            if i + 1 < len(w) and (w[i], w[i + 1]) in hot:
                keep.append(w[i])
                i += 2
                drop += 1
            else:
                keep.append(w[i])
                i += 1
        out.append(keep)
        removed.append(drop)
    return out, removed, len(hot)


def strip_random(word_lists, removed):
    out = []
    for w, k in zip(word_lists, removed):
        if k <= 0 or len(w) <= k:
            out.append(list(w))
            continue
        drop = set(rng.sample(range(len(w)), k))
        out.append([x for i, x in enumerate(w) if i not in drop])
    return out


def main() -> int:
    frac = float(sys.argv[1]) if len(sys.argv) > 1 else 0.10
    pool = json.loads((REPO / "artifacts/generalized_card/seed_pools/camera_product_150_seed42.json").read_text())["seed_posts"]
    by = {int(p["seed_index"]): p for p in pool}
    cache, base, abl, ctl, real, hot_counts = {}, [], [], [], [], []
    for x in sorted((REPO / "artifacts/generalized_card/runs/generalized_card_camera_gpt54_paper_20260825_v1/cleaned").glob("run_*_sampled_reddit")):
        cbt, _ = load_generated_comments(x)
        for tid, cs in cbt.items():
            p = by.get(int(tid.split("seed")[-1]))
            if not p:
                continue
            d = REPO / "data/raw/discussions/camera_product" / p["source_product_dir"]
            if d not in cache:
                try:
                    cache[d] = load_real_comments(d)[0]
                except Exception:
                    cache[d] = {}
            rcs = cache[d].get(p["source_raw_post_id"]) or []
            if len(rcs) < 12 or len(cs) < 12:
                continue
            g = [TOK.findall(c.text.lower()) for c in cs]
            r = [TOK.findall(c.text.lower()) for c in rcs]
            stripped, removed, nhot = strip_high_df(g, frac)
            base.append(overlap(g))
            abl.append(overlap(stripped))
            ctl.append(overlap(strip_random(g, removed)))
            real.append(overlap(r))
            hot_counts.append(nhot)
    mb, ma, mc, mr = (statistics.mean(v) for v in (base, abl, ctl, real))
    print(f"threads {len(base)}   high-DF bigrams removed per thread: {statistics.mean(hot_counts):.1f} (DF > {100*frac:.0f}%)")
    print(f"  REAL                 {mr:.5f}")
    print(f"  ours, as generated   {mb:.5f}   excess {mb-mr:+.5f}  ({100*(mb-mr)/mr:+.1f}%)")
    print(f"  ours, band removed   {ma:.5f}   excess {ma-mr:+.5f}  ({100*(ma-mr)/mr:+.1f}%)")
    print(f"  ours, random control {mc:.5f}   excess {mc-mr:+.5f}  ({100*(mc-mr)/mr:+.1f}%)")
    net = (mb - ma) - (mb - mc)
    print(f"\n  raw drop from removing the band : {mb-ma:+.5f}")
    print(f"  drop from removing the same mass at random : {mb-mc:+.5f}")
    print(f"  NET band-specific effect : {net:+.5f} = {100*net/(mb-mr):.1f}% of the excess")
    wins = sum(1 for a, c in zip(abl, ctl) if a < c)
    print(f"  band beats its random control in {wins}/{len(abl)} threads")
    return 0


if __name__ == "__main__":
    sys.exit(main())
