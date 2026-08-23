#!/usr/bin/env python3
"""Decompose `self_bleu_4`'s generated-vs-real excess into its BLEU components.

`self_bleu_4` has been stuck at a near-constant offset for eleven versions:
generated 0.0316-0.0333 against a real 0.0278 across v97/v98/v101/v103/v107/v108,
sd ~0.0006. Nothing anyone shipped moved it. The two standing explanations in
`docs/ORIENTATION.md` §6 are (a) it is "a length metric first" and generated
"already matches length", and (b) entity diversity, whose OLS partial r is only
-0.097.

Both deserve a direct test rather than another correlational one, because the
scorer is not a 4-gram metric despite its name. `score_thread_self_bleu.sentence_bleu`
computes the **geometric mean of add-one-smoothed 1-, 2-, 3- and 4-gram modified
precisions**, times a brevity penalty:

    BLEU-4 = BP * exp( (1/4) * sum_{n=1..4} log( (overlap_n + 1) / (total_n + 1) ) )

So unigram precision -- plain vocabulary overlap between two comments -- carries a
full quarter of the log-score, and because 3-/4-gram overlaps are almost always
zero (the +1 smoothing dominates them), the *variation* between threads lives
almost entirely in the 1- and 2-gram terms. A "4-gram overlap" reading of this
metric is wrong, and every mechanism aimed at repeated phrases was aimed at the
term that moves least.

This script decomposes the metric exactly, per order, plus the brevity penalty,
for generated vs matched real. Fidelity is checked first (project rule E6):
recomputed thread means must reproduce the shipped
`cleaned/run_*/self_bleu_results.json` values before any decomposed number is
read (it reproduces all ten v108 threads to 0.00e+00).

No model, no API, runs in seconds:

    python3 generalized_card/analysis/self_bleu_decomposition.py --run <run_dir>
    python3 generalized_card/analysis/self_bleu_decomposition.py --attribute-order 1

RESULTS (2026-08-23, v108/v103/v98 N=10 artifacts) -- and the hypotheses these
killed, recorded so they are not re-run:

1. The gap is NOT a 4-gram phenomenon. Per-order log-excess shares on v108:
   p1 25.9%, p2 36.7%, p3 16.4%, p4 14.0%, brevity penalty 7.0%. The 1- and
   2-gram terms carry ~63%. p1/p2 ratios are also the *stable* part across
   versions (p1 1.146-1.167x, p2 1.215-1.245x on v98/v103/v108) while p3/p4
   swing 1.02-1.19x. Every mechanism this project aimed at repeated phrases was
   aimed at the terms that move least, which is why `self_bleu_4` sat at
   0.0316-0.0333 for eleven versions (sd ~0.0006) against a real 0.0278.

2. REJECTED -- "generated already matches length" / "it is a length metric
   first" as an explanation *of the gap*. Stratifying every comment pair by
   min(token length) shows the excess in all five bands (1.12x, 1.26x, 1.35x,
   1.27x, 1.64x) and re-weighting generated to real's length-band mix leaves
   **97.9%** of the pooled excess intact. The original claim came from
   cross-thread correlation among real threads (a different question) and does
   not transfer to the generated-vs-real gap. A crude rank-matched truncation of
   real text appears to show length explaining ~48%, but that is an artifact:
   truncating real text raises its self-BLEU by destroying content, which is not
   the same operation as comparing at matched length.

3. REJECTED -- the comma. Generated uses commas at 1.43x real (4.378 vs 3.065
   per 100 tokens) and it is the single largest term in the 1-gram attribution
   (16.5% of positive excess mass). But deleting 30/50/100% of commas closes
   only 10.6/14.7/9.7% of the gap, and deleting 30% of `the` as a control closes
   the same amount -- precision is a ratio, so removing shared tokens removes
   overlap and total together. The comma excess is a real criterion-2 register
   tell; it is not a lever for this metric.

4. REJECTED -- `--own-fact-license named` / entity diversity. Two paid N=10 runs
   used `named` (v97, v98) and four used `off` (v101/v103/v107/v108). Distinct
   model designators ranged 37-81 with no separation by arm, and `self_bleu_4`
   was 0.0316/0.0330 under `named` against 0.0325-0.0333 under `off` -- inside
   each other's range. v98 (`named`, 81 designators) has the *worst* p1 ratio of
   the three decomposed versions. Naming more rare entities does not move
   pairwise unigram precision, which is dominated by common words.

5. WHAT SURVIVES: thread vocabulary breadth (types/sqrt(tokens)) is 12.13-12.54
   generated against 14.23 real in every version measured -- a stable 12-15%
   shortfall nothing has moved. The 1-gram attribution shows the register
   direction: generated over-shares `,` `the` `that` `is` `actually` `just`
   `still` `pretty` `that's` (nominal/analytic) and under-shares `to` `i` `be`
   `have` `with` `you` `but` `are` `will` `your` `would` `get` (verbal/
   conversational). Two generated comments in one thread reuse the same words
   and word pairs far more than two real ones do, at every length. This is the
   lexical twin of `self_bertscore_mean_f1`'s semantic finding (G3/G21), but the
   two are NOT one thread-level factor: per-thread gaps correlate r=-0.201
   across the ten v108 threads, while 9/10 threads are positive on both.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
SCORER_DIR = REPO / "scripts" / "evaluation"
if str(SCORER_DIR) not in sys.path:
    sys.path.insert(0, str(SCORER_DIR))

from score_thread_self_bleu import (  # noqa: E402
    clipped_ngram_overlap,
    closest_reference_length,
    pairwise_self_bleu_for_order,
    tokenize,
)
from score_thread_semantic_uniformity import (  # noqa: E402
    load_generated_comments,
    load_real_comments,
)

DEFAULT_RUN = (
    REPO
    / "artifacts/generalized_card/runs"
    / "generalized_card_camera_gpt54_v108_coverage_nonrepeat_n10_20260823_v1"
)
SEED_POOL = REPO / "artifacts/generalized_card/seed_pools/camera_product_150_seed42.json"
REAL_DIR = REPO / "data/raw/discussions/camera_product"
SEED_SUFFIX = re.compile(r"seed(\d+)$")
ORDERS = (1, 2, 3, 4)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else float("nan")


def load_threads(run: Path) -> tuple[dict, dict, list, dict]:
    """Return generated comments, real comments, the pairing, shipped means."""

    pool = _load_json(SEED_POOL)
    pool_by_seed = {int(row["seed_index"]): row for row in pool["seed_posts"]}

    generated: dict[str, list] = {}
    shipped_gen: dict[str, float] = {}
    for sim_dir in sorted(run.glob("cleaned/run_*_sampled_reddit")):
        comments, _ = load_generated_comments(sim_dir)
        generated.update(comments)
        shipped = _load_json(sim_dir / "self_bleu_results.json")
        for row in shipped["threads"]:
            shipped_gen[row["thread_id"]] = row["self_bleu_4"]

    real: dict[str, list] = {}
    pairing: list[tuple[str, str]] = []
    cache: dict[Path, dict] = {}
    for thread_id in generated:
        match = SEED_SUFFIX.search(thread_id)
        if not match:
            raise SystemExit(f"cannot recover seed index from {thread_id!r}")
        seed = pool_by_seed[int(match.group(1))]
        raw_id = str(seed["source_raw_post_id"])
        product_dir = REAL_DIR / str(seed["source_product_dir"])
        if product_dir not in cache:
            all_comments, _ = load_real_comments(product_dir)
            cache[product_dir] = all_comments
        all_comments = cache[product_dir]
        if raw_id not in all_comments:
            raise SystemExit(f"real thread {raw_id!r} not found in {product_dir}")
        real[raw_id] = all_comments[raw_id]
        pairing.append((thread_id, raw_id))

    return generated, real, pairing, shipped_gen


def tokenized(comments: list) -> list[list[str]]:
    return [tokenize(c.text) for c in comments]


def pair_components(a: list[str], b: list[str]) -> dict[str, float]:
    """Per-order precisions and brevity penalty, symmetrised the way the scorer is."""

    out: dict[str, float] = {}
    for order in ORDERS:
        precisions = []
        for hyp, ref in ((a, b), (b, a)):
            overlap, total = clipped_ngram_overlap(hyp, [ref], order)
            precisions.append((overlap + 1.0) / (total + 1.0))
        out[f"p{order}"] = mean(precisions)

    penalties = []
    for hyp, ref in ((a, b), (b, a)):
        closest = closest_reference_length(len(hyp), [len(ref)])
        if len(hyp) > closest:
            penalties.append(1.0)
        else:
            penalties.append(math.exp(1.0 - closest / max(1, len(hyp))))
    out["bp"] = mean(penalties)
    return out


def thread_components(toks: list[list[str]]) -> dict[str, float]:
    """Mean of each BLEU component over every unordered comment pair."""

    acc: dict[str, list[float]] = {f"p{o}": [] for o in ORDERS}
    acc["bp"] = []
    for i in range(len(toks)):
        for j in range(i + 1, len(toks)):
            comp = pair_components(toks[i], toks[j])
            for key, value in comp.items():
                acc[key].append(value)
    return {key: mean(values) for key, values in acc.items()}


def cmd_fidelity(generated: dict, shipped_gen: dict) -> bool:
    print("\n== fidelity: recomputed self_bleu_4 must reproduce the shipped artifact ==\n")
    ok = True
    for thread_id, comments in sorted(generated.items()):
        recomputed = pairwise_self_bleu_for_order(tokenized(comments), 4)
        ship = shipped_gen.get(thread_id, float("nan"))
        delta = recomputed - ship
        flag = "" if abs(delta) < 1e-9 else "   <-- MISMATCH"
        ok = ok and abs(delta) < 1e-9
        print(f"  {thread_id:36s} shipped={ship:.6f} recomputed={recomputed:.6f} delta={delta:+.2e}{flag}")
    print(f"\nfidelity: {'OK' if ok else 'FAILED'}")
    return ok


def cmd_decompose(generated: dict, real: dict, pairing: list) -> None:
    print("\n== per-order decomposition, generated vs matched real ==\n")
    print("BLEU-4 = BP * exp( mean_n log p_n ).  Each log p_n contributes 1/4 of the score.\n")

    gen_rows, real_rows = [], []
    for gen_id, real_id in pairing:
        gen_rows.append(thread_components(tokenized(generated[gen_id])))
        real_rows.append(thread_components(tokenized(real[real_id])))

    def avg(rows: list[dict], key: str) -> float:
        return mean([row[key] for row in rows])

    print(f"{'component':>6s} {'generated':>11s} {'real':>11s} {'ratio':>8s} {'log excess':>11s} {'share of gap':>13s}")
    log_excess = {}
    for order in ORDERS:
        key = f"p{order}"
        g, r = avg(gen_rows, key), avg(real_rows, key)
        log_excess[key] = math.log(g) - math.log(r)
    total_log = sum(log_excess.values()) / len(ORDERS)
    g_bp, r_bp = avg(gen_rows, "bp"), avg(real_rows, "bp")
    bp_log = math.log(g_bp) - math.log(r_bp)
    total_with_bp = total_log + bp_log

    for order in ORDERS:
        key = f"p{order}"
        g, r = avg(gen_rows, key), avg(real_rows, key)
        contrib = log_excess[key] / len(ORDERS)
        share = contrib / total_with_bp if total_with_bp else float("nan")
        print(f"{key:>6s} {g:11.6f} {r:11.6f} {g / r:8.4f} {contrib:+11.5f} {share:12.1%}")
    share_bp = bp_log / total_with_bp if total_with_bp else float("nan")
    print(f"{'BP':>6s} {g_bp:11.6f} {r_bp:11.6f} {g_bp / r_bp:8.4f} {bp_log:+11.5f} {share_bp:12.1%}")
    print(f"\n  total log excess = {total_with_bp:+.5f}  (multiplicative: {math.exp(total_with_bp):.4f}x)")

    print("\n== token-count context (the brevity penalty's input) ==\n")
    gen_lens = [len(t) for gid, _ in pairing for t in tokenized(generated[gid])]
    real_lens = [len(t) for _, rid in pairing for t in tokenized(real[rid])]
    for label, lens in (("generated", gen_lens), ("real", real_lens)):
        srt = sorted(lens)
        print(
            f"  {label:9s} n={len(lens):4d} mean={mean(lens):7.1f} "
            f"median={srt[len(srt) // 2]:4d} p90={srt[int(len(srt) * 0.9)]:4d} max={srt[-1]:4d}"
        )

    print("\n== vocabulary breadth (drives p1) ==\n")
    for label, ids, source in (("generated", [g for g, _ in pairing], generated), ("real", [r for _, r in pairing], real)):
        ttrs, per_comment = [], []
        for tid in ids:
            toks = tokenized(source[tid])
            flat = [t for c in toks for t in c]
            if flat:
                ttrs.append(len(set(flat)) / math.sqrt(len(flat)))
            per_comment.extend([len(set(c)) / max(1, len(c)) for c in toks if c])
        print(f"  {label:9s} thread types/sqrt(tokens)={mean(ttrs):6.2f}   per-comment type-token ratio={mean(per_comment):.4f}")


def _ngram_overlap_mass(toks: list[list[str]], order: int) -> tuple[dict, float]:
    """Total clipped pairwise overlap each n-gram contributes in one thread.

    A pair's clipped overlap for n-gram g is min(count_a(g), count_b(g)), so the
    thread total for g is sum over unordered pairs of that min. This is exactly
    the numerator the scorer accumulates, attributed per n-gram, which is what
    makes the excess actionable: it names the specific tokens two comments in the
    same thread keep sharing.
    """

    from collections import Counter

    counts = [Counter(tuple(c[i : i + order]) for i in range(len(c) - order + 1)) for c in toks]
    mass: dict[tuple, float] = {}
    for i in range(len(counts)):
        for j in range(i + 1, len(counts)):
            a, b = counts[i], counts[j]
            smaller, larger = (a, b) if len(a) <= len(b) else (b, a)
            for gram, ca in smaller.items():
                cb = larger.get(gram)
                if cb:
                    mass[gram] = mass.get(gram, 0.0) + min(ca, cb)
    total = sum(mass.values())
    return mass, total


def cmd_attribute(generated: dict, real: dict, pairing: list, order: int, top: int) -> None:
    """Which specific n-grams carry the generated side's extra pairwise overlap?"""

    print(f"\n== {order}-gram attribution: share of all pairwise clipped overlap ==\n")
    gen_share: dict[tuple, float] = {}
    real_share: dict[tuple, float] = {}
    for gen_id, real_id in pairing:
        for source, tid, acc in ((generated, gen_id, gen_share), (real, real_id, real_share)):
            mass, total = _ngram_overlap_mass(tokenized(source[tid]), order)
            if not total:
                continue
            # Normalise within thread first so one huge thread cannot dominate,
            # matching the metric's own equal-weight-per-thread aggregation.
            for gram, value in mass.items():
                acc[gram] = acc.get(gram, 0.0) + value / total / len(pairing)

    rows = []
    for gram in set(gen_share) | set(real_share):
        g, r = gen_share.get(gram, 0.0), real_share.get(gram, 0.0)
        rows.append((g - r, g, r, gram))
    rows.sort(reverse=True)

    print(f"{'excess':>9s} {'gen share':>10s} {'real share':>11s}  n-gram")
    print("  -- most over-shared in generated --")
    for excess, g, r, gram in rows[:top]:
        print(f"{excess:+9.5f} {g:10.5f} {r:11.5f}  {' '.join(gram)!r}")
    print("\n  -- most under-shared in generated --")
    for excess, g, r, gram in rows[-top:][::-1]:
        print(f"{excess:+9.5f} {g:10.5f} {r:11.5f}  {' '.join(gram)!r}")

    pos = sum(e for e, *_ in rows if e > 0)
    top_pos = sum(e for e, *_ in rows[:top] if e > 0)
    print(f"\n  total positive excess mass = {pos:.5f}; top {top} account for {top_pos / pos:.1%}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--run", default=str(DEFAULT_RUN))
    parser.add_argument("--attribute-order", type=int, default=0, help="1 or 2: attribute the excess per n-gram")
    parser.add_argument("--top", type=int, default=25)
    args = parser.parse_args()

    run = Path(args.run)
    generated, real, pairing, shipped_gen = load_threads(run)
    print(f"run: {run.name}\nthreads: {len(pairing)}")
    if not cmd_fidelity(generated, shipped_gen):
        raise SystemExit("fidelity check failed -- not reading the decomposition")
    if args.attribute_order:
        cmd_attribute(generated, real, pairing, args.attribute_order, args.top)
    else:
        cmd_decompose(generated, real, pairing)


if __name__ == "__main__":
    main()
