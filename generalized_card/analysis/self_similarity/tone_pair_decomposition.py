#!/usr/bin/env python3
"""G83 follow-up: does the tone quota RAISE `self_bertscore` by mixing, or by register?

G83 measured a monotone tradeoff across three arms -- every step that improves the
tone distribution deepens the pairwise-similarity failure (v113 -> v119 -> v120b:
`impolite_rate` deviation 49.7 -> 19.8 -> 11.0%, `self_bertscore` +2.41 -> +4.33
-> +5.59%). Three arms is a pattern, not a mechanism. This script asks which of
two routes carries v113 -> v119's +0.0095 on the thread mean:

  COMPOSITION -- the quota reassigned tones, so a larger share of pairs are now
      SAME-tone pairs, and same-tone pairs are more similar. Mechanical. A fix
      exists: keep the tone rates, break the within-stratum convergence.
  REGISTER -- the stratum means themselves moved, i.e. the quota changed how the
      Writer writes inside every tone bucket. Global. No targeted fix; the arm
      itself is the cost.

Standard Oaxaca split of the thread-mean change, per tone-pair stratum s:

    delta = sum_s (dw_s * m_s)   [composition]  +  sum_s (w_s * dm_s)   [register]

`w_s` (the pair share per stratum) is computed EXACTLY over all pairs from the
Planner's saved `tone_target` -- that needs no scoring and is free. Only `m_s`
(the stratum mean pair F1) needs BERTScore, and a stratum mean does not need
every pair, so it is estimated from a per-thread random sample pooled across
threads. Sampling the means while holding the weights exact is what makes this
tractable on CPU: `build_pair_specs` does not dedupe embeddings, so all 52,235
generated pairs across the two runs would be 104,470 forward passes on
`microsoft/deberta-xlarge-mnli`.

Fidelity is anchored per `tasks/lessons.md` E6 before any stratum is read: the
smallest thread in each run is scored over ALL its pairs and must reproduce the
shipped `cleaned/run_*_sampled_reddit/self_bertscore_results.json` mean to
<1e-6. That validates the scoring path; the sampled strata then inherit it.
A sampled mean is NOT expected to reproduce a thread mean, and this script never
claims it does.

Comparability rests on v113 and v119 sharing seed indices 2-11 (G83, verified:
zero `source_raw_post_id` disagreement and identical matched `real_mean` on all
twelve metrics), so the two runs differ only by `--tone-quota inverted`.

Run with system `python3` (transformers 4.48.0), not `.venv` -- 5.10.1 there
drifts the model hash away from the shipped artifact's.

    python3 generalized_card/analysis/self_similarity/tone_pair_decomposition.py
    python3 .../tone_pair_decomposition.py --max-pairs-per-thread 60   # fast probe
"""
from __future__ import annotations

import argparse
import json
import random
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]
SCORER_DIR = REPO / "scripts" / "evaluation"
if str(SCORER_DIR) not in sys.path:
    sys.path.insert(0, str(SCORER_DIR))

from score_thread_self_bertscore import (  # noqa: E402
    DEFAULT_BERT_SCORE_PATH,
    DEFAULT_MODEL,
    load_bert_scorer,
    score_pairs_with_device_fallback,
)
from score_thread_semantic_uniformity import (  # noqa: E402
    ThreadComment,
    load_generated_comments,
)

RUNS = REPO / "artifacts/generalized_card/runs"
ARMS = {
    "v113": "v113_v112_gate_n10_20260826_v1",
    "v119": "v119_tonequota_only_n10_20260827_v1",
}
# Reported tone labels. `somewhat_polite` absorbs mass and is never reported by
# the project's scorer, but it IS a Planner assignment value, so it gets its own
# stratum here rather than being folded into a reported one.
TONE_ORDER = ("polite", "somewhat_polite", "neutral", "impolite", "unassigned")


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def mean(xs: list[float]) -> float:
    return st.mean(xs) if xs else float("nan")


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #


class Arm:
    """One run's generated threads, plus the Planner tone assigned to each slot."""

    def __init__(self, label: str, tag: str) -> None:
        self.label = label
        self.run = RUNS / tag
        self.comments: dict[str, list[ThreadComment]] = {}
        self.shipped_mean: dict[str, float] = {}
        self.tone: dict[tuple[str, str], str] = {}

        sim_dirs = sorted(self.run.glob("cleaned/run_*_sampled_reddit"))
        if not sim_dirs:
            raise SystemExit(f"{label}: no cleaned sim dirs under {self.run}")
        for sim_dir in sim_dirs:
            comments, _ = load_generated_comments(sim_dir)
            self.comments.update(comments)
            for row in _load_json(sim_dir / "self_bertscore_results.json")["threads"]:
                self.shipped_mean[row["thread_id"]] = row["mean_bert_f1"]
            self._read_tones(sim_dir / "discussion.json")

        missing = [
            (t, c.comment_id)
            for t, cs in self.comments.items()
            for c in cs
            if (t, c.comment_id) not in self.tone
        ]
        if missing:
            raise SystemExit(
                f"{label}: {len(missing)} loaded comments have no saved tone_target, "
                f"first={missing[:3]} -- refusing to guess a stratum"
            )

    def _read_tones(self, path: Path) -> None:
        data = _load_json(path)
        for post in data.get("posts", []):
            thread_id = str(post.get("post_id", ""))

            def walk(nodes: list[dict[str, Any]]) -> None:
                for node in nodes:
                    key = (thread_id, str(node.get("comment_id", "")))
                    raw = node.get("tone_target")
                    label = str(raw).strip() if raw else "unassigned"
                    self.tone[key] = label if label in TONE_ORDER else "unassigned"
                    walk(node.get("replies") or [])

            walk(post.get("comments") or [])

    def tone_of(self, thread_id: str, comment_id: str) -> str:
        return self.tone[(thread_id, comment_id)]

    def smallest_thread(self) -> str:
        return min(self.comments, key=lambda t: len(self.comments[t]))


# --------------------------------------------------------------------------- #
# scoring -- the production scorer, restricted to the pairs asked for
# --------------------------------------------------------------------------- #


def score_specs(specs: list[dict[str, Any]], device: str, batch_size: int) -> list[dict[str, Any]]:
    """Score explicit pair specs with the evaluator's own scorer and config."""

    if not specs:
        return []
    idf_sents = [s["left"].text for s in specs] + [s["right"].text for s in specs]
    kwargs = dict(
        bert_score_path=DEFAULT_BERT_SCORE_PATH,
        model_type=DEFAULT_MODEL,
        num_layers=None,
        idf=False,
        idf_sents=idf_sents,
        rescale_with_baseline=False,
        local_files_only=False,
    )
    (scorer, *_rest, fallback_used) = load_bert_scorer(
        batch_size=batch_size, device=device, **kwargs
    )
    if fallback_used:
        raise SystemExit(
            "microsoft/deberta-xlarge-mnli failed to load; refusing to score with "
            "roberta-large, which would not reproduce the shipped artifact."
        )
    (pair_scores, *_rest2) = score_pairs_with_device_fallback(
        scorer=scorer,
        pair_specs=specs,
        batch_size=batch_size,
        requested_device=device,
        fallback_used=False,
        **kwargs,
    )
    return pair_scores


def all_specs(arm: Arm, thread_id: str) -> list[dict[str, Any]]:
    cs = arm.comments[thread_id]
    return [
        {"thread_id": thread_id, "left": cs[i], "right": cs[j]}
        for i in range(len(cs))
        for j in range(i + 1, len(cs))
    ]


def sampled_specs(arm: Arm, per_thread: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    out: list[dict[str, Any]] = []
    for thread_id in sorted(arm.comments):
        specs = all_specs(arm, thread_id)
        out.extend(specs if len(specs) <= per_thread else rng.sample(specs, per_thread))
    return out


# --------------------------------------------------------------------------- #
# the clean contrast: identical slot, identical assigned tone, different run
# --------------------------------------------------------------------------- #


def invariant_pair_keys(
    a: Arm, b: Arm, per_thread: int, seed: int
) -> list[tuple[str, str, str]]:
    """Pairs whose BOTH members exist in both runs with the SAME assigned tone.

    The Oaxaca split in the main report has a confound it cannot remove: the
    quota changes *which* slots land in a stratum, so a "within-stratum" change
    mixes a true register shift with selection into the stratum. These pairs
    hold slot identity, thread position and assigned tone fixed, so the only
    thing that differs is the run -- i.e. the surrounding thread's tone mix and
    whatever the Writer does in response to it. Any paired difference here is
    register, not selection.
    """
    rng = random.Random(seed)
    out: list[tuple[str, str, str]] = []
    for thread_id in sorted(set(a.comments) & set(b.comments)):
        ida = {c.comment_id for c in a.comments[thread_id]}
        idb = {c.comment_id for c in b.comments[thread_id]}
        inv = sorted(
            cid for cid in (ida & idb)
            if a.tone_of(thread_id, cid) == b.tone_of(thread_id, cid)
        )
        keys = [
            (thread_id, inv[i], inv[j])
            for i in range(len(inv))
            for j in range(i + 1, len(inv))
        ]
        out.extend(keys if len(keys) <= per_thread else rng.sample(keys, per_thread))
    return out


def specs_for_keys(arm: Arm, keys: list[tuple[str, str, str]]) -> list[dict[str, Any]]:
    by_thread = {
        t: {c.comment_id: c for c in cs} for t, cs in arm.comments.items()
    }
    return [
        {"thread_id": t, "left": by_thread[t][lhs], "right": by_thread[t][rhs]}
        for t, lhs, rhs in keys
    ]


def report_invariant(arms: dict[str, Arm], per_thread: int, seed: int,
                     device: str, batch_size: int) -> None:
    a, b = arms["v113"], arms["v119"]
    keys = invariant_pair_keys(a, b, per_thread, seed)
    print(f"\n== CLEAN PAIRED CONTRAST: {len(keys)} tone-invariant pairs "
          f"(same slot, same assigned tone, both runs) ==")
    f113 = [p["bert_f1"] for p in score_specs(specs_for_keys(a, keys), device, batch_size)]
    f119 = [p["bert_f1"] for p in score_specs(specs_for_keys(b, keys), device, batch_size)]
    deltas = [y - x for x, y in zip(f113, f119)]
    up = sum(1 for d in deltas if d > 0)
    print(f"  v113 mean {mean(f113):.4f}   v119 mean {mean(f119):.4f}   "
          f"paired delta {mean(deltas):+.4f}")
    print(f"  pairs that rose: {up}/{len(deltas)} = {up / len(deltas):.3f}")
    try:
        from scipy.stats import wilcoxon  # noqa: PLC0415

        stat, p = wilcoxon(deltas)
        print(f"  Wilcoxon signed-rank on the paired deltas: stat={stat:.0f} p={p:.3g}")
    except Exception as exc:  # pragma: no cover
        print(f"  (scipy unavailable: {exc})")

    print(f"\n  {'tone stratum (unchanged in both)':<34}{'n':>6}{'v113':>9}{'v119':>9}{'delta':>9}")
    by: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for (t, lhs, rhs), x, y in zip(keys, f113, f119):
        by[stratum(a, t, lhs, rhs)].append((x, y))
    for k in sorted(by, key=lambda k: -len(by[k])):
        rows = by[k]
        if len(rows) < 30:
            continue
        mx, my = mean([x for x, _ in rows]), mean([y for _, y in rows])
        print(f"  {k:<34}{len(rows):>6}{mx:>9.4f}{my:>9.4f}{my - mx:>+9.4f}")

    print("\n  per thread:")
    tb: dict[str, list[float]] = defaultdict(list)
    for (t, _l, _r), x, y in zip(keys, f113, f119):
        tb[t].append(y - x)
    for t in sorted(tb):
        print(f"    {t:<34} n={len(tb[t]):>5} paired delta {mean(tb[t]):+.4f}")


# --------------------------------------------------------------------------- #
# strata
# --------------------------------------------------------------------------- #


def stratum(arm: Arm, thread_id: str, a: str, b: str) -> str:
    ta, tb = arm.tone_of(thread_id, a), arm.tone_of(thread_id, b)
    return "|".join(sorted((ta, tb)))


def exact_weights(arm: Arm) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
    """Per-thread stratum pair shares over ALL pairs, plus the thread-averaged share.

    Free: needs only the saved tone assignments, no scoring.
    """
    per_thread: dict[str, dict[str, float]] = {}
    for thread_id, cs in arm.comments.items():
        counts: dict[str, int] = defaultdict(int)
        total = 0
        for i in range(len(cs)):
            for j in range(i + 1, len(cs)):
                counts[stratum(arm, thread_id, cs[i].comment_id, cs[j].comment_id)] += 1
                total += 1
        per_thread[thread_id] = {k: v / total for k, v in counts.items()} if total else {}
    keys = {k for w in per_thread.values() for k in w}
    avg = {k: mean([w.get(k, 0.0) for w in per_thread.values()]) for k in keys}
    return per_thread, avg


def same_tone_share(weights: dict[str, float]) -> float:
    return sum(v for k, v in weights.items() if len(set(k.split("|"))) == 1)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--max-pairs-per-thread", type=int, default=400)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--sample-seed", type=int, default=20260827)
    ap.add_argument("--cache", default="")
    ap.add_argument("--skip-fidelity", action="store_true",
                    help="only for re-reading a cache; never for a first run")
    ap.add_argument("--invariant-only", action="store_true",
                    help="run only the confound-free paired contrast")
    args = ap.parse_args()

    arms = {label: Arm(label, tag) for label, tag in ARMS.items()}

    if args.invariant_only:
        report_invariant(arms, args.max_pairs_per_thread, args.sample_seed,
                         args.device, args.batch_size)
        return

    print("== arms ==")
    for label, arm in arms.items():
        n = sum(len(cs) for cs in arm.comments.values())
        print(f"  {label:<5} threads={len(arm.comments):>3} comments={n:>4} run={arm.run.name}")
    shared = set(arms["v113"].comments) & set(arms["v119"].comments)
    print(f"  thread ids shared by both arms: {len(shared)} "
          f"(v113-only {len(set(arms['v113'].comments) - shared)}, "
          f"v119-only {len(set(arms['v119'].comments) - shared)})")

    # ---- exact composition, free ------------------------------------------ #
    print("\n== EXACT tone-pair composition over ALL pairs (no scoring) ==")
    weights = {}
    for label, arm in arms.items():
        per_thread, avg = exact_weights(arm)
        weights[label] = (per_thread, avg)
        print(f"\n  -- {label}: assigned-tone distribution over slots --")
        tc: dict[str, int] = defaultdict(int)
        for thread_id, cs in arm.comments.items():
            for c in cs:
                tc[arm.tone_of(thread_id, c.comment_id)] += 1
        tot = sum(tc.values())
        for t in TONE_ORDER:
            if tc.get(t):
                print(f"       {t:<16} {tc[t]:>4} = {100 * tc[t] / tot:5.1f}%")
        print(f"       SAME-tone pair share (thread-averaged): {same_tone_share(avg):.4f}")

    keys = sorted(set(weights["v113"][1]) | set(weights["v119"][1]),
                  key=lambda k: -weights["v119"][1].get(k, 0.0))
    print(f"\n  {'stratum':<34}{'v113 w':>9}{'v119 w':>9}{'dw':>9}")
    for k in keys:
        w0, w1 = weights["v113"][1].get(k, 0.0), weights["v119"][1].get(k, 0.0)
        if max(w0, w1) < 0.005:
            continue
        print(f"  {k:<34}{w0:>9.4f}{w1:>9.4f}{w1 - w0:>+9.4f}")
    print(f"  {'SAME-tone total':<34}"
          f"{same_tone_share(weights['v113'][1]):>9.4f}"
          f"{same_tone_share(weights['v119'][1]):>9.4f}"
          f"{same_tone_share(weights['v119'][1]) - same_tone_share(weights['v113'][1]):>+9.4f}")

    # ---- fidelity anchor -------------------------------------------------- #
    scored: dict[str, list[dict[str, Any]]] = {}
    cache_path = Path(args.cache) if args.cache else None
    if cache_path and cache_path.exists():
        print(f"\n== reusing cached pair scores from {cache_path} ==")
        scored = {k: v for k, v in _load_json(cache_path).items()}
    else:
        if not args.skip_fidelity:
            print("\n== FIDELITY: all pairs of the smallest thread must reproduce the artifact ==")
            ok = True
            for label, arm in arms.items():
                tid = arm.smallest_thread()
                specs = all_specs(arm, tid)
                got = mean([p["bert_f1"] for p in score_specs(specs, args.device, args.batch_size)])
                ship = arm.shipped_mean.get(tid, float("nan"))
                bad = not (abs(got - ship) < 1e-6)
                ok = ok and not bad
                print(f"  {label:<5} {tid:<34} n_pairs={len(specs):>5} "
                      f"shipped={ship:.6f} recomputed={got:.6f} "
                      f"delta={got - ship:+.2e}{'  <-- MISMATCH' if bad else ''}")
            if not ok:
                raise SystemExit("FIDELITY FAILED -- refusing to report strata.")
            print("  fidelity OK on both arms")

        print(f"\n== scoring sampled pairs (<= {args.max_pairs_per_thread}/thread/arm) ==")
        for label, arm in arms.items():
            specs = sampled_specs(arm, args.max_pairs_per_thread, args.sample_seed)
            print(f"  {label}: {len(specs)} pairs ...", flush=True)
            rows = score_specs(specs, args.device, args.batch_size)
            for spec, row in zip(specs, rows):
                row["stratum"] = stratum(
                    arm, spec["thread_id"], spec["left"].comment_id, spec["right"].comment_id
                )
                row["same_tone"] = len(set(row["stratum"].split("|"))) == 1
            scored[label] = [
                {k: r[k] for k in ("thread_id", "bert_f1", "stratum", "same_tone")} for r in rows
            ]
        if cache_path:
            cache_path.write_text(json.dumps(scored), encoding="utf-8")
            print(f"  cached -> {cache_path}")

    # ---- stratum means ---------------------------------------------------- #
    print("\n== stratum mean pair F1 (pooled across threads, from the sample) ==")
    means: dict[str, dict[str, float]] = {}
    for label in ARMS:
        by: dict[str, list[float]] = defaultdict(list)
        for r in scored[label]:
            by[r["stratum"]].append(r["bert_f1"])
        means[label] = {k: mean(v) for k, v in by.items()}
        st_same = [r["bert_f1"] for r in scored[label] if r["same_tone"]]
        st_diff = [r["bert_f1"] for r in scored[label] if not r["same_tone"]]
        print(f"  {label}: same-tone n={len(st_same):<5} mean={mean(st_same):.4f}   "
              f"cross-tone n={len(st_diff):<5} mean={mean(st_diff):.4f}   "
              f"same-minus-cross={mean(st_same) - mean(st_diff):+.4f}")

    print(f"\n  {'stratum':<34}{'v113 n':>8}{'v113 m':>9}{'v119 n':>8}{'v119 m':>9}{'dm':>9}")
    n113: dict[str, int] = defaultdict(int)
    n119: dict[str, int] = defaultdict(int)
    for r in scored["v113"]:
        n113[r["stratum"]] += 1
    for r in scored["v119"]:
        n119[r["stratum"]] += 1
    for k in keys:
        if max(n113.get(k, 0), n119.get(k, 0)) < 30:
            continue
        m0, m1 = means["v113"].get(k, float("nan")), means["v119"].get(k, float("nan"))
        print(f"  {k:<34}{n113.get(k, 0):>8}{m0:>9.4f}{n119.get(k, 0):>8}{m1:>9.4f}{m1 - m0:>+9.4f}")

    # ---- Oaxaca decomposition -------------------------------------------- #
    print("\n== decomposition of the v113 -> v119 thread-mean change ==")
    w0, w1 = weights["v113"][1], weights["v119"][1]
    m0, m1 = means["v113"], means["v119"]
    common = [k for k in set(w0) | set(w1)
              if k in m0 and k in m1 and not (m0[k] != m0[k] or m1[k] != m1[k])]
    comp = sum((w1.get(k, 0.0) - w0.get(k, 0.0)) * ((m0[k] + m1[k]) / 2) for k in common)
    reg = sum(((w0.get(k, 0.0) + w1.get(k, 0.0)) / 2) * (m1[k] - m0[k]) for k in common)
    ship0 = mean([arms["v113"].shipped_mean[t] for t in arms["v113"].comments])
    ship1 = mean([arms["v119"].shipped_mean[t] for t in arms["v119"].comments])
    print(f"  shipped thread-mean F1:  v113 {ship0:.4f} -> v119 {ship1:.4f}   "
          f"actual delta {ship1 - ship0:+.4f}")
    print(f"  strata covered by both arms: {len(common)} of {len(set(w0) | set(w1))}"
          f"   (weight covered: v113 {sum(w0.get(k, 0.0) for k in common):.3f}, "
          f"v119 {sum(w1.get(k, 0.0) for k in common):.3f})")
    print(f"  COMPOSITION (tone mix shifted)      {comp:+.4f}")
    print(f"  REGISTER    (stratum means shifted) {reg:+.4f}")
    print(f"  sum of the two terms                {comp + reg:+.4f}")
    tot = comp + reg
    if abs(tot) > 1e-9:
        print(f"  -> composition carries {100 * comp / tot:+.1f}% of the modelled move, "
              f"register {100 * reg / tot:+.1f}%")
    print("\n  Read the split, not the total: the total is a sampled reconstruction of a")
    print("  shipped delta and will not match it exactly. What the split answers is")
    print("  whether a targeted fix exists (composition) or the arm itself is the cost")
    print("  (register).")


if __name__ == "__main__":
    main()
