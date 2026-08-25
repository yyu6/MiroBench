"""The honest version: stratify the inversion by stance, and never extrapolate a cell.

inversion_feasibility.py found P(realize polite | assign polite) stable across
comment_function, payload_type, evidence_mode and speaker_role (0.310-0.474 around
a 0.384 base) -- but NOT across stance. 261 of the 289 polite assignments sit on
`agree` slots and realize at 0.402; the 17 on `uncertain` realize at 0.059. `agree`
is only 34.3% of slots, so the pooled solution's 55.4% polite share has to put ~21pp
of polite onto mixed/uncertain/neutral slots whose only observation is that 0.059.

The pooled 86% is therefore an extrapolation into cells the Planner has never been
asked to fill. This version:
  - estimates C separately per stance stratum,
  - refuses to use any cell with fewer than MIN_CELL observations,
  - forbids polite on disagreeing stances outright, matching prompts.py:951,
  - and reports three regimes for the unobserved cells (optimistic / pooled /
    pessimistic) so the answer is a range, not a point.

A number that needs an unobserved cell is labelled as such. Spending a run to
measure those cells is the alternative, and the range says whether it is worth it.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

REPO = Path("/Users/yaoningyu/Desktop/UIUC/GEO")
sys.path.insert(0, str(REPO / "scripts" / "evaluation"))
from score_thread_semantic_uniformity import load_real_comments, load_generated_comments  # noqa: E402
from score_thread_politeness import PolitenessScorer  # noqa: E402

RUNS = ("v110_length_transfer_n10_20260824_v1", "v113_v112_gate_n10_20260826_v1")
SEEDS = range(2, 12)
LABELS = ("polite", "somewhat_polite", "neutral", "impolite")
STRATA = ("agree", "mixed", "uncertain", "neutral", "disagree")
NO_POLITE = {"disagree"}          # prompts.py:951
MIN_CELL = 12
pool = json.load(open(REPO / "artifacts/generalized_card/seed_pools/camera_product_150_seed42.json"))["seed_posts"]
by_seed = {int(x["seed_index"]): x for x in pool}
scorer = PolitenessScorer("Intel/polite-guard", "auto", 256)


class T:
    def __init__(self, t):
        self.text = t
        self.thread_id = self.thread_title = self.comment_id = self.parent_id = self.author = ""
        self.depth = 0


def score(texts):
    return scorer.score_comments([T(t) for t in texts], batch_size=64, include_text=False)


rows = []
for tag in RUNS:
    root = REPO / "artifacts/generalized_card/runs" / tag
    meta = {}
    for d in sorted((root / "cleaned").glob("run_*_sampled_reddit")):
        for post in json.load(open(d / "discussion.json"))["posts"]:
            for rec in post.get("generation_records") or []:
                cid = str((rec.get("comment") or {}).get("comment_id", ""))
                t = rec.get("task") or {}
                if cid:
                    meta[cid] = (str(t.get("tone_target") or ""), str(t.get("stance") or ""))
    batch = []
    for d in sorted((root / "cleaned").glob("run_*_sampled_reddit")):
        cbt, _ = load_generated_comments(d)
        for cs in cbt.values():
            for c in cs:
                if str(c.comment_id) in meta:
                    batch.append((c.text, meta[str(c.comment_id)]))
    sc = score([t for t, _ in batch])
    rows.extend(((a, s), r["pred_label"]) for (_, (a, s)), r in zip(batch, sc))

real, cache = [], {}
for s in SEEDS:
    p = by_seed[s]
    d = REPO / "data/raw/discussions/camera_product" / p["source_product_dir"]
    if d not in cache:
        cache[d] = load_real_comments(d)[0]
    real.extend(c.text for c in (cache[d].get(p["source_raw_post_id"]) or []))
rr = score(real)
target = np.array([sum(1 for r in rr if r["pred_label"] == lb) / len(rr) for lb in LABELS])

weights = {s: sum(1 for (_, st), _ in rows if st == s) / len(rows) for s in STRATA}
print(f"pooled slots {len(rows)}   stratum weights: "
      + "  ".join(f"{s}={weights[s]:.3f}" for s in STRATA))

print(f"\n== observed cells: realized mix per (stance, assigned tone), n>={MIN_CELL} ==")
print(f"{'stance':<12}{'assigned':<18}{'n':>5}" + "".join(f"{lb[:9]:>11}" for lb in LABELS))
cells = {}
for s in STRATA:
    for i, lb in enumerate(LABELS):
        g = [r for (a, st), r in rows if st == s and a == lb]
        if not g:
            continue
        vec = np.array([sum(1 for r in g if r == lb2) / len(g) for lb2 in LABELS])
        flag = "" if len(g) >= MIN_CELL else "  (thin, unused)"
        print(f"{s:<12}{lb:<18}{len(g):>5}" + "".join(f"{v:>11.3f}" for v in vec) + flag)
        if len(g) >= MIN_CELL:
            cells[(s, lb)] = vec

pooled = {}
for i, lb in enumerate(LABELS):
    g = [r for (a, _), r in rows if a == lb]
    pooled[lb] = np.array([sum(1 for r in g if r == lb2) / len(g) for lb2 in LABELS])

WORST = {"polite": np.array([0.059, 0.176, 0.118, 0.647])}  # the `uncertain` cell, n=17


def matrix(stratum, regime):
    M = np.zeros((4, 4))
    for i, lb in enumerate(LABELS):
        if (stratum, lb) in cells:
            M[i] = cells[(stratum, lb)]
        elif regime == "optimistic":
            M[i] = pooled[lb]
        elif regime == "pooled":
            M[i] = pooled[lb]
        else:
            M[i] = WORST.get(lb, pooled[lb])
    return M


def solve(regime):
    """Per-stratum assignment vectors, minimising the pooled realized mix vs target."""
    mats = {s: matrix(s, regime) for s in STRATA}
    a = {s: np.array([0.25, 0.25, 0.25, 0.25]) for s in STRATA}
    for s in NO_POLITE:
        a[s] = np.array([0.0, 1 / 3, 1 / 3, 1 / 3])
    for _ in range(60000):
        realized = sum(weights[s] * (mats[s].T @ a[s]) for s in STRATA)
        err = realized - target
        for s in STRATA:
            g = 2 * weights[s] * (mats[s] @ err)
            a[s] = np.clip(a[s] - 0.5 * g, 0.0, None)
            if s in NO_POLITE:
                a[s][0] = 0.0
            a[s] = a[s] / a[s].sum() if a[s].sum() > 0 else np.array([0.25] * 4)
    return a, mats, sum(weights[s] * (mats[s].T @ a[s]) for s in STRATA)


a_now = np.zeros(4)
for i, lb in enumerate(LABELS):
    a_now[i] = sum(1 for (a_, _), _ in rows if a_ == lb) / len(rows)
r_now = np.array([sum(1 for _, r in rows if r == lb) / len(rows) for lb in LABELS])
l2_now = float(np.linalg.norm(r_now - target))

print(f"\n{'regime':<16}{'polite share':>14}" + "".join(f"{lb[:9]:>11}" for lb in LABELS)
      + f"{'L2':>9}{'closed':>9}")
print(f"{'today':<16}{a_now[0]:>14.4f}" + "".join(f"{v:>11.4f}" for v in r_now)
      + f"{l2_now:>9.4f}{0:>8.0f}%")
for regime in ("optimistic", "pessimistic"):
    a, mats, realized = solve(regime)
    share = sum(weights[s] * a[s][0] for s in STRATA)
    l2 = float(np.linalg.norm(realized - target))
    print(f"{regime:<16}{share:>14.4f}" + "".join(f"{v:>11.4f}" for v in realized)
          + f"{l2:>9.4f}{100*(1-l2/l2_now):>8.0f}%")
print(f"{'real target':<16}{'':>14}" + "".join(f"{v:>11.4f}" for v in target))

print("\n== bias per metric, today vs the pessimistic (fully observed-cell) solution ==")
a, mats, realized = solve("pessimistic")
print(f"{'metric':<18}{'real':>10}{'today':>10}{'bias':>9}{'solved':>10}{'bias':>9}")
for i, lb in enumerate(LABELS):
    print(f"{lb:<18}{target[i]:>10.4f}{r_now[i]:>10.4f}{100*(r_now[i]-target[i])/target[i]:>8.1f}%"
          f"{realized[i]:>10.4f}{100*(realized[i]-target[i])/target[i]:>8.1f}%")
print("\n   per-stratum assignment the pessimistic solution asks for:")
for s in STRATA:
    print(f"     {s:<12} w={weights[s]:.3f}  " + "  ".join(f"{lb[:4]}={a[s][i]:.3f}"
                                                          for i, lb in enumerate(LABELS)))
print("\n   UNOBSERVED cells this solution leans on (no n>=%d measurement exists):" % MIN_CELL)
need = [(s, lb) for s in STRATA for i, lb in enumerate(LABELS)
        if a[s][i] > 0.05 and (s, lb) not in cells]
for s, lb in need:
    print(f"     stance={s:<12} assign={lb:<16} -> rate assumed from "
          f"{'the uncertain cell (0.059)' if lb=='polite' else 'the pooled row'}")
if not need:
    print("     none -- every cell it uses is measured")
