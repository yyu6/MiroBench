"""Did v112's length repair move polite conversion? The open question in FINDINGS.md s4.

FINDINGS.md s4 established that real P(polite) rises steeply with length while the
generator is both short and converts worse inside every band, and closed with: "What
has not been tested is whether length repair alone moves it -- v112 is the arm that
would answer it." v110 (arms off) and the v113/v112 gate ran the SAME ten seeds 2-11,
so this is that A/B, free, on artifacts already paid for.

CONFOUND, recorded not hidden: v110 ran length_transfer=refit and the gate ran v97.
Commit 21e793c measured the refit arm firing 532/532 slots and moving nothing, so the
difference is believed inert, but it is not a controlled variable here.

Comment-level shares are reported alongside thread-level ones because the shipped
metric is a thread mean over threads of very unequal size, which is noisy at N=10.
"""
from __future__ import annotations
import json, statistics as st, sys
from pathlib import Path

REPO = Path("/Users/yaoningyu/Desktop/UIUC/GEO")
sys.path.insert(0, str(REPO / "scripts" / "evaluation"))
from score_thread_semantic_uniformity import load_real_comments, load_generated_comments  # noqa: E402
from score_thread_politeness import PolitenessScorer  # noqa: E402

RUNS = {
    "v110 (arms off)": "v110_length_transfer_n10_20260824_v1",
    "gate (v112+v113)": "v113_v112_gate_n10_20260826_v1",
}
SEEDS = range(2, 12)
BANDS = ((0, 14), (15, 29), (30, 59), (60, 119), (120, 10**9))

pool = json.load(open(REPO / "artifacts/generalized_card/seed_pools/camera_product_150_seed42.json"))["seed_posts"]
by_seed = {int(x["seed_index"]): x for x in pool}

scorer = PolitenessScorer("Intel/polite-guard", "auto", 256)


class T:
    def __init__(self, t):
        self.text = t
        self.thread_id = self.thread_title = self.comment_id = self.parent_id = self.author = ""
        self.depth = 0


def score(texts):
    return scorer.score_comments([T(t) for t in texts], batch_size=32, include_text=False)


def load_run(tag):
    """-> {seed: [(text, tone_target)]}"""
    root = REPO / "artifacts/generalized_card/runs" / tag
    assigned = {}
    for d in sorted((root / "cleaned").glob("run_*_sampled_reddit")):
        for post in json.load(open(d / "discussion.json"))["posts"]:
            for rec in post.get("generation_records") or []:
                cid = str((rec.get("comment") or {}).get("comment_id", ""))
                if cid:
                    assigned[cid] = (rec.get("task") or {}).get("tone_target")
    out = {}
    for d in sorted((root / "cleaned").glob("run_*_sampled_reddit")):
        cbt, _ = load_generated_comments(d)
        for tid, cs in cbt.items():
            out[int(tid.split("seed")[-1])] = [(c.text, assigned.get(str(c.comment_id))) for c in cs]
    return out


def load_real():
    want = {by_seed[s]["source_raw_post_id"]: s for s in SEEDS}
    out = {}
    for d in sorted((REPO / "data/raw/discussions/camera_product").iterdir()):
        if not d.is_dir():
            continue
        try:
            cbt, _ = load_real_comments(d)
        except Exception:
            continue
        for tid, cs in cbt.items():
            if tid in want:
                out[want[tid]] = [(c.text, None) for c in cs]
    return out


def band(n):
    for lo, hi in BANDS:
        if lo <= n <= hi:
            return (lo, hi)
    raise AssertionError(n)


sides = {"real": load_real()}
for label, tag in RUNS.items():
    sides[label] = load_run(tag)

scored = {}
for label, per_seed in sides.items():
    scored[label] = {s: list(zip([t for t, _ in per_seed[s]],
                                 [a for _, a in per_seed[s]],
                                 score([t for t, _ in per_seed[s]])))
                     for s in sorted(per_seed)}

print(f"{'side':<20}{'threads':>8}{'comments':>10}{'words/cmt':>11}")
for label in sides:
    rows = [r for s in scored[label] for r in scored[label][s]]
    print(f"{label:<20}{len(scored[label]):>8}{len(rows):>10}"
          f"{st.mean(len(t.split()) for t, _, _ in rows):>11.1f}")

print("\n== comment-level label shares (all comments) ==")
print(f"{'side':<20}" + "".join(f"{lbl:>17}" for lbl in ("polite", "somewhat_polite", "neutral", "impolite")))
for label in sides:
    rows = [r for s in scored[label] for r in scored[label][s]]
    print(f"{label:<20}" + "".join(
        f"{sum(1 for _, _, b in rows if b['pred_label'] == lbl) / len(rows):>17.4f}"
        for lbl in ("polite", "somewhat_polite", "neutral", "impolite")))

print("\n== thread-level means (what the shipped metric reports) ==")
print(f"{'side':<20}{'polite':>10}{'impolite':>10}{'neutral':>10}")
for label in sides:
    per = {lbl: [sum(1 for _, _, b in scored[label][s] if b["pred_label"] == lbl) / len(scored[label][s])
                 for s in scored[label]] for lbl in ("polite", "impolite", "neutral")}
    print(f"{label:<20}" + "".join(f"{st.mean(per[lbl]):>10.4f}" for lbl in ("polite", "impolite", "neutral")))

print("\n== P(polite) by word band -- the FINDINGS.md s4 table, rebuilt on these ten seeds ==")
hdr = f"{'band':<12}" + "".join(f"{lbl:>20}" for lbl in sides)
print(hdr)
for lo, hi in BANDS:
    cells = []
    for label in sides:
        rows = [r for s in scored[label] for r in scored[label][s] if band(len(r[0].split())) == (lo, hi)]
        cells.append(f"{sum(1 for _, _, b in rows if b['pred_label'] == 'polite') / len(rows):.3f} (n={len(rows)})"
                     if rows else "-")
    name = f"{lo}-{hi}" if hi < 10**9 else f"{lo}+"
    print(f"{name:<12}" + "".join(f"{c:>20}" for c in cells))

print("\n== band OCCUPANCY (did v112 actually move comments into the long bands?) ==")
print(hdr)
for lo, hi in BANDS:
    cells = []
    for label in sides:
        rows = [r for s in scored[label] for r in scored[label][s]]
        n = sum(1 for r in rows if band(len(r[0].split())) == (lo, hi))
        cells.append(f"{n / len(rows):.3f} (n={n})")
    name = f"{lo}-{hi}" if hi < 10**9 else f"{lo}+"
    print(f"{name:<12}" + "".join(f"{c:>20}" for c in cells))

print("\n== realization of tone_target=polite (generated sides only) ==")
for label in RUNS:
    pa = [r for s in scored[label] for r in scored[label][s] if r[1] == "polite"]
    mix = {lbl: sum(1 for _, _, b in pa if b["pred_label"] == lbl)
           for lbl in ("polite", "somewhat_polite", "neutral", "impolite")}
    allr = [r for s in scored[label] for r in scored[label][s]]
    print(f"{label:<20} assigned polite {len(pa):>4}/{len(allr)} = {len(pa)/len(allr):.3f}   "
          f"realized polite {mix['polite']/len(pa):.3f}   mix {mix}")
    pi = [r for s in scored[label] for r in scored[label][s] if r[1] == "impolite"]
    print(f"{'':<20} assigned impolite {len(pi):>4}/{len(allr)} = {len(pi)/len(allr):.3f}   "
          f"realized impolite {sum(1 for _, _, b in pi if b['pred_label']=='impolite')/len(pi):.3f}")

print("\n== DECOMPOSITION: how much of the polite gap is occupancy vs conversion? ==")
real_rows = [r for s in scored["real"] for r in scored["real"][s]]
real_occ = {b: sum(1 for r in real_rows if band(len(r[0].split())) == b) / len(real_rows) for b in BANDS}
real_cv = {b: (lambda g: sum(1 for _, _, x in g if x["pred_label"] == "polite") / len(g) if g else 0.0)(
    [r for r in real_rows if band(len(r[0].split())) == b]) for b in BANDS}
print(f"{'side':<20}{'actual':>10}{'real occ x own cv':>20}{'own occ x real cv':>20}")
print(f"{'real':<20}{sum(real_occ[b]*real_cv[b] for b in BANDS):>10.4f}{'':>20}{'':>20}")
for label in RUNS:
    rows = [r for s in scored[label] for r in scored[label][s]]
    occ = {b: sum(1 for r in rows if band(len(r[0].split())) == b) / len(rows) for b in BANDS}
    cv = {b: (lambda g: sum(1 for _, _, x in g if x["pred_label"] == "polite") / len(g) if g else 0.0)(
        [r for r in rows if band(len(r[0].split())) == b]) for b in BANDS}
    print(f"{label:<20}{sum(occ[b]*cv[b] for b in BANDS):>10.4f}"
          f"{sum(real_occ[b]*cv[b] for b in BANDS):>20.4f}{sum(occ[b]*real_cv[b] for b in BANDS):>20.4f}")
