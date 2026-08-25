"""Is polite a per-SENTENCE lottery that long comments win by buying more tickets?

v112_length_conversion.py showed the polite gap is 100% conversion and 0% length
occupancy, and that generated matches real exactly at 0-29 words and collapses at
30+. polite-guard takes an argmax over the WHOLE comment, so a comment reads polite
when it contains an appreciative sentence. If real's rise with length is just more
sentences at a constant per-sentence rate, then the generated defect is a per-
sentence rate, and the buildable question becomes "why does an extra generated
sentence carry less appreciation than an extra real one".

Scores every sentence independently on the shipped classifier and reports the
per-sentence polite rate by comment length band, plus how well "any polite
sentence" predicts the shipped comment-level label.
"""
from __future__ import annotations
import json, re, statistics as st, sys
from pathlib import Path

REPO = Path("/Users/yaoningyu/Desktop/UIUC/GEO")
sys.path.insert(0, str(REPO / "scripts" / "evaluation"))
from score_thread_semantic_uniformity import load_real_comments, load_generated_comments  # noqa: E402
from score_thread_politeness import PolitenessScorer  # noqa: E402

RUNS = {"gate (v112+v113)": "v113_v112_gate_n10_20260826_v1"}
SEEDS = range(2, 12)
BANDS = ((0, 14), (15, 29), (30, 59), (60, 119), (120, 10**9))
SENT = re.compile(r"(?<=[.!?])\s+|\n+")

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


def sentences(text):
    return [s.strip() for s in SENT.split(text) if len(s.strip().split()) >= 2]


def band(n):
    for lo, hi in BANDS:
        if lo <= n <= hi:
            return (lo, hi)
    raise AssertionError(n)


def load_gen(tag):
    root = REPO / "artifacts/generalized_card/runs" / tag
    out = []
    for d in sorted((root / "cleaned").glob("run_*_sampled_reddit")):
        cbt, _ = load_generated_comments(d)
        for cs in cbt.values():
            out.extend(c.text for c in cs)
    return out


def load_real_texts():
    want = {by_seed[s]["source_raw_post_id"] for s in SEEDS}
    out = []
    for d in sorted((REPO / "data/raw/discussions/camera_product").iterdir()):
        if not d.is_dir():
            continue
        try:
            cbt, _ = load_real_comments(d)
        except Exception:
            continue
        for tid, cs in cbt.items():
            if tid in want:
                out.extend(c.text for c in cs)
    return out


sides = {"real": load_real_texts()}
for label, tag in RUNS.items():
    sides[label] = load_gen(tag)

data = {}
for label, texts in sides.items():
    comment_rows = score(texts)
    flat, index = [], []
    for i, t in enumerate(texts):
        for s in sentences(t):
            flat.append(s)
            index.append(i)
    sent_rows = score(flat)
    per_comment = [[] for _ in texts]
    for i, r in zip(index, sent_rows):
        per_comment[i].append(r)
    data[label] = list(zip(texts, comment_rows, per_comment))

print(f"{'side':<20}{'comments':>10}{'sentences':>11}{'sent/cmt':>10}{'words/sent':>12}")
for label, rows in data.items():
    ns = sum(len(ss) for _, _, ss in rows)
    nw = sum(len(s.split()) for t, _, _ in rows for s in sentences(t))
    print(f"{label:<20}{len(rows):>10}{ns:>11}{ns/len(rows):>10.2f}{nw/max(ns,1):>12.1f}")

print("\n== 'any polite sentence' vs the shipped comment-level label ==")
for label, rows in data.items():
    tp = sum(1 for _, c, ss in rows if c["pred_label"] == "polite" and any(s["pred_label"] == "polite" for s in ss))
    pol = sum(1 for _, c, _ in rows if c["pred_label"] == "polite")
    anyp = sum(1 for _, _, ss in rows if any(s["pred_label"] == "polite" for s in ss))
    print(f"{label:<20} comment polite {pol:>4}  has>=1 polite sentence {anyp:>4}  "
          f"recall {tp/max(pol,1):.3f}  precision {tp/max(anyp,1):.3f}")

print("\n== per-SENTENCE polite rate, by the comment's length band ==")
print(f"{'band':<12}" + "".join(f"{lbl:>26}" for lbl in data))
for lo, hi in BANDS:
    cells = []
    for label, rows in data.items():
        ss = [s for t, _, sl in rows if band(len(t.split())) == (lo, hi) for s in sl]
        cells.append(f"{sum(1 for s in ss if s['pred_label']=='polite')/len(ss):.3f} (n={len(ss)})" if ss else "-")
    name = f"{lo}-{hi}" if hi < 10**9 else f"{lo}+"
    print(f"{name:<12}" + "".join(f"{c:>26}" for c in cells))

print("\n== sentences per comment, by band (the 'more tickets' term) ==")
print(f"{'band':<12}" + "".join(f"{lbl:>26}" for lbl in data))
for lo, hi in BANDS:
    cells = []
    for label, rows in data.items():
        g = [len(sl) for t, _, sl in rows if band(len(t.split())) == (lo, hi)]
        cells.append(f"{st.mean(g):.2f}" if g else "-")
    name = f"{lo}-{hi}" if hi < 10**9 else f"{lo}+"
    print(f"{name:<12}" + "".join(f"{c:>26}" for c in cells))

print("\n== independence check: observed P(>=1 polite sentence) vs 1-(1-r)^k ==")
for label, rows in data.items():
    print(f"  {label}")
    for lo, hi in BANDS:
        g = [sl for t, _, sl in rows if band(len(t.split())) == (lo, hi)]
        if not g:
            continue
        ss = [s for sl in g for s in sl]
        r = sum(1 for s in ss if s["pred_label"] == "polite") / len(ss)
        obs = sum(1 for sl in g if any(s["pred_label"] == "polite" for s in sl)) / len(g)
        pred = st.mean(1 - (1 - r) ** len(sl) for sl in g)
        name = f"{lo}-{hi}" if hi < 10**9 else f"{lo}+"
        print(f"    {name:<10} r={r:.3f}  k={st.mean(len(sl) for sl in g):>5.2f}  "
              f"observed={obs:.3f}  independent={pred:.3f}  ratio={obs/max(pred,1e-9):.2f}")

print("\n== where in the comment does the polite sentence sit? (deciles of position) ==")
print(f"{'decile':<10}" + "".join(f"{lbl:>26}" for lbl in data))
for d10 in range(10):
    cells = []
    for label, rows in data.items():
        ss = [s for _, _, sl in rows if len(sl) >= 4
              for j, s in enumerate(sl) if int(10 * j / len(sl)) == d10]
        cells.append(f"{sum(1 for s in ss if s['pred_label']=='polite')/len(ss):.3f} (n={len(ss)})" if ss else "-")
    print(f"{d10/10:<10.1f}" + "".join(f"{c:>26}" for c in cells))
