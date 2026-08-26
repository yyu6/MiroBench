"""Auditing v115 against itself: does fixing polite_rate cost self_bertscore?

real_thread_correlates.py turned up an interaction nobody had costed. Across 763
real threads `polite_rate` correlates **+0.22** with `self_bertscore`, and the
generator sits **0.5 sd LOW** on polite_rate. So the tone deficit is currently
*helping* self_bertscore, and v115 -- which exists to raise polite_rate and cut
impolite_rate -- may hand back part of what the link and parenthetical arms are
fighting for.

A cross-thread correlation is not that quantity. The quantity is: inside a thread,
what does a polite comment contribute to the mean pairwise F1 relative to a
non-polite one? That decomposes exactly, since self_bertscore is the mean over
pairs and each comment's leverage is the mean F1 of the pairs it appears in.

LIMIT, stated because it bounds the conclusion: v115 does not relabel existing
text, it changes what the Planner asks for, so the Writer produces different
comments rather than the same ones re-scored. This measures the association in
today's text, which is the best available proxy and not the causal effect.

Reports both sides, because if REAL polite comments also carry high leverage then
the association is a property of polite text and not of the generator.
"""
from __future__ import annotations
import json, statistics as st, sys
from pathlib import Path

REPO = Path("/Users/yaoningyu/Desktop/UIUC/GEO")
sys.path.insert(0, str(REPO / "scripts" / "evaluation"))
from score_thread_semantic_uniformity import load_real_comments, load_generated_comments  # noqa: E402
from score_thread_self_bertscore import load_bert_scorer  # noqa: E402
from score_thread_politeness import PolitenessScorer  # noqa: E402

GATE = REPO / "artifacts/generalized_card/runs/v113_v112_gate_n10_20260826_v1"
SEEDS = range(2, 12)
LABELS = ("polite", "somewhat_polite", "neutral", "impolite")
pool = json.load(open(REPO / "artifacts/generalized_card/seed_pools/camera_product_150_seed42.json"))["seed_posts"]
by_seed = {int(x["seed_index"]): x for x in pool}

gen, real, cache = {}, {}, {}
for d in sorted((GATE / "cleaned").glob("run_*_sampled_reddit")):
    cbt, _ = load_generated_comments(d)
    for tid, cs in cbt.items():
        gen[int(tid.split("seed")[-1])] = [c.text for c in cs]
for s in SEEDS:
    p = by_seed[s]
    dd = REPO / "data/raw/discussions/camera_product" / p["source_product_dir"]
    if dd not in cache:
        cache[dd] = load_real_comments(dd)[0]
    real[s] = [c.text for c in (cache[dd].get(p["source_raw_post_id"]) or [])]

pol = PolitenessScorer("Intel/polite-guard", "auto", 256)


class T:
    def __init__(self, t):
        self.text = t
        self.thread_id = self.thread_title = self.comment_id = self.parent_id = self.author = ""
        self.depth = 0


scorer, _, _, _, _, _ = load_bert_scorer(
    bert_score_path=REPO / "bert_score-master", model_type="microsoft/deberta-xlarge-mnli",
    num_layers=None, batch_size=8, device="auto", idf=False, idf_sents=[],
    rescale_with_baseline=False, local_files_only=True)


def leverage(texts):
    n = len(texts)
    c, r, idx = [], [], []
    for i in range(n):
        for j in range(i + 1, n):
            c.append(texts[i]); r.append(texts[j]); idx.append((i, j))
    if not c:
        return None, None
    _, _, f1 = scorer.score(c, r, batch_size=8)
    vals = [float(x) for x in f1]
    per = [[] for _ in range(n)]
    for (i, j), v in zip(idx, vals):
        per[i].append(v); per[j].append(v)
    return [st.mean(p) if p else None for p in per], st.mean(vals)


rows = {}
for label, threads in (("real", real), ("gate", gen)):
    acc = []
    for s in sorted(threads):
        texts = threads[s]
        if len(texts) < 4:
            continue
        lev, mean = leverage(texts)
        lab = [x["pred_label"] for x in
               pol.score_comments([T(t) for t in texts], batch_size=64, include_text=False)]
        mu = st.mean(v for v in lev if v is not None)
        for v, lb in zip(lev, lab):
            if v is not None:
                acc.append((v - mu, lb))
    rows[label] = acc
    print(f"{label}: {len(acc)} comments")

print(f"\n== within-thread leverage by politeness label (centred per thread) ==")
print(f"{'side':<7}" + "".join(f"{lb[:9]:>20}" for lb in LABELS))
for label in ("real", "gate"):
    cells = []
    for lb in LABELS:
        g = [v for v, x in rows[label] if x == lb]
        cells.append(f"{st.mean(g):+.4f} (n={len(g)})" if len(g) >= 10 else f"n={len(g)}")
    print(f"{label:<7}" + "".join(f"{c:>20}" for c in cells))

print("\n== what v115 would cost, using today's association ==")
gate = rows["gate"]
lev = {lb: st.mean([v for v, x in gate if x == lb]) for lb in LABELS
       if len([v for v, x in gate if x == lb]) >= 10}
now = {"polite": 0.1294, "somewhat_polite": 0.1180, "neutral": 0.1454, "impolite": 0.6072}
after = {"polite": 0.1736, "somewhat_polite": 0.1508, "neutral": 0.1967, "impolite": 0.4789}
base = sum(now[k] * lev[k] for k in lev if k in now)
newv = sum(after[k] * lev[k] for k in lev if k in after)
print(f"  mix-weighted leverage today   {base:+.5f}")
print(f"  under v115's projected mix    {newv:+.5f}")
print(f"  change to self_bertscore      {newv - base:+.5f}")
print(f"  today's gap is +0.0119, so this is {100*(newv-base)/0.0119:+.1f}% of it")
print(f"\n  for scale: the link arm at real's URL mass buys -0.0027 (23% of the gap)")
