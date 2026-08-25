"""The frame, not the word: what shape does a polite sentence have?

sentence_move_conversion.py: conditioned on the SAME lexical move, real sentences
read polite at 0.165-0.331 and generated ones at 0.045-0.090 -- a conversion ratio
of 0.26-0.45. Reading the text, every generated positive is attached to a
concessive, comparative or conditional frame ("great and still says nothing",
"a pretty weak compromise", "not a tiny difference, a pretty noticeable one")
while real states the positive flat and stops.

`plain_verdict`'s cue already forbids exactly this -- "Do not convert it into a
trade-off, a condition, or an abstract appraisal" -- and gets 0.475 compliance on
the WORD. E4 says naming the concrete token gets ~1.0 compliance and naming the
category 0.23, and "one thing that is plainly good" is a category.

This defines a crisp BARE ASSERTION predicate, measures how much of the polite
label it buys on each side, and prices closing the prevalence gap. If real polite
sentences are bare and generated ones are not, the buildable arm is a sentence
SHAPE constraint on the slots register_realization already cues -- which is the
mechanism sentence_rhythm used to move seven habits off zero.
"""
from __future__ import annotations
import json, re, sys
from pathlib import Path

REPO = Path("/Users/yaoningyu/Desktop/UIUC/GEO")
sys.path.insert(0, str(REPO / "scripts" / "evaluation"))
sys.path.insert(0, str(REPO / "generalized_card"))
from score_thread_semantic_uniformity import load_real_comments, load_generated_comments  # noqa: E402
from score_thread_politeness import PolitenessScorer  # noqa: E402
from generalized_card.register_realization import REGISTER_MOVES  # noqa: E402

GATE = "v113_v112_gate_n10_20260826_v1"
SEEDS = range(2, 12)
SENT = re.compile(r"(?<=[.!?])\s+|\n+")
pool = json.load(open(REPO / "artifacts/generalized_card/seed_pools/camera_product_150_seed42.json"))["seed_posts"]
by_seed = {int(x["seed_index"]): x for x in pool}
scorer = PolitenessScorer("Intel/polite-guard", "auto", 256)

POSITIVE = re.compile("|".join(f"(?:{s['pattern']})" for s in REGISTER_MOVES
                               if s["name"] in ("plain_verdict", "love_like")), re.I)
CONTRAST = re.compile(r"\b(?:but|however|though|although|still|yet|whereas|while|"
                      r"instead|rather than|except|unless|other than|aside from|"
                      r"even if|even though|that said|then again)\b|,\s*(?:but|though)\b", re.I)
CONDITION = re.compile(r"\b(?:if|when|once|unless|depends|depending|assuming|"
                       r"provided|as long as|in case)\b", re.I)
COMPARE = re.compile(r"\b(?:more|less|better|worse|than|versus|vs\.?|compared|"
                     r"over the|instead of)\b", re.I)
NEG = re.compile(r"\b(?:not|n't|no|never|nothing|hardly|barely|rarely|little)\b", re.I)


class T:
    def __init__(self, t):
        self.text = t
        self.thread_id = self.thread_title = self.comment_id = self.parent_id = self.author = ""
        self.depth = 0


def score(texts):
    return scorer.score_comments([T(t) for t in texts], batch_size=64, include_text=False)


def sentences(t):
    return [s.strip() for s in SENT.split(t) if len(s.strip().split()) >= 2]


def bare(s):
    """A positive assertion with no qualifying frame around it, short enough to stand alone."""
    return bool(POSITIVE.search(s)) and not (
        CONTRAST.search(s) or CONDITION.search(s) or COMPARE.search(s) or NEG.search(s)
    ) and len(s.split()) <= 20


real, cache = [], {}
for s in SEEDS:
    p = by_seed[s]
    d = REPO / "data/raw/discussions/camera_product" / p["source_product_dir"]
    if d not in cache:
        cache[d] = load_real_comments(d)[0]
    real.extend(c.text for c in (cache[d].get(p["source_raw_post_id"]) or []))
gen = []
for d in sorted((REPO / "artifacts/generalized_card/runs" / GATE / "cleaned").glob("run_*_sampled_reddit")):
    cbt, _ = load_generated_comments(d)
    for cs in cbt.values():
        gen.extend(c.text for c in cs)

sides = {}
for label, corpus in (("real", real), ("gate", gen)):
    ss = [s for t in corpus for s in sentences(t)]
    sides[label] = list(zip(ss, score(ss)))

print("== the frame, decomposed over sentences carrying a positive word ==")
print(f"{'side':<8}{'pos sents':>11}{'bare':>8}{'contrast':>10}{'condition':>11}"
      f"{'compare':>9}{'negation':>10}{'>20 words':>11}")
for label, rows in sides.items():
    pos = [s for s, _ in rows if POSITIVE.search(s)]
    n = len(pos)
    print(f"{label:<8}{n:>11}{sum(map(bare, pos))/n:>8.3f}"
          f"{sum(1 for s in pos if CONTRAST.search(s))/n:>10.3f}"
          f"{sum(1 for s in pos if CONDITION.search(s))/n:>11.3f}"
          f"{sum(1 for s in pos if COMPARE.search(s))/n:>9.3f}"
          f"{sum(1 for s in pos if NEG.search(s))/n:>10.3f}"
          f"{sum(1 for s in pos if len(s.split())>20)/n:>11.3f}")

print("\n== P(polite) by frame ==")
print(f"{'bucket':<34}{'real':>20}{'gate':>20}")
BUCKETS = {
    "bare positive assertion": lambda s: bare(s),
    "positive + contrast": lambda s: POSITIVE.search(s) and CONTRAST.search(s),
    "positive + condition": lambda s: POSITIVE.search(s) and CONDITION.search(s),
    "positive + comparison": lambda s: POSITIVE.search(s) and COMPARE.search(s),
    "positive, >20 words, no frame": lambda s: POSITIVE.search(s) and len(s.split()) > 20
    and not (CONTRAST.search(s) or CONDITION.search(s) or COMPARE.search(s) or NEG.search(s)),
    "no positive word at all": lambda s: not POSITIVE.search(s),
}
for name, fn in BUCKETS.items():
    cells = []
    for label, rows in sides.items():
        hit = [r for s, r in rows if fn(s)]
        cells.append(f"{sum(1 for r in hit if r['pred_label']=='polite')/len(hit):.3f} (n={len(hit)})"
                     if hit else "-")
    print(f"{name:<34}" + "".join(f"{c:>20}" for c in cells))

print("\n== per-sentence polite rate r, and what closing the bare-assertion gap buys ==")
for label, rows in sides.items():
    n = len(rows)
    r = sum(1 for _, x in rows if x["pred_label"] == "polite") / n
    b = [x for s, x in rows if bare(s)]
    print(f"  {label:<6} r={r:.4f}   bare prevalence {len(b)/n:.4f}   "
          f"bare conversion {sum(1 for x in b if x['pred_label']=='polite')/max(len(b),1):.3f}")

rr = sides["real"]; gg = sides["gate"]
r_prev = sum(1 for s, _ in rr if bare(s)) / len(rr)
g_prev = sum(1 for s, _ in gg if bare(s)) / len(gg)
g_bare = [x for s, x in gg if bare(s)]
g_conv = sum(1 for x in g_bare if x["pred_label"] == "polite") / max(len(g_bare), 1)
g_other = [x for s, x in gg if not bare(s)]
o_conv = sum(1 for x in g_other if x["pred_label"] == "polite") / len(g_other)
proj = r_prev * g_conv + (1 - r_prev) * o_conv
gr = sum(1 for _, x in gg if x["pred_label"] == "polite") / len(gg)
realr = sum(1 for _, x in rr if x["pred_label"] == "polite") / len(rr)
print(f"\n  generated r today                       {gr:.4f}")
print(f"  generated r at REAL's bare prevalence   {proj:.4f}   (its own bare conversion, J7 upper bound)")
print(f"  real r                                  {realr:.4f}")
print(f"  closure of the r gap                    {100*(proj-gr)/(realr-gr):.0f}%")

print("\n-- generated bare positive assertions that DID convert --")
for s, r in [(s, r) for s, r in gg if bare(s) and r["pred_label"] == "polite"][:8]:
    print(f"   [{r['polite_probability']:.2f}] {s[:130]}")
print("\n-- real bare positive assertions (the target shape) --")
for s, r in [(s, r) for s, r in rr if bare(s) and r["pred_label"] == "polite"][:10]:
    print(f"   [{r['polite_probability']:.2f}] {s[:130]}")
