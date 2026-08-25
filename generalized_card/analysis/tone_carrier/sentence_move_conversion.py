"""A realized register move is not a polite sentence. Which frame is the classifier buying?

sentence_accumulation.py: polite is a per-sentence independent lottery (observed
P(>=1 polite sentence) tracks 1-(1-r)^k at ratio 0.85-1.05 on both sides), and
generated's per-sentence rate r collapses from real's flat ~0.10 to 0.02-0.04 in
comments of 30+ words while matching real below 30.

register_realization already cues the five moves per band per register, and the
120+ polite band cues ~2.3 of them, so the moves are not missing. This asks the
next question: conditioned on a sentence CONTAINING a given move, how often does
the shipped classifier call that sentence polite, real vs generated? If real
converts and generated does not on the same lexical move, the deficit is in the
frame around the word, not the word.

Prints matched examples at the end because the answer to "which frame" is read,
not computed.
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
LONG = 30  # the band where the per-sentence rate collapses
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


def sentences(t):
    return [s.strip() for s in SENT.split(t) if len(s.strip().split()) >= 2]


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
    ss = [s for t in corpus if len(t.split()) >= LONG for s in sentences(t)]
    sides[label] = list(zip(ss, score(ss)))
    print(f"{label:<6} sentences from {LONG}+ word comments: {len(ss)}  "
          f"polite {sum(1 for _, r in sides[label] if r['pred_label']=='polite')/len(ss):.3f}")

print(f"\n== P(sentence polite | sentence contains the move), {LONG}+ word comments ==")
print(f"{'move':<20}{'real prev':>11}{'real conv':>11}{'gen prev':>10}{'gen conv':>10}{'conv ratio':>12}")
for spec in REGISTER_MOVES:
    pat = re.compile(spec["pattern"], re.I)
    cells = []
    for label in sides:
        hit = [r for s, r in sides[label] if pat.search(s)]
        cells.append((len(hit) / len(sides[label]),
                      sum(1 for r in hit if r["pred_label"] == "polite") / len(hit) if hit else 0.0))
    (rp, rc), (gp, gc) = cells
    print(f"{spec['name']:<20}{rp:>11.4f}{rc:>11.3f}{gp:>10.4f}{gc:>10.3f}"
          f"{(gc/rc if rc else 0):>12.2f}")

print("\n== the same, for sentences carrying NO move ==")
allpat = re.compile("|".join(f"(?:{s['pattern']})" for s in REGISTER_MOVES), re.I)
for label in sides:
    none = [r for s, r in sides[label] if not allpat.search(s)]
    print(f"  {label:<6} n={len(none):<5} polite {sum(1 for r in none if r['pred_label']=='polite')/len(none):.3f}")

print("\n== first person / addressee grammar of a polite sentence ==")
FEATS = {
    "1sg subject (I/my/mine)": r"\b(?:I|my|mine|I'm|I've|I'd)\b",
    "2nd person (you/your)": r"\b(?:you|your|you're|yours)\b",
    "copula verdict (is/are + adj)": r"\b(?:is|are|was|were)\s+(?:really\s+|very\s+|so\s+|pretty\s+)?[a-z]+\b",
    "comparative (better/more/than)": r"\b(?:better|worse|more|less|than|vs)\b",
    "negation": r"\b(?:not|n't|no|never|nothing)\b",
    "modal advice (should/would/could)": r"\b(?:should|would|could|might|can)\b",
    "conditional (if/unless)": r"\b(?:if|unless|depends|depending)\b",
}
print(f"{'feature':<34}" + "".join(f"{c:>22}" for c in ("real polite/other", "gate polite/other")))
for name, pat in FEATS.items():
    rx = re.compile(pat, re.I)
    cells = []
    for label in sides:
        pol = [s for s, r in sides[label] if r["pred_label"] == "polite"]
        oth = [s for s, r in sides[label] if r["pred_label"] != "polite"]
        cells.append(f"{sum(1 for s in pol if rx.search(s))/max(len(pol),1):.2f} / "
                     f"{sum(1 for s in oth if rx.search(s))/max(len(oth),1):.2f}")
    print(f"{name:<34}" + "".join(f"{c:>22}" for c in cells))

print("\n== READ THIS: real polite sentences vs generated sentences with the same move, unpolite ==")
pv = re.compile("|".join(f"(?:{s['pattern']})" for s in REGISTER_MOVES if s["name"] in
                         ("plain_verdict", "love_like", "any_intensifier")), re.I)
print("\n-- real, scored polite --")
for s, r in [(s, r) for s, r in sides["real"] if r["pred_label"] == "polite" and pv.search(s)][:12]:
    print(f"   [{r['polite_probability']:.2f}] {s[:150]}")
print("\n-- generated, same moves, NOT scored polite --")
for s, r in [(s, r) for s, r in sides["gate"] if r["pred_label"] != "polite" and pv.search(s)][:12]:
    print(f"   [{r['polite_probability']:.2f}/{r['pred_label']}] {s[:150]}")
print("\n-- generated, scored polite (what works today) --")
for s, r in [(s, r) for s, r in sides["gate"] if r["pred_label"] == "polite"][:10]:
    print(f"   [{r['polite_probability']:.2f}] {s[:150]}")
