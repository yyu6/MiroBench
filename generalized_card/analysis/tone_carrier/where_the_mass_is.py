"""78% of real's polite sentences carry none of the five register-move words. Read those.

bare_assertion_frame.py corrected an over-read of mine: the frame effect is real
(positive+contrast converts 0.203 real vs 0.031 generated) but lives in 5% of
sentences, and closing the bare-assertion prevalence gap buys 9% of the r gap.
The mass is elsewhere -- the "no positive word at all" bucket is 91% of sentences
and real converts it at 0.073 against generated's 0.035, which is 121 of real's
~180 polite sentences.

So the question is what the shipped classifier buys in a sentence with no
`great/love/very/my/thanks` in it. This mines the discriminative vocabulary
directly instead of testing another guessed pattern -- log-odds of each token in
polite vs non-polite sentences, fitted on REAL text only, then applied to both
sides to see which of real's polite vocabulary the generator does not produce.

Real text here is the ten matched evaluation threads, used only to describe the
target -- no Writer candidate is selected on it (ORIENTATION s4).
"""
from __future__ import annotations
import json, math, re, sys
from collections import Counter
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
MOVES = re.compile("|".join(f"(?:{s['pattern']})" for s in REGISTER_MOVES), re.I)
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
    ss = [s for t in corpus for s in sentences(t)]
    sides[label] = list(zip(ss, score(ss)))

tok = lambda s: re.findall(r"[a-z']+", s.lower())  # noqa: E731

print("== real polite sentences carrying NO register move -- where 78% of the mass is ==")
mass = [(s, r) for s, r in sides["real"] if r["pred_label"] == "polite" and not MOVES.search(s)]
print(f"n = {len(mass)} of {sum(1 for _, r in sides['real'] if r['pred_label']=='polite')} real polite sentences\n")
for s, r in mass[:28]:
    print(f"   [{r['polite_probability']:.2f}] {s[:135]}")

print("\n== discriminative vocabulary, fitted on REAL sentences only (log-odds, min 6 occurrences) ==")
pol = [s for s, r in sides["real"] if r["pred_label"] == "polite"]
oth = [s for s, r in sides["real"] if r["pred_label"] != "polite"]
cp, co = Counter(w for s in pol for w in set(tok(s))), Counter(w for s in oth for w in set(tok(s)))
lo = {}
for w in set(cp) | set(co):
    if cp[w] + co[w] < 6:
        continue
    lo[w] = math.log(((cp[w] + 0.5) / (len(pol) + 1)) / ((co[w] + 0.5) / (len(oth) + 1)))
top = sorted(lo.items(), key=lambda kv: -kv[1])[:45]

gp = [s for s, r in sides["gate"] if r["pred_label"] == "polite"]
go = [s for s, r in sides["gate"] if r["pred_label"] != "polite"]
gall = [s for s, _ in sides["gate"]]
rall = [s for s, _ in sides["real"]]
gc = Counter(w for s in gall for w in set(tok(s)))
rc = Counter(w for s in rall for w in set(tok(s)))
print(f"{'token':<16}{'log-odds':>10}{'real polite':>13}{'real all':>10}{'gen all':>10}{'gen/real':>10}{'in moves':>10}")
for w, v in top:
    rp = rc[w] / len(rall)
    gpv = gc[w] / len(gall)
    print(f"{w:<16}{v:>10.2f}{cp[w]/len(pol):>13.3f}{rp:>10.4f}{gpv:>10.4f}"
          f"{(gpv/rp if rp else 0):>10.2f}{'yes' if MOVES.search(w) else '':>10}")

print("\n== how much of real's polite vocabulary is missing from generated? ==")
miss = [(w, rc[w] / len(rall), gc[w] / len(gall)) for w, _ in top]
under = [m for m in miss if m[2] < 0.6 * m[1]]
print(f"  {len(under)} of the top {len(top)} tokens run below 0.6x real prevalence:")
print("   " + ", ".join(f"{w}({g/r:.2f}x)" for w, r, g in under))
print(f"  summed real prevalence of the top {len(top)}: {sum(m[1] for m in miss):.3f}"
      f"   generated: {sum(m[2] for m in miss):.3f}   ratio {sum(m[2] for m in miss)/sum(m[1] for m in miss):.2f}")
