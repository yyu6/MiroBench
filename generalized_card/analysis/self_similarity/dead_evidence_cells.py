"""Three of the seven evidence modes are at zero. Are they at zero in real text too?

The Planner's `evidence_mode` taxonomy has SEVEN values (prompts.py /
reply_planning.py:421). On the v113 gate it used four:

    technical_or_policy_reasoning  0.504
    none_assertion                 0.174
    small_observation              0.168
    firsthand_experience           0.147
    link_quote_reference           0.006
    hearsay_consensus              0.002
    calculation_math               0.000

discourse_function_floor.py measured `evidence_mode` as the largest per-pair
collision channel -- 41.4% of cross-branch pairs share one, worth +0.0228 -- and
the concentration is what drives it. But flattening toward uniform is not a legal
target and the real distribution is not observable without labelling real text.

Three of the seven cells are effectively unused, and those three happen to be
detectable in real text by surface pattern with decent precision: a link or quote,
a hearsay/consensus frame, an arithmetic statement. So this measures the part of
the question that does NOT need an LLM: **if real comments carry these three at a
material rate, the taxonomy is being under-used and the target has a floor that
costs nothing to establish.**

Precision, not recall: each pattern is written to fire only on clear cases, so the
real rates below are LOWER bounds on real's usage.
"""
from __future__ import annotations
import json, re, sys
from pathlib import Path

REPO = Path("/Users/yaoningyu/Desktop/UIUC/GEO")
sys.path.insert(0, str(REPO / "scripts" / "evaluation"))
sys.path.insert(0, str(REPO / "generalized_card"))
from score_thread_semantic_uniformity import load_generated_comments  # noqa: E402
from generalized_card.reference_link import extract_urls  # noqa: E402

GATE = REPO / "artifacts/generalized_card/runs/v113_v112_gate_n10_20260826_v1"
pool = json.load(open(REPO / "artifacts/generalized_card/seed_pools/camera_product_150_seed42.json"))["seed_posts"]
eval_ids = {str(p["source_raw_post_id"]) for p in pool}

PATTERNS = {
    "link_quote_reference": lambda t: bool(extract_urls(t)) or bool(re.search(r"(?m)^\s*>", t)),
    "hearsay_consensus": lambda t: bool(re.search(
        r"\b(?:i(?:'ve| have)\s+heard|i hear|people say|everyone say\w*|most people|"
        r"supposedly|apparently|from what i(?:'ve| have)? (?:heard|read|seen)|"
        r"the consensus|everybody says|word is|rumou?r)\b", t, re.I)),
    "calculation_math": lambda t: bool(re.search(
        r"\d[\d,.]*\s*(?:[-+*/x×]|plus|minus|times|divided)\s*\d"
        r"|\b\d[\d,.]*\s*%\s*(?:of|off|more|less)"
        r"|\$\s?\d[\d,.]*\s*(?:-|to|vs\.?|versus)\s*\$?\s?\d", t, re.I)),
}

real, tot = {k: 0 for k in PATTERNS}, 0
for f in sorted((REPO / "data/raw/discussions/camera_product").rglob("*.comments.jsonl")):
    for line in f.open():
        try:
            row = json.loads(line)
        except Exception:
            continue
        tid = str(row.get("link_id") or row.get("post_id") or "").split("_")[-1]
        if tid in eval_ids:
            continue
        body = str(row.get("body") or "").strip()
        if not body or body in ("[deleted]", "[removed]"):
            continue
        tot += 1
        for k, fn in PATTERNS.items():
            if fn(body):
                real[k] += 1

gen = []
for d in sorted((GATE / "cleaned").glob("run_*_sampled_reddit")):
    cbt, _ = load_generated_comments(d)
    for cs in cbt.values():
        gen.extend(c.text for c in cs)

ASSIGNED = {"link_quote_reference": 0.006, "hearsay_consensus": 0.002, "calculation_math": 0.000}
print(f"evaluation-excluded real comments {tot}   generated {len(gen)}\n")
print(f"{'evidence mode':<26}{'real (lower bd)':>17}{'gen surface':>13}{'gen ASSIGNED':>14}{'ratio':>8}")
total_real = 0.0
for k, fn in PATTERNS.items():
    r = real[k] / tot
    g = sum(1 for t in gen if fn(t)) / len(gen)
    total_real += r
    print(f"{k:<26}{r:>17.4f}{g:>13.4f}{ASSIGNED[k]:>14.4f}"
          f"{(g / r if r else 0):>8.2f}")
print(f"\n  these three carry at least {total_real:.3f} of real comments;"
      f" the Planner assigns them {sum(ASSIGNED.values()):.3f}")
print(f"  collision rate today sum(p^2) = 0.3338")
p = [0.504, 0.174, 0.168, 0.147, 0.006, 0.002, 0.0]
moved = total_real - sum(ASSIGNED.values())
q = [max(0.0, p[0] - moved)] + p[1:4] + [real["link_quote_reference"] / tot,
                                         real["hearsay_consensus"] / tot,
                                         real["calculation_math"] / tot]
q = [x / sum(q) for x in q]
print(f"  if those three were assigned at real's floor, taken from the dominant cell:")
print("     " + "  ".join(f"{x:.3f}" for x in q))
print(f"     collision sum(p^2) -> {sum(x*x for x in q):.4f}")
print(f"     that arithmetic would say {100*(0.3338 - sum(x*x for x in q)) * 0.0228 / 0.0119:.1f}%"
      " of the +0.0119 gap --")
print("""
  DO NOT USE THAT NUMBER. It assumes assigning these three at real's rate is
  headroom, and the surface column above says it is not: generated already
  exhibits link/quote at 0.86x real and hearsay at 1.08x real WITHOUT the labels.
  The Writer writes those moves regardless of what evidence_mode says, so raising
  the labels would push the surface ABOVE real rather than closing a gap. Only
  `calculation_math` shows a real surface deficit (0.44x).

  What this establishes: evidence_mode is loosely coupled to surface behaviour, so
  the per-pair collision effect discourse_function_floor.py measured (+0.0228, the
  largest of six fields) cannot be assumed to convert into text change. The target
  distribution for the dominant cell -- is real also ~50% technical reasoning? --
  is not measurable by pattern and would need real text labelled by an LLM. On this
  evidence that spend is not justified.""")
