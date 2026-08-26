"""Is the long-comment collapse one root cause, or specific to one cue? Measured: specific.

VERDICT UP FRONT, because the hypothesis this script was written to test is WRONG.
The long-slot prompt is not crowded: it carries the FEWEST rule lines of any band
(68.9 against medium's 82.0) and only 28% more characters than a short one. And
the compliance collapse is confined to `parenthetical` -- ellipsis, exclamation and
digit hold or improve with length. The unifying reading is retracted.

Three independent measurements share a signature:
  - polite conversion per sentence collapses above 30 words (0.098 -> 0.020)
  - the parenthetical rhythm cue is drawn on 51.0% of very_long slots and realized
    on 6.1% of them -- 0.12 compliance, against 0.87 at medium
  - generated long comments develop one claim where real ones change subject

If EVERY surface cue degrades with length, the cause is one thing -- the long-slot
prompt carries more competing instructions and the surface cues lose -- and it is
addressable once rather than three times. If only some do, the cause is specific
to those cues and this reading is wrong.

Measures per-band compliance for every `sentence_rhythm` habit that carries a cue,
plus the prompt's own size, on the gate's saved prompts. Compliance is
`realized | cued`, so it is unaffected by how often each habit is drawn.
"""
from __future__ import annotations
import json, re, statistics as st
from collections import Counter
from pathlib import Path

REPO = Path("/Users/yaoningyu/Desktop/UIUC/GEO")
GATE = REPO / "artifacts/generalized_card/runs/v113_v112_gate_n10_20260826_v1"
BANDS = (("short", 0, 19), ("medium", 20, 49), ("long", 50, 119), ("very_long", 120, 10**9))

# (habit, the cue's identifying substring, the realization pattern)
# Needles are the EXACT cue text from `sentence_rhythm.RHYTHM_HABITS`. A loose
# needle silently matches an unrelated rule: "semicolon" and "dash" matched on
# 522/532 and 532/532 slots in the first version of this script and produced two
# meaningless 0.04 compliance rows. `semicolon` carries an EMPTY cue by design --
# generated over-produces it -- so it is not a cueable habit at all and is dropped.
CUES = (
    ("parenthetical", "Put one aside in parentheses.", r"\([^)]{2,}\)"),
    ("ellipsis", "Let one thought trail off with ...", r"\.\.\.|…"),
    ("exclamation", "End one sentence with an exclamation mark.", r"!"),
    ("digit", "Put a number in this one", r"\d"),
    ("short_sentence", "Let one sentence be very short, under five words",
     r"(?:^|[.!?]\s)[^.!?]{1,28}[.!?]"),
    ("dash_clause", "Hang one clause off the last one with a dash.", r"\s[-\u2013\u2014]\s|--"),
)


def band(n):
    for name, lo, hi in BANDS:
        if lo <= n <= hi:
            return name
    raise AssertionError(n)


rows = []
for d in sorted((GATE / "cleaned").glob("run_*_sampled_reddit")):
    for post in json.load(open(d / "discussion.json"))["posts"]:
        for rec in post.get("generation_records") or []:
            prompt = json.dumps(rec.get("prompt") or "")
            comment = rec.get("comment") or {}
            text = str(comment.get("content") or comment.get("text") or "")
            rows.append((prompt, text, len(text.split())))

print(f"slots {len(rows)}")
print(f"\n{'band':<12}{'slots':>7}{'prompt chars':>14}{'rule lines':>12}{'words asked':>13}")
for name, lo, hi in BANDS:
    g = [r for r in rows if band(r[2]) == name]
    if not g:
        continue
    rule_marker = "\\n- "
    rules = st.mean(p.count(rule_marker) for p, _, _ in g)
    print(f"{name:<12}{len(g):>7}{st.mean(len(p) for p, _, _ in g):>14.0f}"
          f"{rules:>12.1f}{st.mean(w for _, _, w in g):>13.1f}")

print(f"\n== compliance: realized | cued, per habit per band ==")
print(f"{'habit':<16}" + "".join(f"{n:>22}" for n, _, _ in BANDS))
overall = {}
for habit, needle, pattern in CUES:
    rx = re.compile(pattern)
    cells = []
    tot_c = tot_r = 0
    for name, lo, hi in BANDS:
        g = [r for r in rows if band(r[2]) == name and needle in r[0]]
        if len(g) < 5:
            cells.append(f"- (n={len(g)})")
            continue
        hit = sum(1 for _, t, _ in g if rx.search(t))
        tot_c += len(g); tot_r += hit
        cells.append(f"{hit/len(g):.2f} (n={len(g)})")
    overall[habit] = (tot_r / tot_c) if tot_c else None
    print(f"{habit:<16}" + "".join(f"{c:>22}" for c in cells))

print(f"\n{'habit':<16}{'overall compliance':>20}{'cued slots':>13}")
for habit, needle, _ in CUES:
    n = sum(1 for p, _, _ in rows if needle in p)
    v = overall[habit]
    print(f"{habit:<16}{('%.3f' % v) if v is not None else '-':>20}{n:>13}")

print("\n== is it the prompt, or the comment? cue position within the prompt ==")
for habit, needle, _ in CUES[:3]:
    pos = [p.index(needle) / len(p) for p, _, _ in rows if needle in p]
    if pos:
        print(f"  {habit:<16} mean relative position {st.mean(pos):.3f}  "
              f"(n={len(pos)})")

print("\n== control: does compliance track prompt SIZE within a band? ==")
rx = re.compile(r"\([^)]{2,}\)")
cued = [(len(p), bool(rx.search(t)), w) for p, t, w in rows if "Put one aside in parentheses" in p]
for name, lo, hi in BANDS:
    g = [(L, h) for L, h, w in cued if band(w) == name]
    if len(g) < 8:
        continue
    med = st.median(L for L, _ in g)
    lo_g = [h for L, h in g if L <= med]
    hi_g = [h for L, h in g if L > med]
    print(f"  {name:<12} prompt<=median {sum(lo_g)}/{len(lo_g)}={sum(lo_g)/max(len(lo_g),1):.2f}   "
          f"prompt>median {sum(hi_g)}/{len(hi_g)}={sum(hi_g)/max(len(hi_g),1):.2f}")
