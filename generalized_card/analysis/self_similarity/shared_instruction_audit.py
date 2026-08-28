#!/usr/bin/env python3
"""Which shared Writer instructions make comments alike, and which protect a metric?

Roughly half of an 8,836-char Writer prompt is instruction text that also
appears in a quarter or more of its sibling prompts. Those instructions are not
uniformly applied, which makes the run its own natural experiment: for each one,
compare the slots whose prompt carried it against the slots whose prompt did
not.

Observational, not causal -- an instruction is rendered because of the slot's
plan, so its group differs in more than the instruction. Read as a screen that
says where to look, never as an effect size.
"""
from __future__ import annotations
import glob, json, re, statistics, sys
from collections import Counter
from pathlib import Path
REPO = Path("/Users/yaoningyu/Desktop/UIUC/GEO")
TOK = re.compile(r"[a-z0-9']+")
ASSERT = re.compile(r"\b(is|are|isn't|aren't)\s+the\b|\bthat's\s+the\b|\bthe\s+(real|only|actual|whole|key|main)\b", re.I)


def pair_overlap(texts, n=2, cap=4000):
    grams = []
    for t in texts:
        w = TOK.findall(t.lower())
        grams.append(set(zip(*[w[i:] for i in range(n)])) if len(w) >= n else set())
    vals = []
    for i in range(len(grams)):
        for j in range(i + 1, len(grams)):
            u = grams[i] | grams[j]
            if u:
                vals.append(len(grams[i] & grams[j]) / len(u))
            if len(vals) >= cap:
                return statistics.mean(vals)
    return statistics.mean(vals) if vals else float("nan")


def main() -> int:
    rows = []
    for p in sorted(glob.glob(str(REPO / "artifacts/generalized_card/runs/v128_interaction_n10_20260828_v1/cleaned/run_*_sampled_reddit/discussion.json"))):
        d = json.loads(Path(p).read_text())
        for post in d.get("posts") or []:
            seed = int(post.get("seed_index") or 0)
            for r in post.get("generation_records") or []:
                c, pr = r.get("comment"), r.get("prompt")
                if isinstance(c, dict) and isinstance(pr, str) and c.get("content"):
                    rows.append((pr, str(c["content"]), seed))
    n = len(rows)
    counts = Counter()
    for pr, _t, _s in rows:
        for line in set(pr.split("\n")):
            s = line.strip()
            if len(s) > 25:
                counts[s] += 1
    # only lines with a real control group on both sides
    testable = [(v, k) for k, v in counts.items() if 0.12 * n <= v <= 0.92 * n]
    testable.sort(reverse=True)
    print(f"{n} slots; {len(testable)} shared instructions have both groups; comparisons are WITHIN-THREAD\n")
    print(f"{'share':>6} {'n_with':>7} {'2gram_with':>11} {'2gram_without':>14} {'ratio':>7} {'assert_w':>9} {'assert_wo':>10}  instruction")
    print("-" * 150)
    out = []
    for v, line in testable:
        # Compare only WITHIN a thread. Slots sharing a ledger line are all
        # from the same thread, and same-thread comments are alike for reasons
        # that have nothing to do with the instruction -- that confound alone
        # produces a ratio near 1.6, which is the floor to read against.
        ws, os_ = [], []
        for seed in {s for _p, _t, s in rows}:
            w = [t for pr, t, s in rows if s == seed and line in pr]
            o = [t for pr, t, s in rows if s == seed and line not in pr]
            if len(w) >= 5 and len(o) >= 5:
                ws.append(pair_overlap(w))
                os_.append(pair_overlap(o))
        if len(ws) < 3:
            continue
        with_t = [t for pr, t, s in rows if line in pr]
        without_t = [t for pr, t, s in rows if line not in pr]
        ow, oo = statistics.mean(ws), statistics.mean(os_)
        aw = sum(bool(ASSERT.search(t)) for t in with_t) / len(with_t)
        ao = sum(bool(ASSERT.search(t)) for t in without_t) / len(without_t)
        out.append((ow / oo if oo else float("nan"), v, line, ow, oo, aw, ao))
    out.sort(reverse=True)
    for ratio, v, line, ow, oo, aw, ao in out:
        print(f"{100*v//n:5d}% {v:7d} {ow:11.5f} {oo:14.5f} {ratio:7.2f} {aw:9.3f} {ao:10.3f}  {line[:88]}")
    print("\nWithin-thread, so the same-thread confound is removed. ratio > 1: slots carrying")
    print("this instruction are more alike than their own thread-mates that lack it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
