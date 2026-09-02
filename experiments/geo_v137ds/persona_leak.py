#!/usr/bin/env python3
"""Did the persona leak into the comment as stated biography?

  python3 experiments/geo_v137ds/persona_leak.py v153_20260903 a5dsfit_20260902

`--persona-projection register` adds `urbanicity`, `socioeconomic_band`,
`english_proficiency` and `multilingualism` to what the Writer is told. Those
are register determinants, not statable facts, and the official template ends
with "Never state the profile or invent biography, expertise, personal
experience, or facts from it" -- but a rule in a prompt is a hypothesis about
behaviour, not a guarantee, and these four are the first identity-adjacent
dimensions the Writer has ever seen.

This looks for the failure directly: a comment whose text asserts the persona's
own dimension value. It compares against a baseline arm that never had those
dimensions, so a phrase that is simply common on Reddit ("as a parent") shows up
in both and is not evidence of leakage.
"""
import collections
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "generalized_card"))
MARKER = re.compile(r'persona-id="([^"]+)"')

# Phrases that assert a demographic fact about the speaker. Deliberately broad:
# a false positive costs one line of reading, a false negative ships leakage.
SELF_ASSERT = re.compile(
    r"\b(as an?|i'?m an?|i am an?|being an?|speaking as an?)\s+"
    r"(\w+\s+){0,3}"
    r"(suburban|urban|rural|city|small[- ]town|working[- ]class|middle[- ]class|"
    r"low[- ]income|poor|wealthy|rich|native speaker|non[- ]native|bilingual|"
    r"immigrant|esl|autistic|adhd|neurodivergent)\b",
    re.I,
)
# Second-language self-reference, the most likely form for english_proficiency.
ESL = re.compile(
    r"\b(english is(n't| not)? my (first|native) language|not a native (english )?speaker|"
    r"sorry (for )?my english|excuse my english|as a non[- ]native)\b",
    re.I,
)


def load(prefix):
    for d in sorted((REPO / "artifacts/generalized_card/runs").glob(f"{prefix}_p*")):
        f = d / "generated/run_00_sampled_reddit/generation_records.json"
        if f.exists():
            yield d.name, json.load(open(f))


def main():
    for prefix in sys.argv[1:] or ["v153_20260903"]:
        total = 0
        hits = []
        for name, recs in load(prefix):
            for r in recs:
                text = str(r.get("raw") or "").strip()
                if not text:
                    continue
                total += 1
                for pattern, label in ((SELF_ASSERT, "自述人口属性"), (ESL, "自述英语非母语")):
                    m = pattern.search(text)
                    if m:
                        pid = MARKER.search(r.get("prompt") or "")
                        hits.append((label, m.group(0), pid.group(1) if pid else "?", text[:110]))
        print(f"\n{prefix}: {total} 条评论，{len(hits)} 条疑似泄露 ({len(hits)/max(total,1):.1%})")
        for label, phrase, pid, text in hits[:12]:
            print(f"  [{label}] persona {pid}  «{phrase}»")
            print(f"      {text}")


if __name__ == "__main__":
    main()
