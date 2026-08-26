#!/usr/bin/env python3
"""Free post-run gate audit for the v112-v120 arms.

Reads a finished run and answers the one question a gate exists to answer:
**did each arm fire, and did it produce the surface behaviour it was built for?**
It reads no metric and makes no p-value claim -- an N=10 p-value is optimistic by
construction (`ORIENTATION.md` s2 trap 1) and choosing N to improve one is
forbidden by s4.

Usage:
    python3 generalized_card/analysis/gate_audit.py --tag <run tag>
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RUNS = REPO / "artifacts/generalized_card/runs"
sys.path.insert(0, str(REPO / "generalized_card"))
from generalized_card.audit import _malformed_urls, _urls  # noqa: E402
from generalized_card.reference_link import folded_host  # noqa: E402
from generalized_card.tone_donor import load_donor_inventory  # noqa: E402
BANDS = ((1, 9), (10, 19), (20, 34), (35, 49), (50, 69), (70, 100), (101, 150), (151, 10**9))


def load(tag: str):
    root = RUNS / tag
    source = root / "cleaned"
    if not source.exists():
        source = root / "generated"
    rows = []
    for folder in sorted(source.glob("run_*_sampled_reddit")):
        payload = json.loads((folder / "discussion.json").read_text())
        for post in payload.get("posts") or []:
            for record in post.get("generation_records") or []:
                task = record.get("task") or {}
                comment = record.get("comment") or {}
                text = str(comment.get("content") or "")
                if not text.strip():
                    continue
                rows.append(
                    {
                        "post_id": post.get("post_id"),
                        "text": text,
                        "prompt": str(record.get("prompt") or ""),
                        "assigned": int(task.get("real_word_count") or 0),
                        "realized": len(text.split()),
                        "texture": task.get("surface_texture"),
                        "evidence": task.get("evidence_mode"),
                        "tone_target": task.get("tone_target"),
                        "development_plan": task.get("development_plan"),
                    }
                )
    return root, rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()
    root, rows = load(args.tag)
    if not rows:
        raise SystemExit(f"no generation records under {root}")
    config = json.loads((root / "run_config.json").read_text())
    profile = json.loads(Path(config["domain_profile"]).read_text())
    inventory = set((profile.get("reference_link_inventory") or {}).get("urls") or ())
    print(f"tag {args.tag}   slots {len(rows)}   posts {len({r['post_id'] for r in rows})}")
    print(f"arms: reference_link={config.get('reference_link')} "
          f"development_scope={config.get('development_scope')}")

    print("\n=== v113  drawn reference link ===")
    routed = [r for r in rows if r["texture"] == "link_reference"
              or r["evidence"] == "link_quote_reference"]
    offered = [r for r in rows if "Include this exact URL once" in r["prompt"]]
    carrying = [r for r in rows if _urls(r["text"])]
    used = [u for r in carrying for u in dict.fromkeys(_urls(r["text"]))]
    invented = [u for u in used if u not in inventory]
    # dedupe inside a comment first: `[url](url)` yields the same URL twice from
    # one comment and that is the markdown defect above, not a repeat across
    # comments, which is what the guardrail is about.
    per_post = Counter(
        (r["post_id"], u) for r in carrying for u in dict.fromkeys(_urls(r["text"]))
    )
    malformed = [m for r in rows for m in _malformed_urls(r["text"])]
    print(f"  routed slots (matched comment carried a link) : {len(routed)}"
          f" = {100 * len(routed) / len(rows):.2f}% of slots")
    print(f"  slots actually offered a URL in the prompt    : {len(offered)}")
    print(f"  comments carrying a URL in the output         : {len(carrying)}"
          f" = {100 * len(carrying) / len(rows):.2f}%   (real 4.4%)")
    if offered:
        print(f"  compliance, offered -> written               : "
              f"{sum(1 for r in offered if _urls(r['text'])) / len(offered):.3f}")
    print(f"  links written as markdown/escaped garbage     : {len(malformed)}"
          f"   {'FAIL' if malformed else 'ok'}")
    if malformed:
        print(f"    example: {malformed[0][:90]}")
    print(f"  distinct URLs written                         : {len(set(used))}")
    print(f"  URLs NOT in the held-out inventory (invented)  : {len(invented)}"
          f"   {'FAIL' if invented else 'ok'}")
    print(f"  same URL twice inside one post                : "
          f"{sum(1 for v in per_post.values() if v > 1)}"
          f"   {'FAIL' if any(v > 1 for v in per_post.values()) else 'ok'}")
    if invented:
        print(f"    examples: {invented[:3]}")

    print("\n=== v112  development scope ===")
    cue = "One-shot development sequence"
    print(f"  {'assigned band':<14}{'slots':>7}{'has plan':>10}{'plan cue':>10}"
          f"{'realized/assigned':>19}")
    for low, high in BANDS:
        band = [r for r in rows if low <= r["assigned"] <= high]
        if not band:
            continue
        planned = sum(1 for r in band if str(r["development_plan"] or "").strip()
                      not in {"", "none"})
        cued = sum(1 for r in band if cue in r["prompt"])
        ratio = sum(r["realized"] for r in band) / max(1, sum(r["assigned"] for r in band))
        print(f"  {f'{low}-{high}':<14}{len(band):>7}{planned / len(band):>10.3f}"
              f"{cued / len(band):>10.3f}{ratio:>19.3f}")
    total = sum(r["realized"] for r in rows) / max(1, sum(r["assigned"] for r in rows))
    print(f"  {'TOTAL':<14}{len(rows):>7}{'':>10}{'':>10}{total:>19.3f}   (paper run 0.891)")

    print("\n=== v116: parenthetical count (--rhythm-count measured) ===")
    import re as _re
    paren = _re.compile(r"\([^)]{2,}\)")
    one_cue = "Put one aside in parentheses."
    many_cue = "separate asides in parentheses"
    cued = [r for r in rows if one_cue in r["prompt"] or many_cue in r["prompt"]]
    carried = [r for r in rows if paren.search(r["text"])]
    print(f"  slots cued a parenthetical        : {len(cued)} = {len(cued)/len(rows):.4f}"
          f"   (real comment prevalence 0.172)")
    print(f"  slots that wrote one              : {len(carried)} = {len(carried)/len(rows):.4f}")
    hit = [r for r in cued if paren.search(r["text"])]
    print(f"  compliance, realized | cued       : {len(hit)}/{len(cued)} = "
          f"{len(hit)/max(1,len(cued)):.3f}   (v113 gate 0.380)")
    asked = Counter()
    for r in cued:
        m = _re.search(r"Put (one|two|three|four|five) (?:separate )?asides?", r["prompt"])
        asked[m.group(1) if m else "?"] += 1
    print(f"  counts ASKED for                  : {dict(sorted(asked.items()))}")
    got = Counter(len(paren.findall(r["text"])) for r in carried)
    print(f"  counts WRITTEN                    : {dict(sorted(got.items()))}"
          f"   (v113 gate {{1: 48}} -- the arm is inert if this is still all 1s)")
    if carried:
        words = [len(m.split()) for r in carried for m in paren.findall(r["text"])]
        print(f"  parens per carrying comment       : "
              f"{sum(got[k]*k for k in got)/len(carried):.2f}   (real 1.76 on the matched seeds)")
        print(f"  words per parenthetical           : {sum(words)/len(words):.1f}"
              f"   (real 5.7 at long, 6.7 at very_long)")

    print("\n=== v117: reference link count (--reference-link-count measured) ===")
    one_link = "Include this exact URL once"
    many_link = "Include these exact URLs"
    offered = [r for r in rows if one_link in r["prompt"] or many_link in r["prompt"]]
    asked_n = Counter()
    for r in offered:
        m = _re.search(r"Include these exact URLs, (\d+) of them", r["prompt"])
        asked_n[int(m.group(1)) if m else 1] += 1
    written = Counter(len(_urls(r["text"])) for r in rows if _urls(r["text"]))
    print(f"  slots offered a link              : {len(offered)} = {len(offered)/len(rows):.4f}"
          f"   (matched threads carry 0.0492)")
    print(f"  counts OFFERED                    : {dict(sorted(asked_n.items()))}"
          f"   (excluded real 1:.699 2:.172 3:.046 4+:.083)")
    print(f"  counts WRITTEN                    : {dict(sorted(written.items()))}"
          f"   (v113 gate was all 1s -- the arm is inert if this still is)")
    carr = [r for r in rows if _urls(r["text"])]
    if carr:
        # A URL has no spaces, so split() would always be 1. Count characters and
        # DISTINCT urls: v113's `[url](url)` markdown yields the same URL twice
        # from one comment, which is why its counts read {1: 18, 2: 5} while every
        # slot was offered exactly one. v114 fixed the reader; if 2s survive with
        # distinct urls, the count arm is genuinely firing.
        chars = [len(u) for r in carr for u in _urls(r["text"])]
        n_per = sum(written[k] * k for k in written) / len(carr)
        distinct = Counter(len(set(_urls(r["text"]))) for r in carr)
        print(f"  URLs per carrying comment         : {n_per:.2f}   (excluded real 1.67, cap 4 -> 1.51)")
        print(f"  DISTINCT urls per carrier         : {dict(sorted(distinct.items()))}"
              f"   (duplicates here are the v113 markdown defect, not the arm)")
        print(f"  characters per URL                : {sum(chars)/len(chars):.0f}"
              f"   (clean inventory 61)")
    hit = [r for r in offered if _urls(r["text"])]
    print(f"  compliance, wrote | offered       : {len(hit)}/{len(offered)} = "
          f"{len(hit)/max(1,len(offered)):.3f}   (v113 gate 0.958)")

    print("\n=== v118: host coherence (--reference-link-host measured) ===")
    # Measured on the 150-seed evaluation-excluded corpus, k in 2..4 only:
    # 0.771 / 0.640 / 0.417, pooled 0.695. `url_host_coherence.py` reproduces it.
    TARGET = {2: 0.771, 3: 0.640, 4: 0.417}
    multi = [sorted(set(_urls(r["text"]))) for r in carr]
    multi = [u for u in multi if 2 <= len(u) <= 4]
    if not multi:
        print("  no multi-link carriers in this run -- the arm cannot be judged here")
    else:
        print(f"{'k':>3}{'carriers':>10}{'all one host':>14}{'real':>8}")
        for k in (2, 3, 4):
            # NOT `rows`: that name holds every slot in this run and the v115 and
            # v120 sections below still need it. Shadowing it here silently broke
            # the v115 block, and printing only this section's own output hid it.
            at_k = [u for u in multi if len(u) == k]
            if not at_k:
                continue
            same = sum(1 for u in at_k if len({folded_host(x) for x in u}) == 1)
            print(f"{k:>3}{len(at_k):>10}{same/len(at_k):>14.3f}{TARGET[k]:>8.3f}")
        same_all = sum(1 for u in multi if len({folded_host(x) for x in u}) == 1)
        print(f"  pooled 2<=k<=4                    : {same_all/len(multi):.3f}"
              f"   (real 0.695, v117 drew ~0.00)   n={len(multi)}")

    print("\n=== v120: donor sentence (--tone-donor measured) ===")
    # v119 sets the ASSIGNMENT, v120 sets whether the assignment realizes. They are
    # separable in one run: the mix below is v119's outcome, the compliance row is
    # v120's, and the realized tone rates are the joint result.
    inventory = load_donor_inventory("camera_product")
    donors = {s.strip().lower() for s in (inventory.get("sentences") or ())}
    polite_slots = [r for r in rows if str(r["tone_target"] or "") == "polite"]
    if not donors:
        print("  no donor inventory on disk -- run harvest_donor_sentences.py")
    elif not polite_slots:
        print("  no polite-assigned slots in this run")
    else:
        opened = 0
        for r in polite_slots:
            head = " ".join(str(r["text"] or "").split())[:160].lower()
            if any(head.startswith(d) for d in donors):
                opened += 1
        print(f"  inventory sentences                : {len(donors)}")
        print(f"  polite-ASSIGNED slots (v119's job) : {len(polite_slots)}"
              f" = {len(polite_slots)/len(rows):.3f}   (v119 cap 0.56 -> ~0.56)")
        print(f"  opened with the drawn sentence     : {opened}/{len(polite_slots)}"
              f" = {opened/len(polite_slots):.3f}   (v113 link arm reached 0.958)")
        print("  the arm is inert if that share is ~0; a low share caps the "
              "polite row at 0.384")

    print("\n=== v115: tone quota (--tone-quota inverted) ===")
    tones = Counter(str(r["tone_target"] or "") for r in rows)
    for label in ("polite", "somewhat_polite", "neutral", "impolite"):
        print(f"  assigned {label:<16}: {tones[label]:>4} = {tones[label]/len(rows):.4f}")
    print("  (arm off assigns ~.273/.087/.147/.493; inverted at cap 0.35 asks "
          "~.350/.109/.320/.220)")

    print("\n=== guardrails ===")
    pol = [r for r in rows if r["tone_target"] == "polite"]
    print(f"  slots assigned tone_target=polite : {len(pol)} = {100 * len(pol) / len(rows):.1f}%"
          f"   (real polite_rate 0.302)")
    print(f"  mean realized words               : {sum(r['realized'] for r in rows) / len(rows):.1f}"
          f"   (paper run 41.2, real 46.2)")


if __name__ == "__main__":
    main()
