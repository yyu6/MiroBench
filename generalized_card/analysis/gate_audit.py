#!/usr/bin/env python3
"""Free post-run gate audit for the v112 and v113 arms.

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

    print("\n=== guardrails ===")
    pol = [r for r in rows if r["tone_target"] == "polite"]
    print(f"  slots assigned tone_target=polite : {len(pol)} = {100 * len(pol) / len(rows):.1f}%"
          f"   (real polite_rate 0.302)")
    print(f"  mean realized words               : {sum(r['realized'] for r in rows) / len(rows):.1f}"
          f"   (paper run 41.2, real 46.2)")


if __name__ == "__main__":
    main()
