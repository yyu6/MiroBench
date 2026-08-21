#!/usr/bin/env python3
"""Check the v104 predictions against a run artifact, one line per prediction.

The predictions were written into `generalized_card/VERSION_LOG.md` before the
paid run. This reads them back mechanically so the check cannot drift into a
retelling. Offline; no API call and no model load.

    python3 generalized_card/analysis/check_v104_predictions.py --tag <run tag>
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "generalized_card"))

from generalized_card import evaluative_register as ev  # noqa: E402

# The version this is measured against. The baseline is recomputed from its
# artifact rather than hardcoded: a rounded constant puts the direction test on
# a floating-point knife edge, and a hardcoded one silently rots.
BASELINE_TAG = "generalized_card_camera_gpt54_v103_stance_opening_n10_20260821_v1"

# (field, predicted low, predicted high, measured real, direction)
REALIZED = (
    ("downtoner_tag_per_1k_sentences", 0.0, 8.0, 0.51, "down"),
    ("partitive_comment_rate", 0.0, 0.08, 0.0177, "down"),
    ("hot_share_of_positive", 0.30, 0.45, 0.4821, "up"),
)
# Guardrail: this one must NOT rise. The generator already evaluates more than
# real; the arm changes strength, not count.
GUARD = ("positive_per_1k_sentences", 130.26)


def single_thread(run: Path) -> bool:
    """Whether this run holds exactly one thread, i.e. it is a gate."""

    threads = 0
    for path in sorted(glob.glob(str(run / "cleaned/*/politeness_results.json"))):
        threads += len(json.loads(Path(path).read_text()).get("threads") or [])
    if threads:
        return threads == 1
    for path in sorted(glob.glob(str(run / "generated/*/discussion.json"))):
        threads += len(json.loads(Path(path).read_text()).get("posts") or [])
    return threads == 1


def largest_thread(run: Path) -> list[dict]:
    """The biggest thread in a run, for comparing a gate like for like.

    A gate is one thread and the baseline is a ten-thread pool. Comparing the
    two pools crosses thread identity, which is how the v102 prediction band was
    set against the wrong population -- see `tasks/lessons.md`.
    """

    best: list[dict] = []
    for path in sorted(glob.glob(str(run / "cleaned/*/politeness_results.json"))):
        for thread in json.loads(Path(path).read_text()).get("threads") or []:
            rows = [
                {"content": str(c.get("text") or ""), "label": str(c.get("pred_label") or "")}
                for c in thread.get("comments") or []
                if str(c.get("text") or "").strip()
            ]
            if len(rows) > len(best):
                best = rows
    return best


def comments_of(run: Path) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(glob.glob(str(run / "cleaned/*/politeness_results.json"))):
        payload = json.loads(Path(path).read_text())
        for thread in payload.get("threads") or []:
            for comment in thread.get("comments") or []:
                text = str(comment.get("text") or "")
                if text.strip():
                    rows.append({"content": text, "label": str(comment.get("pred_label") or "")})
    if rows:
        return rows
    # Before evaluation has run, read the generator's own output instead.
    def walk(items, out):
        for item in items or []:
            out.append(item)
            walk(item.get("replies"), out)
    for path in sorted(glob.glob(str(run / "generated/*/discussion.json"))):
        payload = json.loads(Path(path).read_text())
        for post in payload.get("posts") or []:
            flat: list[dict] = []
            walk(post.get("comments"), flat)
            rows.extend({"content": str(c.get("content") or ""), "label": ""} for c in flat)
    return [row for row in rows if row["content"].strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()
    run = REPO / "artifacts/generalized_card/runs" / args.tag
    rows = comments_of(run)
    if not rows:
        raise SystemExit(f"no comments found under {run}")
    baseline_run = REPO / "artifacts/generalized_card/runs" / BASELINE_TAG
    gate = single_thread(run)
    baseline_rows = largest_thread(baseline_run) if gate else comments_of(baseline_run)
    if not baseline_rows:
        raise SystemExit(f"baseline run {BASELINE_TAG} is not on disk")
    baseline = ev.realized_evaluative_shares(baseline_rows)
    scored = any(row["label"] for row in rows)
    audit = ev.realized_evaluative_shares(rows)
    print(f"run {args.tag}   comments {len(rows)}   sentences {int(audit['sentences'])}"
          f"   {'evaluated' if scored else 'generation only'}")
    print(f"baseline {BASELINE_TAG}"
          f"   comments {len(baseline_rows)}"
          f"   {'largest thread only (this is a gate)' if gate else 'whole run'}")
    print()
    print(f"{'prediction':<34}{'v103':>10}{'predicted':>16}{'v104':>10}{'real':>9}  verdict")
    ok = 0
    for field, low, high, real, direction in REALIZED:
        got = float(audit[field])
        base = float(baseline[field])
        hit = low <= got <= high
        ok += hit
        moved = "toward real" if (
            (direction == "down" and got < base) or (direction == "up" and got > base)
        ) else "WRONG WAY"
        print(f"{field:<34}{base:>10.4f}{f'{low:g}-{high:g}':>16}{got:>10.4f}{real:>9.4f}"
              f"  {'HIT ' if hit else 'MISS'} ({moved})")
    field, real = GUARD
    got = float(audit[field])
    base = float(baseline[field])
    print(f"{field + ' (guard)':<34}{base:>10.4f}{'must not rise':>16}{got:>10.4f}{real:>9.4f}"
          f"  {'OK' if got <= base else 'BREACHED'}")
    print()
    print(f"realized predictions hit: {ok}/{len(REALIZED)}")
    if scored:
        n = len(rows)
        print()
        print(f"{'tone':<18}{'v104':>9}{'v103':>9}{'predicted':>14}{'matched real':>14}")
        base_n = len(baseline_rows)
        for label, pred, real in (
            ("polite", "0.14-0.18", 0.2883),
            ("impolite", "0.57-0.61", 0.4431),
        ):
            got = sum(1 for row in rows if row["label"] == label) / n
            v103 = sum(1 for row in baseline_rows if row["label"] == label) / base_n
            print(f"{label:<18}{got:>9.4f}{v103:>9.4f}{pred:>14}{real:>14.4f}")
        print("  (one thread: read these as direction, never as an inference)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
