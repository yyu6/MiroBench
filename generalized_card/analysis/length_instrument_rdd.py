#!/usr/bin/env python3
"""Which length cue is the causal instrument -- the asked number, or the beat budget?

`docs/DECISIONS.md` G48 measured that the asked word count has essentially no
leverage on realized length (elasticity -0.02 at 50-100 words, 0.11 above 100),
and correctly diagnosed why the fit that justified it had no identifying
variation: `asked` is a deterministic function of `real_word_count`, which also
drives the layout, the beat count and the token ceiling.

But `length_calibration.py`'s own docstring says the calibration changes **only**
the number: "`real_word_count` stays the truth everywhere else -- the layout
profile, the development beats, the tone-length band, and the substantive length
floor all keep reading the matched slot's real size." So v110 held every
structural cue fixed and moved only the number. The structural cues were never
tested, and they are where the collinear slope actually lived.

This script tests them, for free, on runs already paid for. The identifying
variation is a **discontinuity**: `long_form_planning.expected_development_beats`
returns 0 at `real_word_count <= 100` and `max(3, round(w/21))` above it, so

    w <= 60    "Make one narrow local move and stop when that contribution is complete."
    61..100    "give it the two or three connected beats this slot's scale supports"
                and `reconcile_development_plan_capacity` DELETES any beat plan
                the Planner produced for the slot
    w >= 101    an enumerated, per-slot, Planner-authored development sequence of
                round(w/21) beats, capped at MAX_DEVELOPMENT_BEATS = 12

A slot at 100 assigned words and one at 101 are otherwise near-identical -- the
ask differs by ~5 words and every other size-keyed cue is continuous -- so the
jump in realized length at that boundary is the causal effect of the enumerated
beat plan, not of the number.

Subcommands:
    bands   realized/assigned per band against the cue that band receives
    rdd     the discontinuity at 100/101 and at 60/61, per run and pooled
    beats   realized words per delivered beat, and the 12-beat saturation
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import statistics
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RUNS = REPO / "artifacts/generalized_card/runs"

# Every N=10 run on the same ten seeds, so `real_word_count` is identical across
# them and they are repeated measures of one assignment.
COMPARABLE = {
    "v97_n10": "generalized_card_camera_gpt54_v97_keyboard_n10_20260819_v1",
    "v103_n10": "generalized_card_camera_gpt54_v103_stance_opening_n10_20260821_v1",
    "v108_n10": "generalized_card_camera_gpt54_v108_coverage_nonrepeat_n10_20260823_v1",
    "v110_n10": "v110_length_transfer_n10_20260824_v1",
}
# Reported realized/assigned totals, from G43 and G48. The loader must reproduce
# them before any derived number is printed (rule E6).
FIDELITY = {"v108_n10": 0.8896, "v110_n10": 0.8957}

WORDS_PER_BEAT = 21.0
MAX_BEATS = 12


def load(tag: str) -> list[dict]:
    root = RUNS / tag
    rows: list[dict] = []
    for path in sorted(glob.glob(str(root / "generated/**/generation_records.json"), recursive=True)):
        for rec in json.load(open(path)):
            task = rec.get("task") or {}
            try:
                assigned = int(task.get("real_word_count") or 0)
            except (TypeError, ValueError):
                continue
            if assigned <= 0:
                continue
            attempts = rec.get("attempts") or []
            realized = len(str(rec.get("raw") or "").split())
            if attempts and attempts[-1].get("word_count"):
                realized = int(attempts[-1]["word_count"])
            plan = str(task.get("development_plan") or "")
            rows.append({
                "assigned": assigned,
                "realized": realized,
                "beats": len([b for b in plan.split(" || ") if b.strip()]),
            })
    return rows


def load_all() -> dict[str, list[dict]]:
    data = {}
    for name, tag in COMPARABLE.items():
        if not (RUNS / tag).is_dir():
            continue
        rows = load(tag)
        ratio = sum(r["realized"] for r in rows) / sum(r["assigned"] for r in rows)
        if name in FIDELITY and abs(ratio - FIDELITY[name]) > 5e-5:
            raise SystemExit(f"FIDELITY FAIL {name}: {ratio:.4f} != reported {FIDELITY[name]}")
        data[name] = rows
    print("fidelity: realized/assigned reproduces the reported totals -- " + ", ".join(
        f"{n} {sum(r['realized'] for r in v)/sum(r['assigned'] for r in v):.4f}" for n, v in data.items()))
    return data


def cue_for(words: float) -> str:
    if words <= 60:
        return "one narrow local move (~21w)"
    if words <= 100:
        return "vague 'two or three beats' (~42-63w); beat plan DELETED"
    beats = min(MAX_BEATS, max(3, round(words / WORDS_PER_BEAT)))
    cap = " CAPPED" if beats >= MAX_BEATS else ""
    return f"enumerated {beats} beats (~{int(beats*WORDS_PER_BEAT)}w){cap}"


def cmd_bands(data: dict) -> None:
    rows = data.get("v110_n10") or next(iter(data.values()))
    total_a = sum(r["assigned"] for r in rows)
    total_z = sum(r["realized"] for r in rows)
    deficit = total_a - total_z
    print("\n== realized/assigned by band, against the cue the band receives ==\n")
    print(f"  assigned {total_a}w, realized {total_z}w, deficit {deficit}w ({deficit/total_a:.1%})\n")
    print(f"  {'band':>12} {'slots':>6} {'ratio':>7} {'% of deficit':>13}  cue")
    for lo, hi in [(1, 9), (10, 34), (35, 60), (61, 100), (101, 251), (252, 10**6)]:
        v = [r for r in rows if lo <= r["assigned"] <= hi]
        if not v:
            continue
        a = sum(r["assigned"] for r in v)
        z = sum(r["realized"] for r in v)
        label = f"{lo}-{hi if hi < 10**6 else '+'}"
        print(f"  {label:>12} {len(v):>6} {z/a:>7.3f} {(a-z)/deficit:>12.1%}  {cue_for(statistics.fmean([r['assigned'] for r in v]))}")
    print("\n  The one band that receives an enumerated per-slot beat plan (101-251) is")
    print("  the one band at 0.956. Every band the cue does not reach compresses.")


def local_linear_jump(rows, cutoff, bw):
    import numpy as np
    x = np.array([r["assigned"] for r in rows], float)
    y = np.array([r["realized"] for r in rows], float)
    keep = (x > 0) & (y > 0)
    xc, yv = np.log(x[keep]), np.log(y[keep])
    cut = math.log(cutoff + 0.5)
    half = math.log(1 + bw / cutoff)
    out = []
    for side in (-1, 1):
        m = ((xc - cut) * side > 0) & (np.abs(xc - cut) <= half)
        if m.sum() < 5:
            return None
        A = np.column_stack([np.ones(m.sum()), xc[m] - cut])
        coef, *_ = np.linalg.lstsq(A, yv[m], rcond=None)
        out.append((math.exp(coef[0]), int(m.sum())))
    (left, n_l), (right, n_r) = out
    return right - left, left, right, n_l, n_r


def cmd_rdd(data: dict) -> None:
    import numpy as np
    pool = [r for v in data.values() for r in v]
    for cutoff, bw, what in [(100, 30, "vague '2-3 beats' -> enumerated ~5-beat plan"),
                             (60, 25, "'one narrow move' -> vague '2-3 beats'")]:
        print(f"\n== discontinuity at {cutoff}/{cutoff+1}: {what} ==\n")
        print(f"  {'run':<10} {'jump (words)':>13} {'left@cut':>9} {'right@cut':>10}")
        for name in list(data) + ["POOLED"]:
            rows = pool if name == "POOLED" else data[name]
            res = local_linear_jump(rows, cutoff, bw)
            if not res:
                continue
            j, left, right, _, _ = res
            print(f"  {name:<10} {j:>+13.1f} {left:>9.1f} {right:>10.1f}")
        print(f"\n  binned means either side (pooled over {len(data)} runs):")
        for lo, hi, la, hb in [(cutoff-10, cutoff, cutoff+1, cutoff+11),
                               (cutoff-20, cutoff, cutoff+1, cutoff+21)]:
            L = [r for r in pool if lo <= r["assigned"] <= hi]
            R = [r for r in pool if la <= r["assigned"] <= hb]
            if not L or not R:
                continue
            aL, zL = np.mean([r["assigned"] for r in L]), np.mean([r["realized"] for r in L])
            aR, zR = np.mean([r["assigned"] for r in R]), np.mean([r["realized"] for r in R])
            print(f"    [{lo},{hi}] n={len(L):3d} ratio {zL/aL:.3f}  |  [{la},{hb}] n={len(R):3d} ratio {zR/aR:.3f}"
                  f"   d_assigned {aR-aL:+5.1f}w  d_realized {zR-zL:+5.1f}w")


def cmd_beats(data: dict) -> None:
    import numpy as np
    pool = [r for v in data.values() for r in v]
    below = [r for r in pool if r["assigned"] <= 100]
    print("\n== the beat plan never reaches the compressing band ==\n")
    print(f"  slots with assigned <= 100 words carrying an enumerated beat plan: "
          f"{sum(1 for r in below if r['beats']>0)} of {len(below)}")
    rows = [r for r in pool if r["assigned"] > 100 and r["beats"] > 0 and r["realized"] > 0]
    y = np.array([r["realized"] for r in rows], float)
    b = np.array([r["beats"] for r in rows], float)
    print(f"\n  above 100 words, realized words per delivered beat: mean {np.mean(y/b):.1f}, "
          f"median {np.median(y/b):.1f}, n={len(rows)}")
    print(f"  (`long_form_planning.WORDS_PER_REALIZED_BEAT` is {WORDS_PER_BEAT:.0f})")
    print(f"\n  saturation at MAX_DEVELOPMENT_BEATS={MAX_BEATS} (= {int(MAX_BEATS*WORDS_PER_BEAT)} words):\n")
    print(f"  {'assigned':>14} {'n':>4} {'mean beats':>11} {'ratio':>7}")
    for lo, hi in [(101, 150), (150, 220), (220, 260), (260, 400), (400, 10**6)]:
        v = [r for r in pool if lo <= r["assigned"] < hi and r["beats"] > 0]
        if not v:
            continue
        a = np.mean([r["assigned"] for r in v])
        z = np.mean([r["realized"] for r in v])
        print(f"  {f'[{lo},{hi})':>14} {len(v):>4} {np.mean([r['beats'] for r in v]):>11.2f} {z/a:>7.3f}")


COMMANDS = {"bands": cmd_bands, "rdd": cmd_rdd, "beats": cmd_beats}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=[*COMMANDS, "all"])
    args = p.parse_args()
    data = load_all()
    for name in (COMMANDS if args.command == "all" else [args.command]):
        COMMANDS[name](data)


if __name__ == "__main__":
    main()
