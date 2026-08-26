"""Which `POLITE_ASSIGNMENT_CAP` to ship, and against which objective (G66).

G60 left this as "a decision, not a measurement" and framed it as one tradeoff:
cap 0.59 lands `polite_rate` and `impolite_rate` almost exactly while costing
`neutral_rate` -19.7%, against cap 0.35 leaving all three mid-range. Sweeping the
whole range shows two things that reframe it.

**1. The solve saturates.** Under the shipped matrix the unconstrained optimum is
an interior `a_polite = 0.645`, so every cap at or above that is the same
solution. The cap is only a real constraint below it.

**2. The objective was wrong, and that mattered more than the cap.**
`invert_tone_rates` minimised the FOUR-class L2, but `somewhat_polite` is a real
class that absorbs mass and is **never reported**. Error parked there costs
nothing that is judged, so it should not be in the loss. Minimising only the three
reported rates strictly dominates at every cap that matters, and it removes a
landmine: under the four-way loss at the shipped cap 0.35 the arm drives
`neutral_rate` to **4.6x** its current error (closure -463%). `--tone-quota
inverted` has never been run, so it never fired.

**The decision rule.** Judge by the WORST of the three reported metrics' closure,
because acceptance is per-metric (`ORIENTATION.md` s2) -- a metric that gets worse
is a direct risk to its own p-value, while a lower total L2 is judged by nothing.
And take the worst over BOTH known realization matrices, because they disagree:
the shipped one is measured on the evaluation-seed corpus at an unbalanced
289/92/156/522 assignment, the refit on the calibration run's balanced
137/137/140/145. A cap that is excellent under one and negative under the other is
not a decision, it is a bet.

Usage:  python3 generalized_card/analysis/tone_carrier/cap_decision.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path("/Users/yaoningyu/Desktop/UIUC/GEO")
sys.path.insert(0, str(REPO / "generalized_card"))
import generalized_card.tone_realization as tr  # noqa: E402

PROFILE = REPO / "artifacts/generalized_card/runs/v117_calibration_20260826_v1/domain_profile.json"
# fit_tone_matrix.py v117_calibration_20260826_v1, n=559, every row n>=137.
REFIT = (
    (0.3942, 0.1971, 0.1022, 0.3066),
    (0.1241, 0.3650, 0.1460, 0.3650),
    (0.1357, 0.1286, 0.2429, 0.4929),
    (0.0069, 0.0069, 0.0966, 0.8897),
)
REPORT = ("polite", "impolite", "neutral")


def _target() -> dict[str, float]:
    counts = json.loads(PROFILE.read_text())["register_profile"]["tone_sample_counts"]
    total = sum(counts.values())
    return {k: counts[k] / total for k in tr.TONE_ORDER}


def _realized(matrix, assignment):
    return tuple(
        sum(assignment[i] * matrix[i][j] for i in range(4)) for j in range(4)
    )


# The shipped solve uses 0.005. This report sweeps 80+ caps, so it runs at 0.01;
# the reported closures move by well under a point and the ranking is unchanged.
REPORT_GRID_STEP = 0.01


def _solve(matrix, target, cap, *, reported_only: bool):
    idx = [tr.TONE_ORDER.index(m) for m in REPORT] if reported_only else range(4)
    tgt = tuple(target[name] for name in tr.TONE_ORDER)
    best, best_loss = None, float("inf")
    for candidate in tr._simplex_grid(REPORT_GRID_STEP, cap):
        realized = _realized(matrix, candidate)
        loss = sum((realized[j] - tgt[j]) ** 2 for j in idx)
        if loss < best_loss:
            best_loss, best = loss, candidate
    return best


def _closures(matrix, target, assignment):
    legacy = _realized(matrix, tuple(target[name] for name in tr.TONE_ORDER))
    base = {tr.TONE_ORDER[i]: legacy[i] for i in range(4)}
    now = _realized(matrix, assignment)
    got = {tr.TONE_ORDER[i]: now[i] for i in range(4)}
    return {
        m: 1 - abs(got[m] - target[m]) / abs(base[m] - target[m]) for m in REPORT
    }


def main() -> None:
    target = _target()
    matrices = (("shipped", tr.REALIZATION_MATRIX), ("refit", REFIT))
    caps = [round(0.30 + 0.02 * i, 2) for i in range(21)] + [tr.POLITE_ASSIGNMENT_CAP]
    caps = sorted(set(caps))
    print(f"target p/i/n = {target['polite']:.3f}/{target['impolite']:.3f}/{target['neutral']:.3f}")
    for reported_only in (False, True):
        label = "reported three only" if reported_only else "four-way L2 (pre-G66)"
        print(f"\n=== objective: {label} ===")
        print(f"{'cap':>6}{'shipped worst':>15}{'refit worst':>13}{'ROBUST':>9}")
        rows = []
        for cap in caps:
            worst = []
            for _, matrix in matrices:
                a = _solve(matrix, target, cap, reported_only=reported_only)
                worst.append(min(_closures(matrix, target, a).values()))
            rows.append((cap, worst[0], worst[1], min(worst)))
        for cap, a, b, r in rows:
            if True:
                mark = "  <- shipped" if abs(cap - tr.POLITE_ASSIGNMENT_CAP) < 1e-9 else ""
                print(f"{cap:>6.2f}{a:>15.0%}{b:>13.0%}{r:>9.0%}{mark}")
        best = max(rows, key=lambda row: row[3])
        print(f"   robust optimum cap {best[0]:.2f}, worst-metric closure >= {best[3]:+.0%}")


if __name__ == "__main__":
    main()
