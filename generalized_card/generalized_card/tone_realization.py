"""Ask the Planner for the tone mix that comes out right, not the one that goes in.

The Planner's tone quota is rendered from the reference template's own
`polite_rate`/`impolite_rate`/`neutral_rate`, so the ASSIGNED mix matches real to
within a couple of points. The realized mix does not: the Writer realizes an
assigned `impolite` 85.4% of the time and an assigned `polite` 38.4%, so the
output runs 0.607 impolite against real's 0.464 and 0.129 polite against 0.260.

Six realization-side hypotheses are dead (`analysis/tone_carrier/FINDINGS.md`):
more register cues, the omitted conjunction, hedging, length repair, the
bare-assertion frame, and the polite lexicon -- the generator already carries
real's polite-discriminative vocabulary at 1.14x real prevalence. Conditioned on
the same move word it converts at 0.26-0.45x, so the deficit is not nameable as a
surface feature.

But the Writer's failure is *consistent*, and a consistent failure is a transfer
matrix. With `C[i][j] = P(realize j | assign i)` measured, the assignment that
lands the realized mix on the template is the solution of `C^T a = template`.
Nothing about the Writer changes. This is the same inverse calibration
`length_calibration` performs for word counts.

Why the polite share is capped
------------------------------
C is measured at today's assignment mix, and P(realize polite | assign polite) is
stable across `comment_function`, `payload_type`, `evidence_mode` and
`speaker_role` (0.310-0.474 around a 0.384 base) but NOT across `stance`: 261 of
289 polite assignments sit on `agree` slots and realize at 0.402, while the 17 on
`uncertain` realize at 0.059. `prompts.py:951` forbids polite on a disagreeing
stance outright. `agree` is 34.3% of slots, so a polite share above that has to
land somewhere the rate has never been measured.

`POLITE_ASSIGNMENT_CAP` was therefore set at the measured `agree` share, 0.35, so
that no cell of the solution was an extrapolation.

**That constraint has since been measured away, and the cap is now 0.56 (G66).**
The `v117_calibration` run rendered a deliberately flat quota, which spread every
tone class across every stance and gave each row of C n>=137 -- the polite row is
now measured across every stance, not only `agree`, and it moved only 0.3841 ->
0.3942. The hedge the 0.35 cap encoded is discharged.

Which cap to ship is a decision, and it is made against the **three reported
metrics** -- `polite_rate`, `impolite_rate`, `neutral_rate` -- by worst-metric
closure, because the acceptance rule is per-metric (`ORIENTATION.md` s2): a metric
that gets *worse* is a direct risk to its own p-value, while a lower total L2 is
not judged by anything. Robustness matters too: C is known on two corpora that
disagree, so the shipped value maximises the WORST closure over both.

Worst-of-three closure, minimised over the shipped matrix and the calibration
refit, under the reported-three objective (`analysis/tone_carrier/cap_decision.py`):

    cap    0.45   0.50   0.53   0.55   **0.56**   0.57   0.59   0.62
    robust  44%    54%    59%    63%   **65%**    63%    47%    38%

`somewhat_polite` is deliberately excluded from the objective. It is a real fourth
class that absorbs mass and **is never reported**, so error pushed into it costs
nothing that is measured. Optimising the four-way L2 instead is what made the old
setting dangerous: at cap 0.35 it drives `neutral_rate` to **4.6x** its current
error under the shipped matrix (closure -463%). `--tone-quota inverted` has never
been run, so that landmine was never fired.

Provenance
----------
`REALIZATION_MATRIX` is pooled over `v110_length_transfer_n10_20260824_v1` and
`v113_v112_gate_n10_20260826_v1`, 1,059 slots, scored by the shipped
`Intel/polite-guard` harness. Rebuilt by
`analysis/tone_carrier/fit_tone_matrix.py`.

It is a property of this generator, not of any thread, and it is FROZEN. Refitting
it per run against a run's own output would be tuning, and refitting it against
test-set p-values is forbidden outright (`ORIENTATION.md` s4). The matrix was
measured on runs over evaluation seeds 2-11, which is disclosed rather than hidden:
those ten seeds are in-sample for the calibration at N=150. Measuring C on a
calibration run over profile threads instead removes even that, and is the
recommended refit before the paper run.
"""

from __future__ import annotations

from typing import Any, Iterable

# Column and row order for `REALIZATION_MATRIX`.
TONE_ORDER: tuple[str, ...] = ("polite", "somewhat_polite", "neutral", "impolite")

# C[i][j] = P(realize TONE_ORDER[j] | assign TONE_ORDER[i]).
REALIZATION_MATRIX: tuple[tuple[float, ...], ...] = (
    (0.3841, 0.1938, 0.0900, 0.3322),   # assigned polite,          n=289
    (0.0761, 0.4130, 0.0978, 0.4130),   # assigned somewhat_polite, n=92
    (0.0897, 0.0962, 0.4103, 0.4038),   # assigned neutral,         n=156
    (0.0096, 0.0307, 0.1054, 0.8544),   # assigned impolite,        n=522
)
REALIZATION_MATRIX_PROVENANCE: dict[str, Any] = {
    "runs": (
        "v110_length_transfer_n10_20260824_v1",
        "v113_v112_gate_n10_20260826_v1",
    ),
    "slots": 1059,
    "classifier": "Intel/polite-guard",
    "row_counts": {"polite": 289, "somewhat_polite": 92, "neutral": 156, "impolite": 522},
    "fitted_by": "generalized_card/analysis/tone_carrier/fit_tone_matrix.py",
    "note": "frozen; refitting per run against the run's own output would be tuning",
}

# The robust maximin value over the two known realization matrices, judged on the
# three reported metrics. See the module docstring; was 0.35, the `agree` share,
# until the calibration run measured the polite row across every stance (G66).
POLITE_ASSIGNMENT_CAP = 0.56

# Grid step for the constrained solve. The quota becomes integer slot counts over
# a thread of ~45 comments, so 1/45 = 0.022 is the meaningful resolution and this
# is four times finer. A grid keeps the solve deterministic and dependency-free.
_GRID_STEP = 0.005

# The objective's coordinates. `somewhat_polite` is a real class that carries mass
# and is NEVER reported, so error parked there costs nothing that is judged;
# including it in the loss is what made cap 0.35 drive `neutral_rate` 4.6x worse.
REPORTED_TONES = ("polite", "neutral", "impolite")
_REPORTED_INDEXES = tuple(TONE_ORDER.index(name) for name in REPORTED_TONES)

# `off` reproduces every version through v114, where the quota rendered to the
# Planner was the template's own rates.
#
# `calibrate` is a MEASUREMENT value, never a paper artifact. It renders a flat
# quota so the Planner spreads every tone class across every stance, which is the
# only way to populate the cells `POLITE_ASSIGNMENT_CAP` exists because nobody has
# measured: today 261 of 289 polite assignments sit on `agree` slots, so
# P(realize polite | assign polite, stance != agree) rests on n=17. A calibration
# run over threads outside the evaluation pool fills those cells, and the cap can
# then rise on evidence instead of staying pinned at the `agree` share.
#
# It is recorded in `run_config.json` and in `RUN_EXPERIMENT_FIELDS`, so a
# calibration artifact can never be mistaken for a candidate.
TONE_QUOTA_MODE = "off"

# The flat quota `calibrate` renders. Deliberately uniform rather than fitted:
# the point is coverage of the (stance, assigned tone) grid, not realism.
CALIBRATION_RATES: dict[str, float] = {label: 0.25 for label in TONE_ORDER}

_CACHE: dict[tuple[tuple[float, ...], float], dict[str, float]] = {}


def set_tone_quota_mode(mode: str | None) -> bool:
    """Select the tone-quota arm and return whether it changes the rendered quota."""

    global TONE_QUOTA_MODE
    value = str(mode or "off").strip().lower()
    TONE_QUOTA_MODE = value if value in ("inverted", "calibrate") else "off"
    return TONE_QUOTA_MODE != "off"


def tone_quota_inverted() -> bool:
    return TONE_QUOTA_MODE == "inverted"


def tone_quota_calibrating() -> bool:
    return TONE_QUOTA_MODE == "calibrate"


def _simplex_grid(step: float, cap: float) -> list[tuple[float, ...]]:
    """Every 4-class distribution on `step`'s lattice whose first entry is <= cap."""

    n = int(round(1.0 / step))
    cap_units = int(cap * n)
    out: list[tuple[float, ...]] = []
    for i in range(0, cap_units + 1):
        for j in range(0, n - i + 1):
            for k in range(0, n - i - j + 1):
                out.append((i / n, j / n, k / n, (n - i - j - k) / n))
    return out


def _realized(assignment: Iterable[float]) -> tuple[float, ...]:
    a = tuple(assignment)
    return tuple(
        sum(a[i] * REALIZATION_MATRIX[i][j] for i in range(4)) for j in range(4)
    )


def invert_tone_rates(
    rates: dict[str, float] | None,
    *,
    cap: float = POLITE_ASSIGNMENT_CAP,
) -> dict[str, float]:
    """Return the assignment mix whose REALIZED mix best matches ``rates``.

    Returns the input unchanged when the arm is off or the input is empty. The
    four-class guard below is defensive only: the sole caller,
    `generation_distribution.template_tone_rates`, always completes the vector,
    filling `somewhat_polite` from the residual when the profile lacks the field.

    That residual carries the template's whole measurement error, so on an older
    profile the inversion optimises against a partly inferred class. This is not
    a new risk -- the legacy path renders the same residual as a quota -- but it
    is the reason `somewhat_polite` is excluded from the reported metrics.
    """

    if tone_quota_calibrating():
        return dict(CALIBRATION_RATES)
    if not rates or not tone_quota_inverted():
        return dict(rates or {})
    if any(label not in rates for label in TONE_ORDER):
        return dict(rates)
    total = sum(max(0.0, float(rates.get(label) or 0.0)) for label in TONE_ORDER)
    if total <= 0:
        return dict(rates)
    target = tuple(max(0.0, float(rates[label])) / total for label in TONE_ORDER)

    key = (tuple(round(v, 6) for v in target), round(cap, 6))
    if key in _CACHE:
        return dict(_CACHE[key])

    best: tuple[float, ...] | None = None
    best_loss = float("inf")
    for candidate in _simplex_grid(_GRID_STEP, cap):
        realized = _realized(candidate)
        loss = sum((realized[j] - target[j]) ** 2 for j in _REPORTED_INDEXES)
        if loss < best_loss:
            best_loss, best = loss, candidate
    assert best is not None
    solved = {label: best[i] for i, label in enumerate(TONE_ORDER)}
    _CACHE[key] = solved
    return dict(solved)


def realization_report(rates: dict[str, float] | None) -> dict[str, Any]:
    """What the arm did, for the run audit. Never used to select anything."""

    template = {
        label: round(float((rates or {}).get(label) or 0.0), 6) for label in TONE_ORDER
    }
    if tone_quota_calibrating():
        return {
            "mode": TONE_QUOTA_MODE,
            "template_rates": template,
            "assignment_rates": dict(CALIBRATION_RATES),
            "purpose": (
                "measurement only: populates the (stance, assigned tone) grid so "
                "POLITE_ASSIGNMENT_CAP can be set on evidence. Not a candidate artifact."
            ),
        }
    if not tone_quota_inverted():
        return {"mode": TONE_QUOTA_MODE, "template_rates": template}
    solved = invert_tone_rates(rates)
    realized = _realized(tuple(solved.get(label, 0.0) for label in TONE_ORDER))
    return {
        "mode": TONE_QUOTA_MODE,
        "template_rates": template,
        "assignment_rates": {k: round(v, 6) for k, v in solved.items()},
        "projected_realized_rates": {
            label: round(realized[i], 6) for i, label in enumerate(TONE_ORDER)
        },
        "polite_assignment_cap": POLITE_ASSIGNMENT_CAP,
        "matrix_provenance": REALIZATION_MATRIX_PROVENANCE,
    }
