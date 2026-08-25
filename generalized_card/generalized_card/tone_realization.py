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

`POLITE_ASSIGNMENT_CAP` is therefore set at the measured `agree` share. Every
polite slot the inversion asks for then sits inside the regime where C's polite
row was measured, and no cell of the solution is an extrapolation.

The cap costs less than it looks. The dominant move is not more polite, it is
`impolite` assignment shifting to `neutral`: the impolite row realizes impolite at
0.854 and the neutral row at 0.404, and both are richly observed (n=522, n=156).
Measured closure of the four-class L2 gap by cap: 0.30 -> 39%, **0.35 -> 48%**,
0.40 -> 57%, uncapped -> 86%.

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

# The measured `agree` stance share. See the module docstring.
POLITE_ASSIGNMENT_CAP = 0.35

# Grid step for the constrained solve. The quota becomes integer slot counts over
# a thread of ~45 comments, so 1/45 = 0.022 is the meaningful resolution and this
# is four times finer. A grid keeps the solve deterministic and dependency-free.
_GRID_STEP = 0.005

# `off` reproduces every version through v114, where the quota rendered to the
# Planner was the template's own rates.
TONE_QUOTA_MODE = "off"

_CACHE: dict[tuple[tuple[float, ...], float], dict[str, float]] = {}


def set_tone_quota_mode(mode: str | None) -> bool:
    """Select the tone-quota arm and return whether the inversion is active."""

    global TONE_QUOTA_MODE
    TONE_QUOTA_MODE = "inverted" if str(mode or "off").strip().lower() == "inverted" else "off"
    return TONE_QUOTA_MODE == "inverted"


def tone_quota_inverted() -> bool:
    return TONE_QUOTA_MODE == "inverted"


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
        loss = sum((realized[j] - target[j]) ** 2 for j in range(4))
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
