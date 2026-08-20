"""Fit a thread's tone marginal onto slots using a measured length conditional.

The held-out template owns how many comments of each tone register a thread
has. It does not say which slots carry them, and matching the marginal while
getting the joint backwards is a way to fail every tone metric with a correct
target.

That is what happened through v96. `_tone_cost` ranked candidate slots by
distance from the class's median length, so `polite` (median 53 words) went to
the slots nearest 53 words and the longest slots were left for whatever label
was assigned last. In the v96 ten-thread run the plan put `impolite` on 74% of
120-250 word slots and 100% of slots over 250 words. Measured over 15,294
comments in the same domain's evaluation-excluded threads, real comments over
250 words are 72.0% polite and 23.0% impolite, and the trend is monotone:

    band        n     polite  somewhat  neutral  impolite
    micro     2223     0.251     0.064    0.281     0.404
    short     4220     0.162     0.123    0.234     0.481
    medium    4814     0.263     0.115    0.145     0.476
    long      2690     0.520     0.071    0.059     0.350
    very_long 1104     0.638     0.038    0.035     0.289
    essay      243     0.720     0.021    0.029     0.230

The realized output followed the plan: generated comments over 120 words came
out 87% impolite and 9% polite against a real 27% and 71%.

The measured conditional also contradicts the old hard exclusion that a warm
turn cannot be a micro reaction. A quarter of real comments under ten words are
labelled polite, because a short thank-you is one. Compatibility is therefore
left to the measurement instead of a written-down rule.

The fit keeps the template's label totals exact -- they are the contract -- and
spends the remaining freedom on the measured conditional.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from .comment_structure import structure_bucket


TONE_LENGTH_PROFILE_KEY = "tone_length_profile"
# `median` reproduces every version through v96: candidate slots ranked by
# distance from the tone class's median length, which inverted the joint at the
# long tail.
TONE_LENGTH_FIT_ENABLED = True
_MIN_SAMPLES = 400
_MIN_BAND_SAMPLES = 60
# A label the measurement never saw in a band is unlikely, not impossible; the
# floor keeps it assignable when the template's totals leave no alternative.
_SHARE_FLOOR = 1e-4


def set_tone_length_fit(mode: str) -> bool:
    """Select the tone-placement arm and return whether the fit is active."""

    global TONE_LENGTH_FIT_ENABLED
    TONE_LENGTH_FIT_ENABLED = str(mode or "conditional").strip().lower() != "median"
    return TONE_LENGTH_FIT_ENABLED


def build_tone_length_profile(
    raw_discussions_dir: Path,
    *,
    reference_thread_ids: Iterable[str],
    tone_classes: tuple[str, ...],
) -> dict[str, Any]:
    """Measure P(tone class | comment size band) on evaluation-excluded threads.

    Only counts are stored. The per-comment text is read to measure length and
    is never retained.
    """

    reference = {str(value).strip() for value in reference_thread_ids if str(value).strip()}
    counts: dict[str, Counter[str]] = {}
    total = 0
    for path in sorted(Path(raw_discussions_dir).rglob("politeness_results.json")):
        payload = _load_json(path)
        for thread in payload.get("threads") or []:
            if not isinstance(thread, dict):
                continue
            if str(thread.get("thread_id") or "").strip() not in reference:
                continue
            for row in thread.get("comments") or []:
                label = str((row or {}).get("pred_label") or "").strip().lower()
                if label not in tone_classes:
                    continue
                band = structure_bucket(len(str(row.get("text") or "").split()))
                counts.setdefault(band, Counter())[label] += 1
                total += 1
    if total < _MIN_SAMPLES:
        return {"available": False, "sample_count": total, "conditional": {}}
    conditional = {
        band: {
            label: round(row.get(label, 0) / sum(row.values()), 6)
            for label in tone_classes
        }
        for band, row in counts.items()
        if sum(row.values()) >= _MIN_BAND_SAMPLES
    }
    return {
        "available": bool(conditional),
        "method": (
            "share of each evaluation-classifier tone label per comment size band, "
            "over same-domain threads excluded from the evaluation seed pool"
        ),
        "sample_count": total,
        "band_sample_counts": {
            band: sum(row.values()) for band, row in sorted(counts.items())
        },
        "conditional": dict(sorted(conditional.items())),
    }


def band_shares(
    profile: dict[str, Any] | None,
    band: str,
    tone_classes: tuple[str, ...],
) -> dict[str, float]:
    """Return the measured label shares for one band, or a uniform fallback."""

    conditional = (profile or {}).get("conditional") or {}
    row = conditional.get(band)
    if not row:
        return {label: 1.0 / len(tone_classes) for label in tone_classes}
    return {label: max(_SHARE_FLOOR, float(row.get(label, 0.0))) for label in tone_classes}


def fit_tone_labels(
    slots: list[dict[str, Any]],
    totals: Counter[str],
    *,
    profile: dict[str, Any] | None,
    tone_classes: tuple[str, ...],
) -> tuple[dict[int, str], list[str]]:
    """Assign exactly `totals` labels to slots, reproducing the measured joint.

    Both margins are fixed: every slot takes one label, and the template's label
    counts are the contract. Fitting the measured conditional subject to those
    two margins is iterative proportional fitting, so that is what runs here.

    A min-cost assignment was tried first and rejected. It maximizes total
    likelihood, which drives the solution to a corner: on the ten v96 threads it
    produced 100% polite in the largest band against a measured 72%, and 98%
    impolite in the `short` band against a measured 48%. Proportional fitting
    keeps the same corrected direction without inventing a sharper dependence
    than the reference data shows.
    """

    remaining = Counter(
        {label: int(count) for label, count in totals.items() if int(count) > 0}
    )
    if not slots or not remaining:
        return {}, [
            label for label, count in remaining.items() for _ in range(count)
        ]
    labels = sorted(remaining)
    bands: dict[str, list[dict[str, Any]]] = {}
    for slot in slots:
        bands.setdefault(_band(slot), []).append(slot)
    band_names = sorted(bands)
    row_totals = {band: len(bands[band]) for band in band_names}
    # The two margins must agree before fitting. When the template's counts do
    # not cover every slot, the shortfall is reported rather than invented.
    assignable = min(sum(row_totals.values()), sum(remaining.values()))
    cells = _proportional_fit(
        row_totals=row_totals,
        column_totals={label: remaining[label] for label in labels},
        seed={
            band: band_shares(profile, band, tone_classes) for band in band_names
        },
        assignable=assignable,
    )
    assignments: dict[int, str] = {}
    for band in band_names:
        # Inside a band, the longer slots take the labels the measurement ties
        # most strongly to length, so the ordering stays monotone at the edges
        # of a coarse band as well as across bands.
        ordered_slots = sorted(
            bands[band],
            key=lambda slot: (-int(slot.get("words") or 0), int(slot["sample_id"])),
        )
        shares = band_shares(profile, band, tone_classes)
        queue: list[str] = []
        for label in sorted(labels, key=lambda name: (-shares[name], name)):
            queue.extend([label] * cells.get((band, label), 0))
        for slot, label in zip(ordered_slots, queue):
            assignments[int(slot["sample_id"])] = label
            remaining[label] -= 1
    unassigned = [
        label for label, count in remaining.items() for _ in range(max(0, count))
    ]
    return assignments, unassigned


def _proportional_fit(
    *,
    row_totals: dict[str, int],
    column_totals: dict[str, int],
    seed: dict[str, dict[str, float]],
    assignable: int,
    iterations: int = 200,
) -> dict[tuple[str, str], int]:
    """Return integer cell counts matching both margins, closest to `seed`."""

    bands = sorted(row_totals)
    labels = sorted(column_totals)
    if not bands or not labels or assignable <= 0:
        return {}
    scale = assignable / max(1, sum(row_totals.values()))
    rows = {band: row_totals[band] * scale for band in bands}
    columns_scale = assignable / max(1, sum(column_totals.values()))
    columns = {label: column_totals[label] * columns_scale for label in labels}
    matrix = {
        (band, label): max(_SHARE_FLOOR, seed[band][label]) * rows[band]
        for band in bands
        for label in labels
    }
    for _ in range(iterations):
        for band in bands:
            current = sum(matrix[(band, label)] for label in labels)
            if current > 0:
                factor = rows[band] / current
                for label in labels:
                    matrix[(band, label)] *= factor
        for label in labels:
            current = sum(matrix[(band, label)] for band in bands)
            if current > 0:
                factor = columns[label] / current
                for band in bands:
                    matrix[(band, label)] *= factor
    counts = {key: int(value) for key, value in matrix.items()}
    row_left = {
        band: int(round(rows[band])) - sum(counts[(band, label)] for label in labels)
        for band in bands
    }
    column_left = {
        label: int(round(columns[label]))
        - sum(counts[(band, label)] for band in bands)
        for label in labels
    }
    remainders = sorted(
        matrix,
        key=lambda key: (-(matrix[key] - counts[key]), key[0], key[1]),
    )
    for band, label in remainders:
        if row_left.get(band, 0) > 0 and column_left.get(label, 0) > 0:
            counts[(band, label)] += 1
            row_left[band] -= 1
            column_left[label] -= 1
    # Rounding can leave a unit stranded when the largest remainders collide;
    # any cell with capacity on both margins can absorb it.
    for band in bands:
        while row_left.get(band, 0) > 0:
            label = next(
                (name for name in labels if column_left.get(name, 0) > 0), None
            )
            if label is None:
                break
            counts[(band, label)] += 1
            row_left[band] -= 1
            column_left[label] -= 1
    return {key: value for key, value in counts.items() if value > 0}


def realized_joint(
    slots: list[dict[str, Any]],
    assignments: dict[int, str],
) -> dict[str, dict[str, int]]:
    """Report the (band, label) table the schedule actually produced."""

    table: dict[str, Counter[str]] = {}
    by_id = {int(slot["sample_id"]): slot for slot in slots}
    for sample_id, label in assignments.items():
        slot = by_id.get(int(sample_id))
        if slot is None:
            continue
        table.setdefault(_band(slot), Counter())[label] += 1
    return {band: dict(sorted(row.items())) for band, row in sorted(table.items())}


def _band(slot: dict[str, Any]) -> str:
    return structure_bucket(slot.get("words"))


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}
