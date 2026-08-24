"""Hold each slot's realized length inside the length band it was assigned.

**The defect.** Every slot already carries `real_word_count`, copied from the
matched real comment it fills, and the Writer is already given a length cue. It
does not obey it, and it disobeys in a shape that compresses the thread. Measured
on the v109 gate's 186 slots, realized-over-assigned words run

    assigned <10 words   1.44x     (6.2 -> 8.9)
    assigned 10-50       ~1.00x
    assigned 50-100      0.82x     (72.7 -> 59.8, 40 slots)
    assigned >100        0.89x

-- 91.6% of the assigned words overall, with **compression at both tails**. That
is exactly what holds `length_cv` at 0.857 against a real 0.895 and inflates the
pooled brevity-penalty term to 30.1% of `self_bleu_4`'s log excess.

**Why it is the largest verified lever.** Both priority metrics are means over
unordered within-thread comment pairs, so generated's metric can be evaluated
exactly at real's length-cell distribution. Doing that (`docs/DECISIONS.md` G43)
puts length composition at **33-37% of `self_bleu_4`'s gap** and **17-26% of
`self_bertscore_mean_f1`'s**, stable across 5-to-10 bin choices at 100% cell
coverage. Nothing else priced this session comes close: entity variety <=9.4% and
saturating (G40), full Planner de-duplication <=2.4% (G45), the two absent
surface forms 12-15% (G44). And per G42 the target -- `p ~ 0.5-0.6` -- needs
~90% closure at N=150, so only levers of this size are worth building.

**Why it has never worked.** Three independent reasons, all found by reading the
code and the run's own records rather than by hypothesising:

1. `length_policy.soft_length_guidance`'s own docstring already names the defect
   ("a purely permissive cue let every long slot regress toward the mean, which
   compressed the thread's length spread") and deliberately keeps it a cue.
2. `writer_quality.substantive_length_floor_problem`'s floor is
   `max(8, min(32, round(real_words * 0.5)))` -- half the target, **capped at 32
   words**. A slot assigned 100 words passes at 32. Of the 24 slots that
   realized under 80% of an assigned 40+ words, only **4** raised any length
   problem.
3. `--writer-retries` defaults to **0**, so `total_attempts = 1` and the Writer
   validation loop never gets a second attempt. On the v109 gate **65 of 186
   slots failed their own validator and every one was shipped unretried**,
   including 32 `template_phrase_reused` -- the form-duplication visible by eye
   in the highest-similarity realized pairs.

**The rule this module adds.** A slot's realized word count must fall in the
**same measured length band as its assigned word count**, bands being deciles of
the domain's own evaluation-excluded comment lengths.

**Bands are deciles, and that is a design choice, not a tuned one.** The
33-37% price in G43 was computed with the target thread's own quintiles. The
shipped mechanism cannot use those -- a profile may only be built from excluded
threads -- and re-pricing on the excluded corpus's cuts gives **19.5%
(`self_bleu_4`) / 18.0% (`self_bertscore`) at quintiles** and **31.3% / 13.7% at
deciles**. Deciles are chosen because of where the defect sits, not because of
those numbers: camera's quintile cuts are [11, 22, 38, 72], so the top band is
open above 72 words and a slot assigned 100 words has **no upper constraint at
all** -- exactly the 50-100-word band that realizes at 0.82x. Deciles put cuts at
72 and 111, which bounds it. The quintile figures are recorded here so the choice
stays auditable, and the honest expectation for this arm is therefore **~31% of
`self_bleu_4`'s gap and ~14% of `self_bertscore`'s**, not G43's 33-37% / 17-26%.

**Domain adaptivity** is in the cuts, not in a tolerance constant: a domain whose
comments are shorter gets proportionally lower band edges, measured from its own
excluded corpus. Measured decile cuts differ sharply across the four registered
domains -- camera [6, 11, 16, 22, 29, 38, 52, 72, 111] against headphone's
roughly half that -- so the same rule means different word counts per domain. A
band with fewer than `MIN_BAND_COMMENTS` reference comments withholds the check
rather than defaulting it, the same degradation contract `entity_spread` and the
register profiles use.

**Slot preservation.** The problem is registered as *soft*, so it can never make
a slot blocking and be dropped -- `docs/ORIENTATION.md` §4 requires every matched
structural slot to survive. It works purely by making the Writer's retry loop see
a problem, which is why the arm is only meaningful together with a non-zero
`--writer-retries`.

The arm is `--length-fidelity {off,measured}`, default `off`, which registers
nothing and reproduces the previous release byte-for-byte.
"""

from __future__ import annotations

import statistics
from typing import Any

# Installed per run from the frozen domain profile, like ACTIVE_RHYTHM_PROFILE.
ACTIVE_LENGTH_FIDELITY_PROFILE: dict[str, Any] = {}
LENGTH_FIDELITY_ENABLED = False

# Deciles. See the module docstring: quintiles leave the top band open above the
# 80th percentile, which is precisely where the undershoot lives.
BAND_QUANTILES: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)

# A band measured on fewer real comments than this withholds the check for slots
# assigned into it, rather than gating against a cut point built from noise.
MIN_BAND_COMMENTS = 40

PROBLEM_PREFIX = "length_band_mismatch:"


def set_active_length_fidelity_profile(profile: dict[str, Any] | None) -> None:
    """Install the frozen per-domain length-band profile for this run."""

    global ACTIVE_LENGTH_FIDELITY_PROFILE
    ACTIVE_LENGTH_FIDELITY_PROFILE = dict(profile or {})


def set_length_fidelity(mode: str) -> bool:
    """Select the arm and return whether it is active."""

    global LENGTH_FIDELITY_ENABLED
    LENGTH_FIDELITY_ENABLED = str(mode or "off").strip().lower() == "measured"
    return LENGTH_FIDELITY_ENABLED


def build_length_fidelity_profile(threads: Any) -> dict[str, Any]:
    """Measure this domain's own comment-length band edges.

    Word counts come from evaluation-excluded reference threads only, the same
    corpus every other measured profile is built from. The cuts are quintiles of
    the pooled per-comment word count, and each band records how many reference
    comments support it so a thin band can withhold rather than default.
    """

    words: list[int] = []
    for thread in threads or ():
        for row in thread.get("comments") or []:
            text = str(row.get("body") or row.get("content") or "")
            count = len(text.split())
            if count:
                words.append(count)
    if len(words) < MIN_BAND_COMMENTS * (len(BAND_QUANTILES) + 1):
        return {"available": False, "comment_count": float(len(words))}

    percentiles = statistics.quantiles(sorted(words), n=100)
    cuts = [float(percentiles[int(q * 100) - 1]) for q in BAND_QUANTILES]
    counts: dict[str, float] = {}
    for count in words:
        counts[str(band_of(count, cuts))] = counts.get(str(band_of(count, cuts)), 0.0) + 1.0
    return {
        "available": True,
        "cuts": cuts,
        "band_counts": counts,
        "comment_count": float(len(words)),
    }


def band_of(words: Any, cuts: list[float] | tuple[float, ...]) -> int:
    """Index of the measured band this word count falls in."""

    try:
        value = float(words)
    except (TypeError, ValueError):
        return -1
    for index, cut in enumerate(cuts):
        if value <= cut:
            return index
    return len(cuts)


def active_cuts(profile: dict[str, Any] | None = None) -> list[float]:
    """The installed band cuts, or an empty list when unmeasured."""

    data = profile if profile is not None else ACTIVE_LENGTH_FIDELITY_PROFILE
    if not (data or {}).get("available"):
        return []
    return [float(value) for value in (data or {}).get("cuts") or []]


def band_is_supported(band: int, profile: dict[str, Any] | None = None) -> bool:
    """Whether this band rests on enough reference comments to gate against."""

    data = profile if profile is not None else ACTIVE_LENGTH_FIDELITY_PROFILE
    counts = (data or {}).get("band_counts") or {}
    return float(counts.get(str(band), 0.0)) >= MIN_BAND_COMMENTS


def length_band_problem(
    text: str,
    task: Any,
    *,
    profile: dict[str, Any] | None = None,
) -> str:
    """Report a realized length that left its assigned band, or "".

    Withheld -- returning "" -- whenever the arm is off, the domain has no
    measured cuts, the slot carries no assigned count, or either the assigned or
    the realized band is not supported by enough reference comments.
    """

    if not LENGTH_FIDELITY_ENABLED:
        return ""
    cuts = active_cuts(profile)
    if not cuts:
        return ""
    try:
        assigned = int(getattr(task, "real_word_count", 0) or 0)
    except (TypeError, ValueError):
        return ""
    if assigned <= 0:
        return ""
    realized = len(str(text or "").split())
    if not realized:
        return ""
    target_band = band_of(assigned, cuts)
    actual_band = band_of(realized, cuts)
    if target_band == actual_band:
        return ""
    if not band_is_supported(target_band, profile):
        return ""
    if not band_is_supported(actual_band, profile):
        return ""
    low, high = band_bounds(target_band, cuts)
    return f"{PROBLEM_PREFIX}{realized}w in band {actual_band}, assigned {assigned}w in band {target_band} [{low}-{high}]"


def band_bounds(band: int, cuts: list[float] | tuple[float, ...]) -> tuple[int, int]:
    """Inclusive word-count bounds of a band, with 0 standing for unbounded."""

    low = int(cuts[band - 1]) + 1 if band > 0 else 1
    high = int(cuts[band]) if band < len(cuts) else 0
    return low, high


def retry_note(problem: str) -> str:
    """The revision instruction for a length-band miss.

    Worded as a length correction only. It names no content, so it cannot push
    the Writer toward a shared way of saying things -- the failure mode
    `docs/DECISIONS.md` G37 measured for v109's referent cue.
    """

    detail = problem.split(":", 1)[1].strip() if ":" in problem else problem
    return (
        "The length is wrong for this slot: "
        f"{detail}. Rewrite the same single local point at the assigned length. "
        "If it is too short, develop the point you already made with more of its "
        "own specifics rather than adding a new claim, a summary, or a closing "
        "line. If it is too long, cut, do not compress into a denser sentence."
    )
