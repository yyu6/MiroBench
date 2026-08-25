"""One-shot content-depth planning for anonymous long comment slots.

The matched thread contributes only a word-count signal. The Planner supplies
the domain-grounded semantic beats, and the Writer realizes them once. This
module never reads matched comment text and never triggers candidate sampling.
"""

from __future__ import annotations

import re
from typing import Any

from .planner_schema import parse_sample_id


BEAT_SEPARATOR = " || "
_PREFIX_RE = re.compile(r"^\s*(?:[-*]+|(?:beat\s*)?\d+[.):\-])\s*", re.I)


# Measured on generated output, twice. At one beat per 80 words long slots came
# out at 0.72x their matched length; at one per 35 they came out at 0.87x while
# the Planner supplied 92% of the requested beats, so the shortfall was the
# budget rather than the Planner. The realized rate across those slots was
# 246/12, 179/8, and 134/6 words per beat, i.e. about 21. Budget against that.
# Re-measured on v72: realized output is 24.8 words per beat over 77 long slots,
# so the per-beat budget below is right.
#
# Raising the ceiling to 40 did not raise the reachable length, because the
# Planner does not supply what it is asked for above about nine beats. Measured
# over the v96 slots that carried a beat plan: asked ~6 it supplied 5.2 and the
# slot realized 0.95x its matched length; asked ~9, 8.1 and 0.91x; asked ~12,
# 8.3 and 0.74x; asked 14-40, 9.5 and 0.60x. The largest beat plan any slot
# received in the whole run was 26. Beyond the saturation point an unreachable
# request only produces plan-repair traffic, so the ceiling now sits where the
# Planner still delivers and `comment_structure` carries scale above it by
# asking for the paragraph count a comment that long actually has.
WORDS_PER_REALIZED_BEAT = 21.0
MAX_DEVELOPMENT_BEATS = 12
# The largest count the Planner reliably returns, measured above.
PLANNER_RELIABLE_BEATS = 8

# The floor the enumerated plan is extended down to under `--development-scope
# measured`. It is 35 because that is where the compression starts and not one
# word lower: measured over four N=10 runs on the same ten seeds, realized/
# assigned runs 1.157 at 1-9 assigned words and 0.994 at 10-34 -- those bands
# need no more capacity and giving them beats would make them overshoot -- then
# falls to 0.856 at 35-60 and 0.840 at 61-100. See `docs/DECISIONS.md` G50.
DEVELOPMENT_FLOOR_WORDS = 35
# The only place the pre-v111 boundary is written down. Four gates read it --
# see `development_plan_word_threshold`.
LONG_FORM_ONLY_WORD_THRESHOLD = 100
# `long_only` reproduces every release through v110, which returned no beat
# budget at all below 101 assigned words.
DEVELOPMENT_SCOPE_MODE = "long_only"


def set_development_scope(mode: str) -> str:
    """Select how far down the enumerated beat budget reaches."""

    global DEVELOPMENT_SCOPE_MODE
    chosen = str(mode or "long_only").strip().lower()
    DEVELOPMENT_SCOPE_MODE = "measured" if chosen == "measured" else "long_only"
    return DEVELOPMENT_SCOPE_MODE


def expected_development_beats(word_count: Any) -> int:
    """Return a soft content-capacity target for an anonymous slot.

    CORRECTION TO `docs/DECISIONS.md` G50, measured 2026-08-25. G50 read a
    realized/assigned jump of 0.816 -> 0.953 across the boundary this function
    creates and called it the length instrument. Re-measured on the N=50 paper
    artifact (1,974 slots, `analysis/self_similarity/`), the jump is +0.031 to
    +0.051 depending on bandwidth, MWU 0.494 / 0.239 / 0.147 / 0.058, and a
    placebo cut at 80 assigned words -- where no rule changes at all -- gives
    +0.045 on the same data. **The discontinuity does not survive N=50.** The
    *asked word count* remains a non-instrument (G48, elasticity -0.02 to 0.11);
    that part of G48/G50 still holds.

    What does survive is a content difference either side of the boundary. Above
    100 the Planner supplies an enumerated beat list for 94.2% of slots and they
    realize 0.910 of assigned; at or below 100 it supplies one for 0.0% and they
    realize 0.887. `measured` is the hypothesis that the enumerated list is what
    matters -- E4's finding that naming the concrete thing gets ~1.0 compliance
    where naming the category gets 0.23. It is a hypothesis with a written
    prediction, not an established fix.

    Under `long_only` the budget is withheld below 101 words, which deletes the
    Planner's beat plan for those slots (`reconcile_development_plan_capacity`),
    tells the Planner in prose to return `none` for them, keeps the beat count
    off the slot schedule line, and leaves them the categorical cue in
    `length_policy`. Under `measured` all four reach down to
    `DEVELOPMENT_FLOOR_WORDS`.

    The two modes are identical outside 35-100 words: above 100,
    `round(w / 21) >= 5`, so the `max(3, ...)` and `max(2, ...)` floors never
    bind and the value is the same; below 35 both return 0.
    """

    words = _safe_int(word_count)
    if DEVELOPMENT_SCOPE_MODE == "measured":
        if words < DEVELOPMENT_FLOOR_WORDS:
            return 0
        return min(
            MAX_DEVELOPMENT_BEATS,
            max(2, int(round(words / WORDS_PER_REALIZED_BEAT))),
        )
    if words <= LONG_FORM_ONLY_WORD_THRESHOLD:
        return 0
    return min(
        MAX_DEVELOPMENT_BEATS,
        max(3, int(round(words / WORDS_PER_REALIZED_BEAT))),
    )


def development_plan_word_threshold() -> int:
    """Largest anonymous slot size that must return `none` for `development_plan`.

    Four gates make the same capacity decision, and they have to agree:

    1. `prompts._slot_schedule` prints `development_plan=N beats required`;
    2. the Planner's prose rule tells it to return `none` below a threshold;
    3. `reconcile_development_plan_capacity` deletes a plan below the threshold;
    4. `length_policy.length_hint` picks the Writer's cue.

    1, 3 and 4 call `expected_development_beats` directly. 2 is English prose in
    an f-string and has to be handed the number, so it reads it from here rather
    than repeating a literal. Before this function existed the prose carried a
    hard-coded 100: turning the arm on would have printed
    `development_plan=3 beats required` for a 70-word slot while the same prompt
    still said to return `none` for every slot at or below 100 words. That is
    two contradictory instructions in one prompt, and it is why the arm as first
    written would have been inert rather than wrong-in-a-visible-way.

    The invariant the tests hold is exact:
    `expected_development_beats(w) > 0` iff `w > development_plan_word_threshold()`.
    """

    if DEVELOPMENT_SCOPE_MODE == "measured":
        return DEVELOPMENT_FLOOR_WORDS - 1
    return LONG_FORM_ONLY_WORD_THRESHOLD


def normalize_development_plan(value: Any, *, max_beats: int = MAX_DEVELOPMENT_BEATS) -> str:
    """Normalize a Planner list/string without imposing domain semantics."""

    if isinstance(value, (list, tuple)):
        raw_parts = [str(item or "") for item in value]
    else:
        raw_parts = re.split(r"\s*\|\|\s*|[\r\n;]+", str(value or ""))
    result: list[str] = []
    seen: set[str] = set()
    for raw in raw_parts:
        beat = " ".join(_PREFIX_RE.sub("", raw).split()).strip(" -|;")
        if not beat or beat.casefold() in {"none", "n/a", "not needed"}:
            continue
        beat = beat[:220].rstrip()
        key = beat.casefold()
        if key in seen:
            continue
        seen.add(key)
        result.append(beat)
        if len(result) >= max(1, int(max_beats)):
            break
    return BEAT_SEPARATOR.join(result)


def development_beats(value: Any) -> list[str]:
    normalized = normalize_development_plan(value)
    return normalized.split(BEAT_SEPARATOR) if normalized else []


def enrich_development_plan_fields(
    payload: dict[str, Any],
    normalized: dict[int, dict[str, str]],
) -> dict[int, dict[str, str]]:
    """Carry the generalized field through the shared CARD JSON parser."""

    raw_rows = payload.get("comment_plans") or payload.get("plans") or []
    raw_by_sample: dict[int, dict[str, Any]] = {}
    for row in raw_rows if isinstance(raw_rows, list) else []:
        if not isinstance(row, dict):
            continue
        sample_id = parse_sample_id(row.get("sample_id"))
        if sample_id > 0:
            raw_by_sample[sample_id] = row
    for sample_id, plan in normalized.items():
        raw = raw_by_sample.get(sample_id, {})
        plan["development_plan"] = normalize_development_plan(
            raw.get("development_plan")
        )
    return normalized


def development_plan_problem(plan: dict[str, Any]) -> str:
    """Explain when a long slot lacks enough planned semantic capacity."""

    words = _safe_int(plan.get("_slot_word_count"))
    expected = expected_development_beats(words)
    if expected <= 0:
        return ""
    actual = len(development_beats(plan.get("development_plan")))
    # The Planner saturates near nine beats, so requiring one fewer than the
    # capacity estimate on the largest slots produced repair calls that could
    # not succeed. v94 spent 130 of 152 requests on plan repair.
    minimum = max(2, min(expected - 1, PLANNER_RELIABLE_BEATS))
    if actual >= minimum:
        return ""
    return (
        f"the anonymous slot has {words} words but development_plan contains "
        f"{actual} distinct beat(s); supply about {expected} connected beats "
        "that develop the same local contribution without restating it or adding unrelated claims"
    )


def reconcile_development_plan_capacity(
    plan: dict[str, Any],
) -> dict[str, Any] | None:
    """Drop copied schema prose from a slot with no long-form capacity.

    A development plan is meaningful only when the anonymous slot can carry
    multiple beats. Otherwise any value is prompt residue, not semantic
    content, and the Writer would be told to realize it as part of the comment.
    """

    words = _safe_int(plan.get("_slot_word_count"))
    if expected_development_beats(words) > 0:
        return None
    value = normalize_development_plan(plan.get("development_plan"))
    if not value:
        return None
    plan["development_plan"] = ""
    return {
        "field": "development_plan",
        "words": words,
        "before": value,
        "after": "",
        "reason": "slot_has_no_long_form_capacity",
    }


def render_development_guidance(task: Any) -> str:
    """Render the complete one-shot plan for the Writer."""

    words = _safe_int(getattr(task, "real_word_count", 0))
    beats = development_beats(getattr(task, "development_plan", ""))
    expected = expected_development_beats(words)
    if not beats:
        if expected <= 0:
            return ""
        return (
            f"Develop the local contribution through about {expected} distinct, "
            "connected beats rather than repeating one thesis or compressing this "
            "long-tail slot into a generic answer."
        )
    rows = " ".join(f"{index}. {beat}" for index, beat in enumerate(beats, start=1))
    return (
        "One-shot development sequence: "
        f"{rows} Realize each beat once in a natural order. Combine adjacent beats "
        "when useful, but do not omit the long-tail development, restate an earlier "
        "beat, or introduce a separate conclusion."
    )


def _safe_int(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0
