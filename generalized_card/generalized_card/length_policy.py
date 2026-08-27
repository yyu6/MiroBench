"""Soft, domain-neutral length conditioning for generalized generation."""

from __future__ import annotations

import re
from typing import Any, Callable

from .comment_structure import active_layout_guidance
from .length_calibration import ask_multiplier, calibrated_word_ask
from .long_form_planning import expected_development_beats, render_development_guidance


SOFT_LENGTH_PROBLEMS = frozenset(
    {
        "low_info_too_long",
        "length_too_long",
        "real_slot_too_short",
    }
)
SOFT_LENGTH_PROBLEM_PREFIXES = (
    "substantive_length_floor:",
    # `length_fidelity.PROBLEM_PREFIX`. Soft on purpose: it must be able to
    # trigger a Writer retry without ever making a matched structural slot
    # blocking, which `docs/ORIENTATION.md` §4 forbids.
    "length_band_mismatch:",
)


def is_soft_length_problem(problem: str) -> bool:
    """Return whether a diagnostic describes length rather than validity."""

    return problem in SOFT_LENGTH_PROBLEMS or problem.startswith(
        SOFT_LENGTH_PROBLEM_PREFIXES
    )


# G113: sentence architecture is the one measurably narrow place left in the
# realization layer. Within the `long_turn` bucket our coefficient of variation
# on mean sentence length is **0.37x** real's, against 1.10 on word count --
# we are not uniformly narrow, we are narrow here. The cause is not
# non-compliance: the Writer honours the stated sentence count as well as it
# honours the word count (median relative error +0.00 against -0.07, both 63-64%
# within +-25%). It is that the RATIO is never named. The matched real comments
# handed to our slots carry words-per-sentence at CV **0.53**; our realizations
# come out at **0.39**, so the Writer hits both marginals while pulling their
# ratio toward its own preferred ~17 words per sentence.
#
# The targets are therefore already correct and need no new measurement -- they
# are the matched real comment's own values, which is what makes this
# domain-adaptive with no constant in it. What was missing is E4's concrete
# number: `pacing` is a category, and a category buys 0.23 compliance where a
# named number buys ~1.0.
SENTENCE_PACING_MODE = "off"

_SKELETON_SENTENCES_RE = re.compile(
    r"(?:about\s+)?(\d+)[- ]sentence|about\s+(\d+)\s+sentences", re.I
)


def set_sentence_pacing(mode: str) -> None:
    global SENTENCE_PACING_MODE
    value = str(mode or "off").strip().lower()
    if value not in {"off", "measured"}:
        raise ValueError(
            f"unknown sentence-pacing mode {mode!r}; expected off|measured"
        )
    SENTENCE_PACING_MODE = value


def sentence_pacing_enabled() -> bool:
    return SENTENCE_PACING_MODE == "measured"


def skeleton_sentence_count(skeleton: Any) -> int:
    """Read the matched real comment's own sentence count off its skeleton."""

    matched = _SKELETON_SENTENCES_RE.search(str(skeleton or ""))
    if not matched:
        return 0
    for group in matched.groups():
        if group:
            return _safe_int(group, 0)
    return 0


def sentence_pacing_cue(task: Any, *, asked_words: int) -> str:
    """State the slot's own words-per-sentence instead of the word `pacing`.

    The sentence count is rescaled onto the calibrated word ask so the two
    numbers cannot contradict each other; the matched comment's RATIO is what
    is preserved, because that ratio is the quantity real threads vary in and
    ours does not.
    """

    if not sentence_pacing_enabled():
        return ""
    real_words = _safe_int(getattr(task, "real_word_count", 0), 0)
    sentences = skeleton_sentence_count(getattr(task, "surface_skeleton", ""))
    if real_words <= 0 or sentences <= 0 or asked_words <= 0:
        return ""
    ratio = real_words / sentences
    if ratio <= 0:
        return ""
    scaled = max(1, round(asked_words / ratio))
    per = max(1, round(ratio))
    if scaled == 1:
        return f"Write it as a single sentence of about {per} words."
    return (
        f"Spread it over about {scaled} sentences, averaging about {per} words "
        "each. That average is this slot's own, not a house style: some slots "
        "here run in short clipped sentences and others in long ones, so do not "
        "drift toward a comfortable middle length."
    )


def soft_length_guidance(task: Any) -> str:
    """Expose an anonymous continuous length cue without a fixed bucket gate."""

    real_words = _safe_int(getattr(task, "real_word_count", 0), 0)
    if real_words > 0:
        development = local_move_scope_guidance(task)
        # A purely permissive cue ("not an exact count") let every long slot
        # regress toward the mean, which compressed the thread's length spread.
        # State the target and the direction of the common error instead, while
        # keeping it a cue rather than an acceptance gate.
        #
        # Which direction that is comes from the measured transfer function
        # rather than a written-down threshold. The old cutoff was 100 words,
        # but realized/target crosses 1.0 near 35, so every slot between 35 and
        # 100 was being told "do not pad" while its measured error was
        # undershoot: on the v98 seed-2 gate the 56-80 word slots realized 0.48
        # to 0.74 of their target. `ask_multiplier` is the same curve the
        # calibration inverts, so the cue and the ask can no longer disagree.
        scale = (
            "Comments this long are normal here, so do not trim toward a "
            "medium-length answer: land near this scale."
            if ask_multiplier(real_words) >= 1.0
            else "Do not pad past it."
        )
        layout = active_layout_guidance(real_words)
        # The number in the cue is calibrated; every other consumer of
        # `real_word_count` keeps the matched slot's true size, because the
        # layout, the beat count, and the length floor describe what a comment
        # of that real size looks like. See `length_calibration`.
        asked = calibrated_word_ask(real_words) or real_words
        unit = "word" if asked == 1 else "words"
        return " ".join(
            part
            for part in (
                f"The anonymous matched structural slot contains roughly "
                f"{asked} {unit}. Write a turn of that scale, matching its "
                f"information density and pacing. {scale} This is a target, not "
                "a counted requirement; being far short of it is the common "
                "failure.",
                sentence_pacing_cue(task, asked_words=asked),
                layout,
                development,
            )
            if part
        )
    return (
        "Use a natural length for this local turn. Length is a soft surface "
        "choice, not a word-count requirement."
    )


def local_move_scope_guidance(task: Any) -> str:
    """Translate a continuous real-slot length into local development depth.

    The two categorical branches below are what `docs/DECISIONS.md` G50 measured
    as the compression: a slot assigned 90 words was told in one breath that it
    "contains roughly 118 words" and that it should "give it the two or three
    connected beats this slot's scale supports" -- two or three beats being
    about 42-63 realized words. The categorical cue won, which is E4's rule
    (naming the concrete thing gets ~1.0 compliance, naming the category 0.23).

    Both branches are now gated on the slot having no beat budget at all, so
    `--development-scope measured` routes 35-100 word slots to the same
    enumerated, per-slot development sequence that already reaches slots above
    100 and realizes there at 0.956 of assignment. With the arm at `long_only`
    `expected_development_beats` returns 0 for every slot at or below 100 words,
    so both conditions hold exactly where they did before and this function is
    byte-identical to v110.
    """

    real_words = max(0, _safe_int(getattr(task, "real_word_count", 0), 0))
    has_budget = expected_development_beats(real_words) > 0
    if real_words <= 60 and not has_budget:
        return "Make one narrow local move and stop when that contribution is complete."
    if real_words <= 100 and not has_budget:
        return (
            "Keep one local thesis, but give it the two or three connected beats "
            "this slot's scale supports rather than stopping after the first."
        )
    planned = render_development_guidance(task)
    if planned:
        return planned
    beats = max(2, expected_development_beats(real_words))
    return (
        f"Keep one local thesis, but develop it through about {beats} connected "
        "beats drawn from context, observation, reason, consequence, caveat, or "
        "reaction as the planned slot supports. Do not compress this long-tail "
        "slot into one generic advice sentence or add unrelated claims."
    )


def writer_safety_token_cap(
    original: Callable[..., int],
    bucket: str,
    *,
    payload_type: str = "",
    profile: str = "",
    max_writer_tokens: int,
) -> int:
    """Use one provider safety ceiling instead of bucket-specific output caps."""

    if profile in {"osim8b_minimal_context", "osim8b_qwen_style"}:
        return original(
            bucket,
            payload_type=payload_type,
            profile=profile,
            max_writer_tokens=max_writer_tokens,
        )
    if max_writer_tokens > 0:
        return max(16, int(max_writer_tokens))
    return 260


def writer_provider_token_budget(
    task: Any,
    *,
    configured_max: Any,
    # The calibrated ask for the largest slot the domain has (845 words) is
    # 1,238 words, about 1,650 tokens. A ceiling that cuts the ask off mid
    # sentence would turn the calibration into a truncation.
    hard_ceiling: int = 2400,
) -> int:
    """Return a provider ceiling that cannot truncate an anonymous long slot.

    This value is not a requested output length and is never used as an
    acceptance gate. It only prevents the API ceiling from contradicting the
    continuous matched-slot length cue.
    """

    configured = max(16, _safe_int(configured_max, 260))
    real_words = max(0, _safe_int(getattr(task, "real_word_count", 0), 0))
    if real_words <= 0:
        return configured
    # The ceiling has to clear the number the cue actually asks for, which is
    # larger than the matched slot on a long slot; otherwise the calibration
    # would be silently truncated by the provider instead of realized.
    estimated_tokens = int(round(max(real_words, calibrated_word_ask(real_words)) * 1.7)) + 64
    return min(max(configured, estimated_tokens), max(configured, hard_ceiling))


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default
