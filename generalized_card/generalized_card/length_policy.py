"""Soft, domain-neutral length conditioning for generalized generation."""

from __future__ import annotations

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
SOFT_LENGTH_PROBLEM_PREFIXES = ("substantive_length_floor:",)


def is_soft_length_problem(problem: str) -> bool:
    """Return whether a diagnostic describes length rather than validity."""

    return problem in SOFT_LENGTH_PROBLEMS or problem.startswith(
        SOFT_LENGTH_PROBLEM_PREFIXES
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
    """Translate a continuous real-slot length into local development depth."""

    real_words = max(0, _safe_int(getattr(task, "real_word_count", 0), 0))
    if real_words <= 60:
        return "Make one narrow local move and stop when that contribution is complete."
    if real_words <= 100:
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
    if real_words <= 100:
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
