"""Soft, domain-neutral length conditioning for generalized generation."""

from __future__ import annotations

from typing import Any, Callable

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
        scale = (
            "Comments this long are normal here, so do not trim toward a "
            "medium-length answer: land near this scale."
            if real_words > 100
            else "Do not pad past it."
        )
        return (
            f"The anonymous matched structural slot contains roughly {real_words} "
            "words. Write a turn of that scale, matching its information density "
            f"and pacing. {scale} This is a target, not a counted requirement; "
            f"being far short of it is the common failure. {development}"
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
    hard_ceiling: int = 1600,
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
    estimated_tokens = int(round(real_words * 1.7)) + 64
    return min(max(configured, estimated_tokens), max(configured, hard_ceiling))


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default
