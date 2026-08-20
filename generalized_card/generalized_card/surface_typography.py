"""Keyboard-realistic punctuation, calibrated per domain from excluded threads.

A language model emits typographic punctuation: curly apostrophes and quotes,
em dashes, and a single ellipsis character. People typing into a Reddit box
mostly emit the ASCII characters their keyboard produces, and the minority who
do not are the ones whose device substitutes them, so the choice is a stable
property of a *typist* rather than of a sentence.

This matters to two evaluation metrics directly, not only to realism. The
`self_bleu_4` tokenizer (`score_thread_self_bleu.TOKEN_PATTERN`) reads ``it's``
as one token and ``it’s`` as three, so every generated contraction contributed a
shared ``<word> ’ s`` trigram that no real comment produced. Measured over the
ten matched v96 threads, mapping the typographic characters onto their keyboard
equivalents moved `self_bleu_4` from 0.0373 to 0.0324 against a real 0.0280, its
MWU p from 0.0091 to 0.273 and its KS p from 0.052 to 0.787, and lowered
`self_bertscore_mean_f1` by about 0.008 on four separately scored threads.

Shares are measured, never written down: `build_typography_profile` counts each
typographic character and its keyboard equivalent over the domain's
evaluation-excluded threads, and one deterministic draw per speaker per class
reproduces the measured ratio. On the camera domain's 11,817 excluded comments
the typographic share is 0.271 of apostrophe-bearing comments, 0.225 of
quote-bearing comments, 0.105 of dash-bearing comments, and 0.156 of
ellipsis-bearing comments; v96 generated output was at 1.000 for all four.

What this module does not fix is how often the Writer reaches for a
construction at all. After shaping, em dashes remain in 3.0% of generated
comments against a matched-real 0.2%, because the Writer emits a dash-joined
aside four times as often as a person does. That is a register defect and is
handled where the register is chosen, not here.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any, Iterable

from .comment_structure import structure_bucket


# Each class pairs the typographic characters a model emits with the keyboard
# text a person types, plus the regex that finds the keyboard form when the
# share is being measured. The classes are writing-system properties, not
# domain vocabulary, so the same table serves any domain.
TYPOGRAPHY_CLASSES: tuple[dict[str, Any], ...] = (
    {
        "name": "apostrophe",
        "typographic": ("’", "‘"),
        "keyboard": "'",
        "keyboard_pattern": r"'",
    },
    {
        "name": "double_quote",
        "typographic": ("“", "”"),
        "keyboard": '"',
        "keyboard_pattern": r'"',
    },
    {
        "name": "dash",
        "typographic": ("—", "–"),
        "keyboard": " - ",
        "keyboard_pattern": r"\s-\s|--",
    },
    {
        "name": "ellipsis",
        "typographic": ("…",),
        "keyboard": "...",
        "keyboard_pattern": r"\.\.\.",
    },
)

_CLASS_BY_NAME = {row["name"]: row for row in TYPOGRAPHY_CLASSES}
# `off` reproduces the pre-v97 arm, which emitted the model's own typographic
# punctuation on every comment.
REDDIT_TYPOGRAPHY_ENABLED = True
_COLLAPSE_SPACES = re.compile(r"[ \t]{2,}")
_MIN_SAMPLES = 200
_MIN_BAND_SAMPLES = 60


def set_reddit_typography(mode: str) -> bool:
    """Select the keyboard-typography arm and return whether it is active."""

    global REDDIT_TYPOGRAPHY_ENABLED
    REDDIT_TYPOGRAPHY_ENABLED = str(mode or "on").strip().lower() != "off"
    return REDDIT_TYPOGRAPHY_ENABLED


def build_typography_profile(threads: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Measure this domain's typographic share for each punctuation class.

    The share is a document frequency ratio, not an occurrence ratio: the draw
    it feeds is per speaker, so the quantity to reproduce is how many *comments*
    show the typographic form, not how many characters. The two agree for
    apostrophes, where both sides use contractions at the same rate, and differ
    for dashes, where the model reaches for the construction far more often than
    a person does and an occurrence ratio would carry that overuse through.
    """

    typographic: dict[str, int] = {row["name"]: 0 for row in TYPOGRAPHY_CLASSES}
    keyboard: dict[str, int] = {row["name"]: 0 for row in TYPOGRAPHY_CLASSES}
    comment_count = 0
    for thread in threads:
        for row in thread.get("comments") or []:
            body = str(row.get("body") or row.get("content") or "")
            if not body.strip():
                continue
            comment_count += 1
            for spec in TYPOGRAPHY_CLASSES:
                name = spec["name"]
                if any(char in body for char in spec["typographic"]):
                    typographic[name] += 1
                if re.search(spec["keyboard_pattern"], body):
                    keyboard[name] += 1
    if comment_count < _MIN_SAMPLES:
        return {"available": False, "sample_count": comment_count, "shares": {}}
    shares = {}
    for spec in TYPOGRAPHY_CLASSES:
        name = spec["name"]
        total = typographic[name] + keyboard[name]
        shares[name] = round(typographic[name] / total, 6) if total else 0.0
    return {
        "available": True,
        "method": (
            "share of comments using the typographic rather than the keyboard form "
            "of each punctuation class, over same-domain threads excluded from "
            "the evaluation seed pool"
        ),
        "sample_count": comment_count,
        "shares": shares,
        "comment_counts": {
            name: {"typographic": typographic[name], "keyboard": keyboard[name]}
            for name in typographic
        },
    }


# A declarative comment either ends in a period or ends in nothing, and which
# one is a typing habit that depends on how long the comment is. Measured over
# the camera domain's evaluation-excluded comments, the share ending in no
# punctuation at all runs 0.402 / 0.282 / 0.187 / 0.131 / 0.108 / 0.115 across
# the size bands against 0.307 / 0.534 / 0.686 / 0.753 / 0.785 / 0.806 ending in
# a period. v96 output was at 0.043 overall against a real 0.173, and the v97
# seed-2 thread at 0.044 against 0.244. The ratio between the two endings is what
# is reproduced, so a comment that chose a question mark or an exclamation is
# left alone.
_DECLARATIVE_END = "."
_SENTENCE_END = ".!?"


def build_final_punctuation_profile(
    threads: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    """Measure, per size band, how often a declarative ending is left bare."""

    bare: dict[str, int] = {}
    period: dict[str, int] = {}
    total = 0
    for thread in threads:
        for row in thread.get("comments") or []:
            body = str(row.get("body") or row.get("content") or "").strip()
            if not body:
                continue
            total += 1
            band = structure_bucket(len(body.split()))
            if body[-1] in _SENTENCE_END or not body[-1].isalnum():
                if body.endswith(_DECLARATIVE_END):
                    period[band] = period.get(band, 0) + 1
                continue
            bare[band] = bare.get(band, 0) + 1
    if total < _MIN_SAMPLES:
        return {"available": False, "sample_count": total, "bare_share_by_band": {}}
    shares = {}
    for band in set(bare) | set(period):
        denominator = bare.get(band, 0) + period.get(band, 0)
        if denominator >= _MIN_BAND_SAMPLES:
            shares[band] = round(bare.get(band, 0) / denominator, 6)
    return {
        "available": bool(shares),
        "method": (
            "share of declarative endings left with no final punctuation rather "
            "than a period, per comment size band, over same-domain threads "
            "excluded from the evaluation seed pool"
        ),
        "sample_count": total,
        "bare_share_by_band": dict(sorted(shares.items())),
        "band_counts": {
            band: {"bare": bare.get(band, 0), "period": period.get(band, 0)}
            for band in sorted(set(bare) | set(period))
        },
    }


def apply_final_punctuation_habit(
    text: str,
    *,
    speaker_key: str,
    profile: dict[str, Any] | None,
) -> str:
    """Leave a declarative ending bare at the band's measured rate.

    Only a trailing period is dropped. A question mark or an exclamation is a
    choice the Writer made about the turn, not a typing habit.
    """

    original = str(text or "")
    stripped = original.rstrip()
    if not REDDIT_TYPOGRAPHY_ENABLED or not stripped.endswith(_DECLARATIVE_END):
        return original
    if stripped.endswith("..") or not (profile or {}).get("available"):
        return original
    band = structure_bucket(len(stripped.split()))
    share = float(
        ((profile or {}).get("bare_share_by_band") or {}).get(band, 0.0)
    )
    if share <= 0.0:
        return original
    digest = hashlib.sha256(f"final_punct:{speaker_key}".encode("utf-8")).digest()
    draw = int.from_bytes(digest[8:16], "big", signed=False) / float(1 << 64)
    return stripped[:-1].rstrip() if draw < share else original


def speaker_uses_typographic(
    profile: dict[str, Any] | None,
    *,
    speaker_key: str,
    class_name: str,
) -> bool:
    """Return one stable per-speaker draw at the measured typographic share.

    Keying on the speaker rather than the comment is what the underlying
    behaviour is: a device either substitutes the character or it does not. It
    also raises between-author surface variance inside one thread, which is the
    direction both self-similarity metrics need.
    """

    share = float(((profile or {}).get("shares") or {}).get(class_name, 0.0))
    if share <= 0.0:
        return False
    if share >= 1.0:
        return True
    digest = hashlib.sha256(f"{class_name}:{speaker_key}".encode("utf-8")).digest()
    draw = int.from_bytes(digest[:8], "big", signed=False) / float(1 << 64)
    return draw < share


def apply_keyboard_typography(
    text: str,
    *,
    speaker_key: str,
    profile: dict[str, Any] | None,
) -> str:
    """Rewrite typographic punctuation this speaker's keyboard would not emit."""

    original = str(text or "")
    if not original or not REDDIT_TYPOGRAPHY_ENABLED:
        return original
    if not (profile or {}).get("available"):
        return original
    shaped = original
    for spec in TYPOGRAPHY_CLASSES:
        if speaker_uses_typographic(
            profile, speaker_key=speaker_key, class_name=spec["name"]
        ):
            continue
        for char in spec["typographic"]:
            shaped = shaped.replace(char, spec["keyboard"])
    # Leave untouched text byte-identical so an arm that changes nothing here
    # cannot be confused with one that does.
    return _tidy_spacing(shaped) if shaped != original else original


def _tidy_spacing(text: str) -> str:
    """Repair only the spacing the substitutions themselves can create."""

    lines = [
        _COLLAPSE_SPACES.sub(" ", line).rstrip() for line in text.split("\n")
    ]
    return "\n".join(lines).strip()


def typography_class_names() -> tuple[str, ...]:
    """Expose the measured class names for audits and tests."""

    return tuple(_CLASS_BY_NAME)
