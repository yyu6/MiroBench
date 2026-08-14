"""Grammatical opener types, calibrated per domain from excluded threads.

Measured on one matched pair, 23.4% of generated comments opened with a polarity
token ("Yeah", "Yep") against 3.6% in the real thread, while the two largest real
categories, a content-bearing phrase and a first-person clause, were both
under-produced. The Planner was not asking for this: its 197 `opening_style`
values were 196 distinct free-text routes such as "lead with the mismatch
between", and the Writer collapsed them onto its own default opener.

Free text does not control this Writer. Over v64 to v67 an assigned categorical
field was realized about 86% of the time while prompt rules and optional
affordances drew essentially nothing, so the opener is scheduled as a categorical
control alongside tone, affect, and story, and the schedule's shares are measured
from the domain's evaluation-excluded threads.

The categories are grammatical rather than topical, so the same classifier serves
any domain; only the shares are domain-specific, and they are learned rather than
written down.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any, Iterable

OPENER_TYPES = (
    "content_phrase",
    "first_person",
    "noun_phrase",
    "discourse_marker",
    "polarity_token",
    "question",
    "quote",
    "conditional",
    "address",
    "imperative",
    "link",
)

OPENER_INSTRUCTIONS = {
    "content_phrase": "Open on the substance itself, with no preface, acknowledgement, or stance token before it.",
    "first_person": "Open with your own experience or position, in the first person.",
    "noun_phrase": "Open with the thing being discussed as the grammatical subject.",
    "discourse_marker": "Open with a short conversational connective before the point.",
    "polarity_token": "Open with a bare agreement or disagreement token, then the point.",
    "question": "Open with the question itself.",
    "quote": "Open by quoting the exact visible line you are answering, then respond to it.",
    "conditional": "Open with the condition or circumstance, then what follows from it.",
    "address": "Open by addressing the person you are replying to.",
    "imperative": "Open with the action you are recommending.",
    "link": "Open with the reference itself.",
}

_WORDS = re.compile(r"[a-z']+")
_POLARITY = frozenset(
    {"yeah", "yep", "yes", "no", "nope", "nah", "true", "exactly", "agreed", "same"}
)
_QUESTION = frozenset(
    {
        "what", "why", "how", "which", "who", "when", "where", "does", "do",
        "did", "is", "are", "can", "could", "would", "should", "any", "anyone",
    }
)
_CONDITIONAL = frozenset(
    {"if", "when", "unless", "once", "since", "because", "while", "after", "before"}
)
_NOUN = frozenset({"the", "a", "an", "this", "that", "these", "those", "it"})
_ADDRESS = frozenset({"you", "your"})
_IMPERATIVE = frozenset(
    {"get", "try", "check", "use", "go", "buy", "look", "keep", "make", "take", "dont", "just"}
)
_MARKER = frozenset(
    {
        "thanks", "thank", "appreciate", "congrats", "lol", "haha", "lmao", "oof",
        "ah", "ahh", "oh", "well", "honestly", "personally", "also", "and", "but", "so",
    }
)
_FIRST_PERSON = frozenset({"i", "ive", "im"})
_MIN_SAMPLES = 200


def classify_opener(text: str) -> str:
    """Return the grammatical entry type of one comment."""

    stripped = str(text or "").strip()
    lowered = stripped.lower()
    words = _WORDS.findall(lowered)
    if not words:
        return "content_phrase"
    if stripped.startswith(">") or "&gt;" in stripped[:40]:
        return "quote"
    if re.match(r"^(https?://|www\.)", lowered):
        return "link"
    first = words[0]
    if first in _FIRST_PERSON or lowered.startswith("i'"):
        return "first_person"
    if first in _POLARITY:
        return "polarity_token"
    if first in _QUESTION and "?" in stripped[:120]:
        return "question"
    if first in _CONDITIONAL:
        return "conditional"
    if first in _NOUN:
        return "noun_phrase"
    if first in _ADDRESS:
        return "address"
    if first in _IMPERATIVE:
        return "imperative"
    if first in _MARKER:
        return "discourse_marker"
    return "content_phrase"


def build_opener_profile(threads: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Measure this domain's opener-type shares on evaluation-excluded threads."""

    counts: Counter[str] = Counter()
    for thread in threads:
        for row in thread.get("comments") or []:
            body = str(row.get("body") or row.get("content") or "").strip()
            if body:
                counts[classify_opener(body)] += 1
    total = sum(counts.values())
    if total < _MIN_SAMPLES:
        return {"available": False, "sample_count": total, "shares": {}}
    return {
        "available": True,
        "method": (
            "grammatical opener classification over same-domain threads excluded "
            "from the evaluation seed pool"
        ),
        "sample_count": total,
        "shares": {
            name: round(counts.get(name, 0) / total, 6) for name in OPENER_TYPES
        },
    }


def scaled_opener_counts(profile: dict[str, Any] | None, total: int) -> Counter[str]:
    """Scale the measured shares onto one thread's slots."""

    shares = (profile or {}).get("shares") or {}
    if not shares or total <= 0:
        return Counter()
    exact = {name: float(shares.get(name, 0.0)) * total for name in OPENER_TYPES}
    result = Counter({name: int(value) for name, value in exact.items() if value >= 1})
    remaining = total - sum(result.values())
    for name in sorted(exact, key=lambda key: -(exact[key] - result.get(key, 0)))[
        : max(0, remaining)
    ]:
        result[name] = result.get(name, 0) + 1
    return Counter({name: count for name, count in result.items() if count > 0})


def opener_cost(name: str, slot: dict[str, Any]) -> tuple[int, int, int]:
    """Rank slots by how naturally each can carry an opener type."""

    words = int(slot.get("words") or 0)
    depth = int(slot.get("depth") or 0)
    surface = str(slot.get("surface") or "")
    micro = surface in {"micro", "short_question"} or words <= 5
    if name == "quote":
        # Quoting needs a parent to quote and room for the quote plus a reply.
        return (1 if depth == 0 or words < 20 else 0, 0, -words)
    if name == "question":
        return (0 if surface == "short_question" or words <= 40 else 1, 0, words)
    if name == "link":
        return (1 if words > 25 else 0, 0, words)
    if name in {"polarity_token", "discourse_marker"}:
        return (0, 0 if depth > 0 else 1, words)
    if name in {"first_person", "content_phrase", "conditional"}:
        return (1 if micro else 0, 0 if words >= 15 else 1, -words)
    if name == "address":
        return (1 if micro else 0, 0 if depth > 0 else 1, words)
    return (0, 0, abs(words - 30))
