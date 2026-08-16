"""Domain-neutral surface contracts for anonymous matched comment slots.

The matched comment may expose structural features such as length, punctuation,
or the presence of a link.  A marker is a hard surface type only when it
dominates the comment.  Incidental markers in a substantive comment must not
collapse the Planner's semantic move into a short reaction.
"""

from __future__ import annotations

import re
from dataclasses import replace
from typing import Any


URL_RE = re.compile(r"https?://\S+|www\.\S+", re.I)
QUOTE_RE = re.compile(r"(^|\n)\s*>|&gt;|\[[^\]]+\]\(", re.I)

HARD_SHORT_SURFACE_SHAPES = frozenset(
    {
        "deleted_removed",
        "template_notice",
        "link_reference",
        "quote_link_reference",
        "micro_reaction",
        "short_direct_answer",
        "short_question",
        "thanks_ack",
        "joke_reaction",
    }
)


def infer_surface_shape(row: dict[str, Any]) -> str:
    """Infer a shape without treating incidental style markers as semantics."""

    body = str(row.get("body") or "").strip()
    lowered = body.lower()
    words = len(body.split())
    author = str(row.get("author") or "").strip().lower()
    if lowered in {"[deleted]", "[removed]"}:
        return "deleted_removed"
    if author in {"automoderator", "moderator"} or "!template" in lowered:
        return "template_notice"
    if words <= 5:
        return "micro_reaction"
    if "?" in body and words <= 20:
        return "short_question"
    if words <= 10:
        return "short_direct_answer"

    has_link = bool(URL_RE.search(body))
    has_quote = bool(QUOTE_RE.search(body))
    if (has_link or has_quote) and reference_is_dominant(body):
        return "quote_link_reference" if has_quote else "link_reference"
    if re.search(r"\b[A-Z]{2,6}\b|\b[A-Za-z]+\d[A-Za-z0-9/-]*\b", body) and words <= 35:
        return "compact_datapoint"
    if words >= 70:
        return "story_rant"
    if re.search(r"^\s*(side note|unrelated|fwiw|btw)\b", lowered):
        return "side_tangent"
    return "full_answer"


def surface_only_label(text: str) -> str:
    """Render only an anonymous density label for the Comment Planner."""

    body = str(text or "").strip()
    words = len(body.split())
    if words <= 5:
        return "micro"
    if "?" in body and words <= 22:
        return "short_question"
    if words >= 70:
        return "long_turn"
    if (URL_RE.search(body) or QUOTE_RE.search(body)) and reference_is_dominant(body):
        return "reference"
    if words <= 18:
        return "short_turn"
    return "ordinary_turn"


def infer_surface_skeleton(text: str) -> tuple[str, str]:
    """Infer sentence shape while preserving substantive information density."""

    body = str(text or "").strip()
    compacted = re.sub(r"\s+", " ", body)
    lowered = compacted.lower()
    word_count = len(compacted.split())
    sentence_count = max(
        1,
        len(re.findall(r"[.!?]+(?:\s|$)", compacted))
        or len(re.split(r"\s{2,}", body)),
    )
    has_question = "?" in compacted
    has_link_or_quote = bool(URL_RE.search(body) or QUOTE_RE.search(body))
    if lowered in {"[deleted]", "[removed]"}:
        return "deleted placeholder", "Use a deleted or removed placeholder shape."
    if word_count <= 5:
        return "tiny fragment reaction", "Use one tiny fragment with no explanation."
    if has_link_or_quote and reference_is_dominant(body):
        return (
            "reference-led short turn",
            "Keep the reference as the main surface move without inventing its text or URL.",
        )
    if has_question and word_count <= 18:
        return "single narrow question", "Ask one narrow question only."
    if word_count >= 80:
        marker_note = " An incidental reference may appear within it." if has_link_or_quote else ""
        # The move count used to be capped at 6, so an 80-word comment and an
        # 845-word one were described identically. Measured on v72, no generated
        # comment exceeded 215 words while matched real slots reached 845, and
        # the longest slot was handed "6 moves" and produced 197 words -- about
        # six moves' worth. The cap, not the model, was the ceiling. The real
        # comment's own sentence count is the shape; there is nothing to cap.
        # `sentence_count` counts sentences, so the label says sentences. Called
        # "moves" it read as a count of distinct discourse actions, which is a
        # different and much smaller thing: real 845-word comments here run 48
        # sentences at 16-19 words each, not 48 separate arguments.
        return (
            f"long uneven Reddit paragraph, about {sentence_count} sentences",
            "Use a substantive uneven paragraph with the assigned local details and natural pacing."
            + marker_note,
        )
    if has_question:
        return (
            "question embedded in a local reply",
            "Keep the question inside the assigned local contribution rather than reducing the whole comment to a question.",
        )
    if "..." in compacted or "\u2026" in compacted:
        return (
            "elliptical local reply",
            "Use incomplete pacing around the assigned local move without collapsing its information density.",
        )
    if "(" in compacted and ")" in compacted:
        return (
            "local reply with parenthetical aside",
            "Use a local reply with one parenthetical aside and preserve the slot's information density.",
        )
    if word_count <= 10:
        return "short direct answer", "Use a short direct answer with no extra setup."
    # Same reasoning as the long branch: report the shape the real slot has.
    return (
        f"{sentence_count}-sentence local comment",
        "Use the same rough sentence count and pacing for the assigned local move.",
    )


def infer_surface_texture(
    real_text: str,
    *,
    payload_type: str,
    speaker_role: str,
    utterance_mode: str,
) -> str:
    """Infer typography only; social meaning remains Planner-owned.

    The shared CARD classifier treated words such as ``thanks`` and
    ``appreciate`` in the matched evaluation comment as a gratitude assignment.
    That is tone leakage, not an anonymous surface feature. Text inspection here
    is limited to links, quotes, emoji/sarcasm notation, capitalization,
    terminal punctuation, and messy punctuation. Gratitude, jokes, and tangents
    come only from the already-planned payload or speaker role.
    """

    text = str(real_text or "")
    lowered = text.lower()
    tokens = re.findall(r"[A-Za-z0-9/']+", text)
    if URL_RE.search(text):
        return "link_reference"
    if QUOTE_RE.search(text):
        return "markdown_quote"
    if re.search(r"[\U0001F300-\U0001FAFF]", text) or "/s" in lowered:
        return "emoji_or_sarcasm"
    if speaker_role == "gratitude_reply":
        return "gratitude_social"
    if re.search(r"\b[A-Z]{2,}\b", text) and len(tokens) <= 30:
        return "abbrev_shorthand"
    if utterance_mode == "fragment_only" or (
        len(tokens) <= 8 and not re.search(r"[.!?]\s*$", text.strip())
    ):
        return "no_punct_fragment"
    if (
        "..." in text
        or "…" in text
        or "!" in text
        or payload_type in {"joke", "side_tangent"}
        or speaker_role in {"jokester", "side_observer"}
    ):
        return "messy_punctuation"
    return "plain"


def reference_is_dominant(text: str) -> bool:
    """Return whether a reference, rather than surrounding prose, is the turn."""

    body = str(text or "").strip()
    if not (URL_RE.search(body) or QUOTE_RE.search(body)):
        return False
    residual = URL_RE.sub(" ", body)
    residual = re.sub(r"\[[^\]]+\]\([^)]*\)", " ", residual)
    residual = re.sub(r"(^|\n)\s*>[^\n]*", " ", residual)
    residual_words = re.findall(r"[A-Za-z0-9]+(?:'[A-Za-z]+)?", residual)
    return len(body.split()) <= 35 or len(residual_words) <= 12


def substantive_surface_slot(task: Any) -> bool:
    """Return whether the anonymous slot requires a substantive realization."""

    try:
        words = int(getattr(task, "real_word_count", 0) or 0)
    except (TypeError, ValueError):
        words = 0
    shape = str(getattr(task, "real_surface_shape", "") or "")
    return words >= 35 and shape not in HARD_SHORT_SURFACE_SHAPES


def reconcile_substantive_task(task: Any) -> Any:
    """Remove whole-comment short modes from an anonymous substantive slot."""

    if not substantive_surface_slot(task):
        return task
    mode = str(getattr(task, "utterance_mode", "") or "")
    replacement = {
        "fragment_only": "local_answer_with_context",
        "direct_answer": "local_answer_with_context",
        "question_only": "question_with_context",
        "joke_only": "humorous_local_turn",
        "template_notice": "reference_with_context",
    }.get(mode)
    if not replacement:
        return task
    return replace(task, utterance_mode=replacement)
