"""Planned domain claims carried from the Planner to the Writer.

Measured against a matched real thread of the same size, real comments name a
concrete domain noun in about two thirds of cases and contain a number in about
half, and the thread as a whole used 93 distinct model designators. The generated
side reached 0.32, 0.32, and 22. The cause was not the Writer's wording: the
Planner was instructed to abstract only the *discourse move* from its held-out
reference comments and to import no fact, and the Writer was separately forbidden
to invent one, so no path existed for domain specifics to reach the output. What
remained was commentary about the decision, which is why one thread's 197
comments circled roughly four abstract propositions.

A domain claim is the Planner's own restatement of a general fact about the
domain, taken from evaluation-excluded reference threads. It is a planned
categorical field because, measured over v64 to v67, planned fields are the only
control this Writer follows: an assigned `tone_target` was realized 86% of the
time while an optional prompt affordance drew 16% and an exclusion list drew
essentially nothing.

The claim never carries a detail belonging to the reference discussion or its
participants, and never a fact about the seed post, which still cannot be
invented. `CommentTask` is a frozen dataclass in the pinned shared generator, so
claims are held in a registry keyed like actor state rather than added as a task
field.
"""

from __future__ import annotations

import re
from typing import Any

from .planner_schema import parse_sample_id

_SPACE = re.compile(r"\s+")
_REFERENCE_ID = re.compile(r"(?<![A-Za-z0-9])R\d{1,8}(?![A-Za-z0-9])", re.I)
_URL = re.compile(r"https?://\S+|www\.\S+", re.I)
_EMPTY = {"", "none", "n/a", "na", "null", "no claim", "not applicable"}
_MAX_CHARS = 220


def normalized_domain_claim(value: Any) -> str:
    """Return a clean claim, or an empty string when the slot declares none."""

    text = _SPACE.sub(" ", str(value or "")).strip()
    if text.casefold() in _EMPTY:
        return ""
    # Reference ids and URLs are transport metadata; a claim must not expose the
    # bank it came from, and an invented link is a hard Writer failure.
    text = _REFERENCE_ID.sub("general domain knowledge", text)
    text = _URL.sub("a published source", text)
    if len(text) > _MAX_CHARS:
        text = text[: _MAX_CHARS - 3].rstrip() + "..."
    return text


def enrich_domain_claim_fields(
    payload: dict[str, Any],
    normalized: dict[int, dict[str, str]],
) -> dict[int, dict[str, str]]:
    """Retain ``domain_claim`` through the shared CARD JSON parser.

    That parser keeps only the fields it declares, so every generalized planner
    field needs an explicit enrich step. Without one the field is dropped in
    silence: a first attempt shipped the whole mechanism and 0 of 520 slots ever
    carried a claim, while the run still looked healthy.
    """

    rows = payload.get("comment_plans") or payload.get("plans") or []
    raw_by_sample: dict[int, dict[str, Any]] = {}
    for row in rows if isinstance(rows, list) else []:
        if not isinstance(row, dict):
            continue
        sample_id = parse_sample_id(row.get("sample_id"))
        if sample_id > 0:
            raw_by_sample[sample_id] = row
    for sample_id, plan in normalized.items():
        plan["domain_claim"] = normalized_domain_claim(
            raw_by_sample.get(sample_id, {}).get("domain_claim")
        )
    return normalized


def seed_claim_key(seed_post: Any) -> str:
    """Rebuild the seed half of the registry key the Planner recorded under."""

    return str(
        getattr(seed_post, "source_raw_post_id", "")
        or getattr(seed_post, "index", "")
        or getattr(seed_post, "title", "")
    )


def claim_for_task(
    registry: dict[tuple[str, int], str],
    seed_post: Any,
    task: Any,
) -> str:
    """Look the claim up by the same key scheme the Planner recorded."""

    sample_id = getattr(task, "real_sample_id", None) or getattr(task, "local_task_id", 0)
    try:
        sample_id = int(sample_id or 0)
    except (TypeError, ValueError):
        return ""
    return registry.get((seed_claim_key(seed_post), sample_id), "")


def render_domain_claim_rule(claim: str) -> str:
    """Render the Writer's instruction for a planned claim."""

    if not claim:
        return ""
    return (
        "Domain fact this turn states: "
        f"{claim} "
        "State it as ordinary participant knowledge, in your own words and at "
        "whatever length this slot supports. Name the equipment involved. Do not "
        "attribute it to the post, to another commenter, or to a source, and do "
        "not add a second fact beyond it."
    )
