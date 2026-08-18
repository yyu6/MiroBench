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
MODE_OFF = "off"
MODE_PLANNED = "planned"
MODE_SELECTIVE = "selective"
CLAIM_MODES = (MODE_OFF, MODE_PLANNED, MODE_SELECTIVE)
_FACT_BEARING_REFERENCE_ROLES = {
    "personal_datapoint",
    "correction",
    "parent_local_reply",
    "full_answer",
    "explanation",
}
_MIN_SLOT_WORDS = 25
_MIN_REFERENCE_WORDS = 8
_MIN_SELECTIVE_SHARE = 0.25
_MAX_SELECTIVE_SHARE = 0.60


def domain_claim_mode(backend: Any) -> str:
    """Return the configured claim policy, falling back to the safe default."""

    value = str(getattr(backend, "GENERALIZED_DOMAIN_CLAIM_MODE", MODE_SELECTIVE) or "")
    value = value.strip().lower()
    return value if value in CLAIM_MODES else MODE_SELECTIVE


def planner_claims_enabled(backend: Any) -> bool:
    """Return whether this run plans and delivers per-slot domain claims."""

    return domain_claim_mode(backend) != MODE_OFF


def selective_claim_slots(
    comments: list[dict[str, Any]],
    references: list[dict[str, Any]],
) -> tuple[int, ...]:
    """Choose capacity-compatible slots backed by useful excluded references.

    The schedule is derived only from anonymous slot capacity and the surface
    role of evaluation-excluded reference rows. It never reads the matched
    comment's semantics. The upper bound prevents the old `planned` arm's
    508/522 factual convergence, while the lower bound keeps a substantive
    domain discussion from collapsing back to abstract decision language.
    """

    total = len(comments)
    if total <= 0 or not references:
        return ()
    usable_references = [row for row in references if _reference_can_support_claim(row)]
    observed_share = len(usable_references) / max(1, len(references))
    target_share = min(
        _MAX_SELECTIVE_SHARE,
        max(_MIN_SELECTIVE_SHARE, observed_share),
    )
    budget = max(1, round(total * target_share))
    candidates = [
        sample_id
        for sample_id, (comment, reference) in enumerate(
            zip(comments, references, strict=False),
            start=1,
        )
        if _slot_can_carry_claim(comment) and _reference_can_support_claim(reference)
    ]
    if len(candidates) <= budget:
        return tuple(candidates)
    return tuple(
        candidates[index] for index in _evenly_spaced_indices(len(candidates), budget)
    )


def render_selective_claim_schedule(
    sample_ids: list[int], claim_slots: set[int]
) -> str:
    """Render the allowed factual slots for one Planner request."""

    selected = [sample_id for sample_id in sample_ids if sample_id in claim_slots]
    if not selected:
        return "none in this request"
    return ", ".join(f"S{sample_id}" for sample_id in selected)


def _slot_can_carry_claim(comment: dict[str, Any]) -> bool:
    return len(str(comment.get("body") or "").split()) >= _MIN_SLOT_WORDS


def _reference_can_support_claim(reference: dict[str, Any]) -> bool:
    role = str(reference.get("surface_role") or "").strip().lower()
    try:
        words = int(reference.get("word_count") or 0)
    except (TypeError, ValueError):
        words = 0
    if words <= 0:
        words = len(str(reference.get("text") or "").split())
    return role in _FACT_BEARING_REFERENCE_ROLES and words >= _MIN_REFERENCE_WORDS


def _evenly_spaced_indices(size: int, count: int) -> list[int]:
    count = min(max(0, count), max(0, size))
    if count <= 0:
        return []
    if count == 1:
        return [size // 2]
    return sorted({round(step * (size - 1) / (count - 1)) for step in range(count)})


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
    *,
    enabled: bool = True,
    allowed_sample_ids: set[int] | None = None,
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
        slot_enabled = enabled and (
            allowed_sample_ids is None or sample_id in allowed_sample_ids
        )
        plan["domain_claim"] = (
            normalized_domain_claim(
                raw_by_sample.get(sample_id, {}).get("domain_claim")
            )
            if slot_enabled
            else ""
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

    sample_id = getattr(task, "real_sample_id", None) or getattr(
        task, "local_task_id", 0
    )
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
        "not add a second externally checkable domain fact beyond it."
    )
