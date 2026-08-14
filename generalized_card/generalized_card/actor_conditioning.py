"""Thread-local actor conditioning derived during semantic planning.

Actors are not a fixed persona catalog.  The private Planner composes each
actor state from the current visible discussion and evaluation-excluded real
reference comments.  Only abstract behavioral constraints reach the Writer.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass
from typing import Any

from .planner_schema import parse_sample_id


MODE_NONE = "none"
MODE_DOMAIN_DERIVED = "domain-derived"
ACTOR_MODES = (MODE_NONE, MODE_DOMAIN_DERIVED)

ACTOR_FIELDS = (
    "participant_key",
    "knowledge_boundary",
    "participation_goal",
    "evidence_access",
    "attention_focus",
    "interaction_tendency",
    "context_visibility",
    "realization_route",
)

_SPACE_RE = re.compile(r"\s+")
_REFERENCE_ID_RE = re.compile(r"(?<![A-Za-z0-9])R\d{1,8}(?![A-Za-z0-9])", re.I)
_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.I)
_PARTICIPANT_KEY_RE = re.compile(r"^(?:OP|A\d{1,4})$", re.I)


@dataclass(frozen=True)
class ActorState:
    participant_key: str
    knowledge_boundary: str
    participation_goal: str
    evidence_access: str
    attention_focus: str
    interaction_tendency: str
    context_visibility: str
    realization_route: str
    source: str = "heldout-domain-reference-plus-visible-thread"

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


def enrich_normalized_plans(
    payload: dict[str, Any],
    normalized: dict[int, dict[str, str]],
) -> dict[int, dict[str, str]]:
    """Attach free-form actor fields that the shared CARD parser ignores."""

    raw_rows = payload.get("comment_plans") or payload.get("plans") or []
    by_sample: dict[int, dict[str, Any]] = {}
    for row in raw_rows if isinstance(raw_rows, list) else []:
        if not isinstance(row, dict):
            continue
        sample_id = parse_sample_id(row.get("sample_id"))
        if sample_id > 0:
            by_sample[sample_id] = row

    for sample_id, plan in normalized.items():
        raw = by_sample.get(sample_id, {})
        actor = raw.get("actor") if isinstance(raw.get("actor"), dict) else raw
        state = actor_state_from_values(actor, plan=plan, sample_id=sample_id)
        plan.update({f"actor_{key}": value for key, value in state.to_dict().items()})
    return normalized


def actor_state_from_plan(plan: dict[str, Any], *, sample_id: int) -> ActorState:
    values = {
        key: plan.get(f"actor_{key}") or plan.get(key)
        for key in ACTOR_FIELDS
    }
    return actor_state_from_values(values, plan=plan, sample_id=sample_id)


def actor_state_from_values(
    values: dict[str, Any],
    *,
    plan: dict[str, Any],
    sample_id: int,
) -> ActorState:
    participant_key = _participant_key(values.get("participant_key"), sample_id)
    return ActorState(
        participant_key=participant_key,
        knowledge_boundary=_field(
            values.get("knowledge_boundary"),
            "knows only what is visible in the seed or parent and what this local evidence role supports",
        ),
        participation_goal=_field(
            values.get("participation_goal"),
            plan.get("semantic_move") or "make one narrow local contribution",
        ),
        evidence_access=_field(
            values.get("evidence_access"),
            plan.get("evidence_mode") or "no evidence beyond the visible discussion",
        ),
        attention_focus=_field(
            values.get("attention_focus"),
            plan.get("detail_focus") or plan.get("local_topic") or "one visible local detail",
        ),
        interaction_tendency=_field(
            values.get("interaction_tendency"),
            plan.get("reply_relation") or plan.get("comment_function") or "one local turn",
        ),
        context_visibility=_field(
            values.get("context_visibility"),
            plan.get("context_aperture") or "only the visible local context",
        ),
        realization_route=_field(
            values.get("realization_route"),
            plan.get("opening_style") or "a direct, locally appropriate sentence route",
        ),
    )


def render_actor_state(state: ActorState | None) -> str:
    if state is None:
        return "- none"
    return "\n".join(
        (
            f"- thread-local participant: {state.participant_key}",
            f"- knowledge boundary: {state.knowledge_boundary}",
            f"- participation goal: {state.participation_goal}",
            f"- evidence access: {state.evidence_access}",
            f"- attention focus: {state.attention_focus}",
            f"- interaction tendency: {state.interaction_tendency}",
            f"- visible context boundary: {state.context_visibility}",
            f"- one-shot realization route: {state.realization_route}",
        )
    )


def seed_actor_key(seed_post: Any) -> str:
    return str(
        getattr(seed_post, "source_raw_post_id", "")
        or getattr(seed_post, "index", "")
        or getattr(seed_post, "title", "")
    )


def assignment_key(seed_post: Any, sample_id: Any) -> tuple[str, int]:
    try:
        normalized_sample = int(sample_id or 0)
    except (TypeError, ValueError):
        normalized_sample = 0
    return seed_actor_key(seed_post), normalized_sample


def actor_for_task(
    registry: dict[tuple[str, int], ActorState],
    seed_post: Any,
    task: Any,
) -> ActorState | None:
    sample_id = getattr(task, "real_sample_id", None) or getattr(task, "local_task_id", 0)
    return registry.get(assignment_key(seed_post, sample_id))


def actor_author(
    state: ActorState,
    *,
    run_index: int,
    post_slot: int,
) -> str:
    if state.participant_key.upper() == "OP":
        return f"sampled_op_{run_index}_{post_slot}"
    digest = hashlib.sha256(state.participant_key.encode("utf-8")).hexdigest()[:6]
    return f"sampled_actor_{run_index}_{post_slot}_{digest}"


def _participant_key(value: Any, sample_id: int) -> str:
    raw = _SPACE_RE.sub("", str(value or "")).upper()
    if _PARTICIPANT_KEY_RE.fullmatch(raw):
        return raw
    return f"A{max(1, int(sample_id)):03d}"


def _field(value: Any, fallback: Any, *, limit: int = 180) -> str:
    text = _SPACE_RE.sub(" ", str(value or fallback or "")).strip()
    # These substitutions remove transport metadata, not semantic categories.
    text = _REFERENCE_ID_RE.sub("held-out pattern", text)
    text = _URL_RE.sub("visible source", text)
    if len(text) > limit:
        text = text[: limit - 3].rstrip() + "..."
    return text
