"""Conversation-shaped reference fragments for the Planner.

The Planner's only window into real discourse is the reference-viewpoint block,
and `retrieve_reference_viewpoints` caps each source thread at two rows so that
no single reference discussion becomes a semantic template. Measured
consequence: a 36-row window comes from ~22 distinct threads and is 67-83%
depth-0. The Planner has seen thousands of opening statements and almost no
reply -- no rebuttal quoting its parent, no author returning to defend a claim,
no joke landing on an assertion, no link, no reaction to a link. So it plans
what it has been shown: a list of independent opening statements, which is one
speech act repeated N times, which is exactly what `self_bertscore` and
`self_bleu_4` measure.

This module is additive. The ranked rows stay as they are and keep supplying
domain content; a small number of COMPLETE fragments are appended, rendered in
reply order, chosen for structural richness and for topical DISTANCE from the
seed. Distance is the point: a thread with nothing worth copying can only
contribute its discourse shape, which is the thing being transferred. It also
makes the fragments safer than the rows they sit beside, not riskier.

Nothing here names a domain. The fragments are whatever the configured domain's
own evaluation-excluded threads happen to do.
"""

from __future__ import annotations

import re
from typing import Any

from .viewpoint_bank import STOPWORDS, TOKEN_RE

INTERACTION_SCOPE_MODE = "off"

FRAGMENT_COUNT = 3
MIN_FRAGMENT_COMMENTS = 4
_URL_RE = re.compile(r"https?://|www\.", re.I)
_QUOTE_RE = re.compile(r"(^|\n)\s*(>|&gt;)")


def set_interaction_scope(mode: str) -> None:
    global INTERACTION_SCOPE_MODE
    value = str(mode or "off").strip().lower()
    if value not in {"off", "conversation", "full"}:
        raise ValueError(
            f"unknown interaction-scope mode {mode!r}; expected off|conversation|full"
        )
    INTERACTION_SCOPE_MODE = value


def planner_fragments_enabled() -> bool:
    """The Planner sees real exchanges instead of isolated opening statements."""

    return INTERACTION_SCOPE_MODE in {"conversation", "full"}


def reply_material_enabled() -> bool:
    """The Writer may take its parent's point as material instead of excluding it.

    55.1% of shipped prompts carry "treat its own point as an exclusion rather
    than writing material", and 66.7% of slots are replies. That instruction is
    redundant with the mechanical guard -- `parent_copy` is already a hard
    realization failure, waived only for a planned quote opener with a distinct
    reply -- so it buys nothing against copying and costs the exchange. Real
    replies take the parent's point as material: they quote it and argue, or
    agree and say why.
    """

    return INTERACTION_SCOPE_MODE == "full"


def _tokens(text: Any) -> set[str]:
    return {
        token.lower()
        for token in TOKEN_RE.findall(str(text or ""))
        if token.lower() not in STOPWORDS
    }


def _structure_score(rows: list[dict[str, Any]]) -> float:
    """Reward a fragment for carrying the moves the Planner never sees."""

    depths = [int(str(row.get("depth") or 0) or 0) for row in rows]
    replies = sum(1 for d in depths if d >= 1)
    if replies == 0:
        return 0.0
    texts = [str(row.get("text") or "") for row in rows]
    score = float(replies)
    score += 2.0 * max(0, max(depths) - 1)
    score += 3.0 * sum(1 for t in texts if _QUOTE_RE.search(t))
    score += 3.0 * sum(1 for t in texts if _URL_RE.search(t))
    # A very short turn beside long ones is the joke / bare question / reaction
    # slot, which is the other thing an all-opening-statements window omits.
    score += 1.5 * sum(1 for t in texts if len(t.split()) <= 12)
    return score


def select_conversation_fragments(
    profile: dict[str, Any],
    *,
    seed_title: str,
    seed_body: str,
    exclude_post_ids: set[str] | None = None,
    count: int = FRAGMENT_COUNT,
) -> list[list[dict[str, Any]]]:
    """Return whole excluded threads, reply-ordered, distant from this seed."""

    if not planner_fragments_enabled() or count <= 0:
        return []
    rows = [
        row
        for row in (profile.get("reference_viewpoints") or [])
        if isinstance(row, dict)
    ]
    if not rows:
        return []
    blocked = {str(value) for value in (exclude_post_ids or set())}
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        post_id = str(row.get("source_post_id") or "")
        if not post_id or post_id in blocked:
            continue
        grouped.setdefault(post_id, []).append(row)

    seed_tokens = _tokens(f"{seed_title} {seed_body}")
    ranked: list[tuple[float, float, str, list[dict[str, Any]]]] = []
    for post_id, group in grouped.items():
        if len(group) < MIN_FRAGMENT_COMMENTS:
            continue
        structure = _structure_score(group)
        if structure <= 0.0:
            continue
        text = " ".join(str(row.get("text") or "") for row in group)
        overlap = _tokens(text) & seed_tokens
        # Distance, not similarity: the fragment is here for its shape.
        distance = 1.0 / (1.0 + float(len(overlap)))
        ranked.append((structure * distance, distance, post_id, group))
    ranked.sort(key=lambda item: (-item[0], -item[1], item[2]))
    out: list[list[dict[str, Any]]] = []
    for _score, _distance, _post_id, group in ranked[: max(0, int(count))]:
        ordered = sorted(
            group,
            key=lambda row: (
                int(str(row.get("depth") or 0) or 0),
                str(row.get("reference_id") or ""),
            ),
        )
        out.append(ordered)
    return out


def render_conversation_fragments(fragments: list[list[dict[str, Any]]]) -> str:
    """Render fragments as conversations, not as a list of separate rows."""

    if not fragments:
        return ""
    blocks: list[str] = []
    for index, group in enumerate(fragments, start=1):
        lines = [f"Conversation {index} (a real discussion in this community):"]
        for row in group:
            depth = int(str(row.get("depth") or 0) or 0)
            who = "top-level" if depth == 0 else f"reply at depth {depth}"
            text = " ".join(str(row.get("text") or "").split())[:300]
            lines.append(f"  [{who}] {text}")
        blocks.append("\n".join(lines))
    return (
        "How people actually interact here. These are unrelated to this post's "
        "subject and are shown for the SHAPE of the exchange, never for their "
        "content. Read them as conversations rather than as separate comments: "
        "who takes up whose point and runs with it, who pushes back and gets "
        "answered, who agrees and adds one detail, who drops a single line, who "
        "asks something nobody answers, who comes back a second time to defend "
        "what they said, and who simply talks past everyone. Plan a discussion "
        "that behaves like this one does, not a list of separate opinions about "
        "the post. Do not reuse their wording, claims, products, or links.\n\n"
        + "\n\n".join(blocks)
    )
