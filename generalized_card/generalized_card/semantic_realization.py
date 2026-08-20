"""One-shot semantic realization context for the generalized Writer.

These helpers expose only plans and comments generated in the current thread.
They never inspect matched evaluation text and never trigger model resampling.
"""

from __future__ import annotations

from collections import Counter
import re
from typing import Any, Iterable


SEMANTIC_CONTRACT_FIELDS = (
    ("required contribution", "semantic_move"),
    ("local detail", "detail_focus"),
    ("reply relation", "reply_relation"),
    ("stance", "stance"),
    ("evidence role", "evidence_mode"),
    ("decision intent", "domain_intent"),
    ("decision boundary", "decision_boundary"),
    ("owned decision subject", "owned_decision_subject"),
    ("other branch subjects to avoid", "forbidden_decision_subjects"),
    ("reply increment", "reply_delta"),
    ("reply increment type", "reply_delta_type"),
    ("reply novelty anchor", "reply_novelty_anchor"),
    ("parent contribution not to restate", "parent_semantic_move"),
    ("parent boundary not to restate", "parent_decision_boundary"),
    ("branch exclusion", "branch_exclusion"),
    ("development sequence", "development_plan"),
    ("sentence route", "opening_style"),
    ("explicitly avoid", "avoid_repeating"),
)

# Which planned turn kinds actually settle a question.
#
# The Writer prompt rendered "The question your turn settles: ..." on 532 of 532
# v96 slots. That frame is the documented source of the "that's the part that
# actually matters" family, which survived a rewording (v73), a prompt rebuild
# (v74), and a route lock (v75) and was still in 18.4% of v96 comments against
# effectively zero in 39,265 tokens of matched real text.
#
# Measured per planned function, the frame surfaces where it least belongs:
# personal_datapoint 29.1%, explanation_analysis 23.1%, reaction 19.0%,
# correction_caveat 16.0%, verdict_evaluation 12.3%, recommendation_advice
# 11.8%, question_followup 8.1%. A slot told to report an experience and also
# told which question it settles converts the experience into an adjudication.
# So the boundary is rendered only for the kinds of turn that are an
# adjudication, and never for a slot carrying a story.
ADJUDICATIVE_FUNCTIONS = frozenset(
    {
        "correction_caveat",
        "verdict_evaluation",
        "recommendation_advice",
    }
)
NON_ADJUDICATIVE_PAYLOADS = frozenset(
    {
        "fragment_datapoint",
        "joke",
        "low_info_reaction",
        "meta_or_template",
        "narrow_question",
        "personal_story",
        "side_tangent",
    }
)
DECISION_BOUNDARY_FIELDS = ("decision boundary", "owned decision subject")
# `universal` reproduces v96 and earlier, which rendered the boundary on every
# slot.
TURN_FRAME_ADJUDICATIVE_ONLY = True
# `off` reproduces every version through v97, where the reused mid-comment route
# ledger reached the `full` Writer arm only and `focused` -- active since v82 --
# saw openings and short lines but nothing about a phrase reused mid-comment.
ROUTE_LEDGER_ENABLED = True

TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?", re.I)
NON_SEMANTIC_DEFAULTS = frozenset(
    {
        "one seed-grounded local move",
        "one local decision condition",
        "one local thread detail",
        "one specific detail",
        "make one concrete local point",
        "none",
    }
)


def set_turn_frame(mode: str) -> bool:
    """Select the adjudication-frame arm and return whether gating is active."""

    global TURN_FRAME_ADJUDICATIVE_ONLY
    TURN_FRAME_ADJUDICATIVE_ONLY = (
        str(mode or "adjudicative_only").strip().lower() != "universal"
    )
    return TURN_FRAME_ADJUDICATIVE_ONLY


def set_route_ledger(mode: str) -> bool:
    """Select the reused-route ledger arm and return whether it is active."""

    global ROUTE_LEDGER_ENABLED
    ROUTE_LEDGER_ENABLED = str(mode or "on").strip().lower() != "off"
    return ROUTE_LEDGER_ENABLED


def reused_sentence_routes(
    comments: Iterable[dict[str, Any]],
    *,
    limit: int = 16,
) -> list[str]:
    """Return only the routes this thread has actually taken more than once.

    `used_sentence_routes` pads its list with single-use routes so the `full`
    arm's ledger has a fixed size. A ledger headed "already reused" must not
    carry them: on an early slot the padding is every route the thread has used
    once, which is a list of ordinary phrasing rather than a list of habits.
    """

    if not ROUTE_LEDGER_ENABLED:
        return []
    return [
        value
        for value in used_sentence_routes(comments, limit=limit)
        if "(used " in value
    ]


def turn_settles_a_question(task: Any) -> bool:
    """Return whether this planned turn is an adjudication of a question.

    Both Writer paths call this, so the frame cannot survive on one of them.
    """

    if not TURN_FRAME_ADJUDICATIVE_ONLY:
        return True
    story = str(getattr(task, "story_mode", "") or "no_story").strip().lower()
    if story and story != "no_story":
        return False
    payload = str(getattr(task, "payload_type", "") or "").strip().lower()
    if payload in NON_ADJUDICATIVE_PAYLOADS:
        return False
    function = str(getattr(task, "comment_function", "") or "").strip().lower()
    return function in ADJUDICATIVE_FUNCTIONS


def semantic_contract_values(task: Any) -> list[tuple[str, str]]:
    """Return the meaning-bearing controls that every Writer path must see."""

    settles = turn_settles_a_question(task)
    rows: list[tuple[str, str]] = []
    for label, field in SEMANTIC_CONTRACT_FIELDS:
        if label in DECISION_BOUNDARY_FIELDS and not settles:
            continue
        value = " ".join(str(getattr(task, field, "") or "").split())
        if field == "forbidden_decision_subjects":
            value = _compact_subject_exclusions(value)
        if value.casefold() in NON_SEMANTIC_DEFAULTS:
            continue
        if value:
            rows.append((label, value))
    return rows


def _compact_subject_exclusions(value: str, *, limit: int = 3) -> str:
    """Keep the Writer's exclusion contract focused on actionable conflicts.

    The Planner can create one independent root branch per top-level chain.
    Passing every other branch subject verbatim can turn a 20-root thread into
    a multi-thousand-character negative prompt, which dilutes rather than
    protects the current semantic boundary. This keeps the first distinct
    exclusions only; the full thread ledger remains available to the Planner.
    """

    parts = [" ".join(part.split()) for part in str(value or "").split(";")]
    unique = list(dict.fromkeys(part for part in parts if part))
    return "; ".join(unique[: max(1, int(limit))])


def short_utterance_exclusions(
    comments: Iterable[dict[str, Any]],
    *,
    max_words: int = 14,
    limit: int = 32,
) -> list[str]:
    """Keep all short sentence routes visible without copying long comments."""

    unique: dict[str, str] = {}
    for comment in comments:
        text = " ".join(str(comment.get("content") or "").split())
        if not text or len(text.split()) > max_words:
            continue
        unique.setdefault(text.casefold(), text)
    return list(unique.values())[-max(1, limit) :]


def semantic_coverage_entries(
    comments: Iterable[dict[str, Any]],
    *,
    limit: int = 24,
    current_task: Any = None,
) -> list[str]:
    """Summarize covered propositions, bounded and ranked by relevance.

    This block previously rendered every prior plan's six control fields with no
    effective cap. On a 197-comment thread it reached 30,148 characters, 69% of
    the Writer's thread blackboard and 45% of the entire prompt, which crowded out
    the slot's own assignment. Two changes keep it useful at a fraction of the
    size: each entry carries only the fields that identify a covered proposition,
    and entries are selected by lexical relevance to the current slot rather than
    by recency, so what survives the cap is what this slot could actually
    duplicate.
    """

    unique: dict[str, str] = {}
    for comment in comments:
        parts = []
        for label, field in (
            ("move", "semantic_move"),
            ("boundary", "decision_boundary"),
        ):
            value = " ".join(str(comment.get(field) or "").split())
            if value:
                parts.append(f"{label}={value}")
        if parts:
            rendered = "; ".join(parts)
            unique.setdefault(rendered.casefold(), rendered)
    entries = list(unique.values())
    kept = max(1, limit)
    if len(entries) <= kept:
        return entries
    probe = _relevance_tokens(current_task)
    if not probe:
        return entries[-kept:]
    ranked = sorted(
        entries,
        key=lambda entry: (
            -_token_overlap(probe, entry),
            -entries.index(entry),
        ),
    )
    return ranked[:kept]


def _relevance_tokens(task: Any) -> set[str]:
    if task is None:
        return set()
    text = " ".join(
        str(getattr(task, field, "") or "")
        for field in (
            "semantic_move",
            "decision_boundary",
            "claim_key",
            "detail_focus",
            "local_topic",
        )
    )
    return {token for token in TOKEN_RE.findall(text.lower()) if len(token) > 3}


def _token_overlap(probe: set[str], entry: str) -> int:
    if not probe:
        return 0
    tokens = {token for token in TOKEN_RE.findall(entry.lower()) if len(token) > 3}
    return len(probe & tokens)


def opening_route_counts(
    comments: Iterable[dict[str, Any]],
    *,
    width: int = 3,
    limit: int = 12,
) -> dict[str, int]:
    """Compress all earlier generated openings into dynamic token routes."""

    counts: Counter[str] = Counter()
    for comment in comments:
        tokens = TOKEN_RE.findall(str(comment.get("content") or "").lower())
        if tokens:
            counts[" ".join(tokens[: max(1, width)])] += 1
    return _top_repeated(counts, minimum=2, limit=limit)


def repeated_phrase_counts(
    comments: Iterable[dict[str, Any]],
    *,
    width: int = 4,
    limit: int = 12,
) -> dict[str, int]:
    """Find recurring generated n-grams without a domain phrase catalog."""

    counts: Counter[str] = Counter()
    for comment in comments:
        tokens = TOKEN_RE.findall(str(comment.get("content") or "").lower())
        seen_in_comment = {
            " ".join(tokens[index : index + width])
            for index in range(max(0, len(tokens) - width + 1))
        }
        counts.update(seen_in_comment)
    return _top_repeated(counts, minimum=2, limit=limit)


def used_sentence_routes(
    comments: Iterable[dict[str, Any]],
    *,
    width: int = 4,
    limit: int = 24,
) -> list[str]:
    """Return the most reused sentence and clause-entry routes, with counts.

    This is a surface ledger, not a semantic classifier. It derives exclusions
    from the current generated thread and therefore works across domains without
    a catalog of known bad phrases.

    Ranking is by reuse count, not recency. A first-seen-unique list truncated
    from the end dropped exactly the routes that had been established early and
    were still being repeated, and it presented a route used once and a route
    used eighteen times identically. The count is included so the exclusion
    states how entrenched each route already is.
    """

    counts: Counter[str] = Counter()
    for comment in comments:
        text = str(comment.get("content") or "")
        # Comma- and dash-led clauses carry the recurring discourse frames
        # (for example, a generic conclusion setup) that sentence-only memory
        # misses. This segments surface form; it does not classify content.
        seen_in_comment: set[str] = set()
        for segment in re.split(r"[.!?;:,\n]+|\s+[—–]\s+", text):
            tokens = TOKEN_RE.findall(segment.lower())
            if len(tokens) < 3:
                continue
            seen_in_comment.add(" ".join(tokens[: max(3, width)]))
        counts.update(seen_in_comment)
    if not counts:
        return []
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    kept = max(1, limit)
    repeated = [(route, count) for route, count in ranked if count >= 2][:kept]
    remainder = kept - len(repeated)
    singles = [
        (route, count) for route, count in ranked if count < 2
    ][-remainder:] if remainder > 0 else []
    return [
        f"{route} (used {count}x)" if count >= 2 else route
        for route, count in [*repeated, *singles]
    ]


def _top_repeated(
    counts: Counter[str],
    *,
    minimum: int,
    limit: int,
) -> dict[str, int]:
    selected = sorted(
        ((value, count) for value, count in counts.items() if count >= minimum),
        key=lambda item: (-item[1], item[0]),
    )[: max(1, limit)]
    return dict(selected)
