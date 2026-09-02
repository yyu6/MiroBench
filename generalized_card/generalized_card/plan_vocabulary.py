"""One source of truth for the Planner's control vocabularies.

Two defects motivate this module, both measured on celebrity (G202, G205).

**The taxonomies were a shopping list.** `perspectives` came from
`planning_quality.universal_viewpoints()` -- twelve entries whose own `source`
field reads `universal_decision_lens` -- and they are product-purchase lenses:
cost and value, compatibility and ecosystem, cause and troubleshooting, timing
and availability.  `content_angle` was an eight-value enum of the same shape.
Neither is derived from the domain.  On a celebrity thread the Planner has
almost nothing that fits, so it takes the escape hatch: `seed_local` for 45% of
top-level slots and `unclear_mixed` for 62%.  That is not the Planner failing to
choose, it is the Planner correctly reporting that the menu is for another
restaurant.

**The two planners had drifted.** `prompts.py` renders the root schema and
`reply_planning.py` renders the reply schema, independently.  The reply schema
had no `perspective_id` and no `domain_intent` at all, and asked for
`content_angle` as free text which `normalize_plan_rows` then folded to
`unclear_mixed`.  Replies are 49% of a thread, so half of every thread reached
the Writer with three content controls pinned to constants -- `decision intent:
one seed-grounded local move` on 178 of 364 slots.  Both planners now render
these three hints from here, and `test_plan_vocabulary_parity` fails if either
schema drops a field the task builder reads.

Why an open vocabulary and not simply a longer list: G191 priced removing the
constraints and it was WORSE (+83.8% against +30.2%), because dropping the grid
left `seed_local` as the only remaining answer and 58-79% of slots took it.
The failure was the escape hatch, not the constraint.  So `open` does the
opposite of G191: it removes the escape hatch and requires a name, keeps the
universal twelve as illustrations of the FORM a lens takes, and points the
Planner at the R# reference rows -- real comments from evaluation-excluded
threads, already in its prompt -- as the material to abstract its own lens set
from.  That abstraction step is CARD's, and the Planner is the only stage
allowed to see that text.

`closed` reproduces every release through v150 byte-for-byte, including the
reply schema's two missing fields: repairing the drift under the closed taxonomy
would hand replies the same shopping menu that top-level slots already answer
`unclear_mixed` to, so the parity repair is gated on `open` with the vocabulary
it needs. `test_plan_vocabulary_parity` therefore asserts parity under `open`,
which is the arm that claims it.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any

PLAN_VOCABULARY_MODE = "closed"

# The share/collision guards compare control fields by string equality, so
# "media trust and framing" and "media framing and trust" would read as two
# lenses and `max_perspective_share` would not see the concentration. Compare on
# a canonical form and keep the Planner's wording for the prompt.
_WORD_RE = re.compile(r"[a-z0-9]+")
_LENS_STOPWORDS = frozenset(
    "a an the and or of in on to for with about as at by from is are its it this that".split()
)
# Generic head nouns a named lens may or may not carry. Without stripping them,
# the root planner's "practical burden lens" and the reply planner's "practical
# burden" count as two lenses, which is how a 97-slot thread was measured at 54
# distinct lenses when many were the same position under a different suffix.
_LENS_HEAD_NOUNS = frozenset("lens angle read take view framing perspective".split())

MAX_LENS_CHARS = 48
MAX_ANGLE_CHARS = 40


def set_plan_vocabulary(mode: str) -> bool:
    """Select the arm and return whether the open vocabulary is active."""

    global PLAN_VOCABULARY_MODE
    PLAN_VOCABULARY_MODE = str(mode or "closed").strip().lower()
    return PLAN_VOCABULARY_MODE == "open"


def open_vocabulary() -> bool:
    return PLAN_VOCABULARY_MODE == "open"


def canonical_lens(value: object) -> str:
    """Comparison key for a lens or angle: content words, lowercased, sorted."""

    words = _WORD_RE.findall(str(value or "").lower())
    kept = [w for w in words if w not in _LENS_STOPWORDS] or words
    trimmed = [w for w in kept if w not in _LENS_HEAD_NOUNS] or kept
    return " ".join(sorted(trimmed))


def normalize_open_control(value: object, *, fallback: str, limit: int) -> str:
    """Accept a Planner-named control without folding it to the fallback.

    Under `closed` the caller keeps its own vocabulary check; this is only
    reached under `open`, where any non-empty phrase is a legitimate answer. A
    bare branch or slot identifier is not a lens, though -- `B3` and `S12` carry
    no semantic content and were the one value the closed canonicalizer ever had
    to repair -- so those still fall back.
    """

    text = " ".join(str(value or "").strip().split())
    if not text:
        return fallback
    if re.fullmatch(r"[BSRP]\s*\d+", text, re.I):
        return fallback
    if len(text) > limit:
        text = text[:limit].rstrip(" ,;:-")
    return text


# ---------------------------------------------------------------------------
# Schema hints. Both planners render these, so neither can drift from the other.
# ---------------------------------------------------------------------------

def perspective_schema_hint(*, allow_seed_local: bool = True) -> str:
    if not open_vocabulary():
        return (
            "one P## from the frozen domain profile, or seed_local"
            if allow_seed_local
            else "one P## from the frozen domain profile"
        )
    return (
        "the decision lens this slot argues from, 3-6 words, named by you from "
        "this discussion's own material -- reuse a lens you already named when "
        "two slots genuinely share one, and never write seed_local or a P##"
    )


def content_angle_schema_hint(closed_enum: str) -> str:
    if not open_vocabulary():
        return closed_enum
    return (
        "what this comment is about, 2-5 words, named from the discussion "
        "itself rather than chosen from a list"
    )


def domain_intent_schema_hint() -> str:
    if not open_vocabulary():
        return "one short domain-grounded intent that does not import a hidden fact"
    return (
        "what this specific comment is trying to accomplish socially, in your "
        "own words and specific to this slot -- never a generic placeholder"
    )


def reply_shared_field_lines() -> str:
    """The two fields the reply schema never asked for.

    Gated on `open` together with the vocabulary, and not because it is
    convenient: adding `perspective_id` to the reply schema while the twelve
    frozen lenses are still the whole menu hands replies the same
    domain-mismatched taxonomy that makes top-level slots answer
    `unclear_mixed` 62% of the time. The two fixes are one fix. Keeping them
    gated together also keeps `closed` an exact reproduction of v150, so every
    comparison already recorded against it stays valid.
    """

    if not open_vocabulary():
        return ""
    return (
        f'\n      "perspective_id": "{perspective_schema_hint(allow_seed_local=False)}",'
        f'\n      "domain_intent": "{domain_intent_schema_hint()}",'
    )


# Two comments whose embeddings sit below this cosine are treated as occupying
# different positions. Not a new free parameter: `compare_arms.py` and
# `measure_isolation.py` already use 0.35 as the point below which two comments
# are semantically unrelated.
DISTINCT_POSITION_COSINE = 0.35

_POSITION_COUNT_CACHE: dict[str, int] = {}


def real_position_count(bodies, encode) -> int:
    """How many distinct semantic positions this thread's real comments occupy.

    The prompt used to tell the Planner a thread holds "typically five to
    twelve" lenses. That number was invented. Measured on the ten celebrity
    seeds by agglomerative clustering at this threshold, a 97-comment thread
    holds 34 positions, a 61-comment thread 40, and a 29-comment thread 22 --
    so the invented figure was low by a factor of three to seven and was
    suppressing exactly the variety the arm exists to create. Under the closed
    taxonomy the same thread was planned with 6 lenses.

    Greedy single-pass clustering rather than scipy: this runs inside prompt
    assembly, correlates with average-linkage at r=0.991 over those ten threads
    (ratio 0.88), and adds no dependency to the generation path.
    """

    texts = [str(b or "").strip() for b in bodies]
    texts = [t for t in texts if t]
    if len(texts) < 4:
        return 0
    key = hashlib.sha256("\n".join(texts).encode("utf-8")).hexdigest()[:16]
    cached = _POSITION_COUNT_CACHE.get(key)
    if cached is not None:
        return cached
    try:
        vectors = encode(texts)
    except Exception:  # noqa: BLE001 - a prompt must still assemble without it
        return 0
    centroids: list[Any] = []
    for vector in vectors:
        if not centroids or max(
            float(_dot(vector, c)) for c in centroids
        ) < DISTINCT_POSITION_COSINE:
            centroids.append(vector)
    _POSITION_COUNT_CACHE[key] = len(centroids)
    return len(centroids)


def _dot(left: Any, right: Any) -> float:
    try:
        return float(sum(a * b for a, b in zip(left, right)))
    except TypeError:
        return float(left @ right)


def _expected_position_sentence(real_positions: int) -> str:
    if real_positions < 4:
        return ""
    return (
        f"This thread's own real comments occupy about {real_positions} "
        f"distinct semantic positions. Expect a lens set of that order -- not "
        f"a handful. A real discussion this size is many people making many "
        f"different points, not five themes restated.\n"
    )


# A thread cannot name more lenses than it has slots, and the largest celebrity
# thread is 97 comments, so this bounds the list without ever truncating a real
# one. The first version capped it at 24 -- another number I invented -- and by
# batch 12 of a 97-comment thread 19 of the 43 named lenses were hidden from the
# planner that was being asked not to duplicate them. It duplicated them:
# batches 11 and 12 added 7 new lenses each after batch 10 had added 1.
_MAX_LISTED_LENSES = 150


def named_lens_block(prior_plans, *, limit: int = _MAX_LISTED_LENSES) -> str:
    """The lenses this thread has already named, for the REPLY planner.

    Root slots and reply slots are planned by two different prompts. The root
    prompt derives a lens set and carries it forward through a ledger summary;
    the reply prompt shows a parent, its siblings' delta types, and nothing
    about lenses at all -- correct while the vocabulary was twelve frozen P##
    codes that needed no introduction, and wrong the moment the Planner names
    its own. Measured on a 97-comment thread: root batches converged, adding
    0-2 new lenses per batch by batch 8, and then the first reply batch added 8
    fresh ones and the next two added 8 more each, arriving at 54 lenses for one
    thread. They were not new positions -- "practical burden" against the root
    planner's "practical burden lens", "image-management lens" and "damage
    control lens" against "reputation management".

    Empty under `closed`, where the reply schema carries no `perspective_id`.
    """

    if not open_vocabulary():
        return ""
    counts: dict[str, tuple[str, int]] = {}
    for plan in prior_plans or ():
        raw = str((plan or {}).get("perspective_id") or "").strip()
        if not raw or raw.lower() == "seed_local":
            continue
        key = canonical_lens(raw)
        label, seen = counts.get(key, (raw, 0))
        counts[key] = (label, seen + 1)
    if not counts:
        return ""
    ranked = sorted(counts.values(), key=lambda item: (-item[1], item[0]))
    rows = ranked[:limit]
    listed = "\n".join(f"- {label} (used {seen}x)" for label, seen in rows)
    if len(ranked) > len(rows):
        # Say so rather than presenting a partial list as complete: a planner
        # told not to duplicate a list it cannot see will duplicate it.
        listed += (
            f"\n- ... and {len(ranked) - len(rows)} more not shown; avoid "
            f"renaming a position even if it is not listed here"
        )
    return (
        "\nLenses this thread has already named, with how many slots each one "
        "holds:\n"
        f"{listed}\n"
        "This list exists so the same position does not get two names. If a "
        "reply argues from a position already on it, use that exact wording "
        "rather than a synonym -- the root planner wrote \"practical burden "
        "lens\" and a reply wrote \"practical burden\" for the same position. "
        "It is NOT a menu to stay inside: a reply that argues from a position "
        "no listed lens covers must name a new one. Real threads of this size "
        "hold many distinct positions, so a growing list is expected.\n"
    )


def abstraction_block(universal_rows: str = "", real_positions: int = 0) -> str:
    """The section that asks the Planner to derive its own lens set.

    Empty under `closed`, so the assembled prompt is unchanged.
    """

    if not open_vocabulary():
        return ""
    illustration = ""
    if universal_rows.strip():
        illustration = (
            "\nThese are what a lens looks like, from a product-shopping "
            "discussion. They are the FORM, not the menu -- this discussion is "
            "not about buying anything, so expect to name different ones:\n"
            f"{universal_rows.strip()}\n"
        )
    return (
        "\n--- DERIVE THE LENSES BEFORE YOU ASSIGN THEM ---\n"
        "The R# rows above are real comments from other discussions in this "
        "community. Read them for the KINDS of position people take here: what "
        "someone is arguing from when they react, what they treat as the thing "
        "at stake, which angles recur and which appear once.\n"
        f"{_expected_position_sentence(real_positions)}"
        "Name the lenses this thread's slots will argue from and assign one to "
        "each slot in `perspective_id`. A lens is a standing "
        "position, not a topic: \"whether the framing is doing the work\" is a "
        "lens, \"the budget number\" is a topic.\n"
        "Two rules on the set you name:\n"
        "- Every slot gets a real lens. There is no 'none of these' option, "
        "because the set is yours to name; if nothing fits, that means you have "
        "not named the right lens yet.\n"
        "- No lens may cover more than about a third of the slots.\n"
        f"{illustration}"
    )


# ---------------------------------------------------------------------------
# Drift guard. `test_plan_vocabulary_parity` asserts both schemas carry these.
# ---------------------------------------------------------------------------

# Fields the task builder reads off a plan and the Writer prompt renders. A
# field missing from one planner's schema is not a rendering bug -- it silently
# pins that field to its code fallback for every slot that planner owns, which
# is how 49% of comments ended up with no intent.
SHARED_PLAN_FIELDS = (
    "payload_type",
    "comment_function",
    "content_angle",
    "evidence_mode",
    "story_mode",
    "voice",
    "speaker_role",
    "semantic_move",
    "local_topic",
    "reply_relation",
    "stance",
    "detail_focus",
    "avoid_repeating",
    "claim_family",
    "claim_key",
    "perspective_id",
    "domain_intent",
    "decision_boundary",
    "opening_style",
    "context_aperture",
)

# Root-only by design, with the reason. Anything not listed here and not in
# SHARED_PLAN_FIELDS is drift.
ROOT_ONLY_PLAN_FIELDS = {
    "sample_id": "transport",
    "branch_id": "the reply planner is given its parent, not a branch grid",
    "reference_id": "root slots pair with an R# row; replies pair with a parent",
    "tone_class": "the reply schema copies a fixed contract instead of choosing",
    "affect_role": "same fixed contract",
    "opener_type": "same fixed contract",
    "domain_claim": "rendered by both, but through the claim schema, not here",
    "development_plan": "hardcoded none on replies",
    "reply_delta": "reply-only in practice; root slots carry none",
    "reply_delta_type": "reply-only",
    "reply_novelty_anchor": "reply-only",
}


def missing_fields(schema_text: str, *, fields=SHARED_PLAN_FIELDS) -> list[str]:
    """Which shared fields a rendered schema does not ask for."""

    return [f for f in fields if f'"{f}"' not in schema_text]
