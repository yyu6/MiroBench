"""Per-slot closing move, drawn from the domain's measured last sentences.

A comment has to stop somewhere, and the move it stops on is measured. Real
comments close on a concrete fact of the speaker's own; generated ones close on
an abstract verdict about what matters. Over comments of 25 words or more,
measured on the last sentence only:

    closing move                          real    generated   ratio
    abstract verdict                      0.014     0.265     19.1x
    a concrete fact of the speaker's own  0.152     0.048      0.32x
    a figure in the last sentence         0.318     0.200      0.63x
    a conditional about the reader        0.095     0.145      1.53x

The verdict close is the 19x defect. The reader-conditional close is only mildly
over-produced -- real people do end that way -- so it is measured and left alone
rather than suppressed on the strength of how it reads.

Why this is the frame's root and not another symptom
----------------------------------------------------
The "that's the part that actually matters" family has been chased since v73
through phrase bans, a rewording, a route lock, and a prompt rebuild, and it
survived all four because the phrase is not the thing. The thing is the move.
Conditioned on the payload the Planner assigned, the broad frame appears in:

    personal_story      0.448       correction   0.212      advice    0.210
    fragment_datapoint  0.202       soft_helpful 0.187      bare_answer 0.091
                                                           low_info_reaction 0.071

A story slot has the most obvious place to pivot, so it pivots most. And in real
text the frame is at 0.003 among story comments against 0.382 generated -- 127x.
Three Planner-side explanations were measured and rejected first: the rendered
"decision intent" line (lift 1.08x), the "decision boundary" line (0.83x, i.e.
slots receiving it produce the frame *less*), and v97's adjudication gate, whose
gated slots also produce it less (0.175 against 0.210). The Writer is not echoing
a control. It reaches for a verdict because it has no other way to stop.

So this names the two closing moves and draws them, rather than banning a phrase.

Scope
-----
Every slot, whatever tone the plan assigned -- the verdict close is over-produced
on `polite` (0.324 carry the frame), `impolite` (0.287) and `neutral` (0.184)
alike, so gating it to one register would leave most of it in place. That makes
this module's scope deliberately different from `register_realization`, which
fires only on the class the plan marked.

Interaction to watch
--------------------
`register_realization` asks a polite slot to commit to a positive judgement.
"Commit to a judgement" and "do not close on a verdict" are compatible -- one is
about stating an opinion, the other about not generalising it into a ruling on
what matters -- but they are adjacent, and the gate must check that a polite slot
still lands its judgement rather than dropping it.
"""

from __future__ import annotations

import hashlib
import re
import statistics
from typing import Any, Iterable

from .comment_structure import STRUCTURE_BUCKETS, structure_bucket


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
# Below this the "last sentence" is the whole comment and there is no closing
# move to speak of. Real micro comments are the reaction, not a wrap-up.
_MIN_WORDS = 25

CLOSING_MOVES: tuple[dict[str, Any], ...] = (
    {
        "name": "own_concrete_close",
        "pattern": (
            r"\b(?:i|my|mine|we|our)\b[^.!?]{0,80}"
            r"(?:\d|\bstill\b|\byears?\b|\bmonths?\b|\bnever\b|"
            r"\bno (?:issues?|problems?)\b)"
            r"|^\s*(?:no issues|still|never|about \d|i (?:was|had|still))\b"
        ),
        # First written as "how long you have had it, what it did or did not do",
        # which are events, and the v100 gate answered with narrative: story
        # probability rose on every slot group including the ones this cue never
        # reached, and `mean_story_probability` moved from 0.8% error to 29.2%.
        # Real closers of this kind are states and counts -- "No issues yet about
        # 40,000 clicks in." / "Still got mine." -- so the cue now names the state
        # and rules out the recounting.
        "cue": (
            "End on something concrete of your own -- a count, a number, a state "
            "it is still in -- stated flatly, rather than on a summary. Do not "
            "recount what happened; just say what is so. Only what the plan "
            "already lets you claim."
        ),
        "suppress_cue": "",
    },
    {
        "name": "abstract_verdict_close",
        "pattern": (
            r"\b(?:matters?|counts?|settles?|the real|the whole|the part|"
            r"the only thing|my take|the upshot|bottom line|in the end|"
            r"at the end of the day)\b"
        ),
        "cue": "",
        # Suppression-only, and two-sided suppression is unnecessary because the
        # measured rate is 0.014: unlike v98's dash, driving this to zero costs
        # almost nothing, and the draw still leaves it available on the rare slot.
        "suppress_cue": (
            "Do not end by saying what matters, what the real question is, what "
            "it comes down to, or what your take is. Stop on the thing itself."
        ),
    },
)

_MOVE_BY_NAME = {row["name"]: row for row in CLOSING_MOVES}
_MIN_SAMPLES = 200
_MIN_BAND_SAMPLES = 40

ACTIVE_CLOSING_PROFILE: dict[str, Any] = {}
# `off` reproduces every version through v99, none of which said anything about
# how a comment ends.
CLOSING_MOVE_ENABLED = True

# `off` (default) reproduces `abstract_verdict_close`'s suppression wording
# unmodified. `on` widens the *suppression cue only* -- not the measurement
# pattern above, so no domain profile rebuild is needed -- to also name "a
# check"/"a test" as the same move. Measured on the v103 N=10 artifact and
# the v106 gate: even where `abstract_verdict_close`'s existing wording
# reaches the Writer, the closing tic it targets still lands at 0.130-0.166
# of comments 25+ words against a real 0.013 (10-13x), and a "that's the
# check"/"a solid check" variant this wording never named adds another
# 0.007-0.019 against a real 0.0005 (13-37x) on top of it -- on the v106
# gate thread specifically, at nearly 3x the v103 rate, plausibly because
# forcing a different novelty angle per reply (v105) pushes the Writer
# toward this as a generic fallback move when it runs out of new specific
# content to name. See `docs/DECISIONS.md` G14/G15.
VERDICT_CLOSE_GUARD_ENABLED = False
_VERDICT_CLOSE_GUARDED = (
    "Do not end by saying what matters, what the real question is, what it "
    "comes down to, or what your take is. Naming a check or a test at the "
    "very end is the same move in different words -- \"that's the check\", "
    "\"a solid check\", \"that's the real test\" -- so it is included too. "
    "Stop on the thing itself."
)


def set_verdict_close_guard(mode: str) -> bool:
    """Select the verdict-close quantifier-style guard and return whether on."""

    global VERDICT_CLOSE_GUARD_ENABLED
    VERDICT_CLOSE_GUARD_ENABLED = str(mode or "off").strip().lower() == "on"
    return VERDICT_CLOSE_GUARD_ENABLED


def set_active_closing_profile(profile: dict[str, Any] | None) -> None:
    """Install the frozen per-domain closing profile for this run."""

    global ACTIVE_CLOSING_PROFILE
    ACTIVE_CLOSING_PROFILE = dict(profile or {})


def set_closing_move(mode: str) -> bool:
    """Select the closing-move arm and return whether it is active."""

    global CLOSING_MOVE_ENABLED
    CLOSING_MOVE_ENABLED = str(mode or "measured").strip().lower() != "off"
    return CLOSING_MOVE_ENABLED


def last_sentence(text: str) -> str:
    """Return the final sentence, which is where the closing move lives."""

    body = str(text or "").strip()
    if not body:
        return ""
    parts = [part for part in _SENTENCE_SPLIT.split(body) if part.strip()]
    return parts[-1] if parts else body


def build_closing_profile(threads: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Measure each closing move's share per size band on excluded threads."""

    samples: dict[str, list[str]] = {name: [] for name in STRUCTURE_BUCKETS}
    total = 0
    for thread in threads:
        for row in thread.get("comments") or []:
            body = str(row.get("body") or row.get("content") or "").strip()
            words = len(body.split())
            if words < _MIN_WORDS:
                continue
            total += 1
            samples[structure_bucket(words)].append(body)
    if total < _MIN_SAMPLES:
        return {"available": False, "sample_count": total, "bands": {}}
    bands = {
        name: _band_row(bodies)
        for name, bodies in samples.items()
        if len(bodies) >= _MIN_BAND_SAMPLES
    }
    return {
        "available": bool(bands),
        "method": (
            "share of comments whose final sentence carries each closing move, by "
            "size band, over same-domain threads excluded from the evaluation "
            "seed pool; comments under "
            f"{_MIN_WORDS} words are not counted because their last sentence is "
            "the whole comment"
        ),
        "min_words": _MIN_WORDS,
        "sample_count": total,
        "bands": bands,
    }


def _band_row(bodies: list[str]) -> dict[str, Any]:
    finals = [last_sentence(body) for body in bodies]
    shares = {
        spec["name"]: round(
            sum(1 for final in finals if re.search(spec["pattern"], final, re.I))
            / len(finals),
            6,
        )
        for spec in CLOSING_MOVES
    }
    return {
        "sample_count": len(bodies),
        "median_final_words": int(
            statistics.median(len(final.split()) for final in finals)
        ),
        "shares": dict(sorted(shares.items())),
    }


def band_row(profile: dict[str, Any] | None, word_count: Any) -> dict[str, Any]:
    """Return the measured closing row for the band this slot falls in."""

    row = ((profile or {}).get("bands") or {}).get(structure_bucket(word_count))
    return row if isinstance(row, dict) else {}


def slot_uses_move(
    profile: dict[str, Any] | None,
    *,
    slot_key: str,
    move: str,
    word_count: Any,
) -> bool:
    """One stable per-slot draw at this band's measured share for the move.

    Namespaced away from the rhythm and register draws so a slot drawing one does
    not thereby draw another.
    """

    row = band_row(profile, word_count)
    if not row or move not in _MOVE_BY_NAME:
        return False
    share = float((row.get("shares") or {}).get(move, 0.0))
    if share <= 0.0:
        return False
    if share >= 1.0:
        return True
    digest = hashlib.sha256(f"closing:{move}:{slot_key}".encode("utf-8")).digest()
    draw = int.from_bytes(digest[:8], "big", signed=False) / float(1 << 64)
    return draw < share


def closing_guidance(
    profile: dict[str, Any] | None,
    *,
    slot_key: str,
    word_count: Any,
) -> str:
    """Render this slot's drawn closing move as one Writer rule.

    Silent below the measurement floor: a slot too short to have a closing move
    separate from its body must not be told how to end.
    """

    if not CLOSING_MOVE_ENABLED:
        return ""
    try:
        words = int(word_count)
    except (TypeError, ValueError):
        return ""
    if words < int((profile or {}).get("min_words") or _MIN_WORDS):
        return ""
    row = band_row(profile, words)
    if not row:
        return ""
    parts: list[str] = []
    for spec in CLOSING_MOVES:
        if spec["name"] not in (row.get("shares") or {}):
            continue
        drawn = slot_uses_move(
            profile, slot_key=slot_key, move=spec["name"], word_count=words
        )
        if (
            not drawn
            and spec["name"] == "abstract_verdict_close"
            and VERDICT_CLOSE_GUARD_ENABLED
        ):
            cue = _VERDICT_CLOSE_GUARDED
        else:
            cue = spec["cue"] if drawn else spec["suppress_cue"]
        if cue:
            parts.append(cue)
    if not parts:
        return ""
    return "Closing move: " + " ".join(parts)


def active_closing_guidance(*, slot_key: str, word_count: Any) -> str:
    """Render the closing rule for this slot from the run's frozen profile."""

    return closing_guidance(
        ACTIVE_CLOSING_PROFILE, slot_key=slot_key, word_count=word_count
    )


def move_names() -> tuple[str, ...]:
    """Expose the measured move names for audits and tests."""

    return tuple(_MOVE_BY_NAME)


def realized_close_shares(comments: Iterable[dict[str, Any]]) -> dict[str, float]:
    """Measure the closing moves in generated output, for the post-run audit."""

    finals = [
        last_sentence(str(row.get("content") or row.get("body") or ""))
        for row in comments
        if len(str(row.get("content") or row.get("body") or "").split()) >= _MIN_WORDS
    ]
    if not finals:
        return {}
    return {
        spec["name"]: round(
            sum(1 for final in finals if re.search(spec["pattern"], final, re.I))
            / len(finals),
            6,
        )
        for spec in CLOSING_MOVES
    }
