"""Per-slot realization of the tone class the plan already assigned.

`TONE_DEFINITIONS["polite"]` describes a register in prose -- "warm and personally
engaged, commit to a positive evaluation" -- and the Writer realizes it 19.3% of
the time, turning half of its polite assignments into the blunt register instead.
The same Writer realizes `impolite` 89.7% of the time. It has one register, and a
prose description of another one does not reach the output.

`sentence_rhythm` moved seven habits from 0.000-0.299 to their measured rates
because it names a concrete surface act and draws it per slot. This applies that
mechanism to the tone the plan already owns. Nothing here chooses a tone; the
Planner and `tone_length_fit` still do that.

Measured on 15,294 evaluation-excluded real comments carrying polite-guard's own
labels, with the derivation fitted on half the excluded threads and every lift
scored on the other half. Full evidence in `tasks/v99-worklog.md`.

Why these moves and not others
------------------------------
Four moves clear all three bars -- out-of-sample lift on P(polite), a real
prevalence worth spending a slot on, and a generated rate below real:

    move                held-out lift   real prev   generated   ratio
    love_like                   2.69      0.064      0.015      0.24x
    plain_verdict               2.53      0.215      0.085      0.40x
    own_thing                   2.11      0.212      0.117      0.55x
    any_intensifier             2.07      0.384      0.288      0.75x

Three candidates were measured and deliberately left out:

    gratitude       lift 2.46 but generated already runs 1.25x real. Asking for
                    thanks would push a rate that is already too high. The
                    per-band profile also shows it running backwards to every
                    other move -- 0.292 in the 0-15 band against 0.089 above 120
                    words -- so short polite comments are thank-yous and long
                    ones are enthusiastic sharing. One "add warmth" cue would be
                    wrong at both ends.
    reassure_you    lift 2.26 but real prevalence 0.023. Too rare to spend a
                    slot on.
    link            lift 1.97 and generated is 0.000 against a real 0.058, which
                    is a genuine gap and an eye-visible tell. It cannot be cued:
                    a link needs a real URL and inventing one is a hard failure.
                    Recorded, not fixed here.

`intensified_positive` (lift 2.17) is omitted as well: it is the conjunction of
`any_intensifier` and `plain_verdict`, and cueing all three would ask for the
same construction twice.

Why the draw is banded
----------------------
The deficit scales with length and so does the register. Share of real *polite*
comments carrying each move:

    band        any_intensifier  plain_verdict  own_thing  love_like
    0-15 w                0.140          0.240      0.061      0.058
    15-30 w               0.253          0.322      0.263      0.140
    30-60 w               0.399          0.342      0.324      0.124
    60-120 w              0.615          0.417      0.486      0.152
    120+ w                0.861          0.610      0.627      0.228

At 120+ words 86% of real polite comments carry an intensifier and 63% name
something of the speaker's own. Generated 120+ slots are 67.3% planned polite and
realize polite 14.3% of the time, against 76.7% in real text -- the largest gap in
the corpus and the band where a flat cue would understate the register most.

Why only the polite class
-------------------------
No move discriminates `neutral`: every candidate scores a held-out lift below 0.3
for that label, and the neutral-versus-impolite contrast shows neutral to be the
*unmarked* register, characterised by the absence of these moves rather than the
presence of anything. There is nothing additive to ask a neutral slot for. The
planned-neutral bleed into impolite (0.513) is a real problem and it needs a
suppressive mechanism instead; see `tasks/v99-worklog.md`.

Why the cues name an act and never a phrase
-------------------------------------------
`self_bleu_4` is a weak pass at Cliff +0.42 and a fixed cue vocabulary repeats
across slots. Each cue below names the move and leaves the wording to the slot,
the way `sentence_rhythm`'s digit cue names kinds of number rather than a number.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable

from .comment_structure import STRUCTURE_BUCKETS, structure_bucket


# The label whose realization this module repairs. Kept explicit rather than
# implied so a future extension to another class has to state itself.
TARGET_TONE = "polite"

REGISTER_MOVES: tuple[dict[str, Any], ...] = (
    {
        "name": "any_intensifier",
        "pattern": (
            r"\b(?:very|really|super|pretty|so|incredibly|absolutely|definitely)\b"
        ),
        "cue": (
            "Put an ordinary intensifier on one judgement here rather than "
            "stating it flat -- the plain spoken kind, not an emphatic "
            "construction."
        ),
    },
    {
        "name": "plain_verdict",
        "pattern": (
            r"\b(?:great|good|excellent|fantastic|awesome|amazing|perfect|"
            r"lovely|beautiful|incredible|superb|brilliant)\b"
        ),
        "cue": (
            "Commit to the positive judgement in one plain everyday word, and "
            "let it stand. Do not convert it into a trade-off, a condition, or "
            "an abstract appraisal of what matters."
        ),
    },
    {
        "name": "own_thing",
        "pattern": r"\bmy \w+",
        "cue": (
            "Name something of your own in passing -- what you use, what you "
            "carry, what you ended up keeping -- as yours. Only something the "
            "plan already allows you to have; do not invent a possession."
        ),
    },
    {
        "name": "love_like",
        "pattern": r"\b(?:love|loved|loving|adore|enjoy|enjoyed|enjoying)\b",
        "cue": "Say that you like or enjoy it, in your own words, without hedging it.",
    },
)

_MOVE_BY_NAME = {row["name"]: row for row in REGISTER_MOVES}
_MIN_SAMPLES = 200
_MIN_BAND_SAMPLES = 40

# Installed once per run from the frozen domain profile, the same way the rhythm
# profile reaches the Writer prompt.
ACTIVE_REGISTER_PROFILE: dict[str, Any] = {}
# `off` reproduces every version through v98, where the polite register reached
# the Writer only as the prose in `TONE_DEFINITIONS`.
REGISTER_REALIZATION_ENABLED = True


def set_active_register_profile(profile: dict[str, Any] | None) -> None:
    """Install the frozen per-domain register profile for this run."""

    global ACTIVE_REGISTER_PROFILE
    ACTIVE_REGISTER_PROFILE = dict(profile or {})


def set_register_realization(mode: str) -> bool:
    """Select the register-realization arm and return whether it is active."""

    global REGISTER_REALIZATION_ENABLED
    REGISTER_REALIZATION_ENABLED = (
        str(mode or "measured").strip().lower() != "off"
    )
    return REGISTER_REALIZATION_ENABLED


def build_register_profile(
    raw_discussions_dir: Path,
    *,
    reference_thread_ids: Iterable[str],
) -> dict[str, Any]:
    """Measure each move's share among real `polite` comments, per size band.

    Reads the same per-comment `politeness_results.json` tables that
    `tone_length_fit.build_tone_length_profile` reads, filtered to the same
    evaluation-excluded reference threads, because the shares only mean anything
    conditioned on the evaluation classifier's own label. Only counts are
    stored; the comment text is read to match a pattern and never retained.
    """

    reference = {
        str(value).strip() for value in reference_thread_ids if str(value).strip()
    }
    samples: dict[str, list[str]] = {name: [] for name in STRUCTURE_BUCKETS}
    total = 0
    for path in sorted(Path(raw_discussions_dir).rglob("politeness_results.json")):
        payload = _load_json(path)
        for thread in payload.get("threads") or []:
            if not isinstance(thread, dict):
                continue
            if str(thread.get("thread_id") or "").strip() not in reference:
                continue
            for row in thread.get("comments") or []:
                label = str((row or {}).get("pred_label") or "").strip().lower()
                if label != TARGET_TONE:
                    continue
                text = str(row.get("text") or "").strip()
                if not text:
                    continue
                total += 1
                samples[structure_bucket(len(text.split()))].append(text)
    if total < _MIN_SAMPLES:
        return {"available": False, "sample_count": total, "bands": {}}
    bands = {
        name: _band_row(bodies)
        for name, bodies in samples.items()
        if len(bodies) >= _MIN_BAND_SAMPLES
    }
    return {
        "available": bool(bands),
        "tone_class": TARGET_TONE,
        "method": (
            "per-move comment frequency among evaluation-classifier `polite` "
            "comments, by size band, over same-domain threads excluded from the "
            "evaluation seed pool"
        ),
        "sample_count": total,
        "band_sample_counts": {
            name: len(bodies) for name, bodies in sorted(samples.items())
        },
        "bands": bands,
    }


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _band_row(bodies: list[str]) -> dict[str, Any]:
    shares = {
        spec["name"]: round(
            sum(1 for body in bodies if re.search(spec["pattern"], body, re.I))
            / len(bodies),
            6,
        )
        for spec in REGISTER_MOVES
    }
    return {"sample_count": len(bodies), "shares": dict(sorted(shares.items()))}


def band_row(profile: dict[str, Any] | None, word_count: Any) -> dict[str, Any]:
    """Return the measured register row for the band this slot falls in."""

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

    Namespaced away from `sentence_rhythm`'s draw so a slot that draws a rhythm
    habit is not thereby correlated with drawing a register move.
    """

    row = band_row(profile, word_count)
    if not row or move not in _MOVE_BY_NAME:
        return False
    share = float((row.get("shares") or {}).get(move, 0.0))
    if share <= 0.0:
        return False
    if share >= 1.0:
        return True
    digest = hashlib.sha256(f"register:{move}:{slot_key}".encode("utf-8")).digest()
    draw = int.from_bytes(digest[:8], "big", signed=False) / float(1 << 64)
    return draw < share


def slot_moves(
    profile: dict[str, Any] | None,
    *,
    slot_key: str,
    word_count: Any,
) -> tuple[tuple[str, bool], ...]:
    """Return every move this band measures, with the slot's draw for it."""

    row = band_row(profile, word_count)
    if not row:
        return ()
    return tuple(
        (
            spec["name"],
            slot_uses_move(
                profile, slot_key=slot_key, move=spec["name"], word_count=word_count
            ),
        )
        for spec in REGISTER_MOVES
        if spec["name"] in (row.get("shares") or {})
    )


def register_guidance(
    profile: dict[str, Any] | None,
    *,
    slot_key: str,
    word_count: Any,
    tone_class: str,
) -> str:
    """Render this slot's drawn register moves as one Writer rule.

    Returns empty for any tone other than the target class. A slot the plan did
    not assign `polite` must not be nudged warm: the plan marginal already
    matches real text and moving it would trade one failing metric for another.
    """

    if not REGISTER_REALIZATION_ENABLED:
        return ""
    if str(tone_class or "").strip().lower() != TARGET_TONE:
        return ""
    cues = [
        _MOVE_BY_NAME[name]["cue"]
        for name, drawn in slot_moves(
            profile, slot_key=slot_key, word_count=word_count
        )
        if drawn
    ]
    if not cues:
        return ""
    return "Warm register, realized: " + " ".join(cues)


def active_register_guidance(
    *, slot_key: str, word_count: Any, tone_class: str
) -> str:
    """Render the register rule for this slot from the run's frozen profile."""

    return register_guidance(
        ACTIVE_REGISTER_PROFILE,
        slot_key=slot_key,
        word_count=word_count,
        tone_class=tone_class,
    )


def move_names() -> tuple[str, ...]:
    """Expose the measured move names for audits and tests."""

    return tuple(_MOVE_BY_NAME)


def realized_move_shares(comments: Iterable[dict[str, Any]]) -> dict[str, float]:
    """Measure the moves in generated output, for the post-run audit.

    The arm is only believable if the realized shares track the profile, which is
    how `sentence_rhythm` was verified. This is the measurement side of that.
    """

    bodies = [
        str(row.get("content") or row.get("body") or "")
        for row in comments
        if str(row.get("content") or row.get("body") or "").strip()
    ]
    if not bodies:
        return {}
    return {
        spec["name"]: round(
            sum(1 for body in bodies if re.search(spec["pattern"], body, re.I))
            / len(bodies),
            6,
        )
        for spec in REGISTER_MOVES
    }
