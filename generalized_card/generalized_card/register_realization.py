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

Every tone class, at its own measured rate
-----------------------------------------
v99 scoped this to `polite` only, on the grounds that no move *discriminates* the
other labels -- every candidate scored a held-out lift below 0.3 for `neutral`.
The v100 large-thread gate showed that was the wrong test. Discrimination and
rate-matching are different questions: a move can fail to predict a label and
still be exactly the rate real comments of that label carry. Making text read real
is rate-matching.

Real comments of every register carry these moves. Blunt ones included:

    real label        any_intensifier  plain_verdict  own_thing  love_like
    polite                      0.485          0.393      0.375      0.142
    somewhat_polite             0.324          0.202      0.184      0.027
    neutral                     0.130          0.070      0.108      0.004
    impolite                    0.300          0.128      0.182      0.026

Generated output on the gate, by the tone the plan assigned:

    planned                     0.543          0.217      0.239      0.065  polite
                                0.154          0.000      0.000      0.000  somewhat
                                0.054          0.000      0.081      0.000  neutral
                                0.100          0.000      0.011      0.000  impolite

Three of the four moves are at **exactly zero** on every non-polite register, and
`any_intensifier` on planned-impolite slots is 0.100 against a real 0.300.
Decomposed for that move, polite slots run +0.059 at weight 0.25 while every other
slot runs **-0.170 at weight 0.75** -- so the class v99 excluded by design carries
essentially the whole deficit.

The cues are therefore worded to hold in any register. "Name one thing that is
plainly good" is a concession inside a blunt turn and an appraisal inside a warm
one, which is what real comments of each label do; it is not an instruction to
soften. The tone the Planner assigned still owns the stance.

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


# Every register whose realization this module repairs, measured separately.
# `somewhat_polite` is a real polite-guard class that the 12 metrics never report,
# but it is 8.7% of planned slots and its moves are at zero, so it is included.
TARGET_TONES = ("polite", "somewhat_polite", "neutral", "impolite")

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
        # Worded to hold in any register: a concession inside a blunt turn, an
        # appraisal inside a warm one. Real `impolite` comments carry this at
        # 0.128, usually as exactly that concession.
        "cue": (
            "Name one thing here that is plainly good, in an ordinary everyday "
            "word, and let that much stand -- even if your overall judgement is "
            "negative. Do not convert it into a trade-off, a condition, or an "
            "abstract appraisal of what matters."
        ),
    },
    {
        "name": "own_thing",
        "pattern": r"\bmy \w+",
        # "what you ended up keeping" invited a story arc, and it got one: on the
        # v100 gate, generated comments carrying a possessive scored 0.510 mean
        # story probability against 0.279 for real ones carrying the same
        # possessive, and `mean_story_probability` went from 0.8% error to 29.2%.
        # Real text uses the possessive as a bare fact ("my copy is junk") far
        # more often than as a narrative, so the cue now asks for the state and
        # rules out the events.
        "cue": (
            "Refer to something of your own as yours, in passing -- what you "
            "have, what you use -- as a plain present fact. Do not tell the story "
            "of how you came to have it or what happened with it. Only something "
            "the plan already allows you to have; do not invent a possession."
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
    """Measure each move's share per size band, separately for each register.

    Reads the same per-comment `politeness_results.json` tables that
    `tone_length_fit.build_tone_length_profile` reads, filtered to the same
    evaluation-excluded reference threads, because the shares only mean anything
    conditioned on the evaluation classifier's own label. Only counts are stored;
    the comment text is read to match a pattern and never retained.
    """

    reference = {
        str(value).strip() for value in reference_thread_ids if str(value).strip()
    }
    samples: dict[str, dict[str, list[str]]] = {
        tone: {name: [] for name in STRUCTURE_BUCKETS} for tone in TARGET_TONES
    }
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
                if label not in samples:
                    continue
                text = str(row.get("text") or "").strip()
                if not text:
                    continue
                total += 1
                samples[label][structure_bucket(len(text.split()))].append(text)
    if total < _MIN_SAMPLES:
        return {"available": False, "sample_count": total, "tones": {}}
    tones = {
        tone: {
            name: _band_row(bodies)
            for name, bodies in bands.items()
            if len(bodies) >= _MIN_BAND_SAMPLES
        }
        for tone, bands in samples.items()
    }
    tones = {tone: bands for tone, bands in tones.items() if bands}
    return {
        "available": bool(tones),
        "tone_classes": list(TARGET_TONES),
        "method": (
            "per-move comment frequency by evaluation-classifier label and size "
            "band, over same-domain threads excluded from the evaluation seed pool"
        ),
        "sample_count": total,
        "tone_sample_counts": {
            tone: sum(len(b) for b in bands.values())
            for tone, bands in sorted(samples.items())
        },
        "tones": {tone: dict(sorted(bands.items())) for tone, bands in sorted(tones.items())},
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


def band_row(
    profile: dict[str, Any] | None, word_count: Any, tone_class: str
) -> dict[str, Any]:
    """Return the measured row for this slot's register and size band."""

    tone = str(tone_class or "").strip().lower()
    bands = ((profile or {}).get("tones") or {}).get(tone) or {}
    row = bands.get(structure_bucket(word_count))
    return row if isinstance(row, dict) else {}


def slot_uses_move(
    profile: dict[str, Any] | None,
    *,
    slot_key: str,
    move: str,
    word_count: Any,
    tone_class: str,
) -> bool:
    """One stable per-slot draw at this register and band's measured share.

    Namespaced away from `sentence_rhythm`'s draw so a slot that draws a rhythm
    habit is not thereby correlated with drawing a register move.
    """

    row = band_row(profile, word_count, tone_class)
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
    tone_class: str,
) -> tuple[tuple[str, bool], ...]:
    """Return every move this register and band measures, with the slot's draw."""

    row = band_row(profile, word_count, tone_class)
    if not row:
        return ()
    return tuple(
        (
            spec["name"],
            slot_uses_move(
                profile,
                slot_key=slot_key,
                move=spec["name"],
                word_count=word_count,
                tone_class=tone_class,
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

    The rate comes from the register the plan assigned, so a blunt slot is asked
    for what real blunt comments carry and not for warmth. The plan still owns the
    stance; nothing here changes which tone a slot has.
    """

    if not REGISTER_REALIZATION_ENABLED:
        return ""
    tone = str(tone_class or "").strip().lower()
    if tone not in TARGET_TONES:
        return ""
    cues = [
        _MOVE_BY_NAME[name]["cue"]
        for name, drawn in slot_moves(
            profile, slot_key=slot_key, word_count=word_count, tone_class=tone
        )
        if drawn
    ]
    if not cues:
        return ""
    # Not "warm register": the same moves are measured for every register, and
    # naming one would tell a blunt slot to soften.
    return "Register, realized: " + " ".join(cues)


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
