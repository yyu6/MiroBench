"""Per-slot typing rhythm, drawn from the domain's measured habits.

This module was built to close `self_bertscore_mean_f1`, and a falsification
test on real text says it does not. Among evaluation-excluded real comments of
matched length, pairs that *differ* in these habits are only 0.003-0.011 lower
in function-word cosine than pairs that share them, against a generated-vs-real
gap of 0.134; pairs that are both sentence-length-uneven are slightly *more*
alike, not less. The lexical cause that does explain that gap is in
`story_scope`. This claim is recorded rather than deleted because the rejected
hypothesis is why the module exists at all.

What the module does move is measured and separate. v97 wrote **zero**
exclamation marks in 532 comments against 0.079 of matched real ones, and in the
reference corpus a comment containing one is 1.48x as likely to carry a
non-neutral dominant emotion, concentrated on gratitude, admiration, joy, love,
and amusement -- the tail labels `emotion_entropy` is made of. It also carries
the digit rate (0.299 against a real 0.562) and suppresses two constructions the
Writer over-reaches for: the semicolon at 4.5x the real rate and the dash-joined
clause at 4x. `surface_typography` maps an em dash onto " - " but cannot stop the
construction being chosen; this is where it is chosen.

The measured deficits, per size band, real against v97 generated:

    band       words/sentence     has a <=5-word sentence     has "!"
               real    gen        real       gen              real   gen
    short      10.0   13.0        0.635      0.280            0.099  0.000
    medium     14.0   19.0        0.401      0.044            0.101  0.000
    long       16.0   20.0        0.426      0.120            0.105  0.000
    very_long  17.9   22.1        0.551      0.122            0.141  0.000

    and over all 532 v97 comments against their matched real ones:
    parenthetical aside 0.055 / 0.171,  ellipsis 0.017 / 0.079,
    semicolon 0.109 / 0.024,  dash-joined clause 0.299 / 0.075

Each habit is drawn per slot at its band's measured rate, the way
`surface_typography` draws punctuation per speaker, rather than stated as a rule
that would apply to every slot of one size identically.

`caps_emphasis` was measured (real 0.214, generated 0.156) and deliberately left
out. The all-caps-run probe counts domain abbreviations -- ISO, DSLR, JPEG --
far more often than emphasis, so a cue asking for capitals would produce a
different thing from what was measured. See the 2026-08-19 lesson on unreliable
probes.
"""

from __future__ import annotations

import hashlib
import re
import statistics
from typing import Any, Iterable

from .comment_structure import STRUCTURE_BUCKETS, structure_bucket


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
_SHORT_SENTENCE_WORDS = 5

# Each habit carries the pattern that measures it and the cue that asks for it.
# `suppress_cue` is the instruction rendered when the draw comes up negative, and
# is populated only for constructions the Writer over-produces; for the rest,
# silence is the correct negative instruction.
RHYTHM_HABITS: tuple[dict[str, Any], ...] = (
    {
        "name": "short_sentence",
        "cue": (
            "Let one sentence be very short, under five words, so the sentences "
            "are uneven rather than all one size."
        ),
        "suppress_cue": "",
        "needs_two_sentences": True,
    },
    {
        "name": "exclamation",
        "pattern": r"!",
        "cue": "End one sentence with an exclamation mark.",
        "suppress_cue": "",
    },
    {
        "name": "parenthetical",
        "pattern": r"\([^)]{2,}\)",
        "cue": "Put one aside in parentheses.",
        "suppress_cue": "",
    },
    {
        "name": "ellipsis",
        "pattern": r"\.\.\.|…",
        "cue": "Let one thought trail off with ...",
        "suppress_cue": "",
    },
    {
        "name": "digit",
        "pattern": r"\d",
        "cue": (
            "Put a number in this one -- a price, a count, a model number, a "
            "length of time -- written as a figure rather than described in "
            "words. Use only a number you are allowed to name above; if there "
            "is none, leave it out rather than inventing one."
        ),
        "suppress_cue": "",
    },
    {
        "name": "semicolon",
        # Suppression only, unlike `dash_clause`: the measured share runs
        # 0.007-0.016 in the bands that carry most slots, so driving it to zero
        # costs almost nothing, and the generated rate was 4.5x real.
        "pattern": r";",
        "cue": "",
        "suppress_cue": "Use no semicolons.",
    },
    {
        "name": "dash_clause",
        "pattern": r"\s-\s|--|—|–",
        # Two-sided. Suppression alone took this to 0.000 on the seed-2 gate
        # against a real 0.089: the Writer stops reaching for a construction the
        # moment it is named, so a habit measured above zero needs the positive
        # cue as well as the negative one.
        "cue": "Hang one clause off the last one with a dash.",
        "suppress_cue": "Do not join two clauses with a dash.",
    },
)

_HABIT_BY_NAME = {row["name"]: row for row in RHYTHM_HABITS}
_MIN_SAMPLES = 200
_MIN_BAND_SAMPLES = 40

# Set once per run from the frozen domain profile, the same way the layout
# profile reaches `soft_length_guidance`.
ACTIVE_RHYTHM_PROFILE: dict[str, Any] = {}
# `off` reproduces every version through v97, which gave the Writer no rhythm
# instruction and let one register serve every slot.
SENTENCE_RHYTHM_ENABLED = True


def set_active_rhythm_profile(profile: dict[str, Any] | None) -> None:
    """Install the frozen per-domain rhythm profile for this run."""

    global ACTIVE_RHYTHM_PROFILE
    ACTIVE_RHYTHM_PROFILE = dict(profile or {})


def set_sentence_rhythm(mode: str) -> bool:
    """Select the rhythm arm and return whether it is active."""

    global SENTENCE_RHYTHM_ENABLED
    SENTENCE_RHYTHM_ENABLED = str(mode or "measured").strip().lower() != "off"
    return SENTENCE_RHYTHM_ENABLED


def sentence_words(text: str) -> list[int]:
    """Return the word count of each sentence in one comment."""

    body = str(text or "").strip()
    if not body:
        return []
    parts = [part for part in _SENTENCE_SPLIT.split(body) if part.strip()]
    return [len(part.split()) for part in (parts or [body])]


def build_rhythm_profile(threads: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Measure sentence pacing and per-habit frequency for each size band."""

    samples: dict[str, list[str]] = {name: [] for name in STRUCTURE_BUCKETS}
    total = 0
    for thread in threads:
        for row in thread.get("comments") or []:
            body = str(row.get("body") or row.get("content") or "").strip()
            if not body:
                continue
            total += 1
            samples[structure_bucket(len(body.split()))].append(body)
    if total < _MIN_SAMPLES:
        return {"available": False, "sample_count": total, "bands": {}}
    bands: dict[str, dict[str, Any]] = {}
    for name, bodies in samples.items():
        if len(bodies) < _MIN_BAND_SAMPLES:
            continue
        bands[name] = _band_row(bodies)
    return {
        "available": bool(bands),
        "method": (
            "sentence pacing and per-habit comment frequency per size band, over "
            "same-domain threads excluded from the evaluation seed pool"
        ),
        "sample_count": total,
        "bands": bands,
    }


def _band_row(bodies: list[str]) -> dict[str, Any]:
    per_sentence: list[float] = []
    sentence_counts: list[int] = []
    multi_sentence = 0
    with_short = 0
    for body in bodies:
        lengths = sentence_words(body) or [len(body.split())]
        sentence_counts.append(len(lengths))
        per_sentence.append(len(body.split()) / max(1, len(lengths)))
        if len(lengths) >= 2:
            multi_sentence += 1
            if min(lengths) <= _SHORT_SENTENCE_WORDS:
                with_short += 1
    shares: dict[str, float] = {}
    for spec in RHYTHM_HABITS:
        pattern = spec.get("pattern")
        if not pattern:
            continue
        shares[spec["name"]] = round(
            sum(1 for body in bodies if re.search(pattern, body)) / len(bodies), 6
        )
    # Measured only where it is a choice. A one-sentence comment trivially
    # contains its own shortest sentence, which would report a share of 1.0 and
    # ask a four-word comment for a shorter sentence inside it.
    shares["short_sentence"] = (
        round(with_short / multi_sentence, 6) if multi_sentence else 0.0
    )
    return {
        "sample_count": len(bodies),
        "median_words_per_sentence": round(statistics.median(per_sentence), 2),
        "median_sentences": int(statistics.median(sentence_counts)),
        "multi_sentence_count": multi_sentence,
        "shares": dict(sorted(shares.items())),
    }


def band_row(profile: dict[str, Any] | None, word_count: Any) -> dict[str, Any]:
    """Return the measured rhythm row for the band this slot falls in."""

    bands = (profile or {}).get("bands") or {}
    row = bands.get(structure_bucket(word_count))
    return row if isinstance(row, dict) else {}


def slot_uses_habit(
    profile: dict[str, Any] | None,
    *,
    slot_key: str,
    habit: str,
    word_count: Any,
) -> bool:
    """Return one stable per-slot draw at this band's measured habit share.

    Keyed on the slot rather than the speaker, unlike `surface_typography`: a
    keyboard either substitutes a character or it does not, but nobody puts an
    aside in parentheses in every comment they write. Per-slot draws are also
    what breaks the same-size-same-instruction coupling that converged the
    function-word skeleton.
    """

    row = band_row(profile, word_count)
    if not row:
        return False
    spec = _HABIT_BY_NAME.get(habit)
    if spec is None:
        return False
    if spec.get("needs_two_sentences") and int(row.get("median_sentences") or 1) < 2:
        return False
    share = float((row.get("shares") or {}).get(habit, 0.0))
    if share <= 0.0:
        return False
    if share >= 1.0:
        return True
    digest = hashlib.sha256(f"{habit}:{slot_key}".encode("utf-8")).digest()
    draw = int.from_bytes(digest[:8], "big", signed=False) / float(1 << 64)
    return draw < share


def slot_habits(
    profile: dict[str, Any] | None,
    *,
    slot_key: str,
    word_count: Any,
) -> tuple[tuple[str, bool], ...]:
    """Return every habit this band measures with the slot's draw for it."""

    row = band_row(profile, word_count)
    if not row:
        return ()
    drawn = []
    for spec in RHYTHM_HABITS:
        name = spec["name"]
        if name not in (row.get("shares") or {}):
            continue
        drawn.append(
            (
                name,
                slot_uses_habit(
                    profile, slot_key=slot_key, habit=name, word_count=word_count
                ),
            )
        )
    return tuple(drawn)


def rhythm_guidance(
    profile: dict[str, Any] | None,
    *,
    slot_key: str,
    word_count: Any,
    slot_names_sentence_count: bool = False,
) -> str:
    """Render this slot's drawn typing rhythm as one Writer rule.

    `slot_names_sentence_count` reports whether the surface skeleton already
    carries this slot's own sentence count, which it does on 348 of 532 slots.
    Where it does, the band's median words per sentence is both redundant and
    contradictory -- a 115-word slot was told "about 12 sentences" and "about 16
    words" per sentence, which is 192 words against a 133-word ask -- so the
    slot-specific number wins and this one is withheld.
    """

    if not SENTENCE_RHYTHM_ENABLED:
        return ""
    row = band_row(profile, word_count)
    if not row:
        return ""
    words_per_sentence = int(round(float(row.get("median_words_per_sentence") or 0)))
    parts: list[str] = []
    # A one-sentence band has no pacing to describe, and telling a four-word
    # comment its sentences are uneven is not a statement about anything.
    if (
        words_per_sentence > 0
        and int(row.get("median_sentences") or 1) >= 2
        and not slot_names_sentence_count
    ):
        parts.append(
            "Typing rhythm: sentences in a comment this size run about "
            f"{words_per_sentence} words here, and they are uneven."
        )
    elif slot_names_sentence_count and int(row.get("median_sentences") or 1) >= 2:
        parts.append(
            "Typing rhythm: keep those sentences uneven rather than all one "
            "length."
        )
    for name, drawn in slot_habits(profile, slot_key=slot_key, word_count=word_count):
        spec = _HABIT_BY_NAME[name]
        cue = spec["cue"] if drawn else spec["suppress_cue"]
        if cue:
            parts.append(cue)
    if not parts:
        return ""
    return " ".join(parts)


def active_rhythm_guidance(
    *, slot_key: str, word_count: Any, slot_names_sentence_count: bool = False
) -> str:
    """Render the rhythm rule for this slot from the run's frozen profile."""

    return rhythm_guidance(
        ACTIVE_RHYTHM_PROFILE,
        slot_key=slot_key,
        word_count=word_count,
        slot_names_sentence_count=slot_names_sentence_count,
    )


def habit_names() -> tuple[str, ...]:
    """Expose the measured habit names for audits and tests."""

    return tuple(_HABIT_BY_NAME)
