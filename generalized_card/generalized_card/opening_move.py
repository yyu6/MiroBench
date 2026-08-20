"""The concrete first word of a comment, drawn per slot from measured shares.

`opener_profile` already schedules a grammatical entry type per slot at the
domain's measured share, and `prompts._opener_rule` already renders it. The
schedule is faithful -- on the v101 N=10 run the Writer prompt carried
`polarity_token` on 28 of 532 slots, exactly the profile's 0.0526 -- and it is
not obeyed:

    planned             n     obeyed   where the rest went
    discourse_marker    38     0.184   polarity_token 19, content_phrase 11
    content_phrase     224     0.460   noun_phrase 49, first_person 26,
                                       polarity_token 21
    polarity_token      28     0.893   -
    first_person       100     0.960   -

`polarity_token` came out at **0.1274 against a measured 0.0526** and
`discourse_marker` at **0.0247 against 0.0726**. The traffic runs one way,
because "open with a short conversational connective before the point" names a
*category*, and this Writer resolves that category to `Yeah,`.

Why it matters beyond the opener
--------------------------------
A `polarity_token` opening is the highest-disagreement entry there is. Measured
on evaluation-excluded real reply pairs with the evaluation scorer's own labels,
against a reply base rate of 0.180:

    opening token   n     P(hard-labelled disagree)
    agreed          17    0.882
    exactly         20    0.800
    yep             27    0.778
    same            28    0.500
    yes            106    0.462
    yeah           168    0.405
    no              74    0.203
    ---
    thanks         134    0.037
    thank          110    0.055
    but             42    0.119
    also            16    0.125
    ah              28    0.143
    well            72    0.153
    oh              78    0.154

The generator concentrates its polarity openers on the bad end of that list:
`yeah` 31 and `yep` 21 of 71, so `yep` runs at 0.30 of the class against a real
0.047 -- six times its share, on the single worst token.

An exact ablation harness (`analysis/disagreement_diagnosis.py`, which reproduces
the shipped artifact label-for-label before it edits anything) puts a number on
it: stripping only the polarity openers the schedule did **not** assign moves the
reply-pair `hard_disagree_rate` 0.2235 -> 0.1862 against a matched real 0.1433 --
47% of the gap -- and moves `self_bleu_4` 0.03330 -> 0.03297, down in 10 of 10
threads. Full evidence in `tasks/v102-worklog.md`.

Why a prohibition is not the fix
--------------------------------
There is already one. `_opener_rule` has appended "Do not open with a bare
agreement or disagreement token" to every non-`polarity_token` slot since v96,
and on the v101 run it reached **504 of 532 prompts and was violated on 9.1% of
them**. Naming a category does not work in either direction, which is the same
finding `TONE_DEFINITIONS["polite"]` produced at 19.3% realization. What has
worked here is naming a concrete surface form: v98's "Use no semicolons" took the
semicolon 0.109 -> 0.023 and "Do not join two clauses with a dash" took the
dash-joined clause 0.299 -> 0.071.

So this module does two things, both concrete:

  * it draws the actual word for the two entry types whose category resolves to
    the wrong act, and names it; and
  * it replaces the categorical prohibition with the token list it is about.

Why the draw is per register
----------------------------
The opening connective is not register-neutral, and a flat table would tell a
blunt correction slot to open with `Thanks`. Share of real comments opening with
each class, and the words inside it, by polite-guard's own label:

    register          discourse_marker   the words
    polite                     0.1060    thank .31 thanks .28 oh .08 so .06 well .06
    somewhat_polite            0.0944    thanks .37 oh .13 so .10 well .10 ah .09
    neutral                    0.0209    and .36 but .21 also .18 so .10 well .08
    impolite                   0.0601    well .20 oh .16 and .15 so .10 lol .08

    register          polarity_token     the words
    polite                     0.0351    yes .44 yeah .21 no .16 same .06
    somewhat_polite            0.0926    yeah .72 yes .13 agreed .07 no .03
    neutral                    0.0214    yes .47 no .25 agreed .07 yeah .05
    impolite                   0.0663    yeah .26 no .22 yes .21 same .07

Gratitude is 59% of the polite row and absent from the blunt one. That is the
same argument v101 made for measuring register moves per register, and it is why
this profile is keyed the same way. A register the profile does not measure gets
**no** rule rather than a default, so a sparse domain loses the mechanism instead
of getting a wrong word.

Why naming a word here does not repeat the way a phrase would
-------------------------------------------------------------
`register_realization` names an act and never a phrase, because a fixed cue
vocabulary repeats across slots and `self_bleu_4` is a weak pass. A single
opening function word is the exception the measurement supports: the real
distribution *is* concentrated, the draw spreads across a dozen tokens where the
Writer's own default concentrates 73% of them on two, and the ablation moved
`self_bleu_4` down rather than up.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable

from .opener_profile import classify_opener

# The two entry types whose category the Writer resolves to the wrong act. Every
# other type is either obeyed (`quote`, `conditional`, `first_person`) or drifts
# only between two content-bearing classes, which is not a defect.
DRAWN_OPENERS = ("discourse_marker", "polarity_token")

# Keyed the same way as `register_realization`, for the reason in the docstring.
TARGET_TONES = ("polite", "somewhat_polite", "neutral", "impolite")

# A polarity token carries a stance, and the Planner has already assigned one.
# On the v102 gate 2 of 10 polarity slots drew a token that contradicted their
# own plan -- both `stance=agree` slots told to open with "no" -- so the family
# is chosen by the plan and only the token inside it is drawn. Surface forms
# only, no domain vocabulary, so this transfers to any domain.
AFFIRMATIVE_TOKENS = frozenset(
    {"yes", "yeah", "yea", "yep", "yup", "agreed", "exactly", "same", "true",
     "absolutely", "definitely"}
)
NEGATIVE_TOKENS = frozenset({"no", "nope", "nah"})
# The plan's stance values that force a family. `mixed`, `uncertain`, `joking`
# and `neutral` do not commit to one, so they keep the full measured draw.
STANCE_FAMILIES = {"agree": AFFIRMATIVE_TOKENS, "disagree": NEGATIVE_TOKENS}

_FIRST_WORD = re.compile(r"[a-z']+")
_MIN_SAMPLES = 200
# A token needs to be a habit rather than one person's tic before a slot is told
# to use it, and the tail is long: 18 distinct discourse markers open real camera
# comments and the bottom six account for 2% of them.
_MIN_TOKEN_COUNT = 3
_MAX_TOKENS = 12
# A cell needs enough comments for its token shares to mean something. On camera
# the thinnest cell is `neutral` at 53-55, so every cell survives; a sparse
# domain loses the cell and keeps the categorical instruction, which is the
# correct way to degrade.
_MIN_CELL_SAMPLES = 40

# Installed once per run from the frozen domain profile, the same way the
# register profile reaches the Writer prompt.
ACTIVE_OPENING_PROFILE: dict[str, Any] = {}
# `off` reproduces every version through v101, where the entry type reached the
# Writer as a category name and a categorical prohibition.
OPENING_MOVE_ENABLED = True


def set_active_opening_profile(profile: dict[str, Any] | None) -> None:
    """Install the frozen per-domain opening-move profile for this run."""

    global ACTIVE_OPENING_PROFILE
    ACTIVE_OPENING_PROFILE = dict(profile or {})


def set_opening_move(mode: str) -> bool:
    """Select the opening-move arm and return whether it is active."""

    global OPENING_MOVE_ENABLED
    OPENING_MOVE_ENABLED = str(mode or "measured").strip().lower() != "off"
    return OPENING_MOVE_ENABLED


def first_token(text: str) -> str:
    """Return the first word of a comment, lowercased."""

    words = _FIRST_WORD.findall(str(text or "").strip().lower())
    return words[0] if words else ""


def build_opening_profile(
    raw_discussions_dir: Path,
    *,
    reference_thread_ids: Iterable[str],
) -> dict[str, Any]:
    """Measure the opening-word distribution per register and entry type.

    Reads the same per-comment `politeness_results.json` tables
    `register_realization.build_register_profile` reads, filtered to the same
    evaluation-excluded reference threads, because the distribution only means
    anything conditioned on the evaluation classifier's own label. Only counts
    are stored; the comment text is read to classify its opening and never kept.
    """

    reference = {
        str(value).strip() for value in reference_thread_ids if str(value).strip()
    }
    counts: dict[str, dict[str, dict[str, int]]] = {
        tone: {opener: {} for opener in DRAWN_OPENERS} for tone in TARGET_TONES
    }
    register_totals: dict[str, int] = {tone: 0 for tone in TARGET_TONES}
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
                if label not in counts:
                    continue
                text = str(row.get("text") or "").strip()
                if not text:
                    continue
                total += 1
                register_totals[label] += 1
                opener = classify_opener(text)
                if opener not in DRAWN_OPENERS:
                    continue
                token = first_token(text)
                if token:
                    bucket = counts[label][opener]
                    bucket[token] = bucket.get(token, 0) + 1
    if total < _MIN_SAMPLES:
        return {"available": False, "sample_count": total, "tones": {}}
    tones: dict[str, dict[str, Any]] = {}
    for tone in TARGET_TONES:
        rows = {
            opener: _token_row(counts[tone][opener], register_totals[tone])
            for opener in DRAWN_OPENERS
        }
        rows = {opener: row for opener, row in rows.items() if row}
        if rows:
            tones[tone] = rows
    return {
        "available": bool(tones),
        "tone_classes": list(TARGET_TONES),
        "drawn_openers": list(DRAWN_OPENERS),
        "method": (
            "opening-word distribution by evaluation-classifier label and "
            "grammatical entry type, over same-domain threads excluded from the "
            "evaluation seed pool"
        ),
        "sample_count": total,
        "tone_sample_counts": dict(sorted(register_totals.items())),
        "tones": {tone: dict(sorted(rows.items())) for tone, rows in sorted(tones.items())},
    }


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _token_row(counts: dict[str, int], register_total: int) -> dict[str, Any]:
    """Normalise one register's token counts for one entry type."""

    if sum(counts.values()) < _MIN_CELL_SAMPLES or register_total <= 0:
        return {}
    kept = {
        token: count
        for token, count in counts.items()
        if count >= _MIN_TOKEN_COUNT
    }
    if not kept:
        return {}
    top = sorted(kept.items(), key=lambda item: (-item[1], item[0]))[:_MAX_TOKENS]
    kept_total = sum(count for _, count in top)
    return {
        "sample_count": sum(counts.values()),
        "kept_count": kept_total,
        # Share of the register's comments that open this way, for the audit.
        "opener_share": round(sum(counts.values()) / register_total, 6),
        "tokens": [
            {"token": token, "share": round(count / kept_total, 6)}
            for token, count in top
        ],
    }


def token_row(
    profile: dict[str, Any] | None, *, opener: str, tone_class: str
) -> dict[str, Any]:
    """Return the measured token row for this slot's register and entry type."""

    tone = str(tone_class or "").strip().lower()
    rows = ((profile or {}).get("tones") or {}).get(tone) or {}
    row = rows.get(str(opener or "").strip().lower())
    return row if isinstance(row, dict) else {}


def slot_token(
    profile: dict[str, Any] | None,
    *,
    slot_key: str,
    opener: str,
    tone_class: str,
    stance: str = "",
) -> str:
    """Return one stable per-slot draw from this register's token distribution.

    Namespaced away from the rhythm and register draws so that drawing an
    opening word does not correlate with drawing a habit or a register move.

    `stance` is the Planner's assigned stance. The draw runs over the register's
    full measured distribution first; if the plan commits to a polarity and the
    drawn token disagrees, the plan **vetoes** it and the slot redraws inside the
    family, at the measured relative shares. A slot whose first draw already
    agrees with the plan keeps it, so the correction perturbs only the slots that
    were actually contradicting their own plan. A stance that does not commit to
    a polarity keeps the full draw, and a register whose measured table has no
    token of the required family falls back to the full draw rather than
    inventing one.
    """

    row = token_row(profile, opener=opener, tone_class=tone_class)
    tokens = row.get("tokens") or []
    if not tokens:
        return ""
    drawn = _draw(tokens, namespace=f"opening:{opener}", slot_key=slot_key)
    family = _stance_family(tokens, opener=opener, stance=stance)
    if len(family) == len(tokens) or drawn in {
        str(entry.get("token") or "") for entry in family
    }:
        return drawn
    # The plan vetoed the draw, so redraw inside the family it commits to. A
    # separate namespace rather than the same one: reusing the draw value would
    # map the vetoed slice of [0,1) onto the family's CDF and pile those slots
    # onto whichever tokens that slice happens to cover.
    return _draw(family, namespace=f"opening:{opener}:veto", slot_key=slot_key)


def _draw(tokens: list[dict[str, Any]], *, namespace: str, slot_key: str) -> str:
    """One stable draw over a token list, at its shares renormalised to sum to 1."""

    digest = hashlib.sha256(f"{namespace}:{slot_key}".encode("utf-8")).digest()
    draw = int.from_bytes(digest[:8], "big", signed=False) / float(1 << 64)
    total = sum(float(entry.get("share") or 0.0) for entry in tokens)
    if total <= 0.0:
        return str(tokens[-1].get("token") or "")
    cumulative = 0.0
    for entry in tokens:
        cumulative += float(entry.get("share") or 0.0) / total
        if draw < cumulative:
            return str(entry.get("token") or "")
    return str(tokens[-1].get("token") or "")


def _stance_family(
    tokens: list[dict[str, Any]], *, opener: str, stance: str
) -> list[dict[str, Any]]:
    """Restrict a polarity draw to the family the plan's stance commits to."""

    if str(opener or "").strip().lower() != "polarity_token":
        return tokens
    family = STANCE_FAMILIES.get(str(stance or "").strip().lower())
    if not family:
        return tokens
    kept = [entry for entry in tokens if str(entry.get("token") or "") in family]
    # A register whose measured table has no token of this family keeps the full
    # draw: withholding the opener entirely would cost the slot its assigned
    # entry type, and inventing a token would leave the measurement behind.
    return kept or tokens


def forbidden_opening_tokens(
    profile: dict[str, Any] | None = None, *, tone_class: str = ""
) -> tuple[str, ...]:
    """The polarity tokens the prohibition should name, measured not guessed.

    Pooled across registers so the list a slot is given does not depend on the
    register it was assigned -- a slot is being told what *not* to write, and a
    register-specific ban would leak the register into the negative rule.

    Gated on the arm: with `off` the caller falls back to the categorical
    prohibition, which is what every version through v101 rendered. An arm whose
    legacy value does not reproduce the previous release byte-for-byte is not an
    arm.
    """

    del tone_class
    if not OPENING_MOVE_ENABLED:
        return ()
    source = ACTIVE_OPENING_PROFILE if profile is None else profile
    seen: dict[str, float] = {}
    for rows in ((source or {}).get("tones") or {}).values():
        for entry in (rows.get("polarity_token") or {}).get("tokens") or []:
            token = str(entry.get("token") or "")
            if token:
                seen[token] = seen.get(token, 0.0) + float(entry.get("share") or 0.0)
    if not seen:
        return ()
    return tuple(sorted(seen, key=lambda token: (-seen[token], token)))


def opening_guidance(
    profile: dict[str, Any] | None,
    *,
    slot_key: str,
    opener: str,
    tone_class: str,
    stance: str = "",
) -> str:
    """Render this slot's drawn opening word as one clause, or nothing.

    Returns the clause that replaces the entry type's category description. An
    unmeasured register returns "" and the caller keeps the categorical
    instruction, so a sparse domain loses the mechanism rather than being given
    a word its own text does not use.
    """

    if not OPENING_MOVE_ENABLED:
        return ""
    name = str(opener or "").strip().lower()
    if name not in DRAWN_OPENERS:
        return ""
    if str(tone_class or "").strip().lower() not in TARGET_TONES:
        return ""
    token = slot_token(
        profile,
        slot_key=slot_key,
        opener=name,
        tone_class=tone_class,
        stance=stance,
    )
    if not token:
        return ""
    if name == "polarity_token":
        return (
            f'Open with the bare token "{token}" and nothing else before it, '
            "then the point."
        )
    return (
        f'Open with "{token}" and nothing else before it, then go straight into '
        "the point."
    )


def active_opening_guidance(
    *, slot_key: str, opener: str, tone_class: str, stance: str = ""
) -> str:
    """Render the opening-move clause for this slot from the frozen profile."""

    return opening_guidance(
        ACTIVE_OPENING_PROFILE,
        slot_key=slot_key,
        opener=opener,
        tone_class=tone_class,
        stance=stance,
    )


def realized_opening_shares(comments: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Measure the realized entry types and opening words, for the audit."""

    openers: dict[str, int] = {}
    tokens: dict[str, dict[str, int]] = {name: {} for name in DRAWN_OPENERS}
    total = 0
    for row in comments:
        text = str((row or {}).get("content") or (row or {}).get("text") or "").strip()
        if not text:
            continue
        total += 1
        name = classify_opener(text)
        openers[name] = openers.get(name, 0) + 1
        if name in tokens:
            token = first_token(text)
            if token:
                tokens[name][token] = tokens[name].get(token, 0) + 1
    if not total:
        return {"comment_count": 0, "opener_shares": {}, "tokens": {}}
    return {
        "comment_count": total,
        "opener_shares": {
            name: round(count / total, 6) for name, count in sorted(openers.items())
        },
        "tokens": {
            name: dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))
            for name, counts in tokens.items()
            if counts
        },
    }
