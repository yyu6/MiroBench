"""How a positive evaluation lands, and two tics that stop it landing.

`polite_rate` and `impolite_rate` have failed in every release since v96 and
carry the largest statistically real generator bias against the Planner's own
target (−0.1856 and +0.1529, Wilcoxon p = 0.002 each). Eight versions of
sub-sentence marker work moved neither, and `tasks/v104-worklog.md` says why:
Polite Guard is **confident**, not near-degenerate. Median margin on a generated
non-polite comment is −0.934, only 2.1% sit within 0.10 of flipping, and the
median P(impolite) among impolite-labelled generated comments is 0.981. Nothing
that decorates a clause was ever going to tip that.

What does move it is measured at sentence level. A comment reads polite when it
holds one sentence that is unambiguously appreciative on its own; real comments
carry one 0.220 of the time and generated ones 0.062, in every length band. But
the generator already writes the *forms* — `gratitude` at 1.48x the real rate,
`positive_predicate` at 1.39x, `bare_verdict` at parity. They do not land:

    same form, P(sentence is a carrier)     real    generated
    bare_verdict                            0.900       0.111
    react_to_parent                         0.780       0.188
    gratitude                               0.672       0.256
    positive_predicate                      0.262       0.045
    own_gear_verdict                        0.335       0.000

Reading them side by side, three surface differences account for it, measured
over 19,386 excluded-real and 1,674 generated sentences:

    per 1,000 sentences                     real    generated   ratio
    hot-tier evaluative word               78.82       22.70    0.29x
    trailing downtoner tag                  2.11       33.45   15.82x
    partitive reference ("that part")       6.40       98.57   15.41x

Inside a positive sentence the tier inverts: real is 0.532 hot and 0.371
warm-only, generated 0.142 hot and 0.825 warm-only. Real writes "Wonderful
camera.", "The IV is fantastic.", "Fantastic breakdown!"; this generator writes
"Eye AF is good, sure.", "That part was good.", "Pretty useful, honestly." The
classifier is not wrong about those — a person would read them as grudging.

An exact ablation on the shipped v103 artifact, each edit applied to the comment
and the whole comment re-scored with the evaluation's own checkpoint after the
harness reproduced its labels flip-for-flip:

    edit                              polite   impolite   pol gap closed
    strip the trailing tag            0.1212     0.6042             8.3%
    de-partitive                      0.1174     0.6269             6.2%
    warm tier -> hot tier             0.1439     0.6155            20.8%
    CONTROL warm -> warm synonym      0.1061     0.6250             0.0%
    all three                         0.1572     0.5985            28.1%

The control is the falsification: swapping the same 157 comments' evaluative
words for *other warm words* moves `polite_rate` by 0.0000. The effect is the
tier, not the perturbation.

Two things were checked before this module existed. **The saved v103 prompts
carry no rule against any of the three** — the only adjacent text runs the other
way ("Ordinary hedges and brief thanks are allowed when they fit the turn",
292 prompts). And the reuse ledger, which echoes `- that's the bit that (used
3x)` and `- The $200 part is nice, sure, but` back to the Writer, was tested as a
priming source and **rejected**: partitive lift 0.95x, downtoner lift 1.09x, and
flat or lower where the ledger is present once position in the thread is
controlled. The tics are the model's own register, so they are suppressed here
rather than fixed upstream.

Each of the three is its own arm, so one artifact attributes all three.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable

# Length bands, the same ones `register_realization` and `sentence_rhythm` use.
BANDS: tuple[tuple[int, int], ...] = ((0, 15), (15, 30), (30, 60), (60, 120), (120, 10**6))
# Keyed the same way as `register_realization`, so a blunt slot is measured
# against blunt real comments rather than against warm ones.
TARGET_TONES = ("polite", "somewhat_polite", "neutral", "impolite")

# Surface form only, no domain vocabulary, so the finding transfers to any
# domain. The split is by evaluative strength, which is what the tier measures.
HOT_WORDS: tuple[str, ...] = (
    "wonderful", "fantastic", "incredible", "amazing", "awesome", "gorgeous",
    "stunning", "brilliant", "superb", "phenomenal", "beautiful", "excellent",
    "perfect", "great", "terrific", "stellar", "outstanding", "magnificent",
    "delightful", "love", "loved", "adore", "impressive", "impressed",
)
WARM_WORDS: tuple[str, ...] = (
    "good", "nice", "useful", "handy", "solid", "decent", "fine", "sensible",
    "reasonable", "capable", "neat", "tidy", "workable", "serviceable",
    "adequate", "alright", "okay", "ok",
)

_HOT = re.compile(r"\b(?:%s)\b" % "|".join(HOT_WORDS), re.I)
_WARM = re.compile(r"\b(?:%s)\b" % "|".join(WARM_WORDS), re.I)

# The exact trailing tags this Writer appends after an evaluation. Measured, not
# guessed: these are the forms that occur in the v103 artifact and essentially
# never in real text (2.11 per 1,000 real sentences).
DOWNTONER_TAGS: tuple[str, ...] = (
    "sure", "honestly", "really", "i guess", "i suppose", "admittedly",
    "at least", "for now", "mostly", "to be fair", "in fairness", "granted",
    "anyway", "either way", "apparently", "supposedly", "in theory", "on paper",
)
# Matched where the ablation removed it: after the evaluation, at a clause
# boundary or at the end. An end-anchored detector would under-count the same
# construction inside "The $200 part is nice, sure, but ..." and the audit has to
# measure the thing the cue targets.
_TAG = re.compile(
    r",\s*(?:%s)\b(?=\s*[.!?…,]|\s*$)"
    r"|\b(?:on the surface|in principle|up to a point|as far as it goes|"
    r"for what it'?s worth)\b\s*[.!?…]*\s*$" % "|".join(DOWNTONER_TAGS),
    re.I,
)
# The nominalisation that evaluates a slice of a thing instead of the thing.
PARTITIVE_HEADS: tuple[str, ...] = (
    "part", "bit", "piece", "side", "aspect", "element", "portion", "chunk", "angle",
)
_PARTITIVE = re.compile(
    r"\b(?:that|the|this|which)\s+(?:\w+\s+){0,2}(?:%s)\b" % "|".join(PARTITIVE_HEADS),
    re.I,
)
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")

# A profile needs enough comments before its shares mean anything, matched to
# the guards `sentence_rhythm` and `opening_move` already use.
_MIN_SAMPLES = 200
_MIN_CELL_SAMPLES = 40
# How many concrete words a cue names. Naming the word is what gets compliance --
# v102 measured ~1.0 for a named token against 0.23 for the same instruction as a
# category -- but naming one word for every slot would concentrate the lexicon
# and cost `self_bleu_4`, so each slot draws its own short list.
_CUE_WORDS = 3

ACTIVE_EVALUATIVE_PROFILE: dict[str, Any] = {}
# Each arm's `off` reproduces v103 exactly.
EVALUATION_TIER_ENABLED = True
DOWNTONER_TAG_SUPPRESSED = True
PARTITIVE_SUPPRESSED = True


def set_active_evaluative_profile(profile: dict[str, Any] | None) -> None:
    """Install the frozen per-domain evaluative profile for this run."""

    global ACTIVE_EVALUATIVE_PROFILE
    ACTIVE_EVALUATIVE_PROFILE = dict(profile or {})


def set_evaluation_tier(mode: str) -> bool:
    """Select the evaluation-tier arm and return whether it is active."""

    global EVALUATION_TIER_ENABLED
    EVALUATION_TIER_ENABLED = str(mode or "measured").strip().lower() != "off"
    return EVALUATION_TIER_ENABLED


def set_downtoner_tag(mode: str) -> bool:
    """Select the downtoner-tag arm and return whether suppression is active."""

    global DOWNTONER_TAG_SUPPRESSED
    DOWNTONER_TAG_SUPPRESSED = str(mode or "suppress").strip().lower() != "off"
    return DOWNTONER_TAG_SUPPRESSED


def set_partitive_reference(mode: str) -> bool:
    """Select the partitive arm and return whether suppression is active."""

    global PARTITIVE_SUPPRESSED
    PARTITIVE_SUPPRESSED = str(mode or "suppress").strip().lower() != "off"
    return PARTITIVE_SUPPRESSED


def band_of(word_count: Any) -> str:
    """Return the length band key for a comment size."""

    try:
        words = int(word_count or 0)
    except (TypeError, ValueError):
        words = 0
    for low, high in BANDS:
        if low <= words < high:
            return f"{low}-{high if high < 10**6 else 'inf'}"
    return f"{BANDS[-1][0]}-inf"


def sentences(text: str) -> list[str]:
    """Split a comment the way the sentence-level measurement splits it."""

    parts = (part.strip() for part in _SENTENCE_SPLIT.split(" ".join(str(text or "").split())))
    return [part for part in parts if len(part.split()) >= 2]


def is_positive_sentence(sentence: str) -> bool:
    """Whether the sentence carries an evaluative word of either tier."""

    return bool(_HOT.search(sentence) or _WARM.search(sentence))


def is_hot(sentence: str) -> bool:
    """Whether the sentence's evaluation lands in the hot tier."""

    return bool(_HOT.search(sentence))


def has_downtoner_tag(sentence: str) -> bool:
    """Whether the sentence ends on a tag that deflates what precedes it."""

    return bool(_TAG.search(sentence))


def has_partitive(text: str) -> bool:
    """Whether the text evaluates a slice of a thing rather than the thing."""

    return bool(_PARTITIVE.search(str(text or "")))


def build_evaluative_profile(
    raw_discussions_dir: Path,
    *,
    reference_thread_ids: Iterable[str],
) -> dict[str, Any]:
    """Measure the tier and the two tic rates on evaluation-excluded threads.

    Reads the same per-comment `politeness_results.json` tables
    `register_realization` and `opening_move` read, filtered to the same
    reference threads, because the tier only means anything conditioned on the
    evaluation classifier's own register label. Only counts are stored.
    """

    reference = {str(value).strip() for value in reference_thread_ids if str(value).strip()}
    band_counts: dict[str, dict[str, int]] = {
        band_of(low): {"comments": 0, "positive": 0, "hot": 0} for low, _ in BANDS
    }
    tone_counts: dict[str, dict[str, int]] = {
        tone: {"comments": 0, "positive": 0, "hot": 0} for tone in TARGET_TONES
    }
    totals = {"comments": 0, "sentences": 0, "positive": 0, "hot": 0,
              "tag_comments": 0, "partitive_comments": 0}

    root = Path(raw_discussions_dir)
    for product in sorted(p for p in root.iterdir() if p.is_dir()) if root.is_dir() else []:
        payload = _load_json(product / "politeness_results.json")
        for thread in payload.get("threads") or []:
            if not isinstance(thread, dict):
                continue
            if reference and str(thread.get("thread_id") or "").strip() not in reference:
                continue
            for comment in thread.get("comments") or []:
                text = str((comment or {}).get("text") or "")
                if not text.strip():
                    continue
                tone = str(comment.get("pred_label") or "").strip().lower()
                parts = sentences(text)
                if not parts:
                    continue
                positive = [part for part in parts if is_positive_sentence(part)]
                hot = [part for part in positive if is_hot(part)]
                band = band_of(len(text.split()))
                totals["comments"] += 1
                totals["sentences"] += len(parts)
                totals["positive"] += len(positive)
                totals["hot"] += len(hot)
                totals["tag_comments"] += any(has_downtoner_tag(part) for part in parts)
                totals["partitive_comments"] += has_partitive(text)
                for bucket, key in ((band_counts, band), (tone_counts, tone)):
                    row = bucket.get(key)
                    if row is None:
                        continue
                    row["comments"] += 1
                    row["positive"] += len(positive)
                    row["hot"] += len(hot)

    def _row(counts: dict[str, int]) -> dict[str, Any]:
        positive = counts["positive"]
        return {
            "comments": counts["comments"],
            "positive_sentences": positive,
            "hot_share": (counts["hot"] / positive) if positive else 0.0,
        }

    pooled_positive = totals["positive"]
    return {
        "comments": totals["comments"],
        "sentences": totals["sentences"],
        "positive_sentences": pooled_positive,
        "hot_share": (totals["hot"] / pooled_positive) if pooled_positive else 0.0,
        "downtoner_tag_comment_rate": (
            totals["tag_comments"] / totals["comments"] if totals["comments"] else 0.0
        ),
        "partitive_comment_rate": (
            totals["partitive_comments"] / totals["comments"] if totals["comments"] else 0.0
        ),
        "bands": {key: _row(value) for key, value in band_counts.items()},
        "tones": {key: _row(value) for key, value in tone_counts.items()},
    }


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def hot_share(
    profile: dict[str, Any] | None, *, tone_class: str, word_count: Any
) -> float:
    """Return the measured hot-tier share for this slot's register and size.

    The register is the finer signal and is tried first; a register whose cell is
    too thin falls back to the length band, and a domain with too little data
    falls back to the pooled share, which is the correct way to degrade.
    """

    profile = profile or {}
    if int(profile.get("comments") or 0) < _MIN_SAMPLES:
        return 0.0
    tone = str(tone_class or "").strip().lower()
    row = ((profile.get("tones") or {}).get(tone)) or {}
    if int(row.get("positive_sentences") or 0) >= _MIN_CELL_SAMPLES:
        return float(row.get("hot_share") or 0.0)
    row = ((profile.get("bands") or {}).get(band_of(word_count))) or {}
    if int(row.get("positive_sentences") or 0) >= _MIN_CELL_SAMPLES:
        return float(row.get("hot_share") or 0.0)
    return float(profile.get("hot_share") or 0.0)


def _draw(namespace: str, slot_key: str) -> float:
    """One stable draw in [0, 1) for this slot, namespaced away from the others."""

    digest = hashlib.sha256(f"{namespace}:{slot_key}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False) / float(1 << 64)


def slot_uses_hot_tier(
    profile: dict[str, Any] | None, *, slot_key: str, tone_class: str, word_count: Any
) -> bool:
    """Whether this slot's evaluation, if it makes one, lands in the hot tier."""

    share = hot_share(profile, tone_class=tone_class, word_count=word_count)
    if share <= 0.0:
        return False
    return _draw("evaluative:tier", slot_key) < share


def slot_words(slot_key: str) -> tuple[str, ...]:
    """Draw the concrete words a hot-tier cue names for this slot.

    Different slots name different words on purpose. One fixed word would get the
    compliance a named token gets and pay for it in `self_bleu_4`.
    """

    draw = _draw("evaluative:words", slot_key)
    start = int(draw * len(HOT_WORDS)) % len(HOT_WORDS)
    return tuple(HOT_WORDS[(start + step) % len(HOT_WORDS)] for step in range(_CUE_WORDS))


def evaluative_guidance(
    profile: dict[str, Any] | None,
    *,
    slot_key: str,
    tone_class: str,
    word_count: Any,
) -> str:
    """Render this slot's evaluative rules as one Writer line.

    The tier rule is conditional on the comment making an evaluation at all --
    the Planner owns whether a slot praises anything, and this must not change
    that marginal. It changes only how far an evaluation that happens is
    allowed to travel.
    """

    cues: list[str] = []
    if EVALUATION_TIER_ENABLED and slot_uses_hot_tier(
        profile, slot_key=slot_key, tone_class=tone_class, word_count=word_count
    ):
        words = ", ".join(slot_words(slot_key))
        cues.append(
            f"If this comment rates something positively, land it at full "
            f"strength -- a word like {words} -- not at half strength."
        )
    if DOWNTONER_TAG_SUPPRESSED:
        cues.append(
            "Do not close a sentence with a tag that takes the sentence back: "
            "no trailing \", sure\", \", honestly\", \", really\", "
            "\", I guess\", \", to be fair\", \"on the surface\"."
        )
    if PARTITIVE_SUPPRESSED:
        cues.append(
            "Say what you mean about the thing itself, not about a slice of it: "
            "no \"that part\", \"the useful bit\", \"the part that matters\"."
        )
    if not cues:
        return ""
    return "Evaluation: " + " ".join(cues)


def active_evaluative_guidance(
    *, slot_key: str, tone_class: str, word_count: Any
) -> str:
    """Render the evaluative rule for this slot from the run's frozen profile."""

    return evaluative_guidance(
        ACTIVE_EVALUATIVE_PROFILE,
        slot_key=slot_key,
        tone_class=tone_class,
        word_count=word_count,
    )


def realized_evaluative_shares(comments: Iterable[dict[str, Any]]) -> dict[str, float]:
    """Measure the three rates in generated output, for the post-run audit.

    Reported per 1,000 sentences for the two tics and as a share of positive
    sentences for the tier, so the audit reads against the profile directly.
    """

    total_sentences = 0
    positive = 0
    hot = 0
    tag = 0
    partitive_comments = 0
    total_comments = 0
    for comment in comments or []:
        text = str((comment or {}).get("content") or (comment or {}).get("text") or "")
        parts = sentences(text)
        if not parts:
            continue
        total_comments += 1
        total_sentences += len(parts)
        for part in parts:
            if is_positive_sentence(part):
                positive += 1
                hot += is_hot(part)
            tag += has_downtoner_tag(part)
        partitive_comments += has_partitive(text)
    return {
        "sentences": float(total_sentences),
        "hot_share_of_positive": (hot / positive) if positive else 0.0,
        "positive_per_1k_sentences": (positive / total_sentences * 1000.0) if total_sentences else 0.0,
        "downtoner_tag_per_1k_sentences": (tag / total_sentences * 1000.0) if total_sentences else 0.0,
        "partitive_comment_rate": (
            partitive_comments / total_comments if total_comments else 0.0
        ),
    }
