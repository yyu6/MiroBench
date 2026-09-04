#!/usr/bin/env python3
"""Per-metric revision strategy: what to select, what to ask for, what to keep.

Every string here is domain-neutral. The CARD-era revisers named the domain in
their prompts -- "keep the same card/bank/APR/fee/SUB point", "do not add a new
card name" -- which is why they could not move to celebrity or news without an
edit. Domain content reaches the model through three channels instead, all
derived at run time:

  * `anchors`  -- the concrete tokens already in THIS comment (numbers, names,
                  quoted spans, links), so "keep the facts" needs no vocabulary;
  * `protected`-- the domain config's own `protected_entity_terms`, which
                  `enable_domain.sh` derives from the corpus, never hand-written;
  * `neighbours`-- the sibling comments, which show the register of the thread.

The direction of each edit is read off the measured gap, not hardcoded: a
metric that fails HIGH and one that fails LOW get opposite instructions from the
same strategy.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

URL_RE = re.compile(r"https?://\S+")
NUMBER_RE = re.compile(r"\b\d[\d,.:/%-]*\b")
PROPER_RE = re.compile(r"\b[A-Z][a-zA-Z'&.-]+(?:\s+[A-Z][a-zA-Z'&.-]+){0,2}")
QUOTED_RE = re.compile(r"[\"“”']([^\"“”']{3,60})[\"“”']")
_SENTENCE_START = re.compile(r"(?:^|[.!?]\s+|\n\s*)$")
_CAPITAL = re.compile(r"\b[A-Z][a-zA-Z'&.-]{3,}")


def _mid_sentence_capitals(texts: Sequence[str]) -> set[str]:
    """Words this thread capitalizes somewhere other than a sentence start.

    A capitalized word opening a sentence is ambiguous -- "Honestly" and
    "Sophie" look identical there -- and the thread itself resolves it: a name
    gets capitalized mid-sentence too, a discourse marker does not. Measured on
    12 celebrity cohorts (1159 comments, 2026-09-04): treating every
    sentence-initial capital as an opener dropped 1112 spans; this recovers the
    80 that are names (Charlie, Fergie, Trump, Netflix, Ariana, Sophie among
    them) and leaves the other 1032 dropped.
    """
    out: set[str] = set()
    for text in texts:
        for match in _CAPITAL.finditer(text):
            if not _SENTENCE_START.search(text[:match.start()]):
                out.add(match.group(0).lower())
    return out


def anchors_in(text: str, protected: Sequence[str] = (),
               context: Sequence[str] = ()) -> list[str]:
    """Concrete things a rewrite must not silently drop or invent around."""
    found: list[str] = []
    found += URL_RE.findall(text)
    found += NUMBER_RE.findall(text)
    found += QUOTED_RE.findall(text)
    for term in protected:
        if term and term.lower() in text.lower():
            found.append(term)
    mid = _mid_sentence_capitals([text, *context])
    for match in PROPER_RE.finditer(text):
        span = match.group(0)
        if len(span) <= 3 or span.lower() in {f.lower() for f in found}:
            continue
        # A lone capitalized word that opens a sentence is capitalization, not
        # a name, unless the thread capitalizes it elsewhere too. Without this,
        # "Honestly the remaster..." listed "Honestly" as a fact the rewrite had
        # to preserve -- instructing the model to keep exactly the kind of
        # shared opener self_bleu_4 penalises.
        if (" " not in span and _SENTENCE_START.search(text[:match.start()])
                and span.lower() not in mid):
            continue
        found.append(span)
    seen, out = set(), []
    for item in found:
        key = item.strip().lower()
        if key and key not in seen:
            seen.add(key)
            out.append(item.strip())
    return out[:12]


@dataclass(frozen=True)
class Strategy:
    metric: str
    # What the rewrite must change, when the metric is too HIGH / too LOW.
    high: str
    low: str
    # What it must not touch, on top of the shared invariants.
    keep: str
    # How many comments of a thread may be rewritten in one round.
    max_share: float = 0.12
    # Candidates requested per comment.
    candidates: int = 5


SHARED_INVARIANTS = (
    "Write one Reddit comment and nothing else: no labels, no quotes around it, "
    "no explanation of the change.\n"
    "Keep the same speaker, the same stance toward the post, and the same reply "
    "relation to the parent comment.\n"
    "Keep every listed anchor that the original used, and invent no new name, "
    "number, date, price, link, quote, or personal experience.\n"
    "Stay within the same rough length as the original unless told otherwise."
)

STRATEGIES: dict[str, Strategy] = {
    "semantic_mean_cosine": Strategy(
        metric="semantic_mean_cosine",
        high=(
            "This comment is making the same POINT as its neighbours in different "
            "words. Keep its assigned role in the thread, but move it to a "
            "genuinely different contribution: a different consequence, a "
            "different condition under which it matters, a different kind of "
            "evidence, or an aside the topic reminded the speaker of. Changing "
            "the wording alone will not do -- the underlying claim has to be a "
            "different claim."
        ),
        low=(
            "This comment has drifted away from what the thread is about. Bring "
            "it back onto the thread's subject while keeping its own angle."
        ),
        keep="Do not turn it into a summary of the whole discussion.",
        max_share=0.15,
    ),
    "self_bertscore_mean_f1": Strategy(
        metric="self_bertscore_mean_f1",
        high=(
            "This comment is built out of the same WORDS and the same sentence "
            "machinery as its neighbours -- the same connectives, the same "
            "hedges, the same clause order, the same rhythm. Keep the point it "
            "makes exactly, and rebuild it as a different person would type it: "
            "different sentence shape, different entry, different vocabulary for "
            "the same idea. Contractions, fragments and abrupt endings are fine."
        ),
        low=(
            "This comment reads as unrelated to how the rest of the thread talks. "
            "Keep its point and bring its phrasing back toward ordinary replies "
            "in this community."
        ),
        keep="Do not change what it claims. Only how it is worded.",
        max_share=0.15,
    ),
    "self_bleu_4": Strategy(
        metric="self_bleu_4",
        high=(
            "This comment repeats exact word sequences that already appear "
            "elsewhere in the thread. Keep the meaning and replace the repeated "
            "runs of words -- especially the opening -- with different wording."
        ),
        low="Keep the meaning but phrase it in this community's ordinary words.",
        keep="Do not change the claim, the stance, or the length band.",
        max_share=0.10,
    ),
    "mean_story_probability": Strategy(
        metric="mean_story_probability",
        high=(
            "This comment narrates a sequence of events. Keep the same point and "
            "state it directly instead: no second event, no then/after pacing, no "
            "before/after change."
        ),
        low=(
            "This comment states a position abstractly. Keep the same point and "
            "ground it in one concrete thing that happened, told in the speaker's "
            "own voice. One event, not an arc, and invent no new facts beyond "
            "what the original already implies."
        ),
        keep="Do not change the stance or who is speaking.",
        max_share=0.12,
    ),
    "emotion_entropy": Strategy(
        metric="emotion_entropy",
        high="Keep the point but let one clear feeling carry it, instead of several.",
        low=(
            "The thread's comments all carry the same emotional colour. Keep this "
            "comment's point and let it carry a different feeling from its "
            "neighbours -- amusement, irritation, curiosity, resignation, "
            "enthusiasm -- whichever actually fits what it says."
        ),
        keep="Do not add an emotion the content does not support.",
        max_share=0.15,
    ),
    "polite_rate": Strategy(
        metric="polite_rate",
        high="Keep the point and drop the courtesy framing: no thanks, no praise, no softening.",
        low="Keep the point and deliver it with ordinary courtesy, without becoming formal or servile.",
        keep="Do not change the stance from disagreement to agreement or back.",
        max_share=0.12,
    ),
    "impolite_rate": Strategy(
        metric="impolite_rate",
        high="Keep the disagreement and remove the dismissiveness: argue with the claim, not at the person.",
        low="Keep the point and let it land bluntly, without hedging or balancing it away.",
        keep="No slurs, no threats, no personal abuse.",
        max_share=0.12,
    ),
    "neutral_rate": Strategy(
        metric="neutral_rate",
        high="Keep the content and let the speaker's own attitude show.",
        low="Keep the content and state it flatly, with no evaluation attached.",
        keep="Do not change the claim.",
        max_share=0.12,
    ),
    "hard_disagree_rate": Strategy(
        metric="hard_disagree_rate",
        high="Keep the substance but stop contradicting the parent outright; qualify or extend it instead.",
        low="Keep the substance and let it actually contradict what the parent claims.",
        keep="Do not change which parent it replies to.",
        max_share=0.12,
    ),
    "length_cv": Strategy(
        metric="length_cv",
        high="Keep the point and bring the length toward the thread's typical comment.",
        low=(
            "The thread's comments are all a similar length. Keep this comment's "
            "point and give it the length it would naturally have -- much shorter "
            "if it is a reaction, longer if it is an argument."
        ),
        keep="Do not pad and do not truncate mid-thought.",
        max_share=0.15,
    ),
}


def direction(measured: float, target: float) -> str:
    return "high" if measured > target else "low"


def instruction(metric: str, measured: float, target: float) -> str:
    strategy = STRATEGIES[metric]
    return strategy.high if direction(measured, target) == "high" else strategy.low


# ---------------------------------------------------------------- groups
# Metrics that read the same underlying quantity and therefore move together.
# A round targets a whole group rather than one member.
#
# `SIMILARITY` is three readings of one thing -- how much a thread's comments
# repeat each other: self_bleu_4 over exact word runs, semantic_mean_cosine
# over embeddings, self_bertscore over soft token alignment, which sits between
# the two. Targeting one and merely guarding the others throws away the fact
# that a rewrite fixing one usually fixes all three.
SIMILARITY = ("self_bertscore_mean_f1", "semantic_mean_cosine", "self_bleu_4")

# `REGISTER` is how the comments sound rather than what they say. On a cohort
# that reads flat these agree in direction -- less neutral, more feeling, more
# lived detail -- and they FIGHT the similarity group, because courtesy and
# narrative arrive as repeated wording. Hence a separate group and a separate
# round; the guard is what keeps one from undoing the other.
REGISTER = ("polite_rate", "neutral_rate", "emotion_entropy", "mean_story_probability")

GROUPS: dict[str, tuple[str, ...]] = {"similarity": SIMILARITY, "register": REGISTER}

# What a group asks for, when its members are all too HIGH / too LOW. The three
# single-metric strategies contradict each other on purpose -- semantic says
# "the claim has to change", self_bertscore says "do not change what it claims,
# only how it is worded" -- because each was written to move its own metric
# without disturbing the others. Targeted together that guard is unnecessary,
# and the union is simply: both have to change.
GROUP_STRATEGY: dict[str, Strategy] = {
    "similarity": Strategy(
        metric="similarity",
        high=(
            "This comment says the same thing as its neighbours, in the same "
            "kind of language. Both have to change. Give it a genuinely "
            "different contribution -- a different consequence, a different "
            "condition under which it matters, a different kind of evidence, or "
            "an aside the topic reminded the speaker of -- and type it the way "
            "a different person would: different sentence shape, different "
            "opening, different vocabulary for the same kind of idea. Reusing "
            "the wording with a new point, or making the old point in new "
            "words, is not enough on its own."
        ),
        low=(
            "This comment reads as unrelated to the rest of the thread, in "
            "subject and in phrasing. Bring it back onto what the thread is "
            "about and into the way this community actually writes, while "
            "keeping its own angle."
        ),
        keep="Do not turn it into a summary of the discussion.",
        max_share=0.15,
    ),
    "register": Strategy(
        metric="register",
        high=(
            "The thread is louder and more performed than real replies here. "
            "Keep every point and let the comments settle: less evaluation "
            "attached to each claim, fewer stacked feelings, less narration."
        ),
        low=(
            "This comment is flat -- it states a position with no attitude, no "
            "feeling and nothing lived behind it, and so do its neighbours. "
            "Keep what it claims and let a real person say it: an actual "
            "reaction to what the parent said, or one concrete thing that "
            "happened, in the speaker's own voice. Invent no fact the original "
            "does not already imply."
        ),
        keep="Do not change the stance, and do not add courtesy formulas.",
        max_share=0.15,
    ),
}


def strategy_for(target: str) -> Strategy:
    """A round's target is either a group name or a single metric."""
    return GROUP_STRATEGY[target] if target in GROUP_STRATEGY else STRATEGIES[target]


def metrics_of(target: str) -> tuple[str, ...]:
    return GROUPS.get(target, (target,))
