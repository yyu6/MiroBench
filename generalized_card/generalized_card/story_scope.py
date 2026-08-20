"""What a `no_story` slot is actually barred from doing.

`self_bertscore_mean_f1` was the worst metric in the suite through v97: 0.5185
against a matched real 0.4942, Cliff 0.96. Three hypotheses were measured and
two were rejected before this one survived.

*Not* a topic problem: `semantic_mean_cosine` passes on the same output with
Cliff 0.04.

*Not* a duplication tail. Scoring every unordered pair of the six smallest
threads with the evaluator's own BERTScore and trimming the top of both
distributions leaves the gap exactly where it was -- +0.0163 untrimmed, +0.0154
after dropping the top 20% of pairs on each side. The Planner is clean too: zero
exactly duplicated `semantic_move` values over 532 slots and at most 1.3% of
in-thread plan pairs above 0.35 content-word Jaccard.

*Not* the surface register either. Real comments that differ in typing habits
are only 0.003-0.011 less alike in function-word cosine than ones that share
them, against a generated-vs-real gap of 0.134, and real comments with *uneven*
sentence lengths are slightly *more* alike, not less. See `sentence_rhythm`,
which is kept for the metrics it does move.

What is left is a uniform lexical narrowing. Over the ten matched v97 threads:

                          real     generated
    distinct types        3645          2670
    types / sqrt(tokens) 21.02         15.95
    hapax rate           0.502         0.427
    top-500 type coverage 0.783         0.830

Per-comment type-token ratio at a fixed 30 tokens is *higher* in the generated
text (0.891 against 0.866), so no individual comment is lexically thin. The
thread as a whole draws from a smaller lexicon, which lifts every pair equally.

The narrowing has one dominant cause. 453 of 532 slots are planned `no_story`,
and the instruction v96 gave them bans tense rather than narrative:

    "... no past action, event, before/after change, or then/after pacing."

Measured on those 453 slots against their 532 matched real comments:

                        real   generated(no_story)
    past-tense verb    0.543          0.181
    future / 'll       0.226          0.031
    present perfect    0.167          0.031

`have` appears at 11% of its real rate, `will` at 1%, `to` at 54%. Removing the
past and the future from 85% of a thread removes every past participle, every
perfect auxiliary, and every future construction, and what the model falls back
on is a timeless conditional: `the` at 147% of its real rate, `if` at 225%,
`whether` at 1800%, `matters` at 2900%.

StorySeeker scores narrative *sequence*, not tense, and
`mean_story_probability` already passes with Cliff -0.10 -- generated is
slightly below real, so the constraint has headroom in the safe direction. The
`sequence` arm therefore bars the second event and the pacing that makes a
narrative, and allows one completed past fact or an ordinary future statement,
which is what a non-story Reddit comment is full of.
"""

from __future__ import annotations


# `tense` reproduces v96 and v97: no past action or event at all on a no_story
# slot, which was 85% of them.
NO_STORY_SCOPE = "sequence"

TENSE_BAN_INSTRUCTION = (
    "Do not narrate a sequence of events or repeated attempts. A "
    "first-person slot may state current ownership, preference, or one "
    "present-state observation, but no past action, event, before/after "
    "change, or then/after pacing."
)

SEQUENCE_BAN_INSTRUCTION = (
    "Do not narrate a sequence of events or repeated attempts: no second "
    "event, no then/after pacing, no before/after change, and no story arc. "
    "One completed thing you did or saw, stated in a single clause, is not a "
    "story and neither is saying what will happen; use ordinary past and "
    "future tense where the point needs it."
)


def set_no_story_scope(mode: str) -> bool:
    """Select the no-story arm and return whether the narrower scope is active."""

    global NO_STORY_SCOPE
    NO_STORY_SCOPE = (
        "tense" if str(mode or "sequence").strip().lower() == "tense" else "sequence"
    )
    return NO_STORY_SCOPE == "sequence"


def no_story_instruction() -> str:
    """Render what a `no_story` slot is barred from, under the active arm."""

    return (
        SEQUENCE_BAN_INSTRUCTION
        if NO_STORY_SCOPE == "sequence"
        else TENSE_BAN_INSTRUCTION
    )
