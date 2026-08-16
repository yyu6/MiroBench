"""The grounding rules for the Writer, in one place.

Before this module the fact ban was written in eight separate places across
`prompts.py` and the pinned core vocabulary, and the copies disagreed with each
other. Measured over the 522 rendered slots of run v75:

    443 slots (84.9%)  "Do not invent products, specifications, prices,
                        measurements, dates, outcomes, policies, links, or
                        personal experiences."
    249 slots (47.7%)  "Equipment you may claim as your own, if this turn
                        reports personal experience: ..."
    170 slots (32.6%)  both of the above, in the same prompt

A slot cannot satisfy both. The measured result is 0.08 specifications per
comment against 0.55 in the matched real threads, and 6.6 novel brand or model
tokens per generated thread against 47.3 real.

One concept, one definition. A product discussion contains two different kinds
of fact and they need opposite rules:

    seed facts  claims about the product the thread is about. These stay
                grounded in what is visible. A wrong specification here is an
                error a reader can catch, and it also pulls the whole thread
                into one factual space -- the mechanism that made `domain_claim`
                a regression in v71.

    own facts   the speaker's own equipment, settings, prices, dates, places and
                outcomes. Nobody can check these against the world, and they are
                where the real texture lives. Banning them is what leaves "I've
                done that in a packed room before" as the compliant output.

The license is gated on `first_person_experience_slot`, the same predicate that
already gates the equipment block. Licensing own facts on a slot that another
rule bars from a first-person frame would just replace one contradiction with
another.

`--own-fact-license off` restores the single blanket ban and reproduces policy
v75 byte for byte.
"""

from __future__ import annotations

from typing import Any


LICENSE_OFF = "off"
LICENSE_OWN = "own"
LICENSE_NAMED = "named"
LICENSE_MODES = (LICENSE_OFF, LICENSE_OWN, LICENSE_NAMED)

# `own` is kept as a reproducible arm, not as a recommendation. Run v76b measured
# it on seed 8 and it moved concreteness the wrong way: 0.05 -> 0.02 specs per
# comment against a real 0.54, and 0.083 -> 0.024 on the licensed slots
# themselves. Two measurements explain why.
#
#   1. The gate was wrong. Across the ten matched real threads, 78 of 114
#      spec-carrying comments (68%) contain no first-person frame at all.
#      Concreteness is not a property of personal narrative; it is how someone
#      who deals with the subject talks about it.
#   2. The wording made the binding constraint sharper. Replacing a vague
#      blanket ban with an explicit "about the product under discussion, name
#      only what is visible above" put a clearer prohibition on exactly the
#      class of detail that dominates real comments.
#
# `named` is the correction, and it is stated in domain-neutral terms because
# what generalizes is not "specifications". Measured per thread, real against
# generated (v75), the two signals that hold on **all ten** threads are:
#
#     quantities per comment    real 12.3x generated    10/10 threads
#     proper nouns per comment  real  1.85x generated   10/10 threads
#
# while specification-shaped tokens hold on 9 of 10 and vary from 0% of comments
# (seed 1) to 64% (seed 5) depending on whether the thread is technical at all.
# So the rule licenses naming and quantifying, never "give specifications".


def license_mode(backend: Any) -> str:
    """Return the configured concreteness license mode."""

    value = str(getattr(backend, "GENERALIZED_OWN_FACT_LICENSE", LICENSE_OFF) or "")
    value = value.strip().lower()
    return value if value in LICENSE_MODES else LICENSE_OFF


def first_person_experience_slot(task: Any) -> bool:
    """Return whether this slot's plan already licenses personal experience.

    Still the gate for the equipment block, whose permission is specifically
    about the speaker's own things. It is no longer the gate for concreteness.
    """

    if bool(getattr(task, "allow_first_person_frame", False)):
        return True
    if str(getattr(task, "evidence_mode", "") or "") == "firsthand_experience":
        return True
    if str(getattr(task, "story_mode", "no_story") or "no_story") != "no_story":
        return True
    return str(getattr(task, "payload_type", "") or "") in {
        "personal_story",
        "fragment_datapoint",
    }


def substantive_slot(task: Any) -> bool:
    """Return whether this slot has room to name or quantify anything.

    A forced micro reaction has no room for a detail and should not be told to
    find one; that is how `missing_concrete_anchor` fired 84 times against slots
    that could not satisfy it.
    """

    try:
        words = int(getattr(task, "real_word_count", 0) or 0)
    except (TypeError, ValueError):
        words = 0
    return words >= 25 and str(
        getattr(task, "length_bucket", "") or ""
    ) not in {"micro", "short"}


def slot_license(backend: Any, task: Any) -> str:
    """Return the license this particular slot carries.

    One resolver, so a slot can never be handed one mode's permission and
    another mode's rule text.
    """

    mode = license_mode(backend)
    if mode == LICENSE_OWN and first_person_experience_slot(task):
        return LICENSE_OWN
    if mode == LICENSE_NAMED and substantive_slot(task):
        return LICENSE_NAMED
    return LICENSE_OFF


def licensed_for(backend: Any, task: Any) -> bool:
    """Return whether this slot may state concrete detail beyond the visible."""

    return slot_license(backend, task) != LICENSE_OFF


# --- the rules -------------------------------------------------------------


OWN_ENTITY_RULE = (
    "Two kinds of fact here. About the product this thread is about, name only "
    "what is visible above. About your own gear and your own history, be "
    "specific: what you owned, what you shot or set it to, what you paid, when "
    "it was, where you were, and how it turned out."
)

# Deliberately free of any domain vocabulary. "Gear", "shot", "specification"
# describe one domain's texture; naming and quantifying describe all of them,
# and they are the two signals that separate real from generated on every one of
# the ten matched threads.
NAMED_ENTITY_RULE = (
    "Talk like someone who actually deals with this: name the specific things "
    "you mean and give the amounts, and do not retreat into a general "
    "description when a particular one is what you would really say. Stay "
    "consistent with anything the discussion above already establishes about "
    "the thing being discussed, and do not repeat a name or figure another "
    "comment here has already used."
)

# The two Writer paths phrase the unlicensed ban differently. Both are kept
# verbatim so `off` is a true ablation control on either path.
UNLICENSED_ENTITY_RULES = {
    "focused": "Name a product, model, or number only if it is visible above.",
    "full": (
        "Named entities and numbers may appear only when visible in the "
        "discussion, in the visible factual anchors, or, for your own equipment, "
        "in the equipment list above."
    ),
}


def entity_naming_rule(*, mode: str = LICENSE_OFF, variant: str = "focused") -> str:
    """The Writer's closing rule on which entities and numbers it may name."""

    if mode == LICENSE_OWN:
        return OWN_ENTITY_RULE
    if mode == LICENSE_NAMED:
        return NAMED_ENTITY_RULE
    return UNLICENSED_ENTITY_RULES.get(variant, UNLICENSED_ENTITY_RULES["focused"])


def story_fact_rule(task: Any, *, has_domain_claim: bool, mode: str = LICENSE_OFF) -> str:
    """Replace the single fact-safety rule with a mode-appropriate rule."""

    story_slot = str(getattr(task, "story_mode", "") or "") not in {"", "no_story"}

    if mode == LICENSE_OFF:
        return _unlicensed_rule(story_slot=story_slot, has_domain_claim=has_domain_claim)

    if mode == LICENSE_NAMED:
        # One prohibition, and it is the one with a real failure behind it: v71
        # injected a single planned domain fact into 508 of 522 comments and
        # produced 157 extra semantic-overlap flags. Real concreteness is the
        # opposite shape -- 104 quantity tokens over 44 distinct values in one
        # matched thread -- so the guard is against convergence, not detail.
        claim_clause = (
            " The one fact assigned to you above is the exception."
            if has_domain_claim
            else ""
        )
        if story_slot:
            return (
                "This is a story slot: say what happened, in order, and let it "
                "reach an outcome. Name the particulars as you would in "
                "conversation. Do not contradict what the discussion above "
                f"already establishes.{claim_clause}"
            )
        return (
            "Be particular rather than general, and do not contradict what the "
            f"discussion above already establishes.{claim_clause}"
        )

    seed_clause = (
        "Do not state a specification, price, measurement, or outcome for the "
        "product under discussion unless it is visible above."
    )
    if has_domain_claim:
        seed_clause = f"Beyond the domain fact assigned above, {seed_clause[0].lower()}{seed_clause[1:]}"

    if story_slot:
        return (
            "This is a story slot: tell what happened to you, in order -- where "
            "you were, what you were using, what changed or went wrong, and what "
            "came of it. Your own equipment, settings, prices, dates, places and "
            "outcomes are yours to state plainly; a story that stops before its "
            f"consequence is not a story. {seed_clause}"
        )
    return (
        "Your own equipment and your own experience with it are yours to state "
        f"concretely, with the specifics that make them real. {seed_clause}"
    )


def equipment_closing_clause(*, mode: str = LICENSE_OFF) -> str:
    """The clause that closes the `own equipment` block.

    This block is specifically about the speaker's own things, so it stays
    first-person under either license.
    """

    if mode == LICENSE_OFF:
        return (
            "Name at most one, as gear you have used yourself. Do not attribute "
            "it to the post, to another commenter, or to the discussion, and do "
            "not invent a specification, price, measurement, or test result for "
            "it."
        )
    if mode == LICENSE_NAMED:
        return (
            "These are yours. Say what you use them for and how they have held "
            "up, with the particulars. Do not attribute them to the post, to "
            "another commenter, or to the discussion."
        )
    return (
        "This is your own gear. Say what you used it for, how you had it set up, "
        "what you paid, and how it held up -- those specifics are yours to state. "
        "Do not attribute it to the post, to another commenter, or to the "
        "discussion."
    )


def metric_guidance_story_line(*, mode: str = LICENSE_OFF) -> str:
    """The story line of the low-info path's core metric guidance block."""

    if mode == LICENSE_OFF:
        return (
            "- Story prevalence: use a story only when the sampled story mode or "
            "role requires it. Keep synthetic personal context qualitative and "
            "never invent measured facts or definite outcomes."
        )
    if mode == LICENSE_NAMED:
        return (
            "- Story prevalence: use a story only when the sampled story mode or "
            "role requires it. When one is called for, make it particular rather "
            "than a general summary."
        )
    return (
        "- Story prevalence: use a story only when the sampled story mode or "
        "role requires it. When one is called for, give it the concrete "
        "specifics of your own kit and history rather than a qualitative "
        "summary; keep invented facts about the product under discussion out."
    )


def system_prompt_fact_sentence(*, mode: str = LICENSE_OFF) -> str:
    """The sentence appended to the writer system prompt under the license.

    The core system prompt is pinned in `engine/vocabulary.py` and rewritten at
    configure time; this is added there rather than edited into the pinned
    string, so `off` leaves the core contract untouched.
    """

    if mode == LICENSE_NAMED:
        return (
            " Name the particular things you mean and give amounts, the way "
            "someone who deals with this every day would, staying consistent "
            "with whatever the discussion already establishes."
        )
    if mode == LICENSE_OFF:
        return ""
    return (
        " Facts about your own equipment and your own history -- what you owned, "
        "your settings, what you paid, when, where, and how it turned out -- are "
        "yours to state concretely; only the product under discussion has to stay "
        "within what is visible."
    )


# --- the unlicensed rule, preserved verbatim -------------------------------


def _unlicensed_rule(*, story_slot: bool, has_domain_claim: bool) -> str:
    """Reproduce policy v75's `_story_fact_safety_rule` exactly.

    Kept byte-identical so `--own-fact-license off` is a true ablation control
    rather than an approximation of the previous release.
    """

    if not story_slot:
        if has_domain_claim:
            # A blanket ban here would cancel the planned domain fact sitting in
            # the same prompt, which is the contradiction that made an earlier
            # version's tone control unrealizable.
            return (
                "Beyond the domain fact assigned above, do not invent products, "
                "specifications, prices, measurements, dates, outcomes, policies, "
                "links, or personal experiences. Never state a specification, "
                "price, or measurement for the post's own equipment unless it is "
                "visible in the discussion."
            )
        return (
            "Do not invent products, specifications, prices, measurements, dates, outcomes, policies, links, or personal experiences."
        )
    return (
        "This is a synthetic story slot: make the temporal sequence legible with "
        "an ordinary first-person situation or action followed by an observation "
        "or reaction around the existing local point. You may synthesize that "
        "non-verifiable personal sequence, but do not invent a product, "
        "specification, price, measurement, date, policy, link, diagnosis, or "
        "externally checkable outcome."
    )
