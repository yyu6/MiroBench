from __future__ import annotations

from collections import Counter
import re
from typing import Any

from .actor_conditioning import MODE_DOMAIN_DERIVED, actor_for_task, render_actor_state
from .branch_routing import (
    parent_slot_schedule,
    render_branch_requirements,
    root_branch_schedule,
)
from .domain import DomainConfig
from .domain_claim import (
    claim_for_task,
    domain_claim_mode,
    planner_claims_enabled,
    render_domain_claim_rule,
    render_selective_claim_schedule,
    selective_claim_slots,
)
from .domain_profile import render_profile_for_planner
from .entity_inventory import slot_equipment_options
from .entity_spread import slot_referent_block
from .reference_link import (
    draw_reference_link,
    draw_reference_links,
    reference_link_enabled,
    reference_link_offer,
    reference_links_offer,
)
from .generation_distribution import (
    TONE_CLASSES,
    TONE_DEFINITIONS,
    render_planner_distribution_target,
)
from .length_policy import local_move_scope_guidance, soft_length_guidance
from .opener_profile import OPENER_INSTRUCTIONS
from .evaluative_register import active_evaluative_guidance
from .opening_move import active_opening_guidance, forbidden_opening_tokens
from .branch_routing import BRANCH_DICTATION_MODE as _BDM  # noqa: F401
from .planning_quality import isolation_quota_block, outsider_quota_block
from .plan_vocabulary import (
    abstraction_block,
    real_position_count,
    content_angle_schema_hint,
    domain_intent_schema_hint,
    open_vocabulary,
    perspective_schema_hint,
)


def slot_grid_mode() -> str:
    """Read the flag at call time, not at import, or it freezes at `full`."""
    from . import planner_distribution

    return planner_distribution.SLOT_GRID_MODE


# Each of these replaces a rule that, under --slot-grid free, either points at
# machinery that is no longer delivered or contradicts the brief. A dangling
# rule is not harmless: "every S# not explicitly assigned a story_mode in the
# slot schedule is fixed to no_story" reads, with the schedule withheld, as
# every slot being no_story, which would destroy a metric that currently passes.
def _free(text: str, default: str) -> str:
    return text if slot_grid_mode() == "free" else default


def _affect_role_schema() -> str:
    return _free(
        "one GoEmotions label, chosen by you for this slot",
        "one GoEmotions label listed in the frozen target",
    )


def _local_topic_schema() -> str:
    return _free(
        "what this comment is actually about. It may be grounded in the seed or "
        "the parent, and it may be something neither raised -- an incidental "
        "detail, an aside, a remark aimed at another commenter -- because real "
        "comments frequently are",
        "seed-grounded topic or generic parent-local topic",
    )


def _detail_focus_schema() -> str:
    return _free(
        "the detail this comment fixes on. Usually seed-visible; it may instead "
        "be a detail only this commenter would care about",
        "seed-visible detail to use, or a generic detail type if no fact is visible",
    )


def _template_contract_rule() -> str:
    return _free(
        "- Nothing is prescribed for any S#: no labels, no tone or affect "
        "counts, no story schedule, no branch goal, perspective or exclusion, "
        "no opener type. Every field is yours. Keep the set coherent within a "
        "slot, and let the thread's tone and affect fall where the conversation "
        "puts them, including at the extremes.",
        "- A label explicitly listed for an S# is a fixed template contract. Select a\n"
        "  compatible role, payload, stance, and evidence mode in this first plan; do\n"
        "  not expect a later stage to replace it. For a field absent from the S# list,\n"
        "  choose a natural compatible label while satisfying the whole-thread target.",
    )


def _remaining_counts_rule() -> str:
    return _free(
        "",
        "- Satisfy only the remaining story, tone_class, and affect_role counts that\n"
        "  have compatible slots. If the template says a label is unavailable, omit it\n"
        "  rather than attaching it to an unrelated substantive claim. Choose labels\n"
        "  that fit each comment's discourse role; do not change its claim to fit a\n"
        "  label.",
    )


def _story_schedule_rule() -> str:
    return _free(
        "- Decide per slot whether a comment tells a personal story, at roughly "
        "the rate the matched thread shows. Do not add one to make a plan sound "
        "richer, and do not suppress one the slot naturally carries.",
        "- Every S# not explicitly assigned a story_mode in the slot schedule is fixed\n"
        "  to no_story. Do not create an extra story to make a local plan sound richer.",
    )


def _branch_contract_rules() -> str:
    return _free(
        "",
        "- Treat the displayed ``branch_goal`` as a semantic contract, not background\n"
        "  inspiration: ``semantic_move``, ``domain_intent``, and\n"
        "  ``decision_boundary`` must directly develop that goal. A comment that could\n"
        "  instead belong to another displayed B# is invalid.\n"
        "- ``required_perspective`` and ``branch_exclusion`` are also fixed controls.\n"
        "  The former selects the reasoning lens; the latter forbids the adjacent\n"
        "  decision axis already owned by another root branch.\n"
        "- ``owned_decision_subject`` is a fixed branch contract. Establish only that\n"
        "  condition for this chain; do not substitute one of the listed forbidden\n"
        "  subjects just because it is salient in the seed.",
    )


def _branch_axis_rule() -> str:
    return _free(
        "- The branch route is fixed as SHAPE only: keep each slot in its "
        "assigned branch and let replies inherit their parent's. A branch is a "
        "position in the thread, not a topic it owns -- two roots in different "
        "branches need not be discussing related things at all.",
        "- The required branch route is also fixed. Each root discussion chain owns a\n"
        "  distinct decision axis; replies inherit their parent chain's branch. Do not\n"
        "  switch to another branch merely because its topic is easier to write.",
    )


def _register_pairing_rule() -> str:
    return _free(
        "  A register is how a slot speaks, not what it concludes: a ``polite`` "
        "slot can disagree courteously and an ``impolite`` one can be "
        "enthusiastic about something. What is invalid is a move whose manner "
        "contradicts its register.",
        "  Never pair ``polite`` with ``correction_caveat`` or a disagreeing stance, and\n"
        "  never pair ``impolite`` with gratitude or endorsement.",
    )


def _claim_reuse_rule() -> str:
    return _free(
        "- Reuse an earlier claim only for a direct reply relation that needs "
        "it. Otherwise move somewhere else entirely -- a different branch, "
        "detail, stance or social function, and it does not have to be "
        "seed-grounded.",
        "- Reuse an earlier claim only for a direct reply relation that needs it."
        " Otherwise move to a different seed-grounded branch, detail, stance, or"
        " social function.",
    )


def _sec(letter: str, title: str) -> str:
    """Number the blocks between the brief and the rules.

    Those blocks arrive as an unlabelled pile -- lenses, a reference bank,
    objectives, the post, targets, registers, shape, slots, schema -- and a
    reader cannot tell which are inputs, which are constraints, and which is the
    thing being planned. Lettered so they cannot be confused with the brief's
    numbered sections.
    """

    return f"\n--- {letter}. {title} ---\n" if slot_grid_mode() == "free" else ""


def _rule_group(title: str) -> str:
    """A heading between rule groups, so 34 flat bullets read as seven topics.

    Emitted only under `free`; the default list keeps its exact original shape.
    """

    return f"\n{title}\n" if slot_grid_mode() == "free" else ""


def _opener_rules() -> str:
    """No opener_type is assigned when the grid is withheld.

    The measurement the default rule rests on stays true and stays useful --
    the Writer opens 23% of comments with a bare agreement token against 4% in
    the real thread -- so under `free` it becomes a tendency to counter using
    the real comment's own opening, rather than a contract with a value that no
    longer arrives.
    """

    return _free(
        "- No opener type is assigned. Take each slot's entry grammar from how"
        " its own real comment opens, and vary it across nearby slots. One"
        " tendency to counter: left alone, the Writer opens about 23% of"
        " comments with a bare agreement token against 4% in real threads,"
        " while under-producing content-first and first-person entries.\n"
        "- Whatever entry you choose, the rest of the row has to be able to"
        " start that way. An opening question needs"
        " ``comment_function=question_followup`` and"
        " ``payload_type=narrow_question``; an imperative needs a recommending"
        " row; an address needs a row that speaks to the person it replies to."
        " Measured over 520 slots, openers assigned against an incompatible row"
        " were realized 0 of 23 times for questions and 0 of 10 for"
        " imperatives. Choose the function and payload so the entry is"
        " writable.",
        "- ``opener_type`` is a fixed grammatical contract measured from real threads of\n"
        "  this domain. Write ``opening_style`` as a concrete route that begins the way\n"
        "  that type requires. Measured on one matched pair, the Writer opened 23% of\n"
        "  comments with a bare agreement token against 4% in the real thread while\n"
        "  under-producing content-first and first-person entries, so this is the entry\n"
        "  grammar, not a suggestion.\n"
        "- The rest of the row has to be able to start that way. An assigned\n"
        "  ``opener_type`` of ``question`` needs ``comment_function=question_followup``\n"
        "  and ``payload_type=narrow_question``; ``imperative`` needs a recommending row\n"
        "  (``payload_type=advice`` with ``comment_function=recommendation_advice``);\n"
        "  ``address`` needs a row that speaks to the person it replies to. Measured over\n"
        "  520 slots, an assigned opener was realized 43.8% of the time overall but 0 of\n"
        "  23 times for ``question`` and 0 of 10 for ``imperative``, because the rest of\n"
        "  the row made that opening ungrammatical. Choose the row's function and payload\n"
        "  so the assigned entry is writable, rather than keeping both and losing one.",
    )


def _development_plan_trigger() -> str:
    return _free(
        "For a slot long enough to carry several connected beats,",
        "For a slot\n  whose schedule says ``development_plan`` is required,",
    )


def _gratitude_rule_opening() -> str:
    return _free(
        "If you give a slot ``affect_role=gratitude`` or ``affect_role=relief``,",
        "If an S# is assigned ``affect_role=gratitude`` or ``affect_role=relief`` by\n"
        "  the template schedule,",
    )


def _reference_row_framing() -> str:
    """The R# rows are a bank, not a per-slot partner.

    The default text says to pair each displayed slot with a different R# row
    in order, which is a one-to-one mapping onto comments from an unrelated
    thread -- the same slot-filling the brief exists to stop.
    """

    if slot_grid_mode() == "free":
        return (
            "REFERENCE COMMENTS FROM OTHER THREADS (a bank, browse it):\n"
            "The R# rows are real comments from evaluation-excluded threads,"
            " shown so you can see how people in this community actually write."
            " They are not partners for your slots and there is no order to"
            " follow. Use one when it genuinely illuminates a slot; ignore the"
            " rest. Never reuse their wording, facts or named entities."
        )
    return (
        "NON-TEST REFERENCE COMMENTS FOR SEMANTIC ABSTRACTION:\n"
        "The R# rows come from evaluation-excluded threads. Pair each displayed"
        " S# with\na different R# row in order when the viewpoint pattern fits."
        " Abstract the tiny\nsemantic/discourse move and adapt it to the visible"
        " seed or parent."
    )


def _distribution_heading() -> str:
    if slot_grid_mode() == "free":
        return "Whole-thread label counts:"
    return "Frozen whole-thread distribution target:"


def _slot_label_heading() -> str:
    if slot_grid_mode() == "free":
        return "Per-slot labels:"
    return "Required template-derived labels for these displayed slots:"


def _planner_opening() -> str:
    """The first sentence, which sets the frame everything after it is read in.

    The default opens `Assign per-comment semantic and social controls to
    matched-real structural slots`, which defines the task as filling a form
    before any other instruction is read. A brief arriving fifty lines later
    that says to invent instead does not undo it: a probe run under the full
    brief still produced nine plans about one shirt, two of them identical.
    """

    if slot_grid_mode() == "free":
        return (
            "Plan a Reddit discussion: decide what each person in this thread "
            "is doing and why they bothered to type.\n\n"
            "You are not assigning labels to slots. You are deciding what kind "
            "of conversation this is and then populating it. Read the brief "
            "below before anything else in this prompt, including the lenses "
            "and the schema -- those are vocabulary for recording your "
            "decisions, not the decisions."
        )
    return (
        "Assign per-comment semantic and social controls to matched-real "
        "structural slots."
    )


def _brief_at_top() -> str:
    """Deliver the brief first, or the frame is set before it is read."""

    return f"\n{_planner_orientation_block()}\n" if slot_grid_mode() == "free" else ""


def _perspective_field_rule() -> str:
    if open_vocabulary():
        return (
            "Write ``perspective_id`` as the lens you named for this slot, in "
            "your own words. Never ``seed_local``, never a P##, never a branch "
            "or slot identifier."
        )
    return (
        "Use a frozen ``perspective_id`` only when it fits the visible seed or "
        "parent; otherwise use ``seed_local``."
    )


def _real_position_count(backend: Any, rows: list[dict[str, Any]]) -> int:
    """Count the matched real thread's distinct semantic positions.

    The embedding model the plan-quality gate already loads is reused, so this
    adds no model and no dependency. Returns 0 when embeddings are off, and the
    sentence that consumes it is then omitted rather than falling back to a
    guess -- an invented number is what this replaces.
    """

    index = getattr(backend, "GENERALIZED_PLAN_SEMANTIC_INDEX", None)
    if index is None:
        return 0
    return real_position_count(
        [row.get("body") for row in rows or ()], index.encode_texts
    )


def _lens_note() -> str:
    """The paragraph under section A's lens list."""

    if open_vocabulary():
        return (
            "Each row states how a comment REASONS about the local topic -- not "
            "the topic, entity, event, or claim itself. That distinction is the "
            "only thing to carry over from these rows."
        )
    return (
        "Each P## states how a comment reasons about the local topic. It is not "
        "the topic,\nentity, product, feature, event, or claim itself. Derive the "
        "actual local move\nfrom the visible seed/parent and the non-test "
        "reference-comment pattern below."
    )


def _lens_framing() -> str:
    """Demote the lenses from frame to vocabulary when the Planner is free."""

    if open_vocabulary():
        # Under `open` the twelve are not the vocabulary, they are an example of
        # what a lens is. Naming them "available" is what produced 45% seed_local
        # on a domain none of them fit (G205).
        return (
            "Twelve lenses from a product-shopping discussion, shown only so the "
            "SHAPE of a lens is unambiguous. This discussion is not about buying "
            "anything, so most of these will not apply and you are expected to "
            "name your own instead (section B says how). Do not treat this as a "
            "menu and do not use a P## label in your output:"
        )
    if slot_grid_mode() == "free":
        return (
            "Available decision lenses (a vocabulary, not a set of angles to "
            "cover; `seed_local` is always available and is the right answer "
            "whenever none of these honestly fits):"
        )
    return "Frozen domain-neutral decision lenses:"


def _choose_from_objective() -> str:
    """Name the inputs that actually carry direction under the active flags.

    The default sentence sends the Planner to the branch controls, and under
    `--branch-dictation structural` those have been emptied of everything except
    shape. Leaving it unchanged points the Planner at an input that no longer
    says anything.
    """

    if slot_grid_mode() == "free":
        return (
            "Choose discourse function, stance, story/no-story role, tone, "
            "affect and domain perspective yourself, from the kind of "
            "conversation the matched thread turns out to be. The branch "
            "controls carry only shape now, and no per-slot labels are supplied."
        )
    return (
        "Choose discourse function, stance, story/no-story role, and domain "
        "perspective from the visible seed, branch controls, and frozen non-test "
        "domain profile."
    )


def _planner_orientation_block() -> str:
    """One brief, read as a document rather than grown by appending to it.

    Three faults that came from patching it in place, all now fixed here rather
    than answered with another paragraph.

    It contradicted itself on `seed_local`: section 1 said forcing an ill-fitting
    lens was worse than declining one, and axis A said declining was worse. Both
    were written to fix a different observed failure -- the first when the lens
    was being overwritten downstream, the second when freeing it sent 78-100% of
    slots to `seed_local` and within-thread similarity went to 83.8% above real.
    The position that covers both is stated once, in axis A, where spread is the
    subject.

    Its axes had stopped sharing a shape, A carrying five sub-entries and C
    carrying a pre-return check that belongs in the procedure. Each axis is four
    lines: what it measures, what real threads do, how we get it wrong, which
    fields carry it.

    And section 2 said the same thing twice, the second time forward-referencing
    an axis that had not been introduced.

    Reference figures come from this domain's reference corpus, the threads
    excluded from the evaluation pool. No evaluation-set statistic or p-value
    appears in this prompt.
    """

    return "\n".join([
        "================ PLANNER BRIEF ================",
        "",
        "1. WHERE YOU SIT",
        "   a. A structural pass has copied one real thread's skeleton: how many"
        " comments, at what depth, under which parent, at roughly what length."
        " That is fixed. Keep it -- it is why our threads already match real ones"
        " on length variation, reply depth and virality shape.",
        "   b. You see eight slots at a time, in order, with your own earlier"
        " batches for this thread visible. You emit private controls per slot and"
        " never write comment text.",
        "   c. A validator may return your batch up to three times when two slots"
        " plan the same move or one perspective dominates. Repairs cost quality.",
        "   d. The Writer is a DIFFERENT model, called once per slot. It sees"
        " your controls for that slot, the post, and its parent comment --"
        " nothing else. It cannot see that two slots are converging, and it"
        " cannot add variety you did not plan. Its view of the post is degraded"
        " on purpose: truncated for ~42% of slots, reordered for ~32%. Anchor a"
        " plan on something it will plausibly still have.",
        "   e. Withheld from you on purpose, because these decisions are now"
        " yours: the branch routes carry only shape -- which branch, which"
        " parent, which siblings -- and name no subject, perspective or"
        " prohibition; the per-slot grid that used to fix tone, affect, story"
        " mode, opener and surface form; and the whole-thread label counts.",
        "",
        "2. WHAT YOU ARE MAKING",
        "   a. Not a set of complementary angles that jointly cover the topic."
        " That is a briefing document, and it is what we currently produce.",
        "   b. A group of people who happened to open the same post and each had"
        " their own reason to type. Some engage the news. Some react to one"
        " incidental detail. Some are talking to another commenter. Some are"
        " making a joke that would make no sense to anyone who arrived later."
        " Several are barely participating.",
        "   c. You are not filling in a template from the real comments. The real"
        " thread tells you what KIND of conversation this was -- how far apart"
        " its people were, how hard they were pushing, how little most of them"
        " bothered. Take that and invent freely: you may plan subjects the post"
        " never raised, because real commenters do. Freedom means the slots may"
        " go anywhere, not that they may all go to the same place.",
        "   d. Do not reproduce their words, their facts, or their named"
        " entities. What transfers is the character of the conversation.",
        "",
        "3. THE FIVE AXES",
        "   Each is at once a statistic the finished thread is scored on, a thing"
        " to read the real thread for, and a group of fields you fill. For each:"
        " what it measures / what real threads of this kind are like / the way we"
        " get it wrong / the fields that carry it.",
        "",
        "   A. SEMANTIC DIVERSITY -- average likeness of every pair of comments.",
        "      real: roughly a quarter of comments have nothing to do with any"
        " other comment in their own thread. Real people in one thread are"
        " frequently not discussing the same thing at all.",
        "      our tendency: to make every slot another evaluative angle on the"
        " post's subject, so comments differ in wording while meaning the same"
        " thing. This is our largest gap, and it has been reached from both"
        " directions -- once by giving every slot a prescribed subject, once by"
        " prescribing nothing and letting every slot default to the same lens.",
        "      carry: local_topic and detail_focus above all, because two slots"
        " are far apart when they are ABOUT different things and near when they"
        " are not, whatever labels they wear. `perspective_id` is the lever that"
        " moves them: pairs sharing a perspective realize 0.28 similarity, pairs"
        " on different perspectives 0.1953, against a real 0.18. So vary it"
        " deliberately. `seed_local` is not a lens but the absence of one -- right"
        " for the occasional slot that genuinely fits no P##, and a failure state"
        " if most of the thread lands there, since a thread on one lens has no"
        " spread at all.",
        "",
        "   B. WORDING -- overlapping word sequences between comments.",
        "      real: register is shared, phrasing is not. Fragments, missing"
        " punctuation, slang, community in-jokes, capitals for emphasis, blunt"
        " openings, trailing dots.",
        "      our tendency: none -- we already sit slightly below real here, so"
        " repeated wording is not the problem. Different words for the same"
        " meaning is the problem, and that is axis A.",
        "      carry: opening_style, surface_texture, utterance_mode.",
        "",
        "   C. EMOTION -- how many distinct emotions appear across the thread.",
        "      real: delight, contempt, boredom and anger can sit in one thread"
        " with nothing in between mediating them.",
        "      our tendency: to collapse onto one setting -- usually everything"
        " piled in the middle, but a thread that comes out entirely `impolite` is"
        " the same defect wearing different clothes.",
        "      carry: affect_role, tone_class, voice, stance, decided per slot"
        " and never from the thread's average.",
        "",
        "   D. PLAINNESS -- how much of the thread does no argumentative work.",
        "      real: about half of all comments run to eighteen words or fewer,"
        " about a sixth to five or fewer. Bare agreements, one-word reactions,"
        " jokes carrying no argument, replies that only address another"
        " commenter. Explicit gratitude near 1%, stock social filler near 5% --"
        " far less than a polite forum.",
        "      our tendency: to plan substance into slots that should carry none."
        " Each empty slot looks like a wasted one. It is not: these are most of"
        " what keeps a thread from reading like an essay in parts.",
        "      carry: payload_type, comment_function, utterance_mode.",
        "",
        "   E. STORYTELLING -- how much first-person personal narrative appears.",
        "      real: asides are often half a story -- one beat, trailing off, or"
        " a detail with no point attached.",
        "      our tendency: none material; we are close to real. Do not disturb"
        " it while working on the others.",
        "      carry: story_mode, evidence_mode, speaker_role.",
        "",
        "   Also already matching, because the skeleton is copied: length"
        " variation, reply depth, virality shape. Preserve them by keeping each"
        " slot's depth, parent and rough length.",
        "",
        "4. ONE ILLUSTRATION",
        "   From a real thread in this domain, on a post about a film producer"
        " indicted for a $100M fraud. Nine comments:",
        "      - a long quote pasted from the article",
        "      - a joke about an unrelated actor being innocent",
        "      - a three-word reply that means nothing without its parent",
        "      - a pedantic correction: that is not a ponzi scheme, just fraud",
        "      - a pun on the producer's surname",
        "      - a weary one-liner about industry bookkeeping",
        "      - a movie quote riffed into the situation",
        "      - a joke connected to nothing: 'a drum and a cymbal fell down a"
        " cliff'",
        "      - another block quote from the indictment",
        "   Three of the nine engage the news. The rest are people amusing"
        " themselves and each other. This is one instance of a general point, not"
        " a set of categories to reuse -- the next thread will scatter in"
        " completely different directions, and some threads genuinely do stay on"
        " one subject throughout. Read the one in front of you.",
        "",
        "5. PROCEDURE",
        "   a. Read the whole matched thread first and decide what kind of"
        " conversation it was, on axes A to E. Then plan.",
        "   b. Cases that are easy to misjudge, again as illustration rather"
        " than a list to work through:",
        "      - a deep reply usually answers the comment above it, about"
        " something the post never raised;",
        "      - two slots under one parent often disagree with each other or"
        " talk past each other, rather than each adding a tidy increment;",
        "      - a slot addressed to another commenter contributes nothing to the"
        " post's subject, which is exactly why it belongs;",
        "      - a real comment that is only an image, GIF or link: plan the"
        " shortest, least substantial text the slot can carry.",
        "   c. Before returning, look at the batch as a set. Are the"
        " `local_topic` values about different things? Is more than one"
        " `perspective_id` in use? Do the tone and affect labels span a range,"
        " rather than repeating one value? A batch that fails those is the"
        " failure in axes A and C, however good each row looks alone.",
        "   d. Do not manufacture scatter a thread does not have, and do not"
        " treat any figure above as a quota. They describe what threads of this"
        " kind are like. The thread in front of you is the authority.",
        "===============================================",
    ])


def branch_dictation_mode() -> str:
    """Read the flag at call time; importing the value would freeze it at off."""
    from . import branch_routing

    return branch_routing.BRANCH_DICTATION_MODE
from .long_form_planning import (
    development_plan_word_threshold,
    expected_development_beats,
)
from .planner_distribution import render_slot_distribution_schedule
from .persona_bridge import persona_marker_for_task
from .tone_donor import (
    donor_sentence_offer,
    draw_donor_sentence,
    require_donor_inventory,
)
from .reply_planning import (
    SYNTHETIC_STORY_PLANNER_BOUNDARY,
    is_direct_reply_batch,
    render_direct_reply_planner_prompt,
)
from .closing_move import active_closing_guidance
from .register_realization import active_register_guidance
from .semantic_realization import (
    opening_route_counts,
    recurring_function_phrases,
    repeated_phrase_counts,
    reused_sentence_routes,
    semantic_contract_values,
    semantic_coverage_entries,
    short_utterance_exclusions,
    turn_settles_a_question,
    used_sentence_routes,
)
from .sentence_rhythm import active_rhythm_guidance
from .story_scope import no_story_instruction
from .surface_contract import substantive_surface_slot, surface_only_label
from .conversation_reference import (
    render_conversation_fragments,
    reply_material_enabled,
    select_conversation_fragments,
)
from .viewpoint_bank import (
    reference_viewpoint_window,
    render_reference_rows,
    render_reference_viewpoints,
)
from .writer_grounding import (
    entity_naming_rule,
    equipment_closing_clause,
    first_person_experience_slot,
    story_fact_rule,
)
from .writer_grounding import slot_license as writer_grounding_mode


GENERIC_CLAIM_FAMILIES = (
    "direct_answer",
    "product_comparison",
    "recommendation",
    "technical_explanation",
    "troubleshooting",
    "firsthand_datapoint",
    "price_value",
    "compatibility_constraint",
    "reliability_support",
    "availability_timing",
    "clarification_question",
    "correction_caveat",
    "joke_reaction",
    "side_tangent",
    "community_meta",
    "miscellaneous",
)


INTERNAL_CONTROL_ID_RE = re.compile(
    r"(?<![A-Za-z0-9])(?:P\d{2}|S\d+|B\d+)(?![A-Za-z0-9])", re.I
)


def _tone_class_definitions() -> str:
    """Render the shared tone register definitions for Planner and Writer."""

    return "\n".join(
        f"- {label}: {TONE_DEFINITIONS[label]}"
        for label in TONE_CLASSES
        if label in TONE_DEFINITIONS
    )


def _own_equipment_block(
    backend: Any,
    task: Any,
    *,
    has_domain_claim: bool = False,
) -> str:
    """Offer this slot its own equipment from the held-out domain inventory.

    Restricting every comment to the seed post's entities made a whole thread
    circulate the same two or three products, which concentrates 4-gram mass.
    The legacy ``own`` experiment gives a Planner-licensed first-person slot a
    rotating shortlist drawn from evaluation-excluded threads. ``off`` remains
    unlicensed. The broader ``named`` mode also receives the shortlist on a
    first-person slot; v95 showed that a bare permission to be specific still
    left stories without safe particulars. A slot already carrying a planned
    domain claim does not receive a second factual source.
    """

    # This block is an explicit own-fact permission. Rendering it for the
    # unlicensed mode produced 61 permission/revocation conflicts in the frozen
    # 186-slot replay. Keep the first-person gate so the shortlist never assigns
    # a synthetic biography to a non-personal turn.
    if (
        has_domain_claim
        or writer_grounding_mode(backend, task) not in {"own", "named"}
        or not _first_person_experience_slot(task)
    ):
        return ""
    profile = getattr(backend, "GENERALIZED_DOMAIN_PROFILE", {}) or {}
    inventory = profile.get("entity_inventory") or {}
    if not inventory.get("available"):
        return ""
    visible = [
        str(value)
        for value in (getattr(task, "concrete_anchors", ()) or ())
        if str(value).strip()
    ]
    options = slot_equipment_options(
        inventory,
        slot_index=_safe_slot_index(task),
        limit=4,
        excluded=visible,
    )
    if not options:
        return ""
    rendered = ", ".join(options)
    closing = equipment_closing_clause(mode=writer_grounding_mode(backend, task))
    return (
        "\n\nEquipment you may claim as your own, if this turn reports personal "
        f"experience:\n- {rendered}\n{closing}"
    )


def _tone_donor_block(backend: Any, task: Any) -> str:
    """v120's drawn appreciative sentence, for slots the Planner assigned polite.

    Rendered from THREE call sites, not one. G23 records a v108 prompt fix that
    reached only one of the two `writer_prompt` branches, and G41 records that
    `_low_info_writer_prompt` is a third template which has never carried an
    equipment or anchor offer at all. `polite_rate` is a share over every comment,
    so a block that misses a template silently caps the arm.
    """

    inventory = getattr(backend, "GENERALIZED_DONOR_INVENTORY", None)
    if inventory is None:
        inventory = require_donor_inventory(
            getattr(backend, "GENERALIZED_DOMAIN_PROFILE", {}) or {}
        )
        backend.GENERALIZED_DONOR_INVENTORY = inventory
    offer = donor_sentence_offer(draw_donor_sentence(task, inventory))
    return f"\n\n{offer}" if offer else ""


def _equipment_and_referent_block(
    backend: Any,
    task: Any,
    *,
    has_domain_claim: bool = False,
) -> str:
    """Own-gear offer plus the drawn referent offer, for both writer prompts.

    Both blocks are rendered here rather than at each call site because there
    are two writer-prompt builders and v108 shipped a prompt fix that reached
    only one of them (`docs/DECISIONS.md` G23). One helper, two call sites.

    The two offers are deliberately different things: `_own_equipment_block`
    licenses a *possession* on a first-person slot (14.0% of real designator
    mentions), and `entity_spread` offers a *bare referent* on any slot (the
    other 86.0%). See `entity_spread.py`.
    """

    own = _own_equipment_block(backend, task, has_domain_claim=has_domain_claim)
    profile = getattr(backend, "GENERALIZED_DOMAIN_PROFILE", {}) or {}
    seed_key = str(getattr(backend, "GENERALIZED_ACTIVE_SEED_KEY", "") or "")
    visible = [
        str(value)
        for value in (getattr(task, "concrete_anchors", ()) or ())
        if str(value).strip()
    ]
    referent = slot_referent_block(
        profile.get("entity_inventory") or {},
        profile=profile.get("entity_spread_profile") or {},
        slot_key=f"{seed_key}:{_safe_slot_index(task)}",
        slot_index=_safe_slot_index(task),
        comment_count=getattr(backend, "GENERALIZED_ACTIVE_THREAD_COMMENTS", 0),
        excluded=visible,
    )
    link = reference_links_offer(
        draw_reference_links(task, profile.get("reference_link_inventory") or {})
    )
    return own + referent + (f"\n{link}" if link else "")


def _speaker_for_task(backend: Any, task: Any) -> Any:
    """Return the Speaker holding this slot, or None when identity is off."""

    roster = getattr(backend, "GENERALIZED_ACTIVE_SPEAKER_ROSTER", None)
    if roster is None:
        return None
    return roster.speaker_for(getattr(task, "real_sample_id", None))


def _speaker_identity_block(
    backend: Any, task: Any, previous_comments: list[dict[str, Any]] | None
) -> str:
    """Render who this person is and what they already said in this thread.

    The only identity content is structural: OP membership and this anonymous
    participant's own earlier turns. The text already exists in thread history;
    no matched author string or invented biography is rendered.
    """

    speaker = _speaker_for_task(backend, task)
    if speaker is None:
        return ""

    lines: list[str] = []
    if speaker.is_op:
        lines.append("- You are the person who wrote the post.")

    said = [
        str(row.get("content") or "").strip()
        for row in (previous_comments or ())
        if str(row.get("speaker_id") or "") == speaker.speaker_id
        and str(row.get("content") or "").strip()
    ]
    if said:
        # Only the last few: a prolific speaker holds up to 10 slots in the
        # matched threads, and the whole history would crowd out the slot's own
        # controls.
        for text in said[-3:]:
            lines.append(f"- You already wrote here: {backend.compact(text, 200)}")
        lines.append(
            "- Same participant: keep factual self-claims consistent with those "
            "turns, but follow this turn's assigned voice and affect. Make a "
            "different point; do not reintroduce yourself or repeat yourself."
        )
    if not lines:
        return ""
    return "\n\nWho you are in this thread:\n" + "\n".join(lines)


def _opener_rule(assigned: str, *, drawn: str = "", forbidden: tuple[str, ...] = ()) -> str:
    """Render the slot's assigned grammatical entry as a per-slot instruction.

    `drawn` is `opening_move`'s per-slot word for the two entry types whose
    category this Writer resolves to the wrong act; when it is present it
    replaces the category description rather than being added to it, because two
    descriptions of one act is how `discourse_marker` became `Yeah,` in the first
    place. `forbidden` names the tokens the exclusion is about: the categorical
    version of it reached 504 of 532 v101 prompts and was violated on 9.1%.
    """

    name = str(assigned or "").strip().lower()
    instruction = OPENER_INSTRUCTIONS.get(name)
    if not instruction:
        return ""
    if drawn:
        instruction = drawn
    # The schedule is realized 47% of the time and the drift has one direction:
    # v96 opened 20.7% of comments with a bare polarity token against 6.8% of
    # matched real comments and 5.3% scheduled; v101 ran 0.1274 against a
    # measured 0.0526. Naming that one default is narrower than a general style
    # rule and is what the measurement supports.
    if name == "polarity_token":
        exclusion = ""
    elif forbidden:
        listed = ", ".join(f'"{token}"' for token in forbidden)
        exclusion = f" Do not begin this comment with any of: {listed}."
    else:
        exclusion = " Do not open with a bare agreement or disagreement token."
    return (
        f"Opening grammar for this turn: {name}. {instruction}{exclusion} "
        "This is the entry form, not the content; the content is the semantic "
        "route above."
    )


_SKELETON_SENTENCE_COUNT = re.compile(r"(\d+)-sentence|about (\d+) sentences")


def _closing_rule(backend: Any, task: Any) -> str:
    """Render this slot's drawn closing move.

    Applies whatever tone the plan assigned: the verdict close is over-produced
    on every register, so gating it to one would leave most of it in place.
    """

    seed_key = str(getattr(backend, "GENERALIZED_ACTIVE_SEED_KEY", "") or "")
    return active_closing_guidance(
        slot_key=f"{seed_key}:{_safe_slot_index(task)}",
        word_count=getattr(task, "real_word_count", 0),
    )


def _register_rule(backend: Any, task: Any) -> str:
    """Render this slot's drawn warm-register moves.

    Keyed on the same slot as `_rhythm_rule` but namespaced inside
    `register_realization`, so drawing a rhythm habit does not correlate with
    drawing a register move. The rate comes from the register the plan assigned,
    which the rule never changes.
    """

    seed_key = str(getattr(backend, "GENERALIZED_ACTIVE_SEED_KEY", "") or "")
    return active_register_guidance(
        slot_key=f"{seed_key}:{_safe_slot_index(task)}",
        word_count=getattr(task, "real_word_count", 0),
        tone_class=str(getattr(task, "tone_target", "") or ""),
    )


def _evaluative_rule(backend: Any, task: Any) -> str:
    """Render this slot's evaluation-strength rule and the two tic suppressions.

    Keyed on the same slot as `_register_rule` but namespaced inside
    `evaluative_register`, so drawing a tier does not correlate with drawing a
    register move. The tier rule is conditional on the comment evaluating
    anything at all: the Planner owns whether a slot praises something, and this
    changes only how far an evaluation that happens is allowed to travel.
    """

    seed_key = str(getattr(backend, "GENERALIZED_ACTIVE_SEED_KEY", "") or "")
    return active_evaluative_guidance(
        slot_key=f"{seed_key}:{_safe_slot_index(task)}",
        tone_class=str(getattr(task, "tone_target", "") or ""),
        word_count=getattr(task, "real_word_count", 0),
    )


def _opening_move_clause(backend: Any, task: Any, assigned: str) -> str:
    """Render this slot's drawn opening word.

    Keyed on the same slot as `_rhythm_rule` and `_register_rule` but namespaced
    inside `opening_move`, so drawing an opening word does not correlate with
    drawing a rhythm habit or a register move.
    """

    seed_key = str(getattr(backend, "GENERALIZED_ACTIVE_SEED_KEY", "") or "")
    return active_opening_guidance(
        slot_key=f"{seed_key}:{_safe_slot_index(task)}",
        opener=assigned,
        tone_class=str(getattr(task, "tone_target", "") or ""),
        stance=str(getattr(task, "stance", "") or ""),
    )


def _rhythm_rule(backend: Any, task: Any) -> str:
    """Render this slot's drawn typing rhythm.

    Keyed on the run's seed and the slot index so the draw is reproducible and
    so two slots of the same size get different habits. That difference is the
    mechanism: see `sentence_rhythm`.
    """

    seed_key = str(getattr(backend, "GENERALIZED_ACTIVE_SEED_KEY", "") or "")
    skeleton = str(getattr(task, "surface_skeleton", "") or "")
    return active_rhythm_guidance(
        slot_key=f"{seed_key}:{_safe_slot_index(task)}",
        word_count=getattr(task, "real_word_count", 0),
        slot_names_sentence_count=bool(_SKELETON_SENTENCE_COUNT.search(skeleton)),
    )


def _short_output_slot(task: Any) -> bool:
    """Return whether this slot could plausibly reproduce a short prior line."""

    if str(getattr(task, "length_bucket", "") or "") in {"micro", "short"}:
        return True
    try:
        words = int(getattr(task, "real_word_count", 0) or 0)
    except (TypeError, ValueError):
        words = 0
    return 0 < words <= 20


def _first_person_experience_slot(task: Any) -> bool:
    """Return whether this slot's plan already licenses personal experience.

    Defined in `writer_grounding` so the equipment permission and the fact
    license can never drift apart into a slot that may name its gear but not
    describe it.
    """

    return first_person_experience_slot(task)


def _safe_slot_index(task: Any) -> int:
    try:
        return int(getattr(task, "local_task_id", 0) or 0)
    except (TypeError, ValueError):
        return 0


def planner_prompt(
    config: DomainConfig,
    backend: Any,
    *,
    seed_post: Any,
    target: Any,
    matched_real_thread: dict[str, Any] | None,
    matched_real_comments: int,
    global_memory: dict[str, Any],
    minimum_branch_count: int = 3,
) -> str:
    matched = _render_matched_structure(
        matched_real_thread,
        max_comments=matched_real_comments,
    )
    facets = "\n".join(f"- {item}" for item in config.topic_facets)
    profile = getattr(backend, "GENERALIZED_DOMAIN_PROFILE", {})
    perspectives = render_profile_for_planner(profile)
    reference_viewpoints = render_reference_viewpoints(
        profile,
        seed_title=str(seed_post.title or ""),
        seed_body=str(seed_post.body or seed_post.content or ""),
        limit=min(36, max(18, int(target.top_level_comments) * 2)),
    )
    conversation_block = render_conversation_fragments(
        select_conversation_fragments(
            profile,
            seed_title=str(seed_post.title or ""),
            seed_body=str(seed_post.body or seed_post.content or ""),
            exclude_post_ids={str(getattr(seed_post, "source_raw_post_id", "") or "")},
        )
    )
    conversation_section = f"\n{conversation_block}\n" if conversation_block else ""
    recent = backend.render_top_counts(global_memory)
    distribution_target = render_planner_distribution_target(
        getattr(backend, "GENERALIZED_ACTIVE_REFERENCE_TEMPLATE", {}),
        total_comments=int(target.target_comments),
    )
    return f"""Plan the semantic axes of one Reddit discussion in {config.community_context}.

The planner produces private controls. It does not write comments.

Objectives:
- Preserve the matched real thread's comment-count scale, reply-tree shape, local branch behavior, length variation, and mix of social roles.
- Ground every branch in the seed post or in a parent-local issue.
- Keep branches meaningfully different without making them unrelated.
- Use only the supplied matched-real structural and surface labels. The matched comments' text is not provided.
- Concrete facts must come from the seed post or later visible generated parents.
- Allow realistic low-information turns, questions, corrections, jokes, acknowledgements, side tangents, and personal datapoints. Do not plan only polished advice.

Domain: {config.display_name}
Common facets, offered as examples rather than required coverage:
{facets}

    Frozen domain-neutral decision lenses:
{perspectives}
These labels describe how a comment approaches a topic. They are not topic,
entity, feature, or product labels, and they do not replace the reference-comment
semantic moves below.

NON-TEST REFERENCE COMMENTS FOR VIEWPOINT ABSTRACTION:
These comments come only from threads excluded from the evaluation seed pool.
Use them as the original CARD planner used real comments: abstract reusable
viewpoint and discourse moves, then adapt those moves to the visible seed.
Never copy their wording, products, events, measurements, or personal facts.
{reference_viewpoints}
{conversation_section}
Seed post title:
{seed_post.title}

Seed post body:
{backend.compact(seed_post.body or seed_post.content, 2600)}

Target controls:
- target_comments: {target.target_comments}
- top_level_comments: {target.top_level_comments}
- max_depth_goal: {target.max_depth_goal}
- shape_label: {target.shape_label}
- length_mix: {target.length_mix_note}

Frozen whole-thread distribution target:
{distribution_target}

Recent generated-control counts, used only to avoid global templates:
{recent}

Matched real structural sample. Comment text and facts are intentionally hidden:
{matched}

Return strict JSON:
{{
  "branches": [
    {{
      "branch_id": 1,
      "anchor_quote": "short seed-post anchor or abstract local hook",
      "branch_goal": "one narrow branch goal",
      "perspective_id": "{perspective_schema_hint(allow_seed_local=False)}",
      "decision_boundary": "the one decision condition, consequence, or uncertainty this branch owns",
      "owned_decision_subject": "one concrete seed-derived condition or variable owned only by this branch, not a P## label",
      "branch_exclusion": "the adjacent decision axis owned by another branch that this branch must not cover"
    }}
  ]
}}

Planning rules:
- Return at least {minimum_branch_count} genuinely distinct semantic branches.
  This is a topology-derived decision-axis budget, not the number of top-level
  comments. A narrow real discussion may have several independent roots around
  the same axis; those roots should vary their stance, evidence, detail, or
  social function rather than inventing a false new topic.
- Replies inherit their root's branch. When multiple root chains are routed to
  the same branch, use that branch as a shared local subject but give each root
  a different conversational contribution. Do not treat every short agreement,
  vote, question, or counterexample as a separate decision axis.
- Each branch must stay narrow. Do not plan a complete product review.
- Derive substantive branches from different reference viewpoint patterns. Two
  branches may share a topic only when their claim, stance, evidence role, or
  reply function is materially different.
- Make ``owned_decision_subject`` and ``decision_boundary`` mutually exclusive
  across independently rooted branches. Reuse a ``perspective_id`` only when
  needed, but vary it wherever the frozen profile permits; a P## label is a
  decision lens and cannot substitute for a distinct branch subject. Then write
  ``branch_exclusion`` to name the neighboring decision axis that this branch
  must leave to another branch. A root branch may not be a broad restatement of
  another root branch with one adjective changed.
- Do not split one decision into its premise and immediate consequence across
  two root branches. If one branch owns an access, permission, compatibility,
  or eligibility risk, another branch may not merely restate the physical or
  practical condition as the way to satisfy that same risk. Give the other
  branch an independently answerable everyday-use, evidence, performance,
  value, timing, or social question instead.
- ``owned_decision_subject`` must be a concrete condition visible in the seed,
  such as one use constraint, uncertainty, consequence, or tradeoff. It is not
  a topic label, product name, P## lens, or a broad request to compare options.
- Reference comments are semantic examples, not factual sources. Every concrete
  fact in a branch must still be visible in the seed post.
- This is the compact root-coverage contract. The per-comment Planner assigns
  payload, evidence, tone, story, affect, and surface behavior later; do not
  pad each branch with those whole-thread fields.
- Do not infer or invent hidden matched-real content from the structural labels.
"""


def comment_planner_prompt(
    config: DomainConfig,
    backend: Any,
    *,
    seed_post: Any,
    target: Any,
    branches: list[Any],
    matched_real_thread: dict[str, Any] | None,
    comments: list[dict[str, Any]],
    all_comments: list[dict[str, Any]] | None = None,
    sample_offset: int = 0,
    prior_plans: list[dict[str, Any]] | None = None,
    validation_feedback: str = "",
) -> str:
    del matched_real_thread
    real_sample = _render_matched_slots(
        comments,
        all_comments=all_comments,
        sample_offset=sample_offset,
    )
    requested_sample_ids = list(
        range(sample_offset + 1, sample_offset + len(comments) + 1)
    )
    complete_slots = all_comments or comments
    profile = getattr(backend, "GENERALIZED_DOMAIN_PROFILE", {})
    direct_batch = is_direct_reply_batch(
        comments=comments,
        all_comments=complete_slots,
        sample_offset=sample_offset,
        prior_plans=prior_plans or [],
    )
    claim_slots: set[int] = set()
    reference_viewpoints = ""
    if domain_claim_mode(backend) == "selective":
        all_references = reference_viewpoint_window(
            profile,
            seed_title=str(seed_post.title or ""),
            seed_body=str(seed_post.body or seed_post.content or ""),
            limit=len(complete_slots),
        )
        reference_viewpoints = render_reference_rows(
            all_references[sample_offset : sample_offset + len(comments)]
        )
        claim_slots = set(selective_claim_slots(complete_slots, all_references))
    elif not direct_batch:
        reference_viewpoints = render_reference_rows(
            reference_viewpoint_window(
                profile,
                seed_title=str(seed_post.title or ""),
                seed_body=str(seed_post.body or seed_post.content or ""),
                limit=max(1, len(comments)),
                offset=max(0, sample_offset),
            )
        )
    backend.GENERALIZED_ACTIVE_SELECTIVE_CLAIM_SLOTS = set(claim_slots)
    active_schedule = getattr(
        backend, "GENERALIZED_ACTIVE_SLOT_DISTRIBUTION_SCHEDULE", {}
    )
    if direct_batch:
        assignments = active_schedule.get("assignments") or {}
        defaults = active_schedule.get("defaults") or {}
        slot_controls = {
            sample_id: {
                **defaults,
                **dict(
                    assignments.get(str(sample_id)) or assignments.get(sample_id) or {}
                ),
            }
            for sample_id in requested_sample_ids
        }
        return render_direct_reply_planner_prompt(
            config=config,
            backend=backend,
            seed_post=seed_post,
            comments=comments,
            all_comments=all_comments or comments,
            sample_offset=sample_offset,
            prior_plans=prior_plans or [],
            slot_distribution=render_slot_distribution_schedule(
                active_schedule,
                sample_ids=requested_sample_ids,
            ),
            slot_controls=slot_controls,
            reference_viewpoints=reference_viewpoints,
            claim_slots=claim_slots,
            validation_feedback=validation_feedback,
        )
    schema_sample_id = requested_sample_ids[0] if requested_sample_ids else 1
    branches_block = "\n".join(
        f"- B{branch.branch_id}: {branch.branch_goal}; anchor={backend.compact(branch.anchor_quote, 100)}"
        for branch in branches
    )
    families = " | ".join(GENERIC_CLAIM_FAMILIES)
    perspectives = render_profile_for_planner(profile)
    prior_plan_block = _render_prior_comment_plans(backend, prior_plans or [])
    distribution_target = render_planner_distribution_target(
        getattr(backend, "GENERALIZED_ACTIVE_REFERENCE_TEMPLATE", {}),
        total_comments=int(target.target_comments),
        prior_plans=prior_plans or [],
    )
    slot_distribution = render_slot_distribution_schedule(
        active_schedule,
        sample_ids=requested_sample_ids,
    )
    branch_schedule = root_branch_schedule(
        all_comments or comments,
        branch_ids=[int(branch.branch_id) for branch in branches],
    )
    branch_goals = {
        int(branch.branch_id): backend.compact(branch.branch_goal, 180)
        for branch in branches
    }
    branch_perspectives = {
        int(branch.branch_id): str(getattr(branch, "perspective_id", "") or "")
        for branch in branches
    }
    branch_exclusions = {
        int(branch.branch_id): backend.compact(
            getattr(branch, "branch_exclusion", "") or "", 180
        )
        for branch in branches
    }
    branch_subjects = {
        int(branch.branch_id): backend.compact(
            getattr(branch, "owned_decision_subject", "")
            or getattr(branch, "decision_boundary", ""),
            180,
        )
        for branch in branches
    }
    required_branch_routes = render_branch_requirements(
        branch_schedule,
        sample_ids=requested_sample_ids,
        branch_goals=branch_goals,
        branch_perspectives=branch_perspectives,
        branch_exclusions=branch_exclusions,
        branch_subjects=branch_subjects,
        parent_slots=parent_slot_schedule(all_comments or comments),
    )
    feedback_block = (
        f"\nPLAN-QUALITY REPAIR FEEDBACK:\n{validation_feedback}\n"
        if validation_feedback
        else ""
    )
    # v125 (G97): ask for the topical outsiders real threads carry and we ship
    # at zero. Sized against THIS BATCH, not the thread. Sizing it against the
    # thread was a real bug measured in v125's first run: the Planner sees 8
    # slots at a time, so a 186-slot thread was told "exactly 22 of these 186",
    # which is unsatisfiable in a batch of 8 and was ignored outright --
    # compliance was 6.7% on a 45-slot thread and 0.5% on the 186-slot one.
    # The rate is a thread property; the instruction has to be a batch one.
    _outsider = outsider_quota_block(len(comments))
    outsider_block = f"\n{_outsider}\n" if _outsider else ""
    # Sized against the batch for the same reason, and stated separately: the
    # outsider quota asks for distance from the post, this asks for distance
    # from the sibling slots, and on the current configuration only the second
    # gap is still open.
    _isolation = isolation_quota_block(len(comments))
    if _isolation:
        outsider_block = f"{outsider_block}\n{_isolation}\n"
    # v148: with the routes stripped to structure, say what replaces them. This
    # is not another quota -- the quotas were tried and moved the realized
    # isolation rate without moving the metric. It names the real comment as the
    # authority the routes used to be, which is the one input never given that
    # standing before.
    if branch_dictation_mode() == "structural" and slot_grid_mode() != "free":
        outsider_block = (
            f"{outsider_block}\n"
            "WHERE EACH SLOT'S DIRECTION COMES FROM:\n"
            "- The routes above now carry only shape: which branch a slot sits "
            "in, its parent, its siblings. They no longer tell you what the "
            "comment is about, which perspective it takes, or what it may not "
            "touch. That is deliberate.\n"
            "- Each matched slot below shows the real comment that occupied it. "
            "Read what that person actually did -- what they picked up on, how "
            "far they wandered from the post, who they were talking to, how "
            "blunt or warm they were -- and plan a slot that makes an equally "
            "distinct move. Do not reproduce their words or their facts.\n"
            "- Real commenters in a thread are often not discussing the same "
            "thing. Where the real comments at two slots have nothing to do with "
            "each other, your plans for those slots should have nothing to do "
            "with each other either.\n"
            + (
                "- Name the lens each slot argues from yourself, in 3-6 words. "
                "There is no `seed_local` and no P## to fall back on: if no lens "
                "you have named fits what the real comment did, the answer is a "
                "lens you have not named yet, not the absence of one.\n"
                if open_vocabulary()
                else "- `perspective_id` accepts `seed_local`. Use it whenever "
                "no P## honestly fits what the real comment did; forcing a lens "
                "that does not fit is worse than declining one.\n"
            )
        )
    actor_enabled = (
        str(getattr(backend, "GENERALIZED_ACTOR_MODE", "") or "") == MODE_DOMAIN_DERIVED
    )
    actor_schema = ""
    actor_rules = ""
    if actor_enabled:
        actor_schema = """,
      "actor": {
        "participant_key": "thread-local A#; use OP only for an actual OP follow-up and reuse a key only when the same participant returns",
        "knowledge_boundary": "what this participant can reasonably know from the visible seed or parent and the assigned evidence role",
        "participation_goal": "why this participant takes this one local turn",
        "evidence_access": "the evidence this participant can use without inventing a biography or hidden fact",
        "attention_focus": "one seed- or parent-grounded detail this participant notices",
        "interaction_tendency": "how this participant engages this parent or branch",
        "context_visibility": "which part of the visible discussion this participant is responding to",
        "realization_route": "an abstract one-shot sentence construction and cadence; never example wording"
      }"""
        actor_rules = """
- Compose every ``actor`` from the current visible discussion and the abstract
  behavior of its selected evaluation-excluded R# pattern. There is no fixed
  persona catalog. Do not assign demographic biography or hidden history.
- Actor fields describe a local cognitive and interaction state. Ground
  ``attention_focus`` in the seed or parent, and do not import facts from R#.
- Vary ``realization_route`` across nearby slots before Writer generation. It
  describes sentence architecture and cadence, not reusable example wording."""
    claim_mode = domain_claim_mode(backend)
    if claim_mode == "selective":
        selected_claims = render_selective_claim_schedule(
            requested_sample_ids,
            claim_slots,
        )
        claim_knowledge = """
These rows are also a bounded source of this domain's general knowledge. For a
slot explicitly selected below, restate at most one general fact from its paired
R# row: a compatibility relation, procedure, observable behaviour,
specification class, or model comparison. The Writer receives only your
restatement, never the R# text. Never carry over the reference participant's
purchase, photos, personal outcome, dispute, wording, URL, or situation-specific
number. A fact about the seed post itself still cannot be invented."""
        claim_schema = (
            "one general domain fact restated from this slot's paired R# row "
            "only when S# is selected below; otherwise none"
        )
        claim_rules = f"""- Selective factual slots in this request: {selected_claims}.
  Only those S# values may return a ``domain_claim``. Each uses its paired R#
  row and carries one independently useful fact. Every other row returns the
  literal ``none`` even if it is substantive.
- Reusing the discussion's product or model name is normal. Vary the fact,
  condition, procedure, quantity, or comparison; do not invent a different
  name merely to make nearby comments look different.
- If a selected R# row contains no transferable general domain fact, return
  ``none`` rather than turning a participant-specific detail into knowledge."""
        fact_grounding_objective = (
            "Keep factual content grounded in the seed or in the one "
            "evaluation-excluded general fact explicitly licensed for a "
            "selective slot. No matched evaluation comment text or facts are "
            "supplied."
        )
        reference_adaptation_rule = (
            "Use the corresponding R# as a semantic pattern. Only a listed "
            "selective factual slot may also restate one transferable general "
            "domain fact from its paired row; all other slots must not import "
            "reference facts."
        )
        semantic_grounding_rule = (
            "Ground semantic_move, local_topic, and detail_focus in the visible "
            "seed or in this slot's delivered domain_claim. They may explain, "
            "apply, question, or react to that one claim, but may not add a "
            "second hidden fact."
        )
    elif planner_claims_enabled(backend):
        claim_knowledge = """
These rows are also this domain's knowledge. Real participants bring domain
knowledge they acquired elsewhere, so a slot may carry one *general* domain fact
from its R# row into ``domain_claim``: a compatibility relation, a procedure, an
observable behaviour, a specification class, a model comparison. Write it in your
own words as a claim a knowledgeable participant would state. Two limits: never
reproduce an R# row's wording, and never carry a detail that belongs to that
discussion or its participants rather than to the domain — someone's own photos,
their purchase, their argument with another commenter, a number tied to their
specific situation. A fact about the seed post itself still cannot be invented."""
        claim_schema = (
            "one concrete domain fact this slot states in your own words - a "
            "compatibility relation, procedure, observable behaviour, "
            "specification class, or model comparison - or none for a purely "
            "social slot"
        )
        claim_rules = """- Give most substantive slots a ``domain_claim``. Measured against a matched real
  thread, real comments name a concrete domain entity or domain noun in
  about two thirds of cases and contain a number in about half, while a plan
  built only from decision language produced neither. A slot whose entire content
  is how to weigh a decision, how a learning curve feels, or whether advice is
  trustworthy is the failure mode: it reads as commentary about the discussion
  rather than participation in it. Vary the concrete entities named across slots
  instead of returning to whichever one the seed post mentions.
- Micro and purely social slots keep ``domain_claim=none``. Do not attach a fact
  to a reaction that has no room for one."""
        fact_grounding_objective = (
            "Keep factual content grounded in the seed or in the separate "
            "general domain_claim field. No matched evaluation comment text or "
            "facts are supplied."
        )
        reference_adaptation_rule = (
            "Use the corresponding R# as a semantic pattern and, for a "
            "substantive slot, as the source of at most one transferable "
            "general domain claim. Never import participant-specific facts."
        )
        semantic_grounding_rule = (
            "Ground semantic_move, local_topic, and detail_focus in the visible "
            "seed or in this slot's delivered domain_claim; do not add another "
            "hidden fact."
        )
    else:
        claim_knowledge = ""
        claim_schema = "none"
        claim_rules = """- Set ``domain_claim`` to the literal ``none`` for every row. This run does not
  deliver a separate planned fact to the Writer. Put every required semantic
  contribution in ``semantic_move``, ``detail_focus``, and ``domain_intent``;
  do not build the move around information that the Writer will not receive."""
        fact_grounding_objective = (
            "Keep factual content grounded in the seed post or a generic "
            "parent-local relation. No matched evaluation comment text or "
            "facts are supplied."
        )
        reference_adaptation_rule = (
            "Use the corresponding R# only as a semantic pattern. If it does "
            "not fit the seed, choose another displayed R# or a different "
            "seed-local/social move; never import the reference's facts."
        )
        semantic_grounding_rule = (
            "semantic_move, local_topic, and detail_focus must be supported by "
            "the visible seed or remain generic."
        )
    return f"""{_planner_opening()}
{_brief_at_top()}
Domain: {config.display_name}

{_sec('A', 'VOCABULARY AVAILABLE TO YOU')}{_lens_framing()}
{perspectives}
{_lens_note()}

{_sec('B', 'HOW THIS COMMUNITY WRITES -- REFERENCE BANK')}{_reference_row_framing()}
{claim_knowledge}
{reference_viewpoints}
{abstraction_block(perspectives, _real_position_count(backend, all_comments or comments))}

{_sec('C', 'OBJECTIVES')}Objectives:
- Preserve each supplied slot's depth, parent relation, approximate information density, and surface roughness.
- {_choose_from_objective()}
- Produce a reusable instruction, not a comment.
- {fact_grounding_objective}
- Preserve weak comments as weak comments. Do not upgrade reactions, fragments, jokes, or questions into advice.
- Treat the displayed anonymous word count as semantic capacity, not an output
  word-count target. A micro slot (five words or fewer) can only support a
  reaction, fragment, bare acknowledgement, joke, or narrow question. It
  cannot carry a story, personal datapoint, advice, explanation, or multi-step
  claim. A short slot can add one narrow observation or question, but not a
  multi-step story. This does not require the Writer to match an exact count.

{_sec('D', 'THE POST')}Seed post title:
{seed_post.title}

Seed post body:
{backend.compact(seed_post.body or seed_post.content, 2200)}

{_sec('E', 'THE THREAD TO BUILD')}Thread target:
- target_comments: {target.target_comments}
- max_depth_goal: {target.max_depth_goal}
- shape_label: {target.shape_label}

{_sec('F', 'LABELS -- NONE IMPOSED HERE')}{_distribution_heading()}
{distribution_target}

{_slot_label_heading()}
{slot_distribution}

{_sec('G', 'TONE REGISTERS')}Tone class definitions. These are social registers observed in real threads of this kind, not degrees of manners. Assign the label whose register the slot can actually carry:
{_tone_class_definitions()}

{_sec('H', 'THREAD SHAPE')}Available branch controls:
{branches_block}

Required structural branch routes:
{required_branch_routes}

Plans already assigned in earlier batches of this same thread:
{prior_plan_block}
{feedback_block}
{outsider_block}

{_sec('I', 'THE SLOTS, AND THE REAL COMMENTS THAT FILLED THEM')}Matched-real structural slots. Use the displayed global S# values exactly:
{real_sample}

{_sec('J', 'WHAT TO RETURN')}Return strict JSON:
{{
  "comment_plans": [
    {{
      "sample_id": {schema_sample_id},
      "reference_id": "the R# used as the semantic pattern, or none",
      "branch_id": 1,
      "payload_type": "low_info_reaction | bare_answer | fragment_datapoint | soft_helpful | correction | narrow_question | personal_story | rant | joke | side_tangent | meta_or_template | advice",
      "comment_function": "reaction | question_followup | correction_caveat | personal_datapoint | recommendation_advice | verdict_evaluation | explanation_analysis | offtopic_noise",
      "content_angle": "{content_angle_schema_hint('cost_value | rules_constraints | risk_reliability_support | comparison_alternative | setup_troubleshooting | availability_timing | fit_use_case | unclear_mixed')}",
      "evidence_mode": "none_assertion | firsthand_experience | technical_or_policy_reasoning | calculation_math | hearsay_consensus | link_quote_reference | small_observation",
      "story_mode": "no_story | tiny_personal_context | specific_personal_story | messy_multi_step_story",
      "tone_class": "polite | somewhat_polite | neutral | impolite",
      "affect_role": "{_affect_role_schema()}",
      "voice": "blunt | casual_neutral | polite_soft | sarcastic | annoyed | uncertain | grateful",
      "speaker_role": "advisor | confused_asker | op_followup | gratitude_reply | jokester | mod_meta | contrarian | datapoint_only | ranter | side_observer",
      "semantic_move": "one concrete but non-verbatim action for the generated comment",
      "local_topic": "{_local_topic_schema()}",
      "reply_relation": "relation to the seed post using answers_parent | challenges_parent | asks_narrow_followup | adds_datapoint | jokes_aside | corrects_detail | shifts_to_side_detail",
      "stance": "agree | disagree | mixed | uncertain | joking | neutral",
      "detail_focus": "{_detail_focus_schema()}",
      "avoid_repeating": "nearby discourse move to avoid",
      "claim_family": "{families}",
      "claim_key": "short abstract semantic key",
      "perspective_id": "{perspective_schema_hint()}",
      "domain_intent": "{domain_intent_schema_hint()}",
      "domain_claim": "{claim_schema}",
      "decision_boundary": "the one decision condition, consequence, or uncertainty this independent slot owns",
      "reply_delta": "none",
      "reply_delta_type": "none",
      "reply_novelty_anchor": "none",
      "opening_style": "one specific sentence route for this slot that realizes its assigned opener_type, such as constraint then consequence, concrete observation then caveat, or answer embedded after parent detail",
      "development_plan": "none",
      "context_aperture": "full_seed | seed_gist_only | title_only | semantic_only | parent_only"{actor_schema}
    }}
  ]
}}

{_sec('K', 'RULES')}Rules:
{_rule_group('OUTPUT MECHANICS')}- Output one plan per displayed S#.
- The ``sample_id`` value shown in the JSON schema is an example from this
  request, not a fixed constant. Return exactly these global IDs once each:
  {", ".join(f"S{sample_id}" for sample_id in requested_sample_ids)}.
- {reference_adaptation_rule}
- Use controlled vocabulary values exactly.
- {semantic_grounding_rule}
- Never copy a hidden matched-real anecdote, username, URL, or seed-specific fact.
{_rule_group('WHAT YOU DECIDE')}{_template_contract_rule()}
{_rule_group('KEEPING SLOTS DISTINCT FROM EACH OTHER')}- Equivalent local claims should share ``claim_key``; different claims must not.
- Compare every new plan against the earlier-batch ledger. Do not hide a repeated semantic move behind a new ``claim_key`` or different wording.
{_remaining_counts_rule()}
{_rule_group('KEEPING ONE SLOT COHERENT')}- A slot's ``tone_class`` constrains its whole plan, not just its wording. The
  semantic move has to be the kind of move that register makes:
  a ``polite`` slot agrees, corroborates from experience, endorses with a
  reason, or thanks, so pair it with ``stance=agree``, a ``speaker_role`` of
  datapoint_only, op_followup, gratitude_reply, or side_observer, and a
  ``comment_function`` of personal_datapoint, reaction, or verdict_evaluation;
  an ``impolite`` slot contradicts, limits, complains, or corrects;
  a ``neutral`` slot states one fact or reference with no evaluation;
  a ``somewhat_polite`` slot half-agrees with a qualification.
{_register_pairing_rule()} A plan whose move
  contradicts its register is invalid; change the move, not the register.
- Do not make every substantive slot an ``advisor``. Real discussions of this
  kind are mostly participants reporting their own experience, reacting, and
  asking, with advisors a minority. Reserve ``advisor`` for slots whose plan is
  genuinely a recommendation.
{claim_rules}
{_story_schedule_rule()}
- Story is a joint evidence contract. A `no_story` row must not use
  `firsthand_experience` or `personal_story`; it may use a present-state
  first-person appraisal or one small observation, but not a past action,
  event, before/after change, or temporal sequence. A scheduled story row must
  use `firsthand_experience`, `comment_function=personal_datapoint`, and a
  personal-story or fragment-datapoint payload. The semantic move itself must
  describe that narrative evidence rather than advice or abstract analysis.
  {SYNTHETIC_STORY_PLANNER_BOUNDARY}
{_rule_group('STRUCTURE TO PRESERVE')}{_branch_axis_rule()}
{_branch_contract_rules()}
- A displayed ``root_branch_instance`` marks several independent root comments
  routed to the same semantic axis. Keep that axis, but vary the independent
  root's discourse contribution: for example, a concise verdict, causal reason,
  evidence caveat, counterexample, action/timing point, narrow question, or
  local social reaction. Do not manufacture a new topic merely to make the
  root look unique, and do not write the same premise in a polished new form.
- These are independent root slots. Set ``reply_delta``, ``reply_delta_type``,
  and ``reply_novelty_anchor`` to the literal ``none``. ``reply_relation`` uses
  the legacy relation vocabulary only to describe how the root addresses the
  seed post; it does not create a parent or a reply contract.
- Non-dependent substantive comments must not collapse to the same recommendation,
  tradeoff, verdict, or explanation. Repetition is allowed only for a direct
  reply that changes the relation, evidence, stance, or local detail.
{_claim_reuse_rule()}
{_rule_group('HOW TO WRITE SPECIFIC FIELDS')}- {_perspective_field_rule()}
- Nearby comments should not reuse the same perspective and claim key unless the reply relation directly requires it.
- Write ``semantic_move`` as the exact new contribution made by this slot, not
  as a topic label or a paraphrase of the thread's current conclusion.
- Write ``decision_boundary`` as the particular decision condition,
  consequence, tradeoff, or uncertainty owned by this slot. Two independent
  comments may not own the same boundary even if they use different P## labels.
  A direct reply may revisit its parent only when it changes the relation,
  stance, evidence, or detail.
- Write ``avoid_repeating`` as one specific already-covered proposition or
  sentence route that this slot must not reproduce. Do not use generic values
  such as "avoid repetition" or "be different."
{_opener_rules()}
- Write ``opening_style`` as a concrete one-use realization route, not one of a
  small fixed label set. Vary clause order and discourse entry across nearby
  rows; do not repeat routes such as "question first" or "direct answer."
- Use the anonymous slot word count as a content-capacity cue. {_development_plan_trigger()} return roughly the
  displayed number of distinct connected beats separated by ``||``. That
  number comes from the same capacity function as validation; do not replace it
  with a different words-per-beat rule or fixed ceiling. Under-planning a long
  slot is the common failure and collapses the thread's length spread.
  Each beat must add a different observation, reason, consequence, caveat,
  boundary, or reaction around the same local contribution. Do not pad with
  paraphrases, add unrelated claims, or infer the hidden matched comment.
- For every slot at or below {development_plan_word_threshold()} anonymous words, return the literal string
  `none` for `development_plan`; do not copy the schema's explanatory prose.
- A slot labeled ``ordinary_turn`` or ``long_turn`` must retain its information
  density. Incidental humor, a link, a quote, or a question may be embedded in
  that turn; it must not turn the whole plan into ``joke``,
  ``meta_or_template``, ``low_info_reaction``, or ``narrow_question``.
{_rule_group('BEFORE YOU RETURN')}- Before returning JSON, compare every row with every earlier row in this
  response and the earlier-batch ledger. Resolve repeated answers, verdicts,
  tradeoffs, and evidence roles in this first response; the Writer will not
  sample alternate semantic plans.
{actor_rules}
- Generate the complete semantic plan in this response, plus actor fields only
  when the schema requests them. The Writer will realize it once; do not rely
  on later candidate sampling for diversity.
- {_gratitude_rule_opening()} it must be a genuine social close: use
  ``speaker_role=gratitude_reply`` and ``comment_function=reaction``. Its
  semantic move may only acknowledge the visible local help; it may not carry
  a second explanation, recommendation, or correction. For every other affect
  role, do not turn the slot into a generic thank-you.
"""


def _render_prior_comment_plans(backend: Any, plans: list[dict[str, Any]]) -> str:
    if not plans:
        return "- none; this is the first batch"
    rows = []
    family_counts = Counter(
        str(plan.get("claim_family") or "miscellaneous") for plan in plans
    )
    perspective_counts = Counter(
        str(plan.get("perspective_id") or "seed_local") for plan in plans
    )
    angle_counts = Counter(
        str(plan.get("content_angle") or "unclear_mixed") for plan in plans
    )
    rows.append(
        "- coverage summary: "
        f"claim families={_render_counts(dict(family_counts))}; "
        f"perspectives={_render_counts(dict(perspective_counts))}; "
        f"content angles={_render_counts(dict(angle_counts))}. "
        "Shift independent new rows away from dominant combinations when the visible discussion supports it."
    )
    for plan in plans[-100:]:
        actor_suffix = ""
        if plan.get("actor_participation_goal") or plan.get("actor_realization_route"):
            actor_suffix = (
                f"; actor_goal={backend.compact(plan.get('actor_participation_goal') or 'local contribution', 70)}"
                f"; actor_route={backend.compact(plan.get('actor_realization_route') or 'local sentence route', 80)}"
            )
        rows.append(
            "- "
            f"S{plan.get('sample_id')}: "
            f"claim={backend.compact(plan.get('claim_key') or 'local_claim', 60)}; "
            f"perspective={backend.compact(plan.get('perspective_id') or 'seed_local', 24)}; "
            f"relation={backend.compact(plan.get('reply_relation') or 'local_turn', 32)}; "
            f"move={backend.compact(plan.get('semantic_move') or plan.get('local_topic') or 'local move', 120)}; "
            f"detail={backend.compact(plan.get('detail_focus') or plan.get('domain_intent') or 'local detail', 90)}"
            f"; boundary={backend.compact(plan.get('decision_boundary') or 'local condition', 90)}"
            f"; reply_delta={backend.compact(plan.get('reply_delta') or 'none', 90)}"
            f"; story={backend.compact(plan.get('story_mode') or 'no_story', 30)}"
            f"; tone={backend.compact(plan.get('tone_class') or 'unassigned', 30)}"
            f"; affect={backend.compact(plan.get('affect_role') or 'unassigned', 30)}"
            f"; opening={backend.compact(plan.get('opening_style') or 'unassigned', 50)}"
            f"; development={backend.compact(plan.get('development_plan') or 'none', 100)}"
            f"{actor_suffix}"
        )
    return "\n".join(rows)


def writer_prompt(
    config: DomainConfig,
    backend: Any,
    *,
    profile: str,
    seed_post: Any,
    task: Any,
    parent_comment: dict[str, Any] | None,
    recent_openings: list[str] | None = None,
    previous_comments: list[dict[str, Any]] | None = None,
    retry_note: str = "",
) -> str:
    domain_profile = getattr(backend, "GENERALIZED_DOMAIN_PROFILE", {})
    actor_state = actor_for_task(
        getattr(backend, "GENERALIZED_ACTOR_ASSIGNMENTS", {}),
        seed_post,
        task,
    )
    actor_block = render_actor_state(actor_state)
    actor_section = (
        f"\nThread-local actor state composed by the Planner:\n{actor_block}\n"
        if actor_state is not None
        else ""
    )
    actor_hard_rule = (
        "- Realize the thread-local actor state as a knowledge, attention, and "
        "interaction boundary. Do not invent biography, experience, or facts."
        if actor_state is not None
        else ""
    )
    speaker = _speaker_for_task(backend, task)
    persona_marker = persona_marker_for_task(
        seed_post, task, speaker_id=speaker.speaker_id if speaker else ""
    )
    marker_prefix = f"{persona_marker}\n" if persona_marker else ""
    visible = (
        render_parent_context(config, backend, parent_comment=parent_comment, task=task)
        if parent_comment is not None
        else render_seed_context(config, backend, seed_post=seed_post, task=task)
    )
    prior_comments = previous_comments or []
    # This was raised to every prior opening on the theory that a longer ledger
    # prevents more reuse. It does not. Uncapping it (v69's 18 lines -> v71's
    # unlimited) left `self_bleu_4` slightly *worse*, delta 0.46 -> 0.52, and the
    # rebuilt-thread A/B held diversity with 24 entries while cutting the prompt
    # to 13% of its size. Two dozen recent openings is what the Writer uses.
    openings = (
        "\n".join(
            f"- {item}"
            for item in _dedupe(list(recent_openings or []), limit=400)[-24:]
            if item
        )
        or "- none"
    )
    low_info_writer = backend.should_use_low_info_writer(task)
    focused_writer = _writer_prompt_mode(backend) == "focused"
    previous = (
        _focused_thread_ledger(
            backend,
            prior_comments,
            current_task=task,
            recent_openings=recent_openings or [],
        )
        if focused_writer or low_info_writer
        else _thread_memory(
            backend,
            prior_comments,
            current_task=task,
            domain_profile=domain_profile,
        )
    )
    anchors = _writer_visible_anchors(getattr(task, "concrete_anchors", ()))
    anchors_block = "\n".join(f"- {item}" for item in anchors[:8]) or "- none"
    retry = (
        f"\nThe previous attempt failed these guards:\n{retry_note}\n"
        if retry_note
        else ""
    )
    controls = backend.controls_for_task(task)
    controls_block = "\n".join(
        f"- {'perspective' if key == 'perspective_id' else key}: "
        f"{_writer_safe_control_text(value, domain_profile)}"
        for key, value in controls.items()
    )
    sampled_plan = _writer_safe_control_text(
        backend.render_sampled_plan_block(task),
        domain_profile,
    )
    semantic_contract = _semantic_realization_contract(
        backend,
        task,
        domain_profile=domain_profile,
    )
    route_lock = _semantic_route_lock(backend, task, domain_profile=domain_profile)
    prompt_tone_slot, prompt_tone_instruction = backend.real_tone_slot_for_prompt(task)
    tone_slot_rule = _optional_control_rule(
        "Real tone slot",
        prompt_tone_slot,
        prompt_tone_instruction,
        "Preserve the social move, not any reference wording.",
    )
    tone_target_rule = _optional_control_rule(
        "Tone target selector",
        getattr(task, "tone_target", ""),
        getattr(task, "tone_target_instruction", ""),
        "This controls attitude and social function only.",
    )
    story_instruction = str(getattr(task, "story_instruction", "") or "")
    if (
        str(getattr(backend, "GENERALIZED_SOCIAL_CONTRACT_COHERENCE", "on")) != "off"
        and str(getattr(task, "story_mode", "") or "") == "no_story"
    ):
        story_instruction = no_story_instruction()
    story_rule = _optional_control_rule(
        "Story realization",
        getattr(task, "story_mode", ""),
        story_instruction,
        "This controls narrative evidence structure, not interpersonal tone.",
    )
    affect_value = str(getattr(task, "affect_role", "") or "")
    affect_suffix = (
        "Do not add an evaluation, interjection, or hedge merely to display an "
        "emotion; preserve the assigned interpersonal register and local move."
        if affect_value == "neutral"
        else "Make this reaction part of the local move through a natural "
        "evaluation, interjection, or pacing choice. Do not name the label or "
        "add a second fact."
    )
    affect_rule = _optional_control_rule(
        "Affect role",
        affect_value,
        getattr(task, "affect_instruction", ""),
        affect_suffix,
    )
    surface_rule = _optional_control_rule(
        "Real surface skeleton",
        getattr(task, "surface_skeleton", ""),
        getattr(task, "surface_instruction", ""),
        "Use only the shape, never reference wording.",
    )
    rhythm_rule = _rhythm_rule(backend, task)
    register_rule = _register_rule(backend, task)
    evaluative_rule = _evaluative_rule(backend, task)
    closing_rule = _closing_rule(backend, task)
    hard_shape_rule = ""
    if backend.is_hard_real_surface_shape(getattr(task, "real_surface_shape", "")):
        hard_shape_rule = _real_surface_shape_guidance(task.real_surface_shape)
    assigned_opener = claim_for_task(
        getattr(backend, "GENERALIZED_OPENER_TYPES", {}) or {}, seed_post, task
    )
    opener_rule = _opener_rule(
        assigned_opener,
        drawn=_opening_move_clause(backend, task, assigned_opener),
        forbidden=forbidden_opening_tokens(),
    )
    domain_claim_rule = render_domain_claim_rule(
        claim_for_task(
            getattr(backend, "GENERALIZED_DOMAIN_CLAIMS", {}) or {},
            seed_post,
            task,
        )
    )
    guidance = "\n".join(
        item
        for item in (
            soft_length_guidance(task),
            opener_rule,
            domain_claim_rule,
            str(getattr(task, "must_not_do", "") or ""),
            _voice_guidance(task.voice),
            _speaker_role_guidance(task.speaker_role, task=task),
            _utterance_mode_guidance(task.utterance_mode, task=task),
            _surface_texture_guidance(task.surface_texture, task=task),
            _tone_shape_guidance(backend.resolved_tone_shape(task), task=task),
            hard_shape_rule,
            surface_rule,
            rhythm_rule,
            tone_slot_rule,
            tone_target_rule,
            register_rule,
            evaluative_rule,
            story_rule,
            affect_rule,
            closing_rule,
        )
        if item
    )

    # Rendered on every Writer path. v74 converted only the focused path and
    # left 106 of 522 slots on the old one, which made that release impossible
    # to attribute.
    speaker_block = _speaker_identity_block(backend, task, prior_comments)

    if focused_writer and not low_info_writer:
        return marker_prefix + _focused_writer_prompt(
            config,
            backend,
            task=task,
            visible=visible,
            route_lock=route_lock,
            slot_contract=_focused_slot_contract(
                backend,
                task,
                domain_profile=domain_profile,
            ),
            previous=previous,
            openings=openings,
            retry=retry,
            anchors_block=anchors_block,
            own_equipment=_equipment_and_referent_block(
                backend,
                task,
                has_domain_claim=bool(domain_claim_rule),
            )
            + _tone_donor_block(backend, task),
            speaker_block=speaker_block,
            domain_profile=domain_profile,
            domain_claim_rule=domain_claim_rule,
            opener_rule=opener_rule,
            tone_target_rule=tone_target_rule,
            story_rule=story_rule,
            affect_rule=affect_rule,
            surface_rule=surface_rule,
            rhythm_rule=rhythm_rule,
            register_rule=register_rule,
            evaluative_rule=evaluative_rule,
            closing_rule=closing_rule,
            actor_section=actor_section,
            actor_rule=(
                f"{actor_hard_rule}\n" if actor_hard_rule else ""
            ),
        )

    if low_info_writer:
        return marker_prefix + _low_info_writer_prompt(
            config,
            backend,
            seed_post=seed_post,
            task=task,
            parent_comment=parent_comment,
            visible=visible,
            previous=previous,
            openings=openings,
            retry=retry,
            guidance=guidance,
            speaker_block=speaker_block,
            actor_section=actor_section,
            actor_rule=(
                "- Express the assigned actor's local attention and interaction "
                "tendency without inventing biography, experience, or hidden facts."
                if actor_section
                else ""
            ),
        )

    return (
        marker_prefix
        + f"""Write exactly one human Reddit comment in {config.community_context}.

Use all controls as private constraints. Never mention them. This is one local turn, not a complete answer, product review, summary, or assistant response.

What this comment says (highest priority):
{route_lock}

Visible discussion:
{visible}

Private sampled controls:
{controls_block}

{actor_section}

Local anchor:
{backend.compact(_writer_safe_control_text(task.local_anchor, domain_profile), 220)}

Visible factual anchors:
{anchors_block}{_equipment_and_referent_block(backend, task, has_domain_claim=bool(domain_claim_rule))}{_tone_donor_block(backend, task)}{speaker_block}

Planner intent:
{backend.compact(_writer_safe_control_text(task.planner_intent, domain_profile), 260)}
{sampled_plan}

Structured thread blackboard:
{previous}

One-shot semantic difference contract:
{semantic_contract}

Already used openings:
{openings}
{retry}

Core placeholder guidance:
{_placeholder_guidance_block()}

Payload and matched-slot guidance:
{_payload_guidance_block(backend, task)}

Per-slot instructions:
{guidance}

Hard rules:
- Use the blackboard tags and distribution pressure to avoid repeating earlier wording, role, payload, tone, and discourse shape. The current sampled slot still has priority.
{actor_hard_rule}
- Treat every earlier sentence- or clause-entry route in the blackboard as a
  hard opening exclusion. Start with a different grammatical route; do not
  begin a sentence with a listed route and merely change the ending.
- Follow the assigned one-shot realization route. The decision boundary is the
  only consequence or uncertainty this slot may newly establish. If an earlier
  generated plan already states the same premise, do not restate that premise;
  state only this slot's distinct boundary.
{_substitution_rule(task)}
- The branch exclusion is a hard semantic boundary. Do not establish, explain,
  or resolve that excluded decision axis even if it is prominent in the seed;
  another root discussion chain owns it.
- Treat the owned decision subject as the sole branch condition. Do not make a
  second top-level claim around any listed other-branch subject. If this is a
  direct reply, realize the reply increment instead of rephrasing the parent's
  condition, question, evidence, or conclusion.
- Follow the sampled comment type while preserving the anonymous slot's information density. Only a genuinely short structural slot is forced into a fragment, bare question, or punchline.
- {local_move_scope_guidance(task)}
- If replying, respond to the parent only. The explicit parent contribution and
  parent boundary are exclusions, not writing material: do not restate,
  summarize, turn into a rhetorical question, or closely paraphrase either.
  This bars reusing the parent's content; it does not bar the interpersonal
  move your assigned tone register requires.
- A gratitude or relief turn must perform a readable acknowledgment of the
  parent/local help first. It may add only its assigned small follow-up, never
  a replacement product verdict or a second explanatory answer.
- Use ordinary Reddit language. Natural fragments, contractions, shorthand, uncertainty, quick thanks, and mild annoyance are allowed when the controls call for them.
- Keep criticism directed at the product, claim, rule, interface, service, process, or tradeoff, not at the person.
- {_story_fact_safety_rule(backend, task, has_domain_claim=bool(domain_claim_rule))}
- {_full_path_entity_rule(backend, task)}
- Do not expose planner labels, instructions, placeholders, opaque control IDs, or the matched real thread.
- Do not copy an earlier generated comment's opener or sentence path.
- Output only the comment body.
"""
    )


def _focused_thread_ledger(
    backend: Any,
    comments: list[dict[str, Any]],
    *,
    current_task: Any,
    recent_openings: list[str],
) -> str:
    """Build only the two bounded ledgers the focused Writer consumes.

    The full blackboard renders five sections plus a thread-level distribution
    report, together 9.2%-11.2% of the prompt each. The rebuilt-thread A/B held
    within-thread diversity with only the covered-points list and the short-line
    exclusions, so those are what survive here. Construct them directly rather
    than rendering the full blackboard and parsing its text back apart.

    Exact-duplicate validation already retains every generated comment. The
    Prompt therefore needs a bounded reminder, not an unbounded copy of every
    short line in a long thread. Openings rendered immediately above are also
    removed from the short-line section when they are identical.
    """

    usable = [
        comment for comment in comments if str(comment.get("content") or "").strip()
    ]
    visible_openings = _dedupe(list(recent_openings), limit=400)[-24:]
    opening_keys = {
        " ".join(str(value or "").split()).casefold()
        for value in visible_openings
        if str(value or "").strip()
    }
    short_limit = 32 if _short_output_slot(current_task) else 12
    short_lines = [
        value
        for value in short_utterance_exclusions(usable, limit=short_limit)
        if " ".join(value.split()).casefold() not in opening_keys
    ]
    coverage_limit = 8 if _short_output_slot(current_task) else 16
    coverage = semantic_coverage_entries(
        usable,
        limit=coverage_limit,
        current_task=current_task,
    )
    if str(getattr(current_task, "reply_delta_type", "") or "") == "social_close":
        current_move = " ".join(
            str(getattr(current_task, "semantic_move", "") or "").split()
        ).casefold()
        if current_move:
            prefix = f"move={current_move}"
            coverage = [
                value for value in coverage if not value.casefold().startswith(prefix)
            ]

    short_block = (
        "\n".join(f"- {backend.compact(value, 90)}" for value in short_lines)
        if short_lines
        else "- none yet"
    )
    coverage_block = (
        "\n".join(f"- {backend.compact(value, 90)}" for value in coverage)
        if coverage
        else "- none yet"
    )
    # Openings and short lines say nothing about a phrase reused in the middle of
    # a comment. `used_sentence_routes` is frequency-ranked and already bounded,
    # and the `full` arm has rendered it since v66, but `focused` has been the
    # active arm since v82 and never received it. Measured over the v97 N=10
    # output, the adjudication frame persists at 0.144 on slots that never see
    # the boundary line at all, and a late slot in the 91-comment thread was
    # shown 24 openings and 21 short lines and nothing about the route seven of
    # its predecessors had already taken.
    route_limit = 8 if _short_output_slot(current_task) else 16
    routes = [
        value
        for value in reused_sentence_routes(usable, limit=route_limit)
        if " ".join(value.split()).casefold() not in opening_keys
    ]
    route_block = (
        "Sentence routes already reused in this thread, so take a different one:\n"
        + "\n".join(f"- {backend.compact(value, 90)}" for value in routes)
        + "\n"
        if routes
        else ""
    )
    # Sibling of the route ledger, one order lower: routes are 4-grams reused
    # twice, these are the two-word grammatical sequences the thread has already
    # leaned on. Named exactly, because E4 measured that naming a concrete token
    # buys ~1.0 compliance where naming a category buys 0.23.
    phrases = recurring_function_phrases(usable)
    phrase_block = (
        "Two-word sequences this thread has already leaned on -- say it a "
        "different way:\n"
        + "\n".join(f"- {value}" for value in phrases)
        + "\n"
        if phrases
        else ""
    )
    coverage_nonrepeat = (
        f"{SEMANTIC_COVERAGE_NONREPEAT_INSTRUCTION}\n"
        if str(
            getattr(backend, "GENERALIZED_SEMANTIC_COVERAGE_NONREPEAT", "off")
        ).strip().lower()
        == "on"
        else ""
    )
    return (
        "Short utterances already used anywhere in this thread:\n"
        f"{short_block}\n"
        f"{route_block}"
        f"{phrase_block}"
        "Semantic contributions already covered in this thread:\n"
        f"{coverage_block}\n"
        f"{coverage_nonrepeat}"
    )


def _writer_prompt_mode(backend: Any) -> str:
    """Which Writer prompt to render.

    ``full`` reproduces policy v73 exactly. ``focused`` keeps the compact
    proposition, discourse-role, distribution, and grounding contracts without
    the full prompt's repeated control paraphrases. See `_focused_writer_prompt`.
    """

    return str(
        getattr(backend, "GENERALIZED_WRITER_PROMPT_MODE", "focused") or "focused"
    )


def _route_lock_mode(backend: Any) -> str:
    """How the slot's assigned proposition is handed to the Writer.

    ``say_only`` reproduces the v73/v74 wording, "Say this, and only this".
    ``own_words`` states the same requirement as a specification to realize.

    The v73 wording was chosen to kill the "that's the part that actually
    matters" frame, and it did not: the frame stayed at 7.2% while the Writer
    began reproducing the Planner's sentence instead. Longest contiguous shared
    word run between `semantic_move` and its comment, measured over four runs of
    ~520 slots, share at 12 words or more:

        v67 0.4%   v69 1.0%   v73 10.2%   v74 25.8%

    Restricted to comments of 25 words or more, v67 is 0.0% and v74 is 34.7%, so
    a healthy run does not do this at all. The Planner's move is a full
    first-person sentence in 19.3% of slots, and "say this, and only this" in
    front of a finished sentence is an instruction to copy it.
    """

    return str(
        getattr(backend, "GENERALIZED_WRITER_ROUTE_LOCK", "own_words") or "own_words"
    )


def _realization_rule(backend: Any) -> str:
    """State that the assigned move is a specification, not a draft.

    v74 dropped the semantic-difference contract from the Writer prompt because
    no metric depended on it. No metric measured plan echo, and echo went 10.2%
    -> 25.8% in the same release. Every prompt path that renders the route lock
    renders this too: applying half a fix to 80% of slots is what made v74's
    result impossible to attribute.
    """

    if _route_lock_mode(backend) == "say_only":
        return ""
    return (
        "- The point above is a specification of what to say, never wording to\n"
        "  reuse. Reach it with your own sentence shape and vocabulary; do not\n"
        "  repeat its phrasing back."
    )


def _focused_writer_prompt(
    config: DomainConfig,
    backend: Any,
    *,
    task: Any,
    visible: str,
    route_lock: str,
    slot_contract: str,
    previous: str,
    openings: str,
    retry: str,
    anchors_block: str,
    own_equipment: str,
    speaker_block: str,
    domain_profile: dict[str, Any],
    domain_claim_rule: str,
    opener_rule: str,
    tone_target_rule: str,
    story_rule: str,
    affect_rule: str,
    surface_rule: str,
    rhythm_rule: str,
    register_rule: str,
    evaluative_rule: str,
    closing_rule: str,
    actor_section: str = "",
    actor_rule: str = "",
) -> str:
    """Render the minimum complete Planner and grounding contract.

    The full prompt averaged 22,249 characters to produce a 56-word comment, and
    only 945 of those were identical across slots, so the size was control count
    rather than boilerplate. Rebuilding one whole 38-slot thread against a
    2,523-character prompt held within-thread diversity -- `self_bleu_4` 0.0466 ->
    0.0434 and `self_bertscore` 0.5279 -> 0.5226, both toward the thread's real
    0.0362 and 0.494 -- while the converged "the part that" frame went 2.6% -> 0%
    and mean length moved 26.3 -> 28.2 words against a real 32.8. A 30-slot
    single-comment A/B showed the same direction on register.

    What is kept, and why:
      route lock + branch exclusion   preserve the assigned proposition
      compact discourse contract      preserve the assigned conversational act
      length cue                      preserve anonymous slot scale
      tone target                     preserve interpersonal register
      affect role                      preserve emotional variation
      story mode                      preserve narrative evidence structure
      opener grammar                   vary clause-entry routes
      typing rhythm                    vary the sentence skeleton per slot
      anchors, equipment, entity rule  preserve factual grounding
      compressed thread ledger         avoid covered points and short repeats

    What is dropped, with nothing depending on it: the static metric-guidance
    block, the near-static tone/discourse guidance that restates the tone target,
    `planner_intent` and the full semantic-difference contract (the slot's own
    proposition was restated verbatim 3.4 times per prompt), the five overlapping
    surface-label paraphrases for voice/utterance/texture/tone_shape, the bulky
    placeholder and payload guidance blocks, and the bulk of the hard-rule list.
    Function, payload, speaker role, voice, evidence, stance, and the local exclusion
    stay once in a compact contract: without them, planned rants, datapoints,
    corrections, and bare reactions can fall back to generic helpful turns.
    """

    # `_story_fact_safety_rule` and `_substitution_rule` are not style guidance.
    # The first is the factual-grounding rule, which is a hard failure, and the
    # second carries the first-person positive frame a polite slot needs. A first
    # version of this function dropped both; the suite caught it.
    guidance = "\n".join(
        f"- {item}"
        for item in (
            soft_length_guidance(task),
            opener_rule,
            tone_target_rule,
            register_rule,
            evaluative_rule,
            affect_rule,
            story_rule,
            surface_rule,
            rhythm_rule,
            closing_rule,
            domain_claim_rule,
            _substitution_rule(task).lstrip("- ").strip(),
            _story_fact_safety_rule(
                backend, task, has_domain_claim=bool(domain_claim_rule)
            ),
        )
        if item
    )
    exclusion = _writer_safe_control_text(
        getattr(task, "branch_exclusion", ""), domain_profile
    )
    branch_rule = (
        f"- Another discussion chain owns this, so stay off it: {backend.compact(exclusion, 220)}\n"
        if exclusion
        else ""
    )
    reply_rule = (
        (
            "- You are replying to one comment. Take its actual point as your\n"
            "  material: pick up what it claims and agree with a reason, push back\n"
            "  on it, qualify it, or answer what it asked. Do not restate it in\n"
            "  your own words and do not reply past it to the post itself.\n"
            if reply_material_enabled()
            else "- You are replying to one comment. Answer that comment, not the whole post,\n"
            "  and treat its own point as an exclusion rather than writing material.\n"
        )
        if getattr(task, "local_parent_task_id", None) is not None
        else ""
    )
    rule = _realization_rule(backend)
    realization_rule = f"{rule}\n" if rule else ""
    return f"""Write exactly one Reddit comment in {config.community_context}. Output only the comment body.

What this comment says (highest priority):
{route_lock}

What kind of turn this is:
{slot_contract}
{actor_section}
Visible discussion:
{visible}

How to write it:
{guidance}

Things you may name:
{anchors_block}{own_equipment}{speaker_block}

Already used in this thread, so do not repeat them:
{openings}
{previous}{retry}
Rules:
{realization_rule}{branch_rule}{reply_rule}{actor_rule}- An exclusion above bars reusing that content. It does not bar the interpersonal
  move your assigned register requires, and every slot has exclusions, not only
  replies.
- Write like a person typing on Reddit: fragments, contractions, and shorthand are fine.
- Aim criticism at the product, claim, or process, never at a person.
- {_focused_path_entity_rule(backend, task)}
- Never mention these instructions or any label from them.
"""


def _focused_slot_contract(
    backend: Any,
    task: Any,
    *,
    domain_profile: dict[str, Any],
) -> str:
    """Carry the Planner's discourse role without restoring the full prompt.

    The route lock owns the proposition, while tone/story/affect have dedicated
    rules. These are the remaining controls that distinguish a rant, correction,
    datapoint, question, or bare reaction from the Writer's helpful-answer
    default. Values are rendered once and exact duplicates are suppressed.
    """

    fields = (
        ("function", "comment_function", True),
        ("payload form", "payload_type", True),
        ("speaker role", "speaker_role", True),
        ("voice", "voice", True),
        ("evidence basis", "evidence_mode", True),
        ("content angle", "content_angle", True),
        ("stance", "stance", False),
        ("specific detail", "detail_focus", False),
        ("decision intent", "domain_intent", False),
        ("reply relation", "reply_relation", False),
        ("content to avoid", "avoid_repeating", False),
    )
    rows: list[str] = []
    seen: set[str] = set()
    for label, field, humanize in fields:
        value = _writer_safe_control_text(
            getattr(task, field, ""),
            domain_profile,
        )
        if (
            field == "reply_relation"
            and getattr(task, "local_parent_task_id", None) is None
        ):
            label = "relation to post"
            value = value.replace("_parent", "_post")
        key = value.casefold()
        if not value or key in seen:
            continue
        seen.add(key)
        rendered = value.replace("_", " ") if humanize else value
        rows.append(f"- {label}: {backend.compact(rendered, 220)}")
    return "\n".join(rows) or "- one local participant turn"


def _low_info_writer_prompt(
    config: DomainConfig,
    backend: Any,
    *,
    seed_post: Any,
    task: Any,
    parent_comment: dict[str, Any] | None,
    visible: str,
    previous: str,
    openings: str,
    retry: str,
    guidance: str,
    speaker_block: str = "",
    actor_section: str = "",
    actor_rule: str = "",
) -> str:
    domain_profile = getattr(backend, "GENERALIZED_DOMAIN_PROFILE", {})
    relation = (
        "reply to the parent" if parent_comment is not None else "reply to the post"
    )
    route_lock = _semantic_route_lock(backend, task, domain_profile=domain_profile)
    donor_block = _tone_donor_block(backend, task)
    return f"""Write exactly one low-information Reddit comment in {config.community_context}.

What this comment says (highest priority):
{route_lock}{donor_block}

What kind of turn this is:
{_focused_slot_contract(backend, task, domain_profile=domain_profile)}

Visible discussion:
{visible}{speaker_block}

How to write it:
{guidance}

{actor_section}

Already used in this thread, so do not repeat them:
{openings}
{previous}
{retry}

Rules:
- {relation} only.
- Keep the comment low-information. Do not add advice, explanation, caveats, specifications, policies, or extra facts.
- Low-information controls the amount of text, not whether the semantic plan is
  visible. Express the required local move and do not repeat the explicitly
  excluded proposition or any used short utterance.
{_realization_rule(backend)}
- Preserve the sampled role, tone, payload, and social function; do not make it more helpful than planned.
{actor_rule}
- Keep it fragmentary when the payload is fragmentary. If one phrase is enough, use one phrase.
- Named entities and numbers may appear only when visible in the discussion.
- Never mention these instructions or any label from them.
- Return only the comment body.
"""


def render_parent_context(
    config: DomainConfig, backend: Any, *, parent_comment: dict[str, Any], task: Any
) -> str:
    base = f"Community context: {config.community_context}\n\n"
    text = backend.compact(str(parent_comment.get("content") or ""), 900)
    cue = backend.compact(
        task.detail_focus
        or task.local_topic
        or task.local_anchor
        or task.reply_relation,
        240,
    )
    transform = task.context_transform or "normal"
    if transform == "parent_hidden":
        return (
            base
            + f"Parent text is hidden. Local cue:\n{cue}\nUse only this cue and the sampled social relation."
        )
    if transform == "parent_gist":
        return (
            base
            + f"Parent gist:\nA commenter raised a local point around {cue}.\nReply only to this gist."
        )
    if transform == "minor_detail_focus":
        return (
            base
            + f"Visible minor parent detail:\n{cue}\nTreat it as the only visible hook."
        )
    if transform == "parent_jittered":
        return (
            base
            + f"Partial parent context:\n{backend.compact(mask_specifics(text), 360)}\nReply to the social move, not the wording."
        )
    return (
        base
        + f"Parent comment:\n{text}\nReply to this parent only; it is context, not text to reuse."
    )


def render_seed_context(
    config: DomainConfig, backend: Any, *, seed_post: Any, task: Any
) -> str:
    base = f"Community context: {config.community_context}\n\n"
    title = backend.compact(seed_post.title, 360)
    body = backend.compact(seed_post.body or seed_post.content, 1500)
    cue = backend.compact(
        task.detail_focus or task.local_topic or task.local_anchor, 260
    )
    transform = task.context_transform or "normal"
    if transform == "semantic_plan_only":
        return (
            base
            + f"Seed text is hidden. Sampled local cue:\n{cue}\nDo not reconstruct missing facts."
        )
    if transform == "minor_detail_focus":
        return (
            base
            + f"Seed title:\n{title}\n\nVisible minor hook:\n{cue}\nRespond only to this hook."
        )
    if transform == "seed_jittered":
        return (
            base
            + f"Partial seed context:\n{mask_specifics(title + ' ' + body[:500])}\n\nLocal cue:\n{cue}"
        )
    aperture = task.context_aperture or "seed_gist_only"
    if aperture == "full_seed":
        return base + f"Seed title:\n{title}\n\nSeed body:\n{body}"
    if aperture == "title_only":
        return (
            base + f"Seed title:\n{title}\nOnly the title and sampled plan are visible."
        )
    if aperture == "semantic_only":
        return base + f"Only this sampled local cue is visible:\n{cue}"
    return base + f"Seed title:\n{title}\n\nSeed gist:\n{seed_gist(config, seed_post)}"


def seed_gist(config: DomainConfig, seed_post: Any) -> str:
    facets = ", ".join(config.topic_facets[:6])
    return (
        f"The OP is discussing {config.display_name} around the title "
        f"'{str(seed_post.title).strip()}'. Plausible local branches include {facets}, "
        "plus clarification, disagreement, personal experience, jokes, and side details."
    )


def mask_specifics(text: str) -> str:
    value = re.sub(r"https?://\S+|www\.\S+", "[link]", str(text), flags=re.I)
    value = re.sub(r"[$€£]\s?\d[\d,.]*", "[amount]", value)
    value = re.sub(r"\b\d+(?:\.\d+)?\s?%", "[rate]", value)
    value = re.sub(r"\b\d{3,}\b", "[number]", value)
    return " ".join(value.split())


def strip_anchor_source(value: str) -> str:
    return re.sub(
        r"\s+\((?:matched real|planner|local|seed)\)\s*$", "", str(value)
    ).strip()


def extract_product_anchors(config: DomainConfig, text: str) -> list[str]:
    value = str(text or "")
    found: list[str] = []
    for term in config.protected_entity_terms:
        if re.search(rf"(?<!\w){re.escape(term)}(?!\w)", value, flags=re.I):
            found.append(term)
    # Model names often combine letters, digits, hyphens, and Roman suffixes.
    patterns = re.findall(
        r"\b(?:[A-Z][A-Za-z&+.-]*\s+){0,3}(?:[A-Za-z]*\d[A-Za-z0-9+./-]*|[A-Z]{2,}(?:\s+[A-Z0-9][A-Za-z0-9+./-]*){0,2})\b",
        value,
    )
    found.extend(item.strip() for item in patterns if len(item.strip()) >= 2)
    return _dedupe(found, limit=12)


def extract_term_anchors(config: DomainConfig, text: str) -> list[str]:
    value = str(text or "")
    found = [
        term
        for term in config.technical_terms
        if re.search(rf"(?<!\w){re.escape(term)}(?!\w)", value, flags=re.I)
    ]
    return _dedupe(found, limit=12)


def extract_concrete_anchors(
    config: DomainConfig,
    text: str,
    *,
    source_label: str = "",
    max_items: int = 12,
) -> list[str]:
    value = str(text or "")
    if not value.strip():
        return []
    anchors: list[str] = []
    anchors.extend(extract_product_anchors(config, value))
    anchors.extend(extract_term_anchors(config, value))
    anchors.extend(re.findall(r"https?://\S+|www\.\S+", value, flags=re.I))
    anchors.extend(
        re.findall(
            r"(?:[$€£]\s*\d[\d,.]*|\b\d+(?:\.\d+)?\s*(?:%|mm|cm|gb|tb|mah|hz|mp|fps|w|wh|kg|g|inch(?:es)?|x)\b)",
            value,
            flags=re.I,
        )
    )
    result = _dedupe(anchors, limit=max_items)
    if source_label:
        result = [f"{item} ({source_label})" for item in result]
    return result


def protected_entities(config: DomainConfig, text: str) -> set[str]:
    entities = {item.lower() for item in extract_product_anchors(config, text)}
    for domain in re.findall(
        r"\b(?:[a-z0-9-]+\.)+(?:com|org|net|io|co|ca|uk)\b", text, flags=re.I
    ):
        entities.add(domain.lower())
    return entities


def protected_numbers(text: str) -> set[str]:
    pattern = r"(?<![\w.])(?:[$€£]\s*\d[\d,.]*|\d+(?:\.\d+)?\s*%|\d+\s*/\s*\d+|\d+(?:\.\d+)?\s*(?:mm|gb|tb|mah|hz|mp|fps|w|wh|kg|g|inch(?:es)?|in))\b|(?<![\w.])\d+(?:\.\d+)?(?![\w.])"
    return {
        re.sub(r"\s+", "", item.lower())
        for item in re.findall(pattern, str(text), flags=re.I)
    }


# `off` (default) reproduces every version through v107: the "already
# covered" block lists prior semantic contributions with no instruction
# attached, unlike its sibling blocks, which already tell the Writer not to
# reuse what they list. Read against a real chain (v103 N=10, seed002
# comments 40-45): comment 45's own coverage block already surfaced all five
# earlier "compactness doesn't matter once it's in a bag" paraphrases
# verbatim -- the information was present, nothing told the Writer what to
# do with it, and comment 45 restated the same point a sixth time. `on`
# appends the same style of instruction its sibling blocks already carry.
#
# Applies to both prompt builders that render this block --
# `_focused_thread_ledger` (the live default: `--writer-prompt focused`,
# active since v82) and `_thread_memory` (the `full` arm, `--writer-prompt
# full`). The v108 gate on seed 8 shipped only touching `_thread_memory` and
# the flag never reached a single prompt on that run -- `focused` is the
# default and was never checked -- confirmed by grepping the run's own saved
# `generation_records.json` for the instruction string, 0 of 186. Fixed the
# same day; see `docs/DECISIONS.md` G23 and `tasks/lessons.md`.
SEMANTIC_COVERAGE_NONREPEAT_INSTRUCTION = (
    "Do not restate one of these already-covered points in different words. "
    "Add a genuinely new relation, consequence, caveat, or evidence type "
    "beyond what is listed here."
)


def _thread_memory(
    backend: Any,
    comments: list[dict[str, Any]],
    *,
    current_task: Any,
    domain_profile: dict[str, Any] | None = None,
) -> str:
    usable = [
        comment for comment in comments if str(comment.get("content") or "").strip()
    ]
    rows = []
    tail = usable[-8:]
    start_index = max(1, len(usable) - len(tail) + 1)
    for index, comment in enumerate(tail, start=start_index):
        tags = ", ".join(
            _comment_tags(comment, domain_profile=domain_profile or {}, compact=True)
        )
        semantic = "; ".join(
            value
            for value in (
                _memory_field(backend, "move", comment.get("semantic_move"), 70),
                _memory_field(backend, "stance", comment.get("stance"), 20),
            )
            if value
        )
        suffix = f"; {semantic}" if semantic else ""
        # Keep prior meaning and control coverage without replaying long text
        # that can prime the Writer to reuse its phrasing. Exact short lines
        # remain in the dedicated exclusion block below.
        rows.append(f"- #{index} depth={comment.get('depth', 0)} [{tags}]{suffix}")
    rendered = "\n".join(rows) or "- none yet"
    # Exact duplicates in long threads often come from short replies generated
    # far earlier than the recent-text window. All current-thread short lines
    # fit comfortably in the Writer context and must remain explicitly visible.
    # Every ledger below is capped at a constant, not scaled by thread length.
    # Scaled caps made the blackboard 81% of the Writer prompt by comment 140,
    # leaving the slot's own assignment at 19%, and nothing in the rule mass was
    # being attended to. A bounded, relevance-ranked ledger is what the Writer
    # can actually use.
    # A long slot cannot reproduce a five-word line, so the complete short-line
    # ledger is kept only for the slots that can. This preserves the
    # exact-duplicate invariant for long threads at no cost on long slots, where
    # the same list was pure prompt mass.
    short_exclusions = short_utterance_exclusions(
        usable,
        limit=max(32, len(usable)) if _short_output_slot(current_task) else 12,
    )
    short_block = (
        "\n".join(f"- {backend.compact(value, 90)}" for value in short_exclusions)
        if short_exclusions
        else "- none yet"
    )
    # These caps are measured, not guessed. A whole 38-slot thread was rebuilt
    # against a 2,523-character prompt carrying a ledger of this size, and
    # within-thread diversity did not regress: self_bleu_4 0.0466 -> 0.0434 and
    # self_bertscore 0.5279 -> 0.5226, both moving toward the thread's real
    # 0.0362 / 0.494, while the converged "the part that" frame went 2.6% -> 0%
    # and length moved from 26.3 to 28.2 words against a real 32.8. The ledger
    # was 78% of a 19,117-character prompt and the extra mass bought nothing.
    coverage = semantic_coverage_entries(
        usable,
        limit=16,
        current_task=current_task,
    )
    coverage_block = (
        "\n".join(f"- {backend.compact(value, 90)}" for value in coverage)
        if coverage
        else "- none yet"
    )
    sentence_routes = used_sentence_routes(usable, limit=12)
    route_block = (
        "\n".join(f"- {backend.compact(value, 60)}" for value in sentence_routes)
        if sentence_routes
        else "- none yet"
    )
    # E15: the `focused` builder is the shipped default and the `full` builder
    # is the one that keeps getting forgotten. Both render it or neither does.
    phrases = recurring_function_phrases(usable)
    phrase_block = (
        "Two-word sequences this thread has already leaned on -- say it a "
        "different way:\n"
        + "\n".join(f"- {value}" for value in phrases)
        + "\n\n"
        if phrases
        else ""
    )
    coverage_nonrepeat = (
        f"{SEMANTIC_COVERAGE_NONREPEAT_INSTRUCTION}\n"
        if str(
            getattr(backend, "GENERALIZED_SEMANTIC_COVERAGE_NONREPEAT", "off")
        ).strip().lower()
        == "on"
        else ""
    )
    return (
        "Earlier generated comments (generated text and private controls only):\n"
        f"{rendered}\n\n"
        "Short utterances already used anywhere in this thread:\n"
        f"{short_block}\n"
        "Do not output one of these lines again or a trivial polarity-swapped paraphrase.\n\n"
        "Semantic contributions already covered in this thread:\n"
        f"{coverage_block}\n"
        f"{coverage_nonrepeat}\n"
        "Sentence- or clause-entry routes already used in this thread:\n"
        f"{route_block}\n"
        f"{phrase_block}"
        "Do not reuse one of these clause paths; keep domain entities when the local point needs them.\n\n"
        "Thread-level distribution pressure:\n"
        f"{_distribution_pressure(backend, usable, current_task=current_task, domain_profile=domain_profile or {})}"
    )


def _semantic_realization_contract(
    backend: Any,
    task: Any,
    *,
    domain_profile: dict[str, Any],
) -> str:
    rows = []
    for label, value in semantic_contract_values(task):
        safe = _writer_safe_control_text(value, domain_profile)
        if safe:
            limit = 1200 if label == "development sequence" else 220
            rows.append(f"- {label}: {backend.compact(safe, limit)}")
    if not rows:
        return "- make one parent- or seed-grounded local contribution"
    rows.append(
        "- realization rule: express this increment once; do not substitute a nearby thread conclusion, generic acknowledgement, or paraphrase"
    )
    return "\n".join(rows)


def _semantic_route_lock(
    backend: Any,
    task: Any,
    *,
    domain_profile: dict[str, Any],
) -> str:
    """Render the one proposition a single Writer call is allowed to add.

    This is intentionally shorter and earlier than the full blackboard. It
    prevents a salient seed question or parent wording from displacing the
    Planner's already-assigned decision boundary in high-fanout discussions.
    """

    move = _writer_safe_control_text(
        getattr(task, "semantic_move", "")
        or getattr(task, "decision_boundary", "")
        or getattr(task, "planner_intent", ""),
        domain_profile,
    )
    boundary = _writer_safe_control_text(
        getattr(task, "decision_boundary", "")
        or getattr(task, "owned_decision_subject", ""),
        domain_profile,
    )
    # The framing here is what the comment ends up sounding like. Asking a model
    # to "write the one new proposition this slot owns" and to name "the only
    # decision boundary you may establish" produces "that's the part that
    # actually matters": measured on v72, that frame is in 20% of generated
    # comments and 0 times in 39,265 tokens of matched real ones, and it is
    # already 20% in the first comment of a thread, so it is not an echo of
    # earlier output. Say what the turn is about in ordinary words instead.
    # See `_route_lock_mode` for the measurement behind the two wordings. The
    # Planner hands over a finished sentence often enough that the verb in front
    # of it decides whether the Writer realizes it or transcribes it.
    if _route_lock_mode(backend) == "say_only":
        rows = [
            "- Say this, and only this: " + backend.compact(move, 260),
        ]
    else:
        rows = [
            "- The point this comment makes, in your own words: "
            + backend.compact(move, 260),
        ]
    if boundary and turn_settles_a_question(task):
        rows.append(
            "- The question your turn settles: " + backend.compact(boundary, 240)
        )
    reply_delta = _writer_safe_control_text(
        getattr(task, "reply_delta", ""), domain_profile
    )
    novelty_anchor = _writer_safe_control_text(
        getattr(task, "reply_novelty_anchor", ""), domain_profile
    )
    reply_delta_type = _writer_safe_control_text(
        getattr(task, "reply_delta_type", ""), domain_profile
    ).casefold()
    if getattr(task, "local_parent_task_id", None) is not None:
        if reply_delta:
            rows.append(
                "- You are replying. What you add: " + backend.compact(reply_delta, 240)
            )
        if novelty_anchor:
            # The previous rewording of this line ("This concrete object is what
            # your turn adds beyond the parent") did not remove the echo; the
            # frame was still at 20% in v72. The whole block is what the Writer
            # imitates, so this names the thing plainly instead of describing
            # its role in an argument.
            rows.append("- Specifically: " + backend.compact(novelty_anchor, 220))
        realization_by_type = {
            "operational_test": (
                "- Reply form: state one concrete action or observation that could decide "
                "the issue. Do not give the parent's verdict, condition, or agreement."
            ),
            "observable_failure": (
                "- Reply form: name one visible failure signal or boundary case. Do not "
                "restate the parent's general concern."
            ),
            "evidence_requirement": (
                "- Reply form: name the evidence needed before acting and what remains "
                "unjustified without it. Do not repeat the parent's recommendation."
            ),
            "scope_limit": (
                "- Reply form: state the narrower context where the parent no longer "
                "settles the decision. Do not rephrase its main condition."
            ),
            "downstream_consequence": (
                "- Reply form: state one practical effect that follows after the choice. "
                "Do not restate why the parent thinks the choice matters."
            ),
            "countercondition": (
                "- Reply form: state one exception that reverses or limits the parent's "
                "conclusion. Do not restate the conclusion itself."
            ),
            "corroborating_datapoint": (
                "- Reply form: report your own concrete experience of the parent's "
                "situation and what it showed. The experience is the new object; do "
                "not restate the parent's claim as your conclusion."
            ),
            "useful_extension": (
                "- Reply form: supply the adjacent practical detail the parent leaves "
                "out and say why it matters here. Do not re-argue the parent's point."
            ),
            "endorsement_with_reason": (
                "- Reply form: commit to the parent's option and name the specific "
                "reason it works. The reason is the new object; agreement alone is "
                "not the increment."
            ),
            "social_close": (
                "- Reply form: make only a brief human acknowledgement. Add no factual "
                "claim, recommendation, evidence, or restatement."
            ),
        }
        if reply_delta_type in realization_by_type:
            rows.append(realization_by_type[reply_delta_type])
        # The parent's proposition is rendered once, as the structured
        # "parent contribution not to restate" field. Repeating it here made the
        # same sentence appear three times in one prompt under three different
        # prohibitions, and the reply-form line above already names the specific
        # thing this increment type must not repeat.
    else:
        rows.append(
            "- Do not replace this branch-specific contribution with a generic overall answer to the OP."
        )
    return "\n".join(rows)


def _memory_field(backend: Any, label: str, value: Any, limit: int) -> str:
    text = " ".join(str(value or "").split())
    return f"{label}={backend.compact(text, limit)}" if text else ""


def _comment_tags(
    comment: dict[str, Any],
    *,
    domain_profile: dict[str, Any] | None = None,
    compact: bool = False,
) -> list[str]:
    values = (
        ("role", comment.get("speaker_role") or "commenter"),
        ("tone", comment.get("tone_shape") or comment.get("voice") or "neutral"),
        ("payload", comment.get("payload_type") or "local_turn"),
        ("utterance", comment.get("utterance_mode") or "local_point"),
        ("voice", comment.get("voice") or "casual_neutral"),
        ("story", comment.get("story_mode") or "no_story"),
        ("affect", comment.get("affect_role") or "neutral"),
        ("length", comment.get("length_bucket") or "unknown"),
        ("shape", _discourse_shape(comment)),
        (
            "perspective",
            _perspective_label(
                domain_profile or {}, comment.get("perspective_id") or "seed_local"
            ),
        ),
        ("claim", comment.get("claim_key") or "local_claim"),
    )
    if compact:
        # The Writer's blackboard needs the axes it must vary against, not every
        # recorded control. The full set is retained for audits and tests.
        wanted = {"role", "payload", "affect", "shape", "claim"}
        values = tuple(item for item in values if item[0] in wanted)
    return [f"{key}={value}" for key, value in values]


def _discourse_shape(comment: dict[str, Any]) -> str:
    skeleton = str(comment.get("surface_skeleton") or "").strip()
    if skeleton:
        return skeleton
    texture = str(comment.get("surface_texture") or "")
    tone = str(comment.get("tone_shape") or "")
    payload = str(comment.get("payload_type") or "")
    role = str(comment.get("speaker_role") or "")
    if texture == "link_reference":
        return "reference_aside"
    if tone == "soft_ack" or role == "gratitude_reply":
        return "soft_acknowledgement"
    if tone == "personal_dp" or payload in {"fragment_datapoint", "personal_story"}:
        return "datapoint"
    if tone == "plain_question" or payload == "narrow_question":
        return "question_nudge"
    if tone == "mild_caveat":
        return "mild_caveat"
    if tone == "direct_correction":
        return "local_correction"
    if tone == "rant":
        return "complaint"
    if tone == "light_joke":
        return "joke_aside"
    if tone == "bare_answer":
        return "plain_answer"
    return "neutral_observation"


def _distribution_pressure(
    backend: Any,
    comments: list[dict[str, Any]],
    *,
    current_task: Any,
    domain_profile: dict[str, Any] | None = None,
) -> str:
    advice_like = sum(1 for comment in comments if _is_advice_like(comment))
    question_like = sum(1 for comment in comments if _is_question_like(comment))
    social_like = sum(1 for comment in comments if _is_social_like(comment))
    story_like = sum(1 for comment in comments if _is_story_like(comment))
    blunt_like = sum(1 for comment in comments if _is_blunt_like(comment))
    affect_counts = _count_values(comments, "affect_role")
    story_counts = _count_values(comments, "story_mode")
    role_counts = _count_values(comments, "speaker_role")
    tone_counts = _count_values(comments, "tone_shape", fallback="voice")
    payload_counts = _count_values(comments, "payload_type")
    raw_perspective_counts = _count_values(comments, "perspective_id")
    perspective_counts: dict[str, int] = {}
    for perspective, count in raw_perspective_counts.items():
        label = _perspective_label(domain_profile or {}, perspective)
        perspective_counts[label] = perspective_counts.get(label, 0) + count
    claim_counts = _count_values(comments, "claim_key")
    opening_counts = opening_route_counts(comments)
    phrase_counts = repeated_phrase_counts(comments)
    shape_counts: dict[str, int] = {}
    for comment in comments[-12:]:
        shape = _discourse_shape(comment)
        shape_counts[shape] = shape_counts.get(shape, 0) + 1
    current = ", ".join(
        str(value)
        for value in (
            current_task.speaker_role,
            backend.resolved_tone_shape(current_task)
            if hasattr(backend, "resolved_tone_shape")
            else getattr(current_task, "tone_shape", "") or current_task.voice,
            current_task.payload_type,
            current_task.utterance_mode,
            current_task.voice,
            getattr(current_task, "story_mode", ""),
            getattr(current_task, "affect_role", ""),
            current_task.length_bucket,
            _discourse_shape(
                {
                    "surface_skeleton": getattr(current_task, "surface_skeleton", ""),
                    "surface_texture": current_task.surface_texture,
                    "tone_shape": getattr(current_task, "tone_shape", ""),
                    "payload_type": current_task.payload_type,
                    "speaker_role": current_task.speaker_role,
                }
            ),
            _perspective_label(
                domain_profile or {},
                getattr(current_task, "perspective_id", "") or "seed_local",
            ),
            getattr(current_task, "claim_key", ""),
            getattr(current_task, "domain_intent", ""),
            getattr(current_task, "opening_style", ""),
        )
        if value
    )
    return "\n".join(
        [
            f"- So far: total={len(comments)}, advice_like={advice_like}, question_like={question_like}, "
            f"social_ack={social_like}, story_like={story_like}, blunt_or_annoyed={blunt_like}.",
            f"- Role distribution: {_render_counts(role_counts)}.",
            f"- Tone distribution: {_render_counts(tone_counts)}.",
            f"- Story-mode distribution: {_render_counts(story_counts)}.",
            f"- Affect-role distribution: {_render_counts(affect_counts)}.",
            f"- Payload distribution: {_render_counts(payload_counts)}.",
            f"- Perspective distribution: {_render_counts(perspective_counts)}.",
            f"- Claim distribution: {_render_counts(claim_counts)}.",
            f"- Repeated generated opening routes: {_render_counts(opening_counts)}.",
            f"- Repeated generated four-grams: {_render_counts(phrase_counts)}.",
            f"- Recent discourse shapes: {_render_counts(shape_counts)}.",
            f"- Current sampled slot: {current or 'ordinary local turn'}. Obey this slot first.",
            "- Avoid the dominant recent perspective, claim, opener, role posture, payload, tone, affect role, story shape, and discourse shape when the sampled slot permits a natural alternative.",
            "- If the thread already leans helpful or explanatory, keep this turn narrow, local, or socially reactive when allowed.",
            "- If the thread already leans fragmentary or noisy, keep this turn substantive when the sampled slot calls for advice, correction, datapoint, or story.",
        ]
    )


def _perspective_label(profile: dict[str, Any], perspective_id: Any) -> str:
    value = str(perspective_id or "seed_local").strip()
    if value == "seed_local":
        return "seed-local perspective"
    for item in profile.get("perspectives") or []:
        if not isinstance(item, dict):
            continue
        if str(item.get("perspective_id") or "").upper() == value.upper():
            return (
                str(item.get("label") or "domain perspective").strip()
                or "domain perspective"
            )
    if INTERNAL_CONTROL_ID_RE.fullmatch(value):
        return "domain perspective"
    return value


def _writer_safe_control_text(value: Any, profile: dict[str, Any]) -> str:
    text = str(value or "")
    perspective_labels = {
        str(item.get("perspective_id") or "").upper(): str(
            item.get("label") or "domain perspective"
        )
        for item in profile.get("perspectives") or []
        if isinstance(item, dict) and item.get("perspective_id")
    }

    def replace_control(match: re.Match[str]) -> str:
        label = match.group(0).upper()
        if label.startswith("P"):
            return perspective_labels.get(label, "domain perspective")
        return "prior structural slot" if label.startswith("S") else "local branch"

    return re.sub(
        r"\s+", " ", INTERNAL_CONTROL_ID_RE.sub(replace_control, text)
    ).strip()


def _writer_visible_anchors(values: Any) -> list[str]:
    anchors = []
    for value in values or ():
        raw = str(value)
        base = strip_anchor_source(raw)
        source_is_seed = raw.rstrip().lower().endswith("(seed)")
        if INTERNAL_CONTROL_ID_RE.fullmatch(base) and not source_is_seed:
            continue
        anchors.append(base)
    return anchors


# v147 arm. Until now the Planner saw the matched real thread only as shape:
# depth, parent, word count, and a surface label per slot -- never a word of what
# those people actually said. Every attempt to teach it how scattered, how blunt,
# how short real comments are went through proxies instead: a behaviour target
# distilled to one number, a quota phrased as an instruction, a window of
# comments from OTHER threads. The thread it is reproducing was sitting right
# there, unread.
#
# `measured` renders the matched comment bodies alongside the structure. This is
# the one thing ORIENTATION.md s7 forbids -- "The Writer never sees matched
# evaluation comment text" -- so a run with it on is a LEAK ARM: it cannot be
# pooled with, or compared against, any held-out release, and it can never be
# the shipped configuration. It answers one question only: with the real thread
# in front of it, can the Planner reproduce the scatter, and how much of the gap
# does that close? The Writer is a separate model and still never sees this text;
# it reaches only the Planner.
MATCHED_TEXT_MODE = "off"
MATCHED_TEXT_MAX_WORDS = 60


def set_matched_text(mode: str) -> bool:
    global MATCHED_TEXT_MODE
    MATCHED_TEXT_MODE = str(mode or "off").strip().lower()
    return MATCHED_TEXT_MODE == "measured"


def _render_matched_structure(
    thread: dict[str, Any] | None,
    *,
    max_comments: int,
) -> str:
    if not thread:
        return "(matched structural sample unavailable)"
    rows = thread.get("comments") or []
    id_to_slot: dict[str, int] = {}
    visible_rows = rows if max_comments <= 0 else rows[:max_comments]
    for index, row in enumerate(visible_rows, start=1):
        for key in (row.get("comment_id"), row.get("id"), row.get("name")):
            if key:
                id_to_slot[str(key)] = index
                id_to_slot[f"t1_{key}"] = index
    lines = []
    for index, row in enumerate(visible_rows, start=1):
        body = str(row.get("body") or "").strip()
        parent_raw = str(row.get("parent_id") or "")
        parent = (
            "OP"
            if parent_raw.startswith("t3_")
            else f"S{id_to_slot[parent_raw]}"
            if parent_raw in id_to_slot
            else "outside_sample"
        )
        line = (
            f"S{index}: depth={int(row.get('depth') or 0)}; parent={parent}; "
            f"words={len(body.split())}; surface={_surface_only_label(body)}"
        )
        if MATCHED_TEXT_MODE == "measured" and body:
            words = body.split()
            shown = " ".join(words[:MATCHED_TEXT_MAX_WORDS])
            if len(words) > MATCHED_TEXT_MAX_WORDS:
                shown += " ..."
            line += f"; text={shown}"
        lines.append(line)
    return "\n".join(lines) or "(matched structural sample empty)"


def _render_matched_slots(
    comments: list[dict[str, Any]],
    *,
    all_comments: list[dict[str, Any]] | None,
    sample_offset: int,
) -> str:
    reference = all_comments if all_comments is not None else comments
    parent_offset = 0 if all_comments is not None else sample_offset
    id_to_slot: dict[str, int] = {}
    for local_index, row in enumerate(reference, start=1):
        slot = parent_offset + local_index
        for key in (row.get("comment_id"), row.get("id"), row.get("name")):
            if key:
                id_to_slot[str(key)] = slot
                id_to_slot[f"t1_{key}"] = slot
    lines = []
    for local_index, row in enumerate(comments, start=1):
        slot = sample_offset + local_index
        body = str(row.get("body") or "").strip()
        parent_raw = str(row.get("parent_id") or "")
        parent = (
            "OP"
            if parent_raw.startswith("t3_")
            else f"S{id_to_slot[parent_raw]}"
            if parent_raw in id_to_slot
            else "outside_sample"
        )
        words = len(body.split())
        # Naming the required beat count per slot instead of leaving it to a
        # general rule: long slots were silently returned with no
        # development_plan and then realized at a fraction of their scale.
        beats = expected_development_beats(words)
        development = f"; development_plan={beats} beats required" if beats > 0 else ""
        lines.append(
            f"S{slot}: depth={int(row.get('depth') or 0)}; parent={parent}; "
            f"words={words}; surface={_surface_only_label(body)}{development}"
        )
    return "\n".join(lines)


def _surface_only_label(text: str) -> str:
    return surface_only_label(text)


def _count_values(
    comments: list[dict[str, Any]],
    key: str,
    *,
    fallback: str = "",
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for comment in comments:
        value = str(
            comment.get(key) or (comment.get(fallback) if fallback else "") or "unknown"
        )
        counts[value] = counts.get(value, 0) + 1
    return counts


def _render_counts(counts: dict[str, int], *, limit: int = 5) -> str:
    if not counts:
        return "none"
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:limit]
    return ", ".join(f"{key}={value}" for key, value in ranked)


def _is_advice_like(comment: dict[str, Any]) -> bool:
    role = str(comment.get("speaker_role") or "")
    payload = str(comment.get("payload_type") or "")
    function = str(comment.get("comment_function") or comment.get("comment_job") or "")
    text = str(comment.get("content") or "").lower()
    return (
        role == "advisor"
        or "advice" in payload
        or "advice" in function
        or bool(
            re.search(
                r"\b(you should|you can|i would|try|check|make sure|worth)\b", text
            )
        )
    )


def _is_question_like(comment: dict[str, Any]) -> bool:
    role = str(comment.get("speaker_role") or "")
    payload = str(comment.get("payload_type") or "")
    utterance = str(comment.get("utterance_mode") or "")
    return (
        role == "confused_asker"
        or "question" in payload
        or "question" in utterance
        or "?" in str(comment.get("content") or "")
    )


def _is_social_like(comment: dict[str, Any]) -> bool:
    role = str(comment.get("speaker_role") or "")
    voice = str(comment.get("voice") or "")
    texture = str(comment.get("surface_texture") or "")
    text = str(comment.get("content") or "").lower()
    return (
        role == "gratitude_reply"
        or voice == "grateful"
        or texture == "gratitude_social"
        or bool(
            re.search(
                r"\b(thanks|thank you|appreciate|good to know|that helps|fair point)\b",
                text,
            )
        )
    )


def _is_story_like(comment: dict[str, Any]) -> bool:
    story_mode = str(comment.get("story_mode") or "")
    payload = str(comment.get("payload_type") or "")
    evidence = str(comment.get("evidence_mode") or "")
    return (
        story_mode not in {"", "no_story"}
        or payload == "personal_story"
        or evidence == "firsthand_experience"
    )


def _is_blunt_like(comment: dict[str, Any]) -> bool:
    role = str(comment.get("speaker_role") or "")
    voice = str(comment.get("voice") or "")
    text = str(comment.get("content") or "").lower()
    return (
        role in {"contrarian", "ranter"}
        or voice in {"blunt", "annoyed", "sarcastic"}
        or bool(
            re.search(
                r"\b(nope|hard pass|wtf|bullshit|lol|lmao|wrong|doubtful)\b", text
            )
        )
    )


def _optional_control_rule(
    label: str, value: str, instruction: str, suffix: str
) -> str:
    if not value:
        return ""
    parts = [f"{label}: {value}."]
    if instruction:
        parts.append(str(instruction).strip())
    if suffix:
        parts.append(suffix)
    return " ".join(parts)


def _full_path_entity_rule(backend: Any, task: Any) -> str:
    """The full Writer path's entity rule, worded for that path."""

    return entity_naming_rule(mode=writer_grounding_mode(backend, task), variant="full")


def _focused_path_entity_rule(backend: Any, task: Any) -> str:
    """The focused Writer path's entity rule, worded for that path."""

    return entity_naming_rule(
        mode=writer_grounding_mode(backend, task), variant="focused"
    )


def _story_fact_safety_rule(
    backend: Any, task: Any, *, has_domain_claim: bool = False
) -> str:
    """Render the slot's factual-grounding rule.

    The rule itself lives in `writer_grounding`, which owns the split between
    facts about the product under discussion and facts about the speaker's own
    kit and history. All three Writer paths call through here.
    """

    return story_fact_rule(
        task,
        has_domain_claim=has_domain_claim,
        mode=writer_grounding_mode(backend, task),
    )


def _substitution_rule(task: Any | None = None) -> str:
    """Forbid substituting the planned move, without forbidding the tone register.

    A blanket ban on acknowledgement and first-person framing also removed the
    only surfaces through which the warm register is realized, so the ban is
    scoped to the registers that actually exclude those surfaces.
    """

    assigned = str(getattr(task, "tone_target", "") or "").strip().lower()
    if assigned == "polite":
        return (
            "- Do not replace the planned move with a bare recommendation or a\n"
            "  content-free agreement. An appreciative acknowledgement or a\n"
            "  first-person positive frame is required here: carry the planned\n"
            "  move through it rather than instead of it."
        )
    if assigned == "somewhat_polite":
        return (
            "- Do not replace the planned move with a generic recommendation. A\n"
            "  brief qualified concession is required here, but it must lead into\n"
            "  the planned move, not stand in for it."
        )
    if bool(getattr(task, "allow_first_person_frame", False)):
        return (
            "- Do not replace the planned move with a generic agreement, "
            "acknowledgement, or recommendation. A first-person frame may carry "
            "the assigned current-state appraisal, but it must not become an "
            "unplanned anecdote."
        )
    return (
        "- Do not replace it with a generic agreement, acknowledgement,\n"
        "  recommendation, or first-person frame."
    )


def _placeholder_guidance_block() -> str:
    """Forbid invented sources. Under `--reference-link measured` a real URL may
    be supplied for a slot whose matched comment carried one, so the rule has to
    say *invent* rather than *never write*, or the Writer receives the offer and
    a prohibition on using it in the same prompt -- the failure v112 was fixed
    for. The `off` text is unchanged from every release through v112.
    """

    if reference_link_enabled():
        return """- Never output planner labels, control names, skeleton labels, placeholders, fake resource titles, or inferred quote headings.
- Never invent a URL or a source name. If this slot supplies an exact URL, use that one and no other; otherwise write a normal human reference sentence with no URL at all.
- Use a markdown quote marker only for exact text visible in the current discussion."""
    return """- Never output planner labels, control names, skeleton labels, placeholders, fake resource titles, or inferred quote headings.
- If a source, wiki, sidebar, link, or template is requested but no exact text is visible, write a normal human reference sentence without inventing a URL.
- Use a markdown quote marker only for exact text visible in the current discussion."""


def _payload_guidance_block(backend: Any, task: Any) -> str:
    lines: list[str] = []
    if backend.is_meta_template_task(task):
        lines.extend(
            [
                "- Meta/template slot: keep the reference or community nudge abrupt and local.",
                "- Do not invent resource names, URLs, template text, or a paragraph explaining the resource.",
            ]
        )
    if backend.is_gpt_long_helpful_task(task):
        lines.extend(
            [
                "- Long helpful slot: earn the length with one or two visible concrete anchors, a narrow caveat, a firsthand datapoint, or a parent-specific reason.",
                "- If no concrete anchor is visible, compress the reply instead of filling space with an abstract overview.",
            ]
        )
    if substantive_surface_slot(task):
        lines.extend(
            [
                "- Matched substantive slot: do not turn it into a tiny reaction, joke label, or acknowledgement.",
                "- Preserve the local function and use visible anchors for detail density without copying reference wording.",
            ]
        )
    return (
        "\n".join(lines)
        or "- Preserve the sampled payload exactly; do not upgrade or broaden it."
    )


def _voice_guidance(voice: str) -> str:
    mapping = {
        "polite_soft": "Voice: show a light local acknowledgement or caveat without customer-support polish.",
        "grateful": "Voice: use a small, specific, casual sign of appreciation.",
        "annoyed": "Voice: let mild frustration show through the local friction, not contempt.",
        "sarcastic": "Voice: keep any dry aside brief, low-stakes, and aimed at the situation.",
        "blunt": "Voice: be direct and concise without becoming hostile or judge-like.",
        "uncertain": "Voice: sound exploratory rather than authoritative.",
    }
    return mapping.get(voice, "Voice: use a natural Reddit tone for this local point.")


def _speaker_role_guidance(role: str, *, task: Any | None = None) -> str:
    substantive = substantive_surface_slot(task)
    mapping = {
        "confused_asker": "Role: ask from uncertainty; do not answer the question yourself.",
        "op_followup": "Role: clarify, react, thank, or report back like an imperfect OP follow-up.",
        "gratitude_reply": "Role: acknowledge help or report back briefly; do not add a lecture.",
        "jokester": (
            "Role: use humor as the posture around the assigned local contribution; preserve the substantive slot instead of reducing it to a punchline."
            if substantive
            else "Role: make the joke or aside the whole point and do not explain it."
        ),
        "mod_meta": (
            "Role: keep the reference or community context inside the assigned substantive local contribution; do not invent template text."
            if substantive
            else "Role: write a community/meta/template-style comment, not product advice."
        ),
        "contrarian": "Role: push back on one local point without turning it into balanced advice.",
        "datapoint_only": "Role: give one compact datapoint or lived detail, not a recommendation.",
        "ranter": "Role: let the complaint stand without resolving it into a helpful takeaway.",
        "side_observer": "Role: stay on the small side observation instead of solving the OP's main problem.",
    }
    return mapping.get(
        role,
        "Role: make one local contribution rather than acting as a general advisor.",
    )


def _utterance_mode_guidance(mode: str, *, task: Any | None = None) -> str:
    substantive = substantive_surface_slot(task)
    mapping = {
        "fragment_only": "Utterance: write a fragment or tiny reaction without background or advice.",
        "direct_answer": "Utterance: answer directly in one small point.",
        "question_only": (
            "Utterance: embed the assigned question in a substantive local turn; preserve the slot's context and pacing."
            if substantive
            else "Utterance: ask only the narrow question; do not answer it."
        ),
        "one_datapoint": "Utterance: give one datapoint or lived detail without recommendation.",
        "op_followup": "Utterance: react, clarify, thank, or report back; do not become an advisor.",
        "joke_only": (
            "Utterance: let humor color the assigned substantive local move without collapsing it to a one-line joke."
            if substantive
            else "Utterance: make the joke or aside the whole comment."
        ),
        "template_notice": (
            "Utterance: keep the reference or meta cue within the assigned substantive local move."
            if substantive
            else "Utterance: stay meta/template-like; do not add product advice."
        ),
        "complaint_only": "Utterance: let the complaint stand without a solution paragraph.",
        "side_tangent": "Utterance: stay on the side detail.",
        "correction_only": "Utterance: correct one local detail and stop.",
        "local_answer_with_context": "Utterance: state the assigned answer directly, then preserve only the local context needed by this substantive slot.",
        "question_with_context": "Utterance: ground the assigned uncertainty in its local context, then ask the question without answering it.",
        "humorous_local_turn": "Utterance: realize the assigned substantive local move with incidental humor rather than reducing it to a punchline.",
        "reference_with_context": "Utterance: explain the local relevance of the reference cue without inventing a source, URL, or template.",
    }
    return mapping.get(
        mode, "Utterance: make one local point; avoid broad advice unless required."
    )


def _surface_texture_guidance(texture: str, *, task: Any | None = None) -> str:
    substantive = substantive_surface_slot(task)
    mapping = {
        "no_punct_fragment": "Texture: use a clipped fragment that may omit final punctuation.",
        "abbrev_shorthand": "Texture: use only natural shorthand already common or visible in this domain; never invent an acronym.",
        "emoji_or_sarcasm": (
            "Texture: humor or an emoji may be incidental to the substantive local turn; do not let it determine the whole sentence route."
            if substantive
            else "Texture: a tiny emoji, lol, lmao, or /s is allowed when natural; do not explain it."
        ),
        "markdown_quote": "Texture: use an informal quote-like shape only for exact visible wording.",
        "link_reference": (
            (
                "Texture: a reference may be embedded in the substantive local turn. Use the URL supplied for this slot and invent no other."
                if reference_link_enabled()
                else "Texture: a reference may be embedded in the substantive local turn, but do not invent a URL or source name."
            )
            if substantive
            else (
                "Texture: make it a bare source/old-thread reference aside built around the URL supplied for this slot; invent no other."
                if reference_link_enabled()
                else "Texture: make it a bare source/wiki/old-thread reference aside without inventing a URL."
            )
        ),
        "messy_punctuation": "Texture: casual punctuation, ellipses, or a clipped aside is allowed.",
        "gratitude_social": "Texture: make the acknowledgement, thanks, or report-back visibly social and brief.",
    }
    return mapping.get(
        texture, "Texture: keep the wording natural and not overly polished."
    )


def _tone_shape_guidance(shape: str, *, task: Any | None = None) -> str:
    substantive = substantive_surface_slot(task)
    mapping = {
        "soft_ack": "Tone: use a human acknowledgement or report-back without a lecture.",
        "personal_dp": "Tone: give one compact lived datapoint without generalizing it.",
        "neutral_fact": "Tone: make a neutral local observation without a verdict.",
        "plain_question": "Tone: ask one plain narrow question.",
        "mild_caveat": "Tone: state a low-stakes limitation without gotcha language.",
        "light_joke": (
            "Tone: keep humor aimed at the situation while preserving the substantive local move."
            if substantive
            else "Tone: keep the joke short and aimed at the situation."
        ),
        "direct_correction": "Tone: correct one detail directly without a lecture or insult.",
        "rant": "Tone: express concrete local frustration without contempt for another person.",
        "bare_answer": "Tone: keep the answer clipped and plain, not dismissive.",
    }
    return mapping.get(shape, "Tone: follow the sampled local Reddit attitude.")


def _real_surface_shape_guidance(shape: str) -> str:
    mapping = {
        "deleted_removed": "Real shape: output a deleted/removed-style placeholder, not an answer.",
        "template_notice": "Real shape: make it community/meta/template-like, not personal advice.",
        "link_reference": "Real shape: make it a short reference or link-like aside.",
        "quote_link_reference": "Real shape: make it a short quote/reference aside using only visible text.",
        "micro_reaction": "Real shape: use a true one-to-five-word reaction.",
        "short_direct_answer": "Real shape: use a short direct answer, not a paragraph.",
        "short_question": "Real shape: ask only the narrow question.",
    }
    return mapping.get(shape, "")


def _dedupe(values: list[str], *, limit: int) -> list[str]:
    seen: set[str] = set()
    out = []
    for value in values:
        cleaned = " ".join(value.split()).strip(" ,.;:()[]{}")
        key = cleaned.lower()
        if not cleaned or key in seen:
            continue
        seen.add(key)
        out.append(cleaned)
        if len(out) >= limit:
            break
    return out
