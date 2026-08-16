"""Compact, parent-local planning contracts for direct discussion replies.

The ordinary Comment Planner needs whole-thread coverage for root turns. A
direct reply instead needs one thing: an irreducible addition to its already
planned parent. Keeping that request separate prevents the parent contract
from being buried beneath global branch and reference-bank instructions.
"""

from __future__ import annotations

from typing import Any

from .branch_routing import parent_slot_schedule
from .long_form_planning import expected_development_beats
from .planner_schema import parse_sample_id
from .surface_contract import surface_only_label


# A reply still has to add something its parent does not already state, but the
# addition does not have to be adversarial. With only limiting increments
# available, every reply was planned as an expert adjudication, which is a
# register no tone control can turn warm. Corroborating and extending
# increments are equally irreducible and are what most real replies do.
CRITICAL_REPLY_DELTA_TYPES = (
    "operational_test",
    "observable_failure",
    "evidence_requirement",
    "scope_limit",
    "downstream_consequence",
    "countercondition",
)
SUPPORTIVE_REPLY_DELTA_TYPES = (
    "corroborating_datapoint",
    "useful_extension",
    "endorsement_with_reason",
)
SOCIAL_REPLY_DELTA_TYPES = ("social_close",)
REPLY_DELTA_TYPES = (
    CRITICAL_REPLY_DELTA_TYPES
    + SUPPORTIVE_REPLY_DELTA_TYPES
    + SOCIAL_REPLY_DELTA_TYPES
)

# Which increments can carry which tone register. A warm turn cannot be built
# on a scope limit, and a blunt turn cannot be built on an endorsement.
REPLY_DELTA_TYPES_BY_TONE = {
    "polite": SUPPORTIVE_REPLY_DELTA_TYPES + SOCIAL_REPLY_DELTA_TYPES,
    "somewhat_polite": (
        "scope_limit",
        "countercondition",
        "evidence_requirement",
    )
    + SUPPORTIVE_REPLY_DELTA_TYPES,
    "neutral": (
        "operational_test",
        "evidence_requirement",
        "downstream_consequence",
        "useful_extension",
        "corroborating_datapoint",
    ),
    "impolite": CRITICAL_REPLY_DELTA_TYPES,
}

REPLY_DELTA_TYPE_DEFINITIONS = {
    "operational_test": "an observation or check that would settle the parent's open question",
    "observable_failure": "a concrete way the parent's option visibly fails",
    "evidence_requirement": "what must be seen before acting on the parent",
    "scope_limit": "a boundary outside which the parent no longer holds",
    "downstream_consequence": "what changes after the parent's answer is accepted",
    "countercondition": "a condition under which the parent's conclusion reverses",
    "corroborating_datapoint": "your own concrete experience that independently confirms the parent",
    "useful_extension": "an adjacent practical detail the parent leaves out but the reader needs",
    "endorsement_with_reason": "commit to the parent's option and name the specific reason it works",
    "social_close": "acknowledge the parent's help without a second factual claim",
}


def allowed_reply_delta_types(tone_class: str) -> tuple[str, ...]:
    """Return the increments compatible with one assigned tone register."""

    return REPLY_DELTA_TYPES_BY_TONE.get(
        str(tone_class or "").strip().lower(),
        REPLY_DELTA_TYPES,
    )


def _development_requirement(comment: dict[str, Any]) -> str:
    """State the content capacity a long reply slot has to be planned for.

    This planner previously omitted ``development_plan`` entirely, so every long
    slot below the root received no development guidance and was realized at
    roughly three quarters of its matched length.
    """

    words = len(str(comment.get("body") or "").split())
    beats = expected_development_beats(words)
    if beats <= 0:
        return "  development_plan: none; this slot is short"
    return (
        f"  development_plan: required, about {beats} beats "
        f"(anonymous slot is {words} words)"
    )


def is_direct_reply_batch(
    *,
    comments: list[dict[str, Any]],
    all_comments: list[dict[str, Any]],
    sample_offset: int,
    prior_plans: list[dict[str, Any]],
) -> bool:
    """Return whether every displayed slot has an already committed parent."""

    if not comments:
        return False
    parent_slots = parent_slot_schedule(all_comments)
    committed = {
        parse_sample_id(plan.get("sample_id"))
        for plan in prior_plans
        if isinstance(plan, dict)
    }
    return all(
        parent_slots.get(sample_offset + local_index) in committed
        for local_index, _comment in enumerate(comments, start=1)
    )


def render_direct_reply_planner_prompt(
    *,
    config: Any,
    backend: Any,
    seed_post: Any,
    comments: list[dict[str, Any]],
    all_comments: list[dict[str, Any]],
    sample_offset: int,
    prior_plans: list[dict[str, Any]],
    slot_distribution: str,
    slot_controls: dict[int, dict[str, str]] | None = None,
    validation_feedback: str = "",
) -> str:
    """Render a short semantic-planning request for direct replies only.

    This deliberately contains no matched-real text and no reference-viewpoint
    examples. The real structural slot contributes only depth, anonymous word
    count, and surface category. Parent meaning comes solely from the committed
    generated plan ledger.
    """

    parent_slots = parent_slot_schedule(all_comments)
    prior_by_id = {
        parse_sample_id(plan.get("sample_id")): plan
        for plan in prior_plans
        if isinstance(plan, dict) and parse_sample_id(plan.get("sample_id")) > 0
    }
    sample_ids = [sample_offset + index for index in range(1, len(comments) + 1)]
    controls_by_slot = dict(slot_controls or {})
    siblings_by_parent: dict[int, list[int]] = {}
    for child_id, parent_id in parent_slots.items():
        siblings_by_parent.setdefault(parent_id, []).append(child_id)
    for sibling_ids in siblings_by_parent.values():
        sibling_ids.sort()
    parent_rows: list[str] = []
    for sample_id, comment in zip(sample_ids, comments, strict=True):
        parent_id = parent_slots[sample_id]
        parent = prior_by_id[parent_id]
        controls = controls_by_slot.get(sample_id) or {}
        tone = str(controls.get("tone_class") or "").strip().lower()
        story = str(controls.get("story_mode") or "").strip().lower()
        affect = str(controls.get("affect_role") or "neutral").strip().lower()
        opener = str(controls.get("opener_type") or "").strip().lower()
        allowed = [
            value
            for value in allowed_reply_delta_types(tone)
            if value != "social_close" or affect in {"gratitude", "relief"}
        ]
        if story == "no_story":
            allowed = [value for value in allowed if value != "corroborating_datapoint"]
        sibling_ids = siblings_by_parent.get(parent_id, [])
        sibling_contract = ""
        if (
            str(getattr(backend, "GENERALIZED_REPLY_SIBLING_VISIBILITY", "on"))
            != "off"
            and len(sibling_ids) > 1
        ):
            committed_siblings = [
                prior_by_id[sibling_id]
                for sibling_id in sibling_ids
                if sibling_id != sample_id and sibling_id in prior_by_id
            ]
            used = [
                f"S{parse_sample_id(plan.get('sample_id'))}:"
                f"{str(plan.get('reply_delta_type') or 'unset')} / "
                f"{backend.compact(plan.get('reply_novelty_anchor') or 'unset', 90)}"
                for plan in committed_siblings
            ]
            sibling_contract = (
                "  Sibling coverage: "
                + ",".join(f"S{sibling_id}" for sibling_id in sibling_ids)
                + "; use a different delta type and novelty object for each sibling"
                + ("; already committed: " + " | ".join(used) if used else "")
            )
        parent_rows.append(
            "\n".join(
                item
                for item in (
                    f"- S{sample_id}: reply to S{parent_id}; "
                    f"depth={int(comment.get('depth') or 0)}; "
                    f"anonymous_words={len(str(comment.get('body') or '').split())}; "
                    f"surface={surface_only_label(str(comment.get('body') or ''))}",
                    f"  Parent semantic move to exclude: {backend.compact(parent.get('semantic_move') or parent.get('local_topic') or 'parent local point', 180)}",
                    f"  Parent decision boundary to exclude: {backend.compact(parent.get('decision_boundary') or 'parent local condition', 160)}",
                    f"  Parent detail to exclude: {backend.compact(parent.get('detail_focus') or 'parent local detail', 140)}",
                    f"  Parent reply type: {str(parent.get('reply_delta_type') or 'root_turn').strip()}",
                    f"  Tone register: {tone or 'unassigned'}",
                    f"  Story contract: {story or 'unassigned'}",
                    f"  Affect contract: {affect}",
                    f"  Opening grammar: {opener or 'unassigned'}",
                    "  Allowed reply_delta_type: " + (", ".join(allowed) or "any"),
                    sibling_contract,
                    _development_requirement(comment),
                )
                if item
            )
        )
    delta_definitions = "\n".join(
        f"- {name}: {REPLY_DELTA_TYPE_DEFINITIONS[name]}" for name in REPLY_DELTA_TYPES
    )
    feedback = (
        f"\nSchema/quality correction for these exact slots:\n{validation_feedback}\n"
        if validation_feedback
        else ""
    )
    slot_ids = ", ".join(f"S{sample_id}" for sample_id in sample_ids)
    # The root planner interpolates the full enumeration; this request used to
    # ask for "one generic claim family" with no list anywhere in it, so every
    # answer fell outside the vocabulary and normalized to `miscellaneous` --
    # 61 of 61 reply slots in v70, against 14 distinct families across roots.
    # That silently disabled the per-thread claim-family share cap for every
    # reply. Reading it off the backend keeps the prompt and the normalizer on
    # one list.
    claim_families = " | ".join(getattr(backend, "CLAIM_FAMILIES", ()) or ())
    # Asking for "a full sentence stating what this reply asserts" produced
    # finished first-person prose -- 19.3% of all moves open with "I" -- which the
    # Writer then reproduced verbatim. Reply slots echoed their plan at 25.1%
    # against 6.4% for root slots, whose schema has always said "non-verbatim".
    # The scale requirement is what stopped bare noun phrases, so it stays; the
    # demand for a finished sentence is what leaked, so it goes.
    if str(getattr(backend, "GENERALIZED_WRITER_ROUTE_LOCK", "own_words")) == "say_only":
        move_schema = (
            "a full sentence stating what this reply asserts, at the same scale "
            "as a top-level slot's semantic_move - not a bare noun phrase"
        )
    else:
        move_schema = (
            "one concrete but non-verbatim action for this reply, at the same "
            "scale as a top-level slot's semantic_move - name what it asserts, "
            "not a bare noun phrase and not the comment's own drafted sentence"
        )
    return f"""Plan only the direct replies of one Reddit discussion in {config.community_context}.

This is an abstract Planner request, not a request to write comments. The
Writer will realize each plan exactly once. Do not rely on later rewriting.

Visible seed, only for grounding an otherwise generic parent-local addition:
- title: {backend.compact(seed_post.title or '', 260)}
- body: {backend.compact(seed_post.body or seed_post.content or '', 700)}

The parent plan is an exclusion, never text to paraphrase. For every row,
choose exactly one reply_delta_type and name one specific new object in
reply_novelty_anchor that cannot be reconstructed from the parent plan.

The increment types, and what each one contributes:
{delta_definitions}

An addition does not have to be a criticism. A reply that confirms the parent
from its own experience, extends it with a detail the parent omits, or endorses
it for a specific reason is adding just as much as one that limits it, as long
as the named object is genuinely new. Only a social_close may add nothing
factual at all.

Correct contrast: if a parent asks whether X is adequate in condition Y,
an invalid reply says X matters in condition Y. A valid operational_test says
what someone would inspect or try; a valid corroborating_datapoint says what
happened when you personally used X in condition Y; a valid useful_extension
says which adjacent detail changes the answer. Put that same distinct object in
semantic_move, decision_boundary, and reply_novelty_anchor.

Each row lists the reply_delta_type values compatible with its assigned tone
register. Choose only from that list: the increment and the register have to be
the same turn, not two contradictory instructions. A polite row builds its
addition out of agreement, corroboration, or endorsement; an impolite row builds
it out of a limit, failure, or counter-condition.

Required template-derived labels for these slots:
{slot_distribution}

Anonymous direct-reply slots and committed parent-plan exclusions:
{chr(10).join(parent_rows)}
{feedback}
Return strict JSON with exactly one row for each of {slot_ids}:
{{
  "comment_plans": [
    {{
      "sample_id": "S{sample_ids[0]}",
      "payload_type": "bare_answer | fragment_datapoint | soft_helpful | correction | narrow_question | personal_story | rant | joke | side_tangent | meta_or_template | advice",
      "comment_function": "reaction | question_followup | correction_caveat | personal_datapoint | recommendation_advice | verdict_evaluation | explanation_analysis | offtopic_noise",
      "content_angle": "one local angle",
      "evidence_mode": "none_assertion | firsthand_experience | technical_or_policy_reasoning | calculation_math | hearsay_consensus | link_quote_reference | small_observation",
      "story_mode": "copy the fixed story contract shown for this S#",
      "tone_class": "copy the fixed tone register shown for this S#",
      "affect_role": "copy the fixed affect contract shown for this S#",
      "opener_type": "copy the fixed opening grammar shown for this S#",
      "voice": "blunt | casual_neutral | polite_soft | sarcastic | annoyed | uncertain | grateful",
      "speaker_role": "advisor | confused_asker | op_followup | gratitude_reply | jokester | contrarian | datapoint_only | ranter | side_observer",
      "semantic_move": "{move_schema}",
      "local_topic": "the parent-local topic",
      "reply_relation": "answers_parent | challenges_parent | asks_narrow_followup | adds_datapoint | jokes_aside | corrects_detail | shifts_to_side_detail",
      "stance": "agree | disagree | mixed | uncertain | joking | neutral",
      "detail_focus": "the new concrete object, not the parent premise",
      "avoid_repeating": "the exact parent proposition to avoid",
      "claim_family": "{claim_families}",
      "claim_key": "short key for this new increment",
      "decision_boundary": "the one question this reply settles",
      "reply_delta": "how this increment relates to the parent",
      "reply_delta_type": "{' | '.join(REPLY_DELTA_TYPES)}",
      "reply_novelty_anchor": "the one concrete new object this reply introduces",
      "opening_style": "a one-use sentence entry route different from the parent",
      "development_plan": "none",
      "domain_claim": "one concrete domain fact this reply states in your own words, or none for a purely social reply",
      "context_aperture": "parent_only"
    }}
  ]
}}

Rules:
- Return exactly the requested IDs once each. Do not return a root plan.
- Never use ``none`` for reply_delta_type or reply_novelty_anchor unless the
  selected type is social_close.
- Use only a reply_delta_type listed as allowed for that row.
- Choose social_close only when it appears in the row's allowed list. For
  social_close use speaker_role=gratitude_reply and comment_function=reaction,
  and add no recommendation, explanation, or fact.
- If the parent is itself a reply, choose a different reply_delta_type from
  the parent unless this slot is an allowed social_close. Do not repeat the
  parent's test, evidence request, or exception under a new name.
- Match speaker_role, stance, voice, and comment_function to the row's tone
  register. A polite row is not an advisor adjudicating a threshold: use
  stance=agree with datapoint_only, op_followup, or gratitude_reply and a
  comment_function of personal_datapoint, reaction, or verdict_evaluation.
  Do not pair a polite register with correction_caveat or a disagreeing stance.
- Treat the story contract as a joint semantic contract, not a Writer style.
  For `no_story`, do not use `firsthand_experience`, `personal_story`, or
  `corroborating_datapoint`; a first-person preference or current-state
  observation is allowed, but no past action, event, before/after change, or
  sequence. For any other story mode, use `firsthand_experience`,
  `comment_function=personal_datapoint`, and a personal-story or
  fragment-datapoint payload whose semantic move is an actual event sequence.
- Make the affect the reaction already contained in the semantic move. It does
  not authorize a second claim or an invented outcome. Gratitude and relief
  require the listed `social_close`, a no-story reaction, and no factual add-on.
- Make the opening grammar writable by the same row: a question opener needs a
  narrow question or operational test; an imperative needs genuine advice; an
  address needs a reply directed to the parent. Do not leave this repair to the
  Writer.
- Preserve the anonymous slot's information capacity without treating its word
  count as an output-length target.
- When a row states that ``development_plan`` is required, return that many
  distinct beats separated by ``||``. Each beat is realized in about one
  sentence, so returning fewer than requested is what makes a long reply come
  out short. Every beat develops this same one increment through a different
  observation, reason, consequence, caveat, condition, or reaction; none of
  them may restate the increment or introduce an unrelated claim.
- When a row says `development_plan: none`, return the literal string `none`.
- Give a substantive reply a ``domain_claim``: one concrete domain fact stated in
  your own words, naming the relevant entity, action, or condition. Real replies
  are largely specific observations, procedures, relationships, and constraints;
  a reply whose whole content is how to weigh a decision reads as commentary
  about the discussion rather than participation in it. A purely social reply
  uses ``none``.
- Do not reproduce any real discussion's wording, and do not carry a detail that
  belongs to a particular discussion or its participants rather than to the
  domain. A fact about this seed post still cannot be invented.
- Output JSON only."""
