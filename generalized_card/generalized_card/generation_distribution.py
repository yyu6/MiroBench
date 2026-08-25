from __future__ import annotations

import hashlib
import math
import random
from collections import Counter
from dataclasses import replace
from typing import Any

from .planner_schema import parse_sample_id
from .reference_metric_calibration import select_reference_template
from .tone_realization import invert_tone_rates, realization_report


STORY_MODES = (
    "tiny_personal_context",
    "specific_personal_story",
    "messy_multi_step_story",
)

# The evaluation classifier is a four-way single-label model.  Only polite,
# impolite, and neutral are reported, but planning over three classes forced the
# measured fourth-class mass onto the reported three and left the register that
# actually attracts it unnamed.  Plan over the complete partition instead.
TONE_CLASSES = ("polite", "somewhat_polite", "impolite", "neutral")
REPORTED_TONE_CLASSES = ("polite", "impolite", "neutral")
STORY_PROBABILITY_TIERS = (
    "very_low",
    "low",
    "ambiguous",
    "story_mid",
    "story_high",
)

# Every entry here used to be dominated by a prohibition: express the emotion
# "but not with hype", "without exclamation marks", "not broad praise", "do not
# add a new complaint". GoEmotions keys on exactly those surface markers, so the
# set as a whole described `neutral`. Measured over v72, every non-neutral affect
# realized at 0-23% while neutral realized at 48-80%, and this held for pairs
# that are perfectly tone-compatible (impolite+annoyance 0/12,
# polite+admiration 0/30), so it was never a tone conflict. Warm markers reached
# 8.7% of generated comments against 20.6% of matched real ones.
#
# These say what to write. The only negative clauses kept are the ones that
# protect factual grounding, because an invented loss or purchase is a real
# failure rather than a stylistic one.
AFFECT_INSTRUCTIONS = {
    "admiration": "Say plainly what impresses you about the existing local point, and why that specific thing lands.",
    "neutral": (
        "Keep the emotional signal low. State the local observation without sounding flat, clinical, or assistant-like."
    ),
    "curiosity": (
        "Show that you actually want to know. Let the interest carry the question or the uncertainty the task already has."
    ),
    "confusion": (
        "Show that the thing genuinely does not add up for you, in ordinary local terms rather than helplessness."
    ),
    "gratitude": (
        "Thank them the way a person does in a thread: say what the help actually saved you or told you."
    ),
    "amusement": (
        "Let it be funny. A natural laughter token is allowed; land the aside "
        "and move on without explaining it."
    ),
    "anger": "Let the frustration be plain, directed at the product, result, or process rather than at another commenter.",
    "annoyance": "Let the irritation show around the friction the task already has.",
    "approval": "Commit to the positive judgement of the existing local point and say what makes it right.",
    "caring": "Show that you want it to work out for them, tied to their situation as described.",
    "disappointment": (
        "Let the letdown show around the existing product, result, or process, never another commenter."
    ),
    "disapproval": (
        "Make the negative judgement explicit and aim it at the claim, product, or process."
    ),
    "disgust": "Keep the strong negative reaction brief and product-directed.",
    "embarrassment": "Own the awkward moment briefly, where the first-person point supports it.",
    "excitement": (
        "Let real enthusiasm for the existing local point come through in how you say it."
    ),
    "optimism": (
        "Say why you think it works out, without promising an outcome or inventing supporting facts."
    ),
    "desire": "Say what you want here and why that option appeals, without inventing a purchase plan.",
    "fear": "Let the concern about the existing risk read as real, without inventing danger.",
    "grief": "Keep the emotional signal subdued, and only where the task already describes a loss.",
    "joy": "Let the good reaction be warm and specific about what is good.",
    "love": "Say plainly how attached you are to this thing, and to what about it.",
    "nervousness": "Let the unease show without adding a new risk.",
    "pride": "Let the satisfaction in your own call or result show, where the task supports it.",
    "relief": (
        "Let the relief land through the existing datapoint, without inventing a successful outcome."
    ),
    "realization": (
        "Present it as the moment your read changed, not as a lesson for the thread."
    ),
    "remorse": "Own the regret about your own earlier call, where the first-person point supports it.",
    "sadness": "Let the sadness read plainly and tie it to the existing outcome, without inventing a loss.",
    "surprise": "Let it read as genuinely unexpected, through the existing detail rather than a new fact.",
}


def allocate_story_and_affect(
    tasks: list[Any],
    *,
    personal_min_share: float,
    calibration: dict[str, Any] | None = None,
    rng: random.Random | None = None,
    template: dict[str, Any] | None = None,
) -> tuple[list[Any], dict[str, Any]]:
    """Audit Planner-owned distribution labels against one frozen template.

    Slot-level template labels are assigned before comment planning by
    ``planner_distribution``.  This function intentionally does *not* mutate
    a completed Planner task: a post-Planner allocator previously forced
    incompatible affects and tones onto otherwise coherent semantic plans.
    """

    if not tasks:
        return tasks, _empty_report(personal_min_share)

    ordered = sorted(tasks, key=lambda task: int(task.local_task_id))
    requested_share = max(0.0, min(0.5, float(personal_min_share)))
    if not template:
        selector = (rng or random.Random(0)).randrange(2**31)
        template = select_reference_template(
            calibration or {},
            comment_count=len(ordered),
            selector=selector,
        )
    if template:
        requested_share = max(0.0, min(0.5, float(template.get("story_rate") or 0.0)))
        target_story = _scaled_count(
            int(template.get("story_count") or 0),
            source_total=max(1, int(template.get("comment_count") or len(ordered))),
            target_total=len(ordered),
        )
    else:
        target_story = int(math.ceil(len(ordered) * requested_share))
    current_story = sum(_is_story(task) for task in ordered)
    target_affects = _target_affect_counts(template, len(ordered))
    target_tones = _target_tone_counts(template, len(ordered))
    affect_counts = Counter(str(task.affect_role or "neutral") for task in ordered)
    tone_counts = Counter(str(task.tone_target or "neutral") for task in ordered)
    story_modes = Counter(str(task.story_mode or "no_story") for task in ordered)
    report = {
        "task_count": len(ordered),
        "personal_min_share": requested_share,
        "target_story_slots": target_story,
        "story_slots_before": current_story,
        "story_slots_after": current_story,
        "story_target_met": current_story == target_story,
        "unfilled_story_slots": abs(target_story - current_story),
        "converted_task_ids": [],
        "demoted_task_ids": [],
        "story_modes_before": dict(sorted(story_modes.items())),
        "story_modes_after": dict(sorted(story_modes.items())),
        "affect_counts": dict(sorted(affect_counts.items())),
        "affect_assignments": {
            str(task.local_task_id): str(task.affect_role or "neutral")
            for task in ordered
        },
        "affect_target_counts": dict(sorted(target_affects.items())),
        "tone_counts": dict(sorted(tone_counts.items())),
        "tone_assignments": {
            str(task.local_task_id): str(task.tone_target or "neutral")
            for task in ordered
        },
        "tone_target_counts": dict(sorted(target_tones.items())),
        # What the quota arm did. `template_rates` is the mix the metric reports
        # against; `assignment_rates` is what the Planner was asked for. Recorded
        # for the audit and never used to select anything (ORIENTATION.md s4).
        "tone_quota": realization_report(template_tone_rates_raw(template)),
        "reference_template_used": bool(template),
        "reference_template": template or {},
        "calibration_reference_thread_count": int(
            (calibration or {}).get("reference_thread_count") or 0
        ),
        "policy": "audit_planner_template_contract_without_post_planner_reassignment",
    }
    return ordered, report


def select_thread_template(
    calibration: dict[str, Any] | None,
    *,
    comment_count: int,
    seed_key: str,
) -> dict[str, Any] | None:
    """Select one same-size template deterministically before Planner calls."""

    digest = hashlib.sha256(str(seed_key).encode("utf-8")).digest()
    selector = int.from_bytes(digest[:8], "big", signed=False)
    return select_reference_template(
        calibration or {},
        comment_count=comment_count,
        selector=selector,
    )


def enrich_distribution_plan_fields(
    payload: dict[str, Any],
    normalized: dict[int, dict[str, str]],
) -> dict[int, dict[str, str]]:
    """Retain distribution fields omitted by the shared CARD JSON parser."""

    raw_rows = payload.get("comment_plans") or payload.get("plans") or []
    raw_by_sample: dict[int, dict[str, Any]] = {}
    for row in raw_rows if isinstance(raw_rows, list) else []:
        if not isinstance(row, dict):
            continue
        sample_id = parse_sample_id(row.get("sample_id"))
        if sample_id > 0:
            raw_by_sample[sample_id] = row
    for sample_id, plan in normalized.items():
        raw = raw_by_sample.get(sample_id, {})
        tone = str(raw.get("tone_class") or "").strip().lower()
        affect = str(raw.get("affect_role") or "").strip().lower()
        plan["tone_class"] = tone if tone in TONE_CLASSES else ""
        plan["affect_role"] = affect if affect in AFFECT_INSTRUCTIONS else ""
        plan["decision_boundary"] = " ".join(
            str(raw.get("decision_boundary") or "").split()
        )[:220]
    return normalized


def apply_planner_distribution_fields(task: Any, plan: dict[str, Any]) -> Any:
    """Carry the Planner's joint target fields into the immutable task."""

    tone = str(plan.get("tone_class") or "").strip().lower()
    affect = str(plan.get("affect_role") or "").strip().lower()
    story = str(plan.get("story_mode") or "no_story").strip().lower()
    updates: dict[str, Any] = {}
    if tone in TONE_CLASSES:
        updates.update(
            tone_target=tone,
            tone_target_instruction=_tone_instruction(tone),
        )
    if affect in AFFECT_INSTRUCTIONS:
        updates.update(
            affect_role=affect,
            affect_instruction=_affect_instruction(affect, task=task),
        )
    if story in STORY_MODES:
        updates.update(
            story_mode=story,
            story_instruction=_story_instruction(story),
            allow_first_person_frame=True,
        )
    elif story == "no_story":
        updates.update(
            story_mode="no_story",
            story_instruction="",
            # A no-story slot still needs a first-person frame when its tone
            # register is realized through personal appraisal.  Forcing the
            # frame off for every non-story slot removed the main surface of
            # the warm register.  Story content stays governed by story_mode.
            allow_first_person_frame=(
                tone == "polite"
                or bool(getattr(task, "allow_first_person_frame", False))
            ),
        )
    return _replace_supported(task, **updates)


def render_planner_distribution_target(
    template: dict[str, Any] | None,
    *,
    total_comments: int,
    prior_plans: list[dict[str, Any]] | None = None,
) -> str:
    """Render exact whole-thread and remaining text-free label counts."""

    if not template or total_comments <= 0:
        return "- unavailable; preserve the matched structural slots and vary labels naturally"
    story_target = _scaled_count(
        int(template.get("story_count") or 0),
        source_total=max(1, int(template.get("comment_count") or total_comments)),
        target_total=total_comments,
    )
    tone_target = _target_tone_counts(template, total_comments)
    affect_target = _target_affect_counts(template, total_comments)
    tier_target = _scaled_label_counts(
        template.get("story_probability_tier_counts") or {},
        total_comments,
    )
    prior = list(prior_plans or [])
    used_story = sum(
        str(plan.get("story_mode") or "no_story") != "no_story" for plan in prior
    )
    used_tone = Counter(
        str(plan.get("tone_class") or "").strip().lower()
        for plan in prior
        if str(plan.get("tone_class") or "").strip().lower() in TONE_CLASSES
    )
    used_affect = Counter(
        str(plan.get("affect_role") or "").strip().lower()
        for plan in prior
        if str(plan.get("affect_role") or "").strip().lower()
    )
    remaining_slots = max(0, total_comments - len(prior))
    remaining_story = max(0, story_target - used_story)
    remaining_tone = _remaining_counts(tone_target, used_tone)
    remaining_affect = _remaining_counts(affect_target, used_affect)
    return "\n".join(
        (
            "- source: one deterministic, same-size evaluation-excluded real metric template; no text or identity is included",
            f"- whole thread comments: {total_comments}; slots remaining before this batch: {remaining_slots}",
            f"- story slots: {story_target} total, {remaining_story} remaining; all other slots must use no_story",
            f"- story-probability shape: {_format_counts(tier_target)}",
            f"- tone_class exact counts: {_format_counts(tone_target)}; remaining: {_format_counts(remaining_tone)}",
            f"- affect_role exact counts: {_format_counts(affect_target)}; remaining: {_format_counts(remaining_affect)}",
            "- distribute labels across compatible discourse roles; labels do not authorize a new claim, event, or fact",
        )
    )


def _is_story(task: Any) -> bool:
    return str(task.story_mode or "no_story") != "no_story"


def _story_instruction(story_mode: str) -> str:
    if story_mode == "tiny_personal_context":
        return (
            "Use a brief first-person past situation with two visibly connected "
            "beats: one ordinary action or condition, then one subjective "
            "observation or reaction. Make the temporal relation legible without "
            "using a stock anecdote opener. Keep it qualitative and stop before a full anecdote."
        )
    if story_mode == "specific_personal_story":
        return (
            "Use a compact first-person event sequence with a setting, an action, "
            "a small friction or change, and a local reaction. Vary the entry and "
            "clause order rather than using a stock experience template. Ground it "
            "in an already visible anchor; do not invent measured facts or a verifiable external outcome."
        )
    return (
        "Use a slightly messy recollection around the existing local point with "
        "at least three temporally connected actions, conditions, or reversals and "
        "a final reaction. Vary sentence lengths and allow one incidental detail so "
        "the sequence does not read like a template. Preserve uncertainty and avoid "
        "a polished lesson, exact measurement, or invented external fact."
    )


def _target_affect_counts(template: dict[str, Any] | None, total: int) -> Counter[str]:
    if not template:
        return Counter()
    source = Counter(
        {
            str(role): int(count)
            for role, count in (template.get("dominant_emotion_counts") or {}).items()
            if int(count) > 0
        }
    )
    return _scaled_label_counts(source, total)


def _target_tone_counts(template: dict[str, Any] | None, total: int) -> Counter[str]:
    if not template or total <= 0:
        return Counter()
    return _scaled_complete_rate_counts(template_tone_rates(template), total)


def template_tone_rates_raw(template: dict[str, Any] | None) -> dict[str, float]:
    """Return the template's complete four-class tone partition, uninverted.

    ``somewhat_polite`` is measured, not inferred, whenever the reference row
    carries it.  Older profiles without the field fall back to the residual so
    the reported three classes are never inflated by the missing mass.
    """

    if not template:
        return {}
    rates = {
        "polite": max(0.0, float(template.get("polite_rate") or 0.0)),
        "impolite": max(0.0, float(template.get("impolite_rate") or 0.0)),
        "neutral": max(0.0, float(template.get("neutral_rate") or 0.0)),
    }
    if "somewhat_polite_rate" in template:
        somewhat = max(0.0, float(template.get("somewhat_polite_rate") or 0.0))
    else:
        somewhat = max(0.0, 1.0 - sum(rates.values()))
    rates["somewhat_polite"] = somewhat
    return rates


def template_tone_rates(template: dict[str, Any] | None) -> dict[str, float]:
    """The quota rendered to the Planner, which is an ASSIGNMENT target.

    The metric reports the REALIZED mix. Under `--tone-quota inverted` the two
    are held apart: `invert_tone_rates` returns the assignment whose realized mix
    lands on the template's rates. It is the identity when the arm is off, so
    every release through v114 reproduces byte for byte.
    """

    return invert_tone_rates(template_tone_rates_raw(template))


def _scaled_complete_rate_counts(values: dict[str, float], total: int) -> Counter[str]:
    """Normalize the complete tone partition onto every planned slot.

    The four classes are a partition of the classifier's output, so scaling is
    close to the identity.  Making the contract total exactly the thread's
    comment count keeps Planner and Writer controls in agreement.
    """

    if total <= 0:
        return Counter()
    source = {label: max(0.0, float(rate)) for label, rate in values.items()}
    source_total = sum(source.values())
    if source_total <= 0:
        return Counter({"neutral": total})
    exact = {label: rate * total / source_total for label, rate in source.items()}
    result = Counter({label: int(math.floor(value)) for label, value in exact.items()})
    remaining = max(0, total - sum(result.values()))
    for label in sorted(
        exact,
        key=lambda item: (-(exact[item] - result[item]), item),
    )[:remaining]:
        result[label] += 1
    return result


def _scaled_label_counts(values: dict[str, Any], total: int) -> Counter[str]:
    source = {
        str(label): max(0.0, float(value))
        for label, value in values.items()
        if float(value) > 0
    }
    source_total = sum(source.values())
    if source_total <= 0 or total <= 0:
        return Counter()
    exact = {label: value * total / source_total for label, value in source.items()}
    result = Counter({label: int(math.floor(value)) for label, value in exact.items()})
    remaining = total - sum(result.values())
    for label in sorted(
        exact,
        key=lambda item: (-(exact[item] - result[item]), item),
    )[:remaining]:
        result[label] += 1
    return result


def _remaining_counts(target: Counter[str], used: Counter[str]) -> Counter[str]:
    return Counter(
        {
            label: max(0, int(count) - int(used.get(label, 0)))
            for label, count in target.items()
            if max(0, int(count) - int(used.get(label, 0))) > 0
        }
    )


def _format_counts(values: Counter[str] | dict[str, Any]) -> str:
    return ", ".join(
        f"{label}={int(count)}"
        for label, count in sorted(values.items())
        if int(count) > 0
    ) or "none"


def _scaled_count(count: int, *, source_total: int, target_total: int) -> int:
    return max(0, min(target_total, int(round(count * target_total / max(1, source_total)))))


def _affect_instruction(role: str, *, task: Any | None = None) -> str:
    """Render the affect target plus one rotating route for expressing it.

    The routes used to open with "Realize it once", which capped the emotion at
    a single clause on top of instructions that already suppressed its surface
    markers. They now name where the feeling enters without limiting how far it
    carries; rotation still keeps nearby comments from sharing an entry path.
    """

    base = AFFECT_INSTRUCTIONS.get(
        role,
        "Let this reaction read plainly, attached to the existing local point, without inventing a new event or fact.",
    )
    if role == "neutral":
        return base
    task_id = int(getattr(task, "local_task_id", 0) or 0)
    channels = (
        "Let it enter in the opening stance rather than by naming the emotion.",
        "Let it enter through an evaluative clause on the concrete detail.",
        "Let it enter through sentence rhythm, a hedge, or a brief interjection.",
        "Let it enter in the closing local reaction rather than a summary.",
    )
    return f"{base} {channels[task_id % len(channels)]}"


# Calibrated on evaluation-excluded real threads only.  Each description states
# the social register that the evaluation classifier actually assigns to that
# label in this data, not a generic notion of manners.  A softener-and-hedge
# reading of "polite" produced the tentative register the classifier scores as
# somewhat_polite, so the distinction is made explicit here rather than left to
# the model's prior.
TONE_DEFINITIONS = {
    "polite": (
        "Warm and personally engaged. Commit to a positive evaluation of the "
        "result, suggestion, help, or concrete subject already visible, or give "
        "genuine thanks for it. Ordinary hedges and brief thanks are allowed when "
        "they fit the turn; what matters is a readable positive or relieved "
        "reaction by the end. Do not turn the response into customer-service "
        "language or abstract decision framing such as what matters, the real "
        "question, or whether something is worth it."
    ),
    "somewhat_polite": (
        "Agreeable but uncommitted. Concede or half-agree with the visible point "
        "while keeping the judgement tentative and qualified. This is a mild, "
        "low-energy register, not warmth and not criticism."
    ),
    "neutral": (
        "Socially unmarked and direct. Do not soften for the other person and do "
        "not attack them. The separately assigned affect may still show interest, "
        "surprise, enthusiasm, or frustration toward the subject; neutral here "
        "describes interpersonal register, not emotional flatness."
    ),
    "impolite": (
        "Blunt and dismissive toward the claim, product, result, or process. "
        "The negative judgement is unqualified rather than softened into a "
        "balanced weighing. Ordinary non-targeted profanity is allowed when it "
        "fits the reaction. Do not use slurs, threats, or personal abuse, and do "
        "not scold another commenter."
    ),
}


SOCIAL_CONTRACT_COHERENCE = True
_LEGACY_POLITE_DEFINITION = (
    "Warm and openly appreciative. Commit to a positive evaluation of the "
    "product, result, suggestion, or help already visible, or give genuine "
    "thanks for it, and let real enthusiasm show. Speak from your own "
    "positive experience where the plan allows it and give the turn room to "
    "develop. Do not hedge the positive judgement into a maybe, and do not "
    "use customer-service phrasing or a template thank-you. Warmth is shown "
    "through what you are enthusiastic about, which differs every time, not "
    "through a recurring appreciative phrase."
)
_LEGACY_TONE_SCOPE_HINTS = {
    "polite": (
        "A warm turn needs room: develop the appraisal across the slot's planned "
        "scope instead of compressing it into one line."
    ),
    "neutral": "Keep it to the bare informational statement.",
}


def set_social_contract_coherence(mode: str) -> bool:
    """Select v80 social guidance or the byte-stable pre-v80 arm."""

    global SOCIAL_CONTRACT_COHERENCE
    SOCIAL_CONTRACT_COHERENCE = str(mode or "on").strip().lower() != "off"
    return SOCIAL_CONTRACT_COHERENCE


def _tone_instruction(tone: str) -> str:
    base = TONE_DEFINITIONS.get(tone, TONE_DEFINITIONS["neutral"])
    if SOCIAL_CONTRACT_COHERENCE:
        return base
    if tone == "polite":
        base = _LEGACY_POLITE_DEFINITION
    hint = _LEGACY_TONE_SCOPE_HINTS.get(tone)
    return f"{base} {hint}" if hint else base


def _empty_report(personal_min_share: float) -> dict[str, Any]:
    return {
        "task_count": 0,
        "personal_min_share": float(personal_min_share),
        "target_story_slots": 0,
        "story_slots_before": 0,
        "story_slots_after": 0,
        "story_target_met": True,
        "unfilled_story_slots": 0,
        "converted_task_ids": [],
        "demoted_task_ids": [],
        "story_modes_before": {},
        "story_modes_after": {},
        "affect_counts": {},
        "affect_assignments": {},
        "affect_target_counts": {},
        "tone_counts": {},
        "tone_assignments": {},
        "tone_target_counts": {},
        "reference_template_used": False,
        "reference_template": {},
        "calibration_reference_thread_count": 0,
    }


def _replace_supported(task: Any, **updates: Any) -> Any:
    supported = {key: value for key, value in updates.items() if hasattr(task, key)}
    return replace(task, **supported) if supported else task
