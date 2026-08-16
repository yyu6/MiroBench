"""One-shot template contracts for Planner-owned social distributions.

The held-out reference template contains aggregate counts only.  This module
maps those counts onto anonymous structural slots *before* the comment Planner
runs, so the Planner can choose a compatible social function and semantic move
once.  It never sees real comment text and never asks the Writer for variants.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any

from .generation_distribution import AFFECT_INSTRUCTIONS, template_tone_rates
from .opener_profile import opener_cost, scaled_opener_counts
from .surface_contract import surface_only_label


def template_distribution_targets(
    template: dict[str, Any] | None,
    *,
    total_comments: int,
) -> dict[str, Any]:
    """Scale one held-out template's aggregate labels to this thread size."""

    total = max(0, int(total_comments))
    if not template or total <= 0:
        return {"story_slots": 0, "tone_counts": {}, "affect_counts": {}}
    source_total = max(1, _int(template.get("comment_count"), total))
    return {
        "story_slots": _scaled_count(
            _int(template.get("story_count")),
            source_total=source_total,
            target_total=total,
        ),
        "tone_counts": dict(_scaled_complete_rate_counts(
            template_tone_rates(template),
            total,
        )),
        "affect_counts": dict(_widen_affect_targets(
            _scaled_label_counts(
                template.get("dominant_emotion_counts") or {}, total
            ),
            total,
        )),
    }


def build_slot_distribution_schedule(
    *,
    template: dict[str, Any] | None,
    comments: list[dict[str, Any]],
    total_comments: int,
    opener_profile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assign aggregate template labels to anonymous, compatible slot forms.

    Compatibility is derived solely from depth, word count, and anonymous
    surface shape.  The semantic Planner still owns the claim, role, evidence,
    and wording needed to realize the assigned social label naturally.
    """

    total = max(0, int(total_comments))
    slots = [_slot(index, row) for index, row in enumerate(comments[:total], start=1)]
    targets = template_distribution_targets(template, total_comments=total)
    assignments: dict[int, dict[str, str]] = {}
    story_ids, unassigned_story_slots = _assign_story_slots(
        slots,
        int(targets["story_slots"]),
    )
    for ordinal, sample_id in enumerate(story_ids):
        assignments.setdefault(sample_id, {})["story_mode"] = _story_mode_for_slot(
            _slot_by_id(slots, sample_id), ordinal
        )
    tone, tone_unassigned = _assign_labels(
        slots,
        Counter(targets["tone_counts"]),
        compatibility=_tone_cost,
    )
    # Tone and affect were previously optimized independently over the same
    # slots, so a warm slot could be handed disapproval or annoyance. A plan
    # cannot satisfy both, and the negative label won.
    affect_slots = [
        {
            **slot,
            "tone": tone.get(int(slot["sample_id"]), ""),
            "story_mode": (assignments.get(int(slot["sample_id"])) or {}).get(
                "story_mode", "no_story"
            ),
        }
        for slot in slots
    ]
    affect, affect_unassigned = _assign_labels(
        affect_slots,
        Counter(targets["affect_counts"]),
        compatibility=_affect_cost,
        exclude=lambda label, slot: (
            not _affect_fits_tone(label, str(slot.get("tone") or ""))
            or not _affect_fits_story(
                label, str(slot.get("story_mode") or "no_story")
            )
        ),
    )
    for sample_id, label in tone.items():
        assignments.setdefault(sample_id, {})["tone_class"] = label
    for sample_id, label in affect.items():
        assignments.setdefault(sample_id, {})["affect_role"] = label
    # The Writer collapsed 196 distinct free-text opening routes onto its own
    # default token, so the grammatical entry is scheduled as a categorical
    # control with shares measured on evaluation-excluded threads.
    opener, opener_unassigned = _assign_labels(
        slots,
        scaled_opener_counts(opener_profile, total),
        compatibility=opener_cost,
        exclude=_opener_is_unwritable,
    )
    # An entry grammar that no slot can carry must not cost a slot its contract:
    # a slot left without one falls back to the Writer's own default opening,
    # which is how a bare agreement token reached 23% against a real 4%. Give
    # the leftover slots the writable types instead, in their measured
    # proportions.
    opener, opener_unassigned = _refill_openers(
        slots,
        opener,
        opener_unassigned,
        opener_profile,
    )
    for sample_id, label in opener.items():
        assignments.setdefault(sample_id, {})["opener_type"] = label
    return {
        "source": "evaluation-excluded same-size aggregate metric template",
        "raw_text_included": False,
        "targets": targets,
        # Story is a whole-thread count, not an opt-in label.  Without an
        # explicit default, the Comment Planner can add a story to a micro
        # slot that was not selected for the held-out template's story quota.
        # Keep that negative contract separate from the sparse assignments so
        # the Planner prompt remains compact on large threads.
        "defaults": {"story_mode": "no_story"},
        "assignments": {str(key): value for key, value in sorted(assignments.items())},
        "unassigned_story_slots": unassigned_story_slots,
        "unassigned_tone_labels": tone_unassigned,
        "unassigned_affect_labels": affect_unassigned,
        "unassigned_opener_types": opener_unassigned,
    }


def apply_slot_distribution_schedule(
    plans: dict[int, dict[str, str]],
    schedule: dict[str, Any] | None,
    *,
    events: list[dict[str, Any]] | None = None,
) -> dict[int, dict[str, str]]:
    """Apply the predeclared template contract without changing semantics."""

    assignments = (schedule or {}).get("assignments") or {}
    defaults = (schedule or {}).get("defaults") or {}
    for sample_id, plan in plans.items():
        expected = assignments.get(str(sample_id)) or assignments.get(sample_id) or {}
        default_story = str(defaults.get("story_mode") or "").strip().lower()
        if default_story:
            plan["story_mode"] = default_story
        for field in ("story_mode", "tone_class", "affect_role", "opener_type"):
            value = str(expected.get(field) or "").strip().lower()
            if not value:
                continue
            original = str(plan.get(field) or "").strip().lower()
            if original and original != value and events is not None:
                events.append(
                    {
                        "sample_id": int(sample_id),
                        "field": field,
                        "planner_value": original,
                        "template_contract_value": value,
                    }
                )
            plan[field] = value
    return plans


def render_slot_distribution_schedule(
    schedule: dict[str, Any] | None,
    *,
    sample_ids: list[int],
) -> str:
    """Render only the contracts for the displayed Planner slots."""

    assignments = (schedule or {}).get("assignments") or {}
    rows = []
    for sample_id in sample_ids:
        value = assignments.get(str(sample_id)) or assignments.get(sample_id) or {}
        if not value:
            continue
        rows.append(
            f"- S{sample_id}: "
            + "; ".join(f"{field}={label}" for field, label in sorted(value.items()))
        )
    unassigned: list[str] = []
    for field, label_name in (
        ("unassigned_story_slots", "story slot"),
        ("unassigned_tone_labels", "tone"),
        ("unassigned_affect_labels", "affect"),
        ("unassigned_opener_types", "opener"),
    ):
        values = (schedule or {}).get(field) or []
        if field == "unassigned_story_slots":
            if int(values or 0) > 0:
                unassigned.append(f"{int(values)} {label_name}")
            continue
        unassigned.extend(f"{label_name}={value}" for value in values)
    if unassigned:
        rows.append(
            "- unavailable template labels: "
            + "; ".join(unassigned)
            + ". These labels have no compatible anonymous slot in this thread. "
            "Do not force them onto another semantic move."
        )
    return "\n".join(rows) or "- no slot-specific label contract is available"


def _refill_openers(
    slots: list[dict[str, Any]],
    opener: dict[int, str],
    unassigned: list[str],
    opener_profile: dict[str, Any] | None,
) -> tuple[dict[int, str], list[str]]:
    """Re-spend unplaceable opener quota on the types the leftover slots allow."""

    leftover = [slot for slot in slots if int(slot["sample_id"]) not in opener]
    shares = dict((opener_profile or {}).get("shares") or {})
    if not leftover or not shares:
        return opener, unassigned
    dropped = set(unassigned)
    writable = {name: value for name, value in shares.items() if name not in dropped}
    total_share = sum(writable.values())
    if not writable or total_share <= 0:
        return opener, unassigned
    renormalized = {name: value / total_share for name, value in writable.items()}
    extra, still_unassigned = _assign_labels(
        leftover,
        scaled_opener_counts({"shares": renormalized}, len(leftover)),
        compatibility=opener_cost,
        exclude=_opener_is_unwritable,
    )
    opener = {**opener, **extra}
    return opener, [*unassigned, *still_unassigned]


def _opener_is_unwritable(label: str, slot: dict[str, Any]) -> bool:
    """Block entry grammars the anonymous slot cannot structurally carry.

    `opener_cost` only ranks; every type stayed assignable to every slot. Over
    520 slots that put `question` on 23 slots and `imperative` on 10 and none of
    them were ever realized, because a slot that answers something cannot open
    by asking it. An unassignable label is reported to the Planner as an
    unavailable template label instead of being spent on a slot that will drop
    it, which keeps the realized mix closer to the measured real one.
    """

    words = int(slot.get("words") or 0)
    depth = int(slot.get("depth") or 0)
    surface = str(slot.get("surface") or "")
    if label == "question":
        # Asking needs a slot that is short enough to be mostly the question.
        return not (surface == "short_question" or words <= 40)
    if label == "quote":
        # Nothing to quote at the top of a thread, and no room in a micro slot.
        return depth == 0 or words < 20
    if label == "address":
        # Addressing someone needs someone to address.
        return depth == 0
    if label == "link":
        return words <= 25
    return False


def _slot(sample_id: int, row: dict[str, Any]) -> dict[str, Any]:
    body = str(row.get("body") or "")
    return {
        "sample_id": sample_id,
        "words": len(body.split()),
        "depth": _int(row.get("depth")),
        "surface": surface_only_label(body),
    }


def _slot_by_id(slots: list[dict[str, Any]], sample_id: int) -> dict[str, Any]:
    return next(slot for slot in slots if int(slot["sample_id"]) == sample_id)


def _assign_story_slots(
    slots: list[dict[str, Any]], count: int
) -> tuple[list[int], int]:
    eligible = [
        slot
        for slot in slots
        if slot["words"] >= 24 and slot["surface"] not in {"micro", "short_question"}
    ]
    ordered = sorted(
        eligible,
        key=lambda slot: (-int(slot["words"]), -int(slot["depth"]), int(slot["sample_id"])),
    )
    chosen = ordered[:count]
    return [int(slot["sample_id"]) for slot in chosen], max(0, count - len(chosen))


def _story_mode_for_slot(slot: dict[str, Any], ordinal: int) -> str:
    words = int(slot["words"])
    if words >= 95:
        return "messy_multi_step_story" if ordinal % 3 == 2 else "specific_personal_story"
    return "specific_personal_story" if ordinal % 4 == 3 else "tiny_personal_context"


# Only the pairs a single comment genuinely cannot realize are excluded. The
# affect quota also drives emotion entropy, so a broader ban would trade one
# failing metric for another; mildly signed affects stay available everywhere.
TONE_INCOMPATIBLE_AFFECTS = {
    "polite": frozenset(
        {
            "anger",
            "annoyance",
            "disapproval",
            "disappointment",
            "disgust",
            "grief",
            "remorse",
            "sadness",
        }
    ),
    "impolite": frozenset(
        {
            "gratitude",
            "admiration",
            "joy",
            "love",
            "excitement",
            "relief",
            "caring",
            "approval",
            "optimism",
        }
    ),
}


def _affect_fits_tone(affect: str, tone: str) -> bool:
    """Reject affect/tone pairs no single comment can realize."""

    label = str(affect or "").strip().lower()
    register = str(tone or "").strip().lower()
    return label not in TONE_INCOMPATIBLE_AFFECTS.get(register, frozenset())


def _affect_fits_story(affect: str, story_mode: str) -> bool:
    """Keep pure social closes out of slots reserved for a narrative."""

    return not (
        str(story_mode or "no_story") != "no_story"
        and str(affect or "").strip().lower() in {"gratitude", "relief"}
    )


def _assign_labels(
    slots: list[dict[str, Any]],
    counts: Counter[str],
    *,
    compatibility: Any,
    exclude: Any = None,
) -> tuple[dict[int, str], list[str]]:
    available = {int(slot["sample_id"]): slot for slot in slots}
    assignments: dict[int, str] = {}
    unassigned: list[str] = []

    def blocked(label: str, slot: dict[str, Any]) -> bool:
        return bool(exclude) and bool(exclude(label, slot))

    label_order = sorted(
        (label for label, count in counts.items() if count > 0),
        key=lambda label: (
            sum(
                compatibility(label, slot)[0] == 0 and not blocked(label, slot)
                for slot in slots
            ),
            -int(counts[label]),
            label,
        ),
    )
    for label in label_order:
        for _ in range(int(counts[label])):
            compatible = [
                (sample_id, slot)
                for sample_id, slot in available.items()
                if compatibility(label, slot)[0] == 0 and not blocked(label, slot)
            ]
            if not compatible:
                unassigned.append(label)
                continue
            sample_id, slot = min(
                compatible,
                key=lambda item: (*compatibility(label, item[1]), item[0]),
            )
            assignments[sample_id] = label
            del available[sample_id]
    return assignments, unassigned


# Median comment length per predicted class, measured on the evaluation-excluded
# camera reference threads (11,817 comments scored with the evaluation
# classifier).  The previous cost function preferred the *shortest* compatible
# slot for ``polite``, which is the opposite of the observed distribution and
# routed the warm register onto slots too small to carry it.
TONE_TYPICAL_WORDS = {
    "polite": 53,
    "somewhat_polite": 27,
    "neutral": 16,
    "impolite": 27,
}
_MICRO_SURFACES = {"micro", "short_question"}


def _tone_cost(label: str, slot: dict[str, Any]) -> tuple[int, int, int]:
    """Rank slots by how naturally they can carry a tone label.

    The first element is a hard incompatibility flag, the second a soft
    preference, and the third orders the remaining candidates by distance from
    the class's observed typical length.
    """

    words = int(slot["words"])
    depth = int(slot["depth"])
    surface = str(slot["surface"])
    typical = TONE_TYPICAL_WORDS.get(label, 27)
    distance = abs(words - typical)
    if label == "polite":
        # A warm, appraising turn cannot be realized in a micro reaction.
        return (1 if surface in _MICRO_SURFACES or words < 12 else 0, 0 if words >= 40 else 1, distance)
    if label == "somewhat_polite":
        # Hedged half-agreement is predominantly a reply move at mid length.
        return (1 if surface in _MICRO_SURFACES or words < 8 else 0, 0 if depth > 0 else 1, distance)
    if label == "impolite":
        return (1 if words < 6 else 0, 0 if words >= 14 else 1, distance)
    # neutral: flat informational statements concentrate in short slots.
    return (0, 0 if words <= 30 else 1, distance)


def _widen_affect_targets(counts: Counter[str], total: int) -> Counter[str]:
    """Keep the template's rare affect labels instead of rounding them away.

    Scaling a template's label counts to a thread size floors every share, so a
    label the template used once or twice disappears. Measured against a matched
    real thread, that left 14 distinct dominant emotions in the generated output
    against 27 in the real thread, and the entropy metric is a direct function of
    how many labels appear. This gives every label the template actually used at
    least one slot, taking the slots from whichever labels are most over-represented.
    """

    if total <= 0 or not counts:
        return counts
    widened = Counter({label: int(value) for label, value in counts.items() if value > 0})
    missing = [label for label, value in counts.items() if int(value) <= 0]
    for label in sorted(missing):
        donor = max(widened, key=lambda key: (widened[key], key), default=None)
        if donor is None or widened[donor] <= 1:
            break
        widened[donor] -= 1
        widened[label] = 1
    return Counter({label: value for label, value in widened.items() if value > 0})


def _affect_cost(label: str, slot: dict[str, Any]) -> tuple[int, int, int]:
    words = int(slot["words"])
    depth = int(slot["depth"])
    surface = str(slot["surface"])
    tone = str(slot.get("tone") or "")
    tone_preference = _tone_affect_preference(label, tone)
    if label in {"gratitude", "relief"}:
        return (0 if depth > 0 and words <= 48 else 1, tone_preference, words)
    if label in {"curiosity", "confusion"}:
        return (0 if surface == "short_question" or depth == 0 else 1, tone_preference, words)
    if label in {"amusement", "excitement", "surprise"}:
        return (0 if words <= 32 else 1, tone_preference, words)
    if label in {"anger", "annoyance", "disapproval", "disappointment", "fear"}:
        return (0 if words >= 14 and surface not in {"micro", "short_question"} else 1, tone_preference, -words)
    if label in {"admiration", "approval", "optimism", "realization", "desire"}:
        return (0 if words >= 18 else 1, tone_preference, -words)
    return (0, tone_preference, abs(words - 24))


def _tone_affect_preference(label: str, tone: str) -> int:
    """Rank feasible marginal pairings without pretending the axes coincide."""

    positive = {
        "admiration", "approval", "caring", "desire", "excitement",
        "gratitude", "joy", "love", "optimism", "relief",
    }
    negative = {
        "anger", "annoyance", "disappointment", "disapproval", "disgust",
        "fear", "grief", "remorse", "sadness",
    }
    exploratory = {"amusement", "confusion", "curiosity", "realization", "surprise"}
    preferred = {
        "polite": positive | exploratory,
        "somewhat_polite": exploratory | {"neutral", "approval", "optimism"},
        "neutral": exploratory | {"neutral"},
        "impolite": negative | {"neutral", "confusion", "curiosity"},
    }
    return 0 if label in preferred.get(tone, {label}) else 1


def _scaled_complete_rate_counts(values: dict[str, float], total: int) -> Counter[str]:
    source = {label: max(0.0, float(rate)) for label, rate in values.items()}
    source_total = sum(source.values())
    if source_total <= 0:
        return Counter({"neutral": total})
    exact = {label: rate * total / source_total for label, rate in source.items()}
    result = Counter({label: int(math.floor(value)) for label, value in exact.items()})
    for label in sorted(exact, key=lambda key: (-(exact[key] - result[key]), key))[: total - sum(result.values())]:
        result[label] += 1
    return result


def _scaled_label_counts(values: dict[str, Any], total: int) -> Counter[str]:
    source = Counter(
        {
            str(label).strip().lower(): max(0, _int(value))
            for label, value in values.items()
            if str(label).strip().lower() in AFFECT_INSTRUCTIONS and _int(value) > 0
        }
    )
    source_total = sum(source.values())
    if source_total <= 0:
        return Counter({"neutral": total})
    exact = {label: value * total / source_total for label, value in source.items()}
    result = Counter({label: int(math.floor(value)) for label, value in exact.items()})
    for label in sorted(exact, key=lambda key: (-(exact[key] - result[key]), key))[: total - sum(result.values())]:
        result[label] += 1
    return result


def _scaled_count(count: int, *, source_total: int, target_total: int) -> int:
    return max(0, min(target_total, int(round(count * target_total / max(1, source_total)))))


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default
