from __future__ import annotations

from sampling_generator.engine.util import weighted_choice
from sampling_generator.engine.vocabulary import TERSE_PAYLOAD_TYPES
import random

def default_reply_delta(reply_relation: str) -> str:
    """Provide a structural reply contract when a Planner leaves it blank.

    The fallback deliberately names no domain fact or semantic conclusion. It
    only preserves the parent-child discourse requirement so the one Writer
    call must contribute a new relation rather than restating its parent.
    """

    normalized = str(reply_relation or "").strip().lower()
    if normalized == "asks_narrow_followup":
        return (
            "Ask one different narrow follow-up about a consequence, caveat, or "
            "constraint that the parent did not ask about."
        )
    if normalized == "challenges_parent":
        return (
            "State one specific caveat or counter-condition that changes the "
            "parent's conclusion without repeating its premise."
        )
    if normalized == "adds_datapoint":
        return (
            "Add one concrete observation, outcome, or evidence type not already "
            "used by the parent."
        )
    if normalized == "corrects_detail":
        return (
            "Correct one narrow implication of the parent and state the practical "
            "consequence without re-explaining its main point."
        )
    if normalized == "jokes_aside":
        return (
            "Make one small social aside about a different detail, not a paraphrase "
            "of the parent's conclusion."
        )
    return (
        "Add one new relation, consequence, caveat, evidence, or stance beyond "
        "the parent; do not repeat its question or conclusion."
    )

def choose_context_transform(
    *,
    rng: random.Random,
    has_parent: bool,
    payload_type: str,
    comment_function: str,
    reply_relation: str,
    context_aperture: str,
    dropout_rate: float,
    jitter_rate: float,
) -> str:
    """Choose how much visible context the writer receives for this task.

    This is semantic dropout, not text filtering.  The writer still receives the
    sampled plan, but for some comments it does not see the exact OP/parent
    wording that would pull it back to the same high-salience claim.
    """

    rate = max(0.0, min(1.0, float(dropout_rate)))
    jitter = max(0.0, min(1.0, float(jitter_rate)))
    if rate <= 0 and jitter <= 0:
        return "normal"

    relation = str(reply_relation or "")
    if has_parent:
        if jitter > 0 and rng.random() < jitter:
            return "parent_jittered"
        if payload_type in {"joke", "side_tangent", "meta_or_template"} or relation in {"jokes_aside", "shifts_to_side_detail"}:
            return weighted_choice(
                rng,
                (
                    ("parent_hidden", 0.55),
                    ("parent_gist", 0.30),
                    ("minor_detail_focus", 0.15),
                ),
            )
        if payload_type in TERSE_PAYLOAD_TYPES:
            return weighted_choice(
                rng,
                (
                    ("normal", max(0.0, 1.0 - rate)),
                    ("parent_hidden", rate * 0.45),
                    ("parent_gist", rate * 0.40),
                    ("minor_detail_focus", rate * 0.15),
                ),
            )
        if comment_function == "question_followup" or relation == "asks_narrow_followup":
            return weighted_choice(
                rng,
                (
                    ("normal", max(0.0, 1.0 - rate)),
                    ("minor_detail_focus", rate * 0.45),
                    ("parent_gist", rate * 0.35),
                    ("parent_hidden", rate * 0.20),
                ),
            )
        return weighted_choice(
            rng,
            (
                ("normal", max(0.0, 1.0 - rate)),
                ("parent_jittered", rate * 0.20),
                ("parent_gist", rate * 0.42),
                ("minor_detail_focus", rate * 0.38),
                ("parent_hidden", rate * 0.20),
            ),
        )

    if context_aperture in {"semantic_only", "title_only"}:
        return "normal"
    if jitter > 0 and rng.random() < jitter * 0.65:
        return "seed_jittered"
    seed_rate = rate * 0.60
    return weighted_choice(
        rng,
        (
            ("normal", max(0.0, 1.0 - seed_rate)),
            ("seed_jittered", seed_rate * 0.20),
            ("semantic_plan_only", seed_rate * 0.55),
            ("minor_detail_focus", seed_rate * 0.45),
        ),
    )
