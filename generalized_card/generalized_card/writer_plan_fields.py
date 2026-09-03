"""`--writer-plan-fields angle_detail`: stop handing the Writer the Planner's
own sentences about what the comment argues.

Measured on v156's own stored Writer prompts (364 slots, every cosine inside
one thread against that thread's own matched real thread), the six meaning-
bearing things the Writer is shown disperse very differently:

    content_angle      0.1746      domain_intent      0.3031
    detail_focus       0.2127      semantic_move      0.3071

`content_angle` and `detail_focus` are short labels ("public slogan stance",
"shirt as social statement").  `semantic_move` and `domain_intent` are finished
sentences the Planner wrote ("Reject the shirt as a loud virtue signal and
treat the public message as the whole point"), and the Planner writes similar
sentences for every slot of one thread.  Every subset containing either of them
prices at plan cosine ~0.31; dropping both reaches 0.2173.

Fitted on the same run, realized text = 0.384 x plan + 0.0665, so reaching
real's 0.1552 needs plan cosine 0.2310 against the current 0.3657 -- and
`content_angle` + `detail_focus` alone predicts 0.1500.

**That prediction is an extrapolation, not a measurement.** The realization
function was fitted while the Writer could see all six fields; with less
information the Writer must invent the point, and what it invents comes from
its own priors -- which is what the 0.0665 intercept is. The intercept could
rise and eat the gain. That is exactly what the paid run tests, and it is the
reason this ships as an arm whose default reproduces the prior release rather
than as a repair.
"""
from __future__ import annotations

# `full` reproduces every prior release byte for byte.
WRITER_PLAN_FIELDS_MODE = "full"

# The Planner's own prose about what to argue, in the order the Writer sees it.
_ANGLE_DETAIL_HIDDEN = frozenset({"semantic_move", "decision_boundary", "domain_intent"})

_HIDDEN_BY_MODE = {
    "full": frozenset(),
    "angle_detail": _ANGLE_DETAIL_HIDDEN,
}


def set_writer_plan_fields(mode: str) -> str:
    global WRITER_PLAN_FIELDS_MODE
    value = str(mode or "full").strip().lower()
    if value not in _HIDDEN_BY_MODE:
        raise ValueError(f"unknown writer-plan-fields mode: {mode}")
    WRITER_PLAN_FIELDS_MODE = value
    return WRITER_PLAN_FIELDS_MODE


def hidden(field: str) -> bool:
    """Is this plan field withheld from the Writer under the active mode?"""

    return field in _HIDDEN_BY_MODE[WRITER_PLAN_FIELDS_MODE]


def active() -> bool:
    return WRITER_PLAN_FIELDS_MODE != "full"


def substitute_route_lock(detail: str) -> list[str]:
    """What replaces the withheld proposition block.

    The block has a header in three Writer templates, so it cannot simply
    vanish. It names the same detail the turn-kind block already carries and
    hands the point itself back to the Writer, which is the whole intervention.
    """

    text = " ".join(str(detail or "").split())
    rows = []
    if text:
        rows.append(f"- What this comment is about: {text}")
    rows.append(
        "- Decide for yourself what point to make about it. Make one narrow "
        "local point that belongs to this slot alone and stop when it is made."
    )
    return rows
