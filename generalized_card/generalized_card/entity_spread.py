"""Per-slot referent spread, so a thread names as many things as a real one does.

**The defect.** Measured per matched thread on the v108 N=10 artifact, a real
camera thread names **40.8** distinct equipment designators and generated names
**7.4**; real's most frequent designator takes **0.152** of that thread's
mentions and generated's takes **0.485**. Pooled it is 302 against 67, one
generated thread names zero, and the concentration has *degraded* across
releases (top share v98 0.190 -> v103 0.214 -> v108 0.266). This is the
measured form of the user's own framing of the goal -- that a real thread
wanders and a generated one "都是统一的一个 topic 的趋势" -- and it is a
first-order criterion-2 tell independent of any metric.
See `docs/DECISIONS.md` G35.

**Why the existing mechanism does not reach it.** `entity_inventory.py` already
builds a held-out designator vocabulary and `slot_equipment_options` already
rotates it by slot, which is the right shape. But its only consumer,
`prompts._own_equipment_block`, is gated on
`writer_grounding_mode in {own, named}` **and** `_first_person_experience_slot`,
so it renders only for a turn that reports personal experience. Measured over
**18,829 designator mentions** in the evaluation-excluded corpus, only **14.0%**
sit in a possession context (`my X`, `I have`, `I shoot`); **8.9%** are explicit
comparisons and **77.1%** are bare mentions -- so **86.0% of real entity
mentions need no first-person frame at all**, and the existing gate can only
ever reach the smallest slice. That is why the two paid runs that did enable
`--own-fact-license named` (v97, v98) still landed at 81 pooled designators
against a real 302.

**Why this module does not simply widen that gate.** Offering *owned gear* to a
slot that was not planned for personal experience is a known, measured
regression: v67 found that "equipment plus first-person licensing" moved
`mean_story_probability`'s Cliff from 0.06 to 0.26, with per-thread gaps up to
+0.19, because own-gear anecdotes on a `no_story` slot produce text StorySeeker
scores as narrative. v88 then deleted invented kit/tenure/use-case biography for
the same class of reason. So this module offers a designator as a **bare
comparison referent** -- the 86% case -- never as a possession, never as a claim
about the seed post, and it leaves the possession path to
`_own_equipment_block` untouched.

**Priced before it was built.** Exact-ablation on the real scorer, in the
direction of the fix: raising generated's per-thread variety from 7.4 to 13.0
distinct designators (top share 0.485 -> 0.297) closes **5.4%** of
`self_bleu_4`'s +0.00489 gap. Collapsing *real*'s variety to a single designator
costs it 16.9%, so the relationship is real but asymmetric and most of
generated's excess is not reachable this way. **This module is therefore not
sold as a `self_bleu_4` fix.** It is a criterion-2 fix with a small, measured,
correctly-signed metric side-effect -- and per `docs/DECISIONS.md` G35 the
correct shape of work on `self_bleu_4` is several stacked fixes of about this
size, because no single large lever exists.

The arm is `--entity-spread {off,measured}`, default `off`, which renders
nothing and reproduces the previous release byte-for-byte.
"""

from __future__ import annotations

import hashlib
from typing import Any

from .entity_inventory import slot_equipment_options

# Installed per run from the frozen domain profile, like ACTIVE_RHYTHM_PROFILE.
ACTIVE_ENTITY_SPREAD_PROFILE: dict[str, Any] = {}
ENTITY_SPREAD_ENABLED = False

# Comment-count bands. A referent offer only makes sense once a thread is long
# enough that repetition is visible, and the target rate is measured per band.
THREAD_BANDS: tuple[tuple[str, int, int], ...] = (
    ("tiny", 0, 12),
    ("small", 12, 40),
    ("medium", 40, 100),
    ("large", 100, 10**9),
)

# Worded to hold in any domain: no camera vocabulary, no brand, no product noun.
# It names the *relation* (something else in the same category, mentioned in
# passing) rather than the entity, so the domain profile supplies the entity and
# this text stays domain-neutral -- the constraint `docs/ORIENTATION.md` §4
# states as "no domain vocabulary in Writer-facing rule text".
REFERENT_CUE = (
    "Mention one of these by name, in passing, as something else that exists "
    "in this space -- a point of comparison, an alternative, or just a thing "
    "people also bring up. Do not claim to own it, do not tell a story about "
    "it, and do not attribute it to the post or to another commenter. If none "
    "of them fits what you are actually saying, leave them out."
)


def set_active_entity_spread_profile(profile: dict[str, Any] | None) -> None:
    """Install the frozen per-domain referent profile for this run."""

    global ACTIVE_ENTITY_SPREAD_PROFILE
    ACTIVE_ENTITY_SPREAD_PROFILE = dict(profile or {})


def set_entity_spread(mode: str) -> bool:
    """Select the arm and return whether it is active."""

    global ENTITY_SPREAD_ENABLED
    ENTITY_SPREAD_ENABLED = str(mode or "off").strip().lower() == "measured"
    return ENTITY_SPREAD_ENABLED


def thread_band(comment_count: Any) -> str:
    """Name the comment-count band a thread of this size falls in."""

    try:
        size = int(comment_count)
    except (TypeError, ValueError):
        return ""
    for name, low, high in THREAD_BANDS:
        if low <= size < high:
            return name
    return ""


def build_entity_spread_profile(threads: Any) -> dict[str, Any]:
    """Measure, per thread band, how much referent variety a real thread carries.

    Two numbers per band, both from evaluation-excluded threads only:
    `mention_rate`, the share of comments carrying at least one designator, and
    `distinct_per_comment`, distinct designators divided by comment count. The
    second is what sets the per-slot offer rate, because it is the quantity the
    defect is measured in (real 40.8 distinct over ~53 comments against
    generated 7.4).
    """

    from .content_profile_analysis import DESIGNATOR

    bands: dict[str, dict[str, float]] = {}
    raw: dict[str, list[tuple[int, int, int]]] = {}
    for thread in threads or ():
        texts = [
            str(row.get("body") or row.get("content") or "")
            for row in thread.get("comments") or []
            if str(row.get("body") or row.get("content") or "").strip()
        ]
        if len(texts) < 2:
            continue
        band = thread_band(len(texts))
        if not band:
            continue
        found: set[str] = set()
        with_mention = 0
        for text in texts:
            hits = {match.group().casefold() for match in DESIGNATOR.finditer(text)}
            if hits:
                with_mention += 1
                found |= hits
        raw.setdefault(band, []).append((len(texts), len(found), with_mention))

    for band, rows in raw.items():
        comments = sum(row[0] for row in rows)
        if not comments:
            continue
        bands[band] = {
            "mention_rate": sum(row[2] for row in rows) / comments,
            "distinct_per_comment": sum(row[1] for row in rows) / comments,
            "thread_count": float(len(rows)),
            "comment_count": float(comments),
        }
    return {"available": bool(bands), "bands": bands}


def band_row(profile: dict[str, Any] | None, comment_count: Any) -> dict[str, Any]:
    """Return the measured row for this thread size, or empty when unmeasured.

    Empty means the cue is withheld rather than defaulted, so a sparse domain
    gets less of the mechanism instead of a wrong rate -- the same degradation
    contract the register/closing-move profiles use.
    """

    data = profile if profile is not None else ACTIVE_ENTITY_SPREAD_PROFILE
    if not (data or {}).get("available"):
        return {}
    band = thread_band(comment_count)
    if not band:
        return {}
    return dict(((data or {}).get("bands") or {}).get(band) or {})


def slot_offers_referent(
    profile: dict[str, Any] | None,
    *,
    slot_key: str,
    comment_count: Any,
) -> bool:
    """Deterministic per-slot draw at this band's measured distinct-per-comment rate.

    Rate-drawn rather than applied to every slot, for the reason
    `sentence_rhythm` is: a rule that fires on every slot of one size produces
    a uniform tic, which is the defect this module exists to remove.
    """

    if not ENTITY_SPREAD_ENABLED:
        return False
    row = band_row(profile, comment_count)
    share = float(row.get("distinct_per_comment") or 0.0)
    if share <= 0.0:
        return False
    if share >= 1.0:
        return True
    digest = hashlib.sha256(f"entity_spread:{slot_key}".encode("utf-8")).digest()
    draw = int.from_bytes(digest[:8], "big", signed=False) / float(1 << 64)
    return draw < share


def slot_referent_block(
    inventory: dict[str, Any] | None,
    *,
    profile: dict[str, Any] | None = None,
    slot_key: str,
    slot_index: Any,
    comment_count: Any,
    excluded: Any = (),
    limit: int = 3,
) -> str:
    """Render the referent offer for one slot, or "" when it is not drawn."""

    if not slot_offers_referent(profile, slot_key=slot_key, comment_count=comment_count):
        return ""
    options = slot_equipment_options(
        inventory,
        slot_index=int(slot_index or 0),
        limit=max(1, int(limit)),
        excluded=excluded,
    )
    if not options:
        return ""
    return "\n\nOther things in this space you may name:\n- " + ", ".join(options) + "\n" + REFERENT_CUE
