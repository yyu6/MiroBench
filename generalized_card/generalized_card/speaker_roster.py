"""Who is speaking, and what stays true about them across their turns.

Before this module the generator had no concept of a person. Every slot got its
own author name -- `run_sampled_reddit_generator.py:1408` built it as
``f"sampled_user_{run}_{post_slot}_{task.local_task_id}"``, a pure function of
the slot index -- so a 186-comment thread was 186 people who each spoke once.

The matched real threads are not shaped like that. Across the ten evaluation
seeds: 559 comments from 265 authors, 2.11 comments per author, and 68% of all
comment mass written by someone who spoke more than once. Seed 8 alone is 200
comments from 80 authors, its busiest author wrote 10, and in seed 1 the OP
wrote 7 of 15.

Naming 186 different users does not produce 186 different voices; it produces
one voice wearing 186 name tags, which is the shape of the `self_bertscore`
result -- a near-uniform +0.033 overshoot on 9 of 10 threads, a shared register
signature applied evenly rather than topical narrowness.

The structure is free. `expand_matched_real_sample_to_tasks` already binds each
slot to one matched real comment through ``real_sample_id``
(``enumerate(selected, start=1)``), and the real comment carries its author. So
"which slots belong to the same person" is a join, not a new sampling policy,
and it reproduces the real thread's own participation shape exactly. Verified on
seed 8: ``real_word_count`` agrees for 186 of 186 slots.

Leakage: the real author string is used **only** as a grouping key. It is never
stored on a Speaker, never rendered, and never written to an artifact. What
crosses the boundary is the participation structure -- how many people, which
slots each holds, which one is the OP -- which is the same class of matched
structural signal as comment order, parent linkage, depth and word-count bucket
that `expand_matched_real_sample_to_tasks` already declares in its contract.

The identity content is domain-derived, never domain-specific: equipment comes
from the profile's `entity_inventory` (built from evaluation-excluded threads)
and the use case from the domain's own `configured_facets`. Only the tenure
ladder is domain-neutral, because "how long have you had it" needs no knowledge
of what "it" is.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from .entity_inventory import slot_equipment_options


# Domain-neutral: applies to any product, service, or tool a community discusses.
TENURE_LADDER = (
    "just picked it up recently",
    "had it for about a year",
    "used it for a few years now",
    "been on it a long time",
)

ANONYMOUS_AUTHORS = frozenset({"", "[deleted]", "[removed]", "none", "null"})

KIT_SIZE = 3

# Ablation modes. "off" reproduces the pre-v77 author naming exactly: one
# speaker per slot, no continuity. "matched" recovers the matched real thread's
# own participation structure.
SPEAKER_IDENTITY_OFF = "off"
SPEAKER_IDENTITY_MATCHED = "matched"


@dataclass(frozen=True)
class Speaker:
    """One participant, stable across every slot they hold in a thread."""

    speaker_id: str
    display_name: str
    is_op: bool
    slot_ids: tuple[int, ...]
    kit: tuple[str, ...]
    tenure: str
    use_case: str
    anonymous: bool = False

    @property
    def is_recurring(self) -> bool:
        return len(self.slot_ids) > 1


@dataclass(frozen=True)
class SpeakerRoster:
    """The cast of one thread, keyed by `real_sample_id`."""

    speakers: tuple[Speaker, ...]
    _by_slot: dict[int, Speaker]

    def speaker_for(self, real_sample_id: Any) -> Speaker | None:
        try:
            return self._by_slot.get(int(real_sample_id))
        except (TypeError, ValueError):
            return None

    def earlier_slots(self, real_sample_id: Any) -> tuple[int, ...]:
        """Return this speaker's own earlier slots, for their prompt's memory."""

        speaker = self.speaker_for(real_sample_id)
        if speaker is None:
            return ()
        try:
            current = int(real_sample_id)
        except (TypeError, ValueError):
            return ()
        return tuple(slot for slot in speaker.slot_ids if slot < current)

    def summary(self) -> dict[str, Any]:
        """Structural counts for the run log. Carries no real author string."""

        total_slots = sum(len(speaker.slot_ids) for speaker in self.speakers)
        recurring = [speaker for speaker in self.speakers if speaker.is_recurring]
        # Deleted accounts are one-shot by construction. Reporting them inside
        # the headline ratio would understate how much a named participant
        # actually recurs: on seed 8 that is 1.92 against the named-only 2.11.
        named = [speaker for speaker in self.speakers if not speaker.anonymous]
        named_slots = sum(len(speaker.slot_ids) for speaker in named)
        return {
            "speaker_count": len(self.speakers),
            "slot_count": total_slots,
            "comments_per_speaker": (
                round(total_slots / len(self.speakers), 3) if self.speakers else 0.0
            ),
            "named_speaker_count": len(named),
            "anonymous_speaker_count": len(self.speakers) - len(named),
            "comments_per_named_speaker": (
                round(named_slots / len(named), 3) if named else 0.0
            ),
            "recurring_speaker_count": len(recurring),
            "recurring_slot_share": (
                round(sum(len(s.slot_ids) for s in recurring) / total_slots, 3)
                if total_slots
                else 0.0
            ),
            "max_slots_for_one_speaker": (
                max((len(s.slot_ids) for s in self.speakers), default=0)
            ),
            "op_slot_count": sum(
                len(s.slot_ids) for s in self.speakers if s.is_op
            ),
        }


EMPTY_ROSTER = SpeakerRoster(speakers=(), _by_slot={})


def build_speaker_roster(
    selected_rows: Sequence[dict[str, Any]],
    *,
    inventory: dict[str, Any] | None = None,
    facets: Sequence[str] = (),
    name_prefix: str = "sampled_user",
) -> SpeakerRoster:
    """Group the thread's slots into the people who wrote them.

    `selected_rows` is the list `selected_matched_comments` returned, in the same
    order the task expansion consumed it, so row ``i`` is ``real_sample_id``
    ``i + 1``.
    """

    groups: list[list[int]] = []
    group_is_op: list[bool] = []
    group_anonymous: list[bool] = []
    index_by_author: dict[str, int] = {}

    for offset, row in enumerate(selected_rows or ()):
        sample_id = offset + 1
        author = str(row.get("author") or "").strip()
        is_op = bool(row.get("is_submitter"))
        if author.casefold() in ANONYMOUS_AUTHORS:
            # A deleted account is not one shared person. Real threads carry
            # many of these and merging them would invent a prolific speaker.
            groups.append([sample_id])
            group_is_op.append(is_op)
            group_anonymous.append(True)
            continue
        position = index_by_author.get(author)
        if position is None:
            position = len(groups)
            index_by_author[author] = position
            groups.append([])
            group_is_op.append(is_op)
            group_anonymous.append(False)
        groups[position].append(sample_id)
        if is_op:
            group_is_op[position] = True

    facet_list = [str(value).strip() for value in facets if str(value).strip()]
    speakers: list[Speaker] = []
    for position, slot_ids in enumerate(groups):
        speakers.append(
            Speaker(
                speaker_id=f"S{position + 1:03d}",
                display_name=f"{name_prefix}_{position + 1}",
                is_op=group_is_op[position],
                slot_ids=tuple(slot_ids),
                kit=tuple(
                    slot_equipment_options(
                        inventory, slot_index=position, limit=KIT_SIZE
                    )
                ),
                tenure=TENURE_LADDER[position % len(TENURE_LADDER)],
                use_case=(
                    facet_list[position % len(facet_list)] if facet_list else ""
                ),
                anonymous=group_anonymous[position],
            )
        )

    by_slot: dict[int, Speaker] = {}
    for speaker in speakers:
        for slot_id in speaker.slot_ids:
            by_slot[slot_id] = speaker
    return SpeakerRoster(speakers=tuple(speakers), _by_slot=by_slot)


def speaker_kit_for_slot(
    speaker: Speaker | None, *, excluded: Iterable[str] = ()
) -> tuple[str, ...]:
    """Return the speaker's kit minus anything already visible in this slot.

    Preserves the pre-v77 property that a slot never claims an entity the
    discussion already owns as its own gear. If the exclusion empties the kit the
    unfiltered kit is returned, because a speaker with no equipment is a worse
    outcome than one whose gear overlaps the thread's.
    """

    if speaker is None or not speaker.kit:
        return ()
    blocked = {
        str(value).strip().casefold()
        for value in excluded
        if str(value).strip()
    }
    if not blocked:
        return speaker.kit
    kept = tuple(
        item
        for item in speaker.kit
        if not any(part in item.casefold() for part in blocked)
    )
    return kept or speaker.kit
