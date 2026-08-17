"""Recover thread participation structure without importing identity content.

Each matched slot already has a ``real_sample_id``. The source author is used
only to group those slot IDs and identify the OP; it is never stored, rendered,
or persisted. A ``Speaker`` therefore carries no invented kit, tenure, use case,
or biography. Those semantic claims belong to the Planner, not to structural
matching.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

ANONYMOUS_AUTHORS = frozenset({"", "[deleted]", "[removed]", "none", "null"})

# ``off`` is handled by the caller as an empty roster; this module only needs
# the value that activates matched participation.
SPEAKER_IDENTITY_MATCHED = "matched"


@dataclass(frozen=True)
class Speaker:
    """One participant, stable across every slot they hold in a thread."""

    speaker_id: str
    is_op: bool
    slot_ids: tuple[int, ...]
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

    speakers: list[Speaker] = []
    for position, slot_ids in enumerate(groups):
        speakers.append(
            Speaker(
                speaker_id=f"S{position + 1:03d}",
                is_op=group_is_op[position],
                slot_ids=tuple(slot_ids),
                anonymous=group_anonymous[position],
            )
        )

    by_slot: dict[int, Speaker] = {}
    for speaker in speakers:
        for slot_id in speaker.slot_ids:
            by_slot[slot_id] = speaker
    return SpeakerRoster(speakers=tuple(speakers), _by_slot=by_slot)
