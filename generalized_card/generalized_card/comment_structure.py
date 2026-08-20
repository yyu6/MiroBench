"""Measured layout of a comment at a given size, per domain.

A long Reddit comment is not one long paragraph. Measured over the camera
domain's 11,817 evaluation-excluded comments, the median paragraph count rises
with length -- 1 below 60 words, 2 at 60-120, 3 at 120-250, and 6 at 250 and
above, with a p90 of 14 -- and lists and quoted parent excerpts appear in 12.6%
and 26.7% of the longest comments. The v96 output had a blank line in 3.4% of
comments against 33.8% of real ones, and one paragraph at every size.

This is also why long slots came out short. The Writer realizes about 20 words
per planned beat and the Planner saturates near 9 beats however many are asked
for -- the largest v96 slot was asked for 40 -- so the beat mechanism tops out
near 250 words while matched slots reach 845. Asking for a comment's normal
number of paragraphs is both what real text looks like and a request the model
can actually satisfy at that scale.

The profile is measured, not written down, so a domain with different habits
gets its own layout. Nothing here reads matched evaluation text.
"""

from __future__ import annotations

import re
from typing import Any, Iterable


# Upper bound of each bucket in words; the last bucket is open-ended. These are
# size bands, not domain categories.
STRUCTURE_LENGTH_BOUNDS = (10, 25, 60, 120, 250)
STRUCTURE_BUCKETS = ("micro", "short", "medium", "long", "very_long", "essay")

_PARAGRAPH_SPLIT = re.compile(r"\n\s*\n")
_LIST_LINE = re.compile(r"^\s*(?:[-*+]\s+|\d+[.)]\s+)", re.M)
_QUOTE_LINE = re.compile(r"^\s*(?:>|&gt;)", re.M)
_MIN_SAMPLES = 200
_MIN_BUCKET_SAMPLES = 40

# Set once per run from the frozen domain profile. `soft_length_guidance` is
# installed on the generator as a one-argument callable, so the measured profile
# reaches it here rather than through a changed core signature.
ACTIVE_STRUCTURE_PROFILE: dict[str, Any] = {}
# `beats_only` reproduces every version through v96, which asked long slots for
# one thesis developed through up to 40 beats and gave no layout at all.
LONG_FORM_LAYOUT_ENABLED = True


def set_active_structure_profile(profile: dict[str, Any] | None) -> None:
    """Install the frozen per-domain layout profile for this run."""

    global ACTIVE_STRUCTURE_PROFILE
    ACTIVE_STRUCTURE_PROFILE = dict(profile or {})


def set_long_form_layout(mode: str) -> bool:
    """Select the layout arm and return whether it is active."""

    global LONG_FORM_LAYOUT_ENABLED
    LONG_FORM_LAYOUT_ENABLED = str(mode or "measured").strip().lower() != "beats_only"
    return LONG_FORM_LAYOUT_ENABLED


def active_layout_guidance(word_count: Any) -> str:
    """Render the layout cue for this slot from the run's frozen profile."""

    if not LONG_FORM_LAYOUT_ENABLED:
        return ""
    return layout_guidance(ACTIVE_STRUCTURE_PROFILE, word_count)


def structure_bucket(word_count: Any) -> str:
    """Return the size band one comment length falls in."""

    words = _safe_int(word_count)
    for index, bound in enumerate(STRUCTURE_LENGTH_BOUNDS):
        if words < bound:
            return STRUCTURE_BUCKETS[index]
    return STRUCTURE_BUCKETS[-1]


def build_structure_profile(threads: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Measure paragraph, list, and quote habits per size band."""

    samples: dict[str, list[tuple[int, bool, bool, int]]] = {
        name: [] for name in STRUCTURE_BUCKETS
    }
    total = 0
    for thread in threads:
        for row in thread.get("comments") or []:
            body = str(row.get("body") or row.get("content") or "")
            if not body.strip():
                continue
            total += 1
            paragraphs = len(
                [part for part in _PARAGRAPH_SPLIT.split(body) if part.strip()]
            )
            words = len(body.split())
            samples[structure_bucket(words)].append(
                (
                    max(1, paragraphs),
                    bool(_LIST_LINE.search(body)),
                    bool(_QUOTE_LINE.search(body)),
                    max(1, words),
                )
            )
    if total < _MIN_SAMPLES:
        return {"available": False, "sample_count": total, "buckets": {}}
    buckets: dict[str, dict[str, Any]] = {}
    for name, rows in samples.items():
        if len(rows) < _MIN_BUCKET_SAMPLES:
            continue
        paragraphs = sorted(row[0] for row in rows)
        per_paragraph = sorted(row[3] / row[0] for row in rows)
        buckets[name] = {
            "sample_count": len(rows),
            "median_paragraphs": paragraphs[len(paragraphs) // 2],
            "p90_paragraphs": paragraphs[int(0.9 * (len(paragraphs) - 1))],
            # The top band is open-ended, so one median paragraph count
            # under-serves its largest slots. Words per paragraph is nearly flat
            # inside a band (52-66 across the camera domain's essay band) while
            # the paragraph count scales: 6 at 250-350 words, 10 at 500-700, 11
            # above 700. Carrying both lets the cue scale within the band.
            "median_words_per_paragraph": round(
                per_paragraph[len(per_paragraph) // 2], 3
            ),
            "list_share": round(sum(row[1] for row in rows) / len(rows), 6),
            "quote_share": round(sum(row[2] for row in rows) / len(rows), 6),
        }
    return {
        "available": bool(buckets),
        "method": (
            "paragraph, list, and quoted-excerpt frequency per comment size band, "
            "over same-domain threads excluded from the evaluation seed pool"
        ),
        "sample_count": total,
        "buckets": buckets,
    }


def expected_paragraphs(profile: dict[str, Any] | None, word_count: Any) -> int:
    """Return the paragraph count a comment of this size is laid out in.

    The band's median is the floor and its p90 the ceiling; between them the
    count scales with the slot's own length at the band's measured words per
    paragraph, so an 845-word slot is not asked for a 300-word slot's layout.
    """

    words = _safe_int(word_count)
    bucket = ((profile or {}).get("buckets") or {}).get(structure_bucket(words))
    if not bucket:
        return 0
    median = max(1, _safe_int(bucket.get("median_paragraphs")))
    ceiling = max(median, _safe_int(bucket.get("p90_paragraphs")))
    try:
        per_paragraph = float(bucket.get("median_words_per_paragraph") or 0.0)
    except (TypeError, ValueError):
        per_paragraph = 0.0
    if per_paragraph <= 0 or words <= 0:
        return median
    scaled = int(round(words / per_paragraph))
    return max(median, min(ceiling, scaled))


def layout_guidance(profile: dict[str, Any] | None, word_count: Any) -> str:
    """Render the layout a comment of this size normally has here.

    This states how the turn is laid out on the page and that the paragraphs may
    go somewhere new. It deliberately does not prescribe an opening, a clause
    order, or a sentence shape: prescribing those gives every comment in the
    same register a shared entry route, which is what pushed `self_bleu_4` and
    `self_bertscore_mean_f1` the wrong way in an earlier release.
    """

    paragraphs = expected_paragraphs(profile, word_count)
    if paragraphs < 2:
        return ""
    return (
        f"Lay it out as about {paragraphs} short paragraphs separated by a blank "
        "line, the way a comment this long is normally typed here. Give each "
        "paragraph its own point rather than restating the first one, and let a "
        "related side point or aside take one of them if that is where the turn "
        "naturally goes."
    )


def _safe_int(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0
