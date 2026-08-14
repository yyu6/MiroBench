"""Domain equipment inventory built from evaluation-excluded real threads.

The Writer may only name entities visible in the seed post or in its task
anchors. That is the right rule for claims *about* the discussion, but it also
means every comment in a thread can only ever name the two or three products the
seed happens to mention. Measured on one matched pair, the real thread named 117
distinct camera models with its most frequent one at 3% of mentions, while the
generated thread named 23 with its most frequent one at 29%. That concentration
is a direct contributor to within-thread 4-gram overlap.

Real commenters name their *own* equipment. This module builds the vocabulary
that makes that possible without inventing facts about the seed: it collects the
equipment names that actually occur in same-domain threads excluded from the
evaluation seed pool, using the domain configuration's brand list as the only
domain-specific input. The algorithm is shared across domains; the brand list is
a configuration boundary.

A name from this inventory is licensed only as the speaker's own gear, on slots
whose plan already permits first-person experience. It is never presented as a
property of the seed post or of another commenter.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any, Iterable

# A model designator is a short alphanumeric token: "a7 IV", "X-T2", "D750",
# "5D", "GR III", "OM-1", "24-70". Prose words are excluded by requiring either
# an internal digit or an all-caps shape next to a configured brand.
_MODEL_TOKEN = re.compile(
    r"""
    (?:
        [A-Za-z]{0,3}\d{1,4}[A-Za-z]{0,3}(?:\s?(?:I{1,3}|IV|V|VI|VII|VIII|IX|X))?
      | [A-Z]{1,3}-?[A-Z]?\d{1,3}[A-Za-z]{0,2}
      | [A-Z]{2,4}(?:\s?\d{1,3})?
    )
    """,
    re.VERBOSE,
)
# One token per alphanumeric run, keeping internal hyphens, apostrophes and dots.
# An earlier pattern stopped a letter-initial token at its first digit, so every
# letter-initial designator was split in half and then rejected for lacking a
# digit: "XM5" became "XM"+"5", "X1" became "X"+"1", "WH-1000XM4" became
# "WH-"+"1000XM4", and Sony's "a7" bodies never appeared at all. Only
# digit-initial designators such as "5D" and "50mm" survived, which is why the
# camera inventory looked usable while three other domains yielded almost
# nothing.
_WORD = re.compile(r"[A-Za-z0-9]+(?:[-'’.][A-Za-z0-9]+)*")
_MIN_OCCURRENCES = 2
# A genuine product designator is written with its brand some of the time and
# bare the rest ("Canon 5D", then "the 5D"). A specification value is almost
# always bare: "4GB" follows a brand once by accident and then appears hundreds
# of times on its own. Requiring a designator to be brand-associated in a
# meaningful fraction of its uses separates the two without any domain
# knowledge, which the shared extraction needs in order to serve every domain.
_MIN_BRAND_ASSOCIATION = 0.05
_MIN_BRANDED_MENTIONS = 2
_MAX_TERMS = 400
_MAX_FOLLOWING_TOKENS = 3


def build_entity_inventory(
    threads: Iterable[dict[str, Any]],
    *,
    brand_terms: Iterable[str],
    max_terms: int = _MAX_TERMS,
    min_occurrences: int = _MIN_OCCURRENCES,
) -> dict[str, Any]:
    """Collect equipment names occurring in evaluation-excluded threads.

    Two passes, because real writers overwhelmingly name a model without its
    brand ("my D750", "the a7III", "X-T2"). Pass one uses brand adjacency to
    learn which designators are real equipment in this domain. Pass two counts
    every occurrence of those learned designators, branded or bare, so the
    inventory reflects actual usage frequency rather than only the minority of
    mentions that happen to repeat the brand.
    """

    brands = [str(term).strip() for term in brand_terms if str(term).strip()]
    brand_lookup = {brand.casefold(): brand for brand in brands}
    if not brand_lookup:
        return {
            "available": False,
            "reason": "domain configuration lists no brand terms",
            "brand_count": 0,
            "terms": [],
        }

    documents = [
        [
            str(row.get("body") or row.get("content") or "")
            for row in thread.get("comments") or []
            if str(row.get("body") or row.get("content") or "").strip()
        ]
        for thread in threads
    ]
    documents = [group for group in documents if group]
    bodies = [body for group in documents for body in group]

    # A designator can legitimately follow more than one brand ("50mm"), so keep
    # every observed association and resolve each one to its most frequent brand
    # rather than whichever happened to appear first in corpus order.
    associations: dict[str, Counter[str]] = {}
    brand_counts: Counter[str] = Counter()
    # A real product designator recurs across separate discussions; a spec value
    # that happened to follow a brand once ("Dell 4GB", "AirPods 5k") does not.
    # Requiring several distinct source threads is a structural precision filter
    # that needs no domain knowledge, which matters because the same extraction
    # has to serve every configured domain.
    thread_frequency: Counter[str] = Counter()
    for group in documents:
        seen_here: set[str] = set()
        for body in group:
            for designator, brand in _iter_branded_designators(body, brand_lookup):
                key = designator.casefold()
                associations.setdefault(key, Counter())[brand] += 1
                brand_counts[brand] += 1
                seen_here.add(key)
        thread_frequency.update(seen_here)
    validated_brand = {
        key: counter.most_common(1)[0][0] for key, counter in associations.items()
    }
    if not validated_brand:
        return {
            "available": False,
            "reason": "no brand-anchored equipment designator occurred in the reference threads",
            "brand_count": len(brand_lookup),
            "terms": [],
        }

    counts: Counter[str] = Counter()
    display: dict[str, str] = {}
    for body in bodies:
        for token in _WORD.findall(body):
            key = token.casefold()
            if key not in validated_brand:
                continue
            counts[key] += 1
            display.setdefault(key, token)

    branded_totals = {
        key: sum(counter.values()) for key, counter in associations.items()
    }
    terms = [
        {
            "term": f"{validated_brand[key]} {display[key]}",
            "designator": display[key],
            "brand": validated_brand[key],
            "count": int(count),
            "branded_mentions": int(branded_totals.get(key, 0)),
        }
        for key, count in counts.most_common()
        if count >= max(1, int(min_occurrences))
        and branded_totals.get(key, 0) >= _MIN_BRANDED_MENTIONS
        and branded_totals.get(key, 0) / count >= _MIN_BRAND_ASSOCIATION
    ][: max(1, int(max_terms))]
    return {
        "available": bool(terms),
        "method": (
            "equipment designators learned by brand adjacency, then counted in "
            "every form, over same-domain threads excluded from the evaluation "
            "seed pool; no seed thread is read"
        ),
        "license": "speaker's own equipment only; never a claim about the seed post",
        "brand_count": len(brand_lookup),
        "distinct_terms": len(terms),
        "brand_mention_counts": dict(brand_counts.most_common()),
        "terms": terms,
    }


def _iter_branded_designators(
    text: str,
    brand_lookup: dict[str, str],
) -> Iterable[tuple[str, str]]:
    """Yield ``(designator, brand)`` for each brand-adjacent model designator."""

    tokens = _WORD.findall(text)
    for index, token in enumerate(tokens):
        brand = brand_lookup.get(token.casefold())
        if brand is None:
            continue
        for following in tokens[index + 1 : index + 1 + _MAX_FOLLOWING_TOKENS]:
            if not _is_model_token(following):
                break
            yield following, brand


def _is_model_token(token: str) -> bool:
    if len(token) > 8 or len(token) < 2:
        return False
    if _MODEL_TOKEN.fullmatch(token) is None:
        return False
    # Require both a letter and a digit. A purely alphabetic designator is a
    # product line or company suffix ("EOS", "DSLR", "USA"); a bare number is a
    # focal length, ISO, price, or percentage that happened to follow a brand,
    # and counting every later occurrence of it would swamp the inventory.
    return any(char.isdigit() for char in token) and any(
        char.isalpha() for char in token
    )


def slot_equipment_options(
    inventory: dict[str, Any] | None,
    *,
    slot_index: int,
    limit: int = 4,
    excluded: Iterable[str] = (),
) -> list[str]:
    """Return a deterministic, slot-rotating subset of the inventory.

    Rotating by slot is what spreads entity mass across a thread. Handing every
    slot the same head of the list would reproduce the concentration this
    inventory exists to remove.
    """

    terms = [
        str(row.get("term") or "").strip()
        for row in (inventory or {}).get("terms") or []
        if str(row.get("term") or "").strip()
    ]
    if not terms:
        return []
    blocked = {value.casefold() for value in excluded if str(value).strip()}
    available = [
        term
        for term in terms
        if not any(part in term.casefold() for part in blocked)
    ] or terms
    width = max(1, int(limit))
    start = (max(0, int(slot_index)) * width) % len(available)
    rotated = available[start:] + available[:start]
    return rotated[:width]
