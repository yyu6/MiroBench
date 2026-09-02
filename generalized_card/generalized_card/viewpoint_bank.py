from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from typing import Any


TOKEN_RE = re.compile(r"[a-z][a-z0-9+./-]{1,30}", re.I)
STOPWORDS = set(
    """
    a about after all also am an and any are as at be because been before being
    both but by can could did do does doing dont for from get got had has have
    having he her here him how i if im in into is it its ive just like ll make me
    more most much my not now of on one only or other our out over re really same
    she should so some still than that thats the their them then there theres
    these they this those through to too up us use used using very want was way we
    were what when where which while who why will with would you your youre
    """.split()
)


_INDEX_CACHE: dict[str, tuple[list[dict[str, Any]], dict[str, float]]] = {}


def build_reference_viewpoints(
    threads: list[dict[str, Any]],
    *,
    max_comments_per_thread: int = 8,
    max_items: int = 3200,
) -> list[dict[str, Any]]:
    """Build a leakage-safe semantic source bank from non-seed threads.

    The bank keeps real language available to the private Planner, matching the
    original CARD abstraction step. The Writer never receives these records.
    Selection is structural and deterministic; it does not encode domain terms.
    """

    records: list[dict[str, Any]] = []
    for thread in threads:
        comments = [
            row
            for row in (thread.get("comments") or [])
            if _usable_text(row.get("body"))
        ]
        if not comments:
            continue
        selected = _select_comment_indices(
            comments,
            limit=max(1, max_comments_per_thread),
        )
        post_id = str(thread.get("post_id") or "")
        thread_hash = hashlib.sha256(post_id.encode("utf-8")).hexdigest()[:16]
        for index in selected:
            row = comments[index]
            text = _compact(row.get("body"), 420)
            parent_id = str(row.get("parent_id") or "")
            depth = _safe_int(row.get("depth"), 0)
            records.append(
                {
                    "reference_id": f"R{len(records) + 1:05d}",
                    "source_thread_hash": thread_hash,
                    "source_post_id": post_id,
                    "thread_title": _compact(thread.get("title"), 240),
                    "thread_context": _compact(thread.get("body"), 420),
                    "text": text,
                    "depth": depth,
                    "parent_scope": "op"
                    if depth == 0 or parent_id.startswith("t3_")
                    else "reply",
                    "word_count": len(text.split()),
                    "surface_role": _surface_role(text, depth=depth),
                }
            )
            if len(records) >= max_items:
                return records
    return records


# v144 arm. The Planner's picture of "what a real comment looks like" comes from
# this window, and ranking it purely by lexical relevance to the seed post makes
# that picture unrepresentative in two ways at once. Measured on celebrity seed 7
# against the 1,217-comment bank the window is drawn from:
#
#                       relatedness to post   off-topic (<0.10)   median words
#   full bank                       0.083            62.7%              12
#   what the Planner sees           0.190            27.8%              34
#
# BM25 scores by token overlap, so a short comment has few tokens to match and
# ranks low; the window therefore drops both the off-topic examples and the short
# ones. Those are the same comments -- 70% of the real corpus's semantically
# isolated comments are under ten words -- so the Planner is shown a corpus that
# is more coherent and three times wordier than the one it is meant to imitate,
# and then asked to produce scatter it has never been shown.
#
# `measured` keeps relevance as the ranking WITHIN each length band but fills the
# window across bands at the bank's own shares. Nothing is fitted to a p-value:
# the target distribution is the reference bank's, and the bank is built only
# from threads excluded from the seed pool.
REFERENCE_WINDOW_MODE = "off"
# Word-count bands. The boundaries are the isolation analysis's own: under ten
# words is where the real isolated comments live, and 40 is where the quota's
# range ends.
_LENGTH_BANDS = ((0, 5), (6, 10), (11, 20), (21, 40), (41, 10**9))


def set_reference_window(mode: str) -> bool:
    global REFERENCE_WINDOW_MODE
    REFERENCE_WINDOW_MODE = str(mode or "off").strip().lower()
    return REFERENCE_WINDOW_MODE == "measured"


def _band_of(word_count: int) -> int:
    for index, (low, high) in enumerate(_LENGTH_BANDS):
        if low <= word_count <= high:
            return index
    return len(_LENGTH_BANDS) - 1


def _distribution_matched_order(
    ranked: list[tuple[float, float, dict[str, Any]]],
) -> list[tuple[float, float, dict[str, Any]]]:
    """Re-order by relevance within length band, interleaved at bank shares.

    Emitting the band that is furthest below its share makes every prefix of the
    result approximately bank-shaped, so a later request for a wider window
    cannot reorder the rows an earlier Planner batch already saw.
    """

    queues: list[list[tuple[float, float, dict[str, Any]]]] = [[] for _ in _LENGTH_BANDS]
    for item in ranked:
        queues[_band_of(len(item[2].get("_comment_tokens") or []))].append(item)
    total = sum(len(q) for q in queues)
    if not total:
        return ranked
    shares = [len(q) / total for q in queues]
    cursors = [0] * len(queues)
    emitted = [0] * len(queues)
    out: list[tuple[float, float, dict[str, Any]]] = []
    for _ in range(total):
        best, best_debt = -1, None
        for index, queue in enumerate(queues):
            if cursors[index] >= len(queue) or shares[index] <= 0.0:
                continue
            # How far this band is behind where its share says it should be.
            debt = emitted[index] - shares[index] * len(out)
            if best_debt is None or debt < best_debt:
                best, best_debt = index, debt
        if best < 0:
            break
        out.append(queues[best][cursors[best]])
        cursors[best] += 1
        emitted[best] += 1
    return out


def retrieve_reference_viewpoints(
    profile: dict[str, Any],
    *,
    seed_title: str,
    seed_body: str,
    limit: int,
) -> list[dict[str, Any]]:
    """Retrieve relevant, source-diverse non-test comments for one seed.

    Ranking uses a domain-neutral BM25-like lexical score plus deterministic
    tie-breaking. A source-thread cap prevents one reference discussion from
    becoming the semantic template for the generated thread.
    """

    if limit <= 0:
        return []
    references, idf = _reference_index(profile)
    if not references:
        return []
    query_tokens = _tokens(f"{seed_title} {seed_body}")
    query = set(query_tokens)
    query_bigrams = set(zip(query_tokens, query_tokens[1:]))
    seed_key = hashlib.sha256(f"{seed_title}\n{seed_body}".encode("utf-8")).hexdigest()
    ranked: list[tuple[float, float, dict[str, Any]]] = []
    for row in references:
        title_tokens = set(row["_title_tokens"])
        context_tokens = set(row["_context_tokens"])
        comment_tokens = set(row["_comment_tokens"])
        score = 0.0
        for token in query:
            weight = idf.get(token, 1.0)
            if token in title_tokens:
                score += 3.0 * weight
            if token in context_tokens:
                score += 1.6 * weight
            if token in comment_tokens:
                score += 1.0 * weight
        comment_sequence = row["_comment_tokens"]
        comment_bigrams = set(zip(comment_sequence, comment_sequence[1:]))
        score += 2.5 * len(query_bigrams & comment_bigrams)
        # The substantive roles were previously the only ones not given this
        # bonus, and they are also absent from the role round-robin below, so
        # domain-specific content reached the Planner only through the final
        # fill-in pass. Measured against a matched real thread, that left the
        # generated side with 22 distinct model designators against 93, and a
        # concrete-domain-noun share of 0.32 against 0.64. Rank all roles alike.
        score += 0.20
        tie = _stable_fraction(f"{seed_key}:{row.get('reference_id')}")
        ranked.append((score, tie, row))
    ranked.sort(key=lambda item: (-item[0], item[1], str(item[2].get("reference_id"))))

    selected: list[dict[str, Any]] = []
    source_counts: Counter[str] = Counter()
    seen_text: set[str] = set()
    selected_token_sets: list[set[str]] = []

    def append(indexed: dict[str, Any]) -> bool:
        source = str(indexed.get("source_thread_hash") or "")
        if source_counts[source] >= 2:
            return False
        text_key = " ".join(str(indexed.get("text") or "").lower().split())
        if not text_key or text_key in seen_text:
            return False
        token_set = set(indexed.get("_comment_tokens") or [])
        if len(token_set) >= 4 and any(
            len(token_set & old) >= 4
            and len(token_set & old) / max(1, len(token_set | old)) >= 0.68
            for old in selected_token_sets
        ):
            return False
        selected.append(
            {key: value for key, value in indexed.items() if not key.startswith("_")}
        )
        source_counts[source] += 1
        seen_text.add(text_key)
        selected_token_sets.append(token_set)
        return True

    # Preserve the social and discourse variety that the matched real comments
    # supplied in CARD. Relevance still ranks candidates within each role.
    # ``full_answer`` and ``explanation`` are included because they carry the
    # domain specifics — procedures, compatibility, specs, model comparisons —
    # that a real thread is largely made of. Omitting them from the round-robin
    # starved the Planner of exactly the content the generated threads lacked.
    roles = (
        "social_reaction",
        "narrow_question",
        "personal_datapoint",
        "correction",
        "parent_local_reply",
        "full_answer",
        "explanation",
    )
    # The fixed round-robin prefix makes retrieval prefix-stable: requesting
    # R1..R36 later cannot reorder R1..R18 from an earlier planner batch.
    for _cycle in range(2):
        for role in roles:
            for _score, _tie, indexed in ranked:
                if indexed.get("surface_role") == role and append(indexed):
                    break
            if len(selected) >= limit:
                break
        if len(selected) >= limit:
            break
    fill = (
        _distribution_matched_order(ranked)
        if REFERENCE_WINDOW_MODE == "measured"
        else ranked
    )
    for _score, _tie, indexed in fill:
        if len(selected) >= limit:
            break
        append(indexed)
    return selected


def render_reference_viewpoints(
    profile: dict[str, Any],
    *,
    seed_title: str,
    seed_body: str,
    limit: int,
    offset: int = 0,
) -> str:
    references = reference_viewpoint_window(
        profile,
        seed_title=seed_title,
        seed_body=seed_body,
        limit=limit,
        offset=offset,
    )
    return render_reference_rows(references)


def reference_viewpoint_window(
    profile: dict[str, Any],
    *,
    seed_title: str,
    seed_body: str,
    limit: int,
    offset: int = 0,
) -> list[dict[str, Any]]:
    """Return a prefix-stable window from the excluded reference bank."""

    start = max(0, int(offset))
    width = max(0, int(limit))
    references = retrieve_reference_viewpoints(
        profile,
        seed_title=seed_title,
        seed_body=seed_body,
        limit=start + width,
    )
    return references[start : start + width]


def render_reference_rows(references: list[dict[str, Any]]) -> str:
    """Render already-selected reference rows without retrieving them again."""

    rows = []
    for item in references:
        rows.append(
            f"- {item.get('reference_id')}: source_topic={_compact(item.get('thread_title'), 120)}; "
            f"depth={item.get('depth', 0)}; parent={item.get('parent_scope', 'op')}; "
            f"words={item.get('word_count', 0)}; surface={item.get('surface_role', 'local_turn')}; "
            f"text={_compact(item.get('text'), 300)}"
        )
    return "\n".join(rows) or "- No non-test reference viewpoints are available."


def _reference_index(
    profile: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    key = str(profile.get("profile_sha256") or id(profile))
    cached = _INDEX_CACHE.get(key)
    if cached is not None:
        return cached
    raw = [
        row
        for row in (profile.get("reference_viewpoints") or [])
        if isinstance(row, dict)
    ]
    indexed: list[dict[str, Any]] = []
    document_frequency: Counter[str] = Counter()
    for row in raw:
        copied = dict(row)
        copied["_title_tokens"] = _tokens(copied.get("thread_title"))
        copied["_context_tokens"] = _tokens(copied.get("thread_context"))
        copied["_comment_tokens"] = _tokens(copied.get("text"))
        document_frequency.update(
            set(
                copied["_title_tokens"]
                + copied["_context_tokens"]
                + copied["_comment_tokens"]
            )
        )
        indexed.append(copied)
    count = max(1, len(indexed))
    idf = {
        token: math.log((count + 1) / (frequency + 1)) + 1.0
        for token, frequency in document_frequency.items()
    }
    _INDEX_CACHE[key] = (indexed, idf)
    return indexed, idf


def _select_comment_indices(comments: list[dict[str, Any]], *, limit: int) -> list[int]:
    if len(comments) <= limit:
        return list(range(len(comments)))
    categories = (
        [
            index
            for index, row in enumerate(comments)
            if _safe_int(row.get("depth"), 0) >= 2
        ],
        [
            index
            for index, row in enumerate(comments)
            if len(str(row.get("body") or "").split()) <= 12
        ],
        [
            index
            for index, row in enumerate(comments)
            if "?" in str(row.get("body") or "")
        ],
        [
            index
            for index, row in enumerate(comments)
            if _surface_role(
                str(row.get("body") or ""), depth=_safe_int(row.get("depth"), 0)
            )
            in {"personal_datapoint", "correction", "social_reaction"}
        ],
    )
    selected: list[int] = []
    for candidates in categories:
        if candidates:
            value = candidates[len(candidates) // 2]
            if value not in selected:
                selected.append(value)
            if len(selected) >= limit:
                return sorted(selected)
    for value in _evenly_spaced_indices(len(comments), limit * 2):
        if value not in selected:
            selected.append(value)
        if len(selected) >= limit:
            break
    return sorted(selected)


def _evenly_spaced_indices(size: int, count: int) -> list[int]:
    count = min(size, max(0, count))
    if count <= 0:
        return []
    if count == 1:
        return [size // 2]
    return sorted({round(step * (size - 1) / (count - 1)) for step in range(count)})


def _surface_role(text: str, *, depth: int) -> str:
    lowered = text.lower()
    words = len(text.split())
    if words <= 7:
        return "social_reaction"
    if "?" in text and words <= 28:
        return "narrow_question"
    if (
        re.search(r"\b(no|wrong|incorrect|not true|actually|except|but)\b", lowered)
        and words <= 60
    ):
        return "correction"
    if re.search(r"\b(i|i'm|i've|my|we|our)\b", lowered):
        return "personal_datapoint"
    if depth >= 2:
        return "parent_local_reply"
    if words >= 65:
        return "explanation"
    return "full_answer"


def _tokens(value: Any) -> list[str]:
    output = []
    for raw in TOKEN_RE.findall(str(value or "")):
        token = raw.lower().replace("'", "").strip("./-")
        if len(token) >= 2 and token not in STOPWORDS:
            output.append(token)
    return output


def _usable_text(value: Any) -> bool:
    text = " ".join(str(value or "").split())
    return bool(text) and text.lower() not in {
        "[deleted]",
        "[removed]",
        "deleted",
        "removed",
    }


def _compact(value: Any, limit: int) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else text[: max(0, limit - 3)].rstrip() + "..."


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _stable_fraction(value: str) -> float:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]
    return int(digest, 16) / float(16**12)
