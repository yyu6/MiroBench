from __future__ import annotations

import math
import re
import statistics
from collections import Counter
from typing import Any, Callable, Iterable

from .content_profile_data import clean_text, optional_float


TOKEN = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")
FIRST_PERSON = re.compile(r"I['’a-z]* ")
DIGIT = re.compile(r"\d")
LIST_SHAPE = re.compile(r"(^|\s)\d[.)]\s|\n\s*[-*]\s")
DESIGNATOR = re.compile(
    r"\b(?=[A-Za-z0-9-]{2,8}\b)(?=[^\s]*\d)(?=[^\s]*[A-Za-z])[A-Za-z0-9-]+\b"
)
END_PUNCT = tuple(".!?)\"”")

# Explicitly weak surface probes. They are never semantic classifiers.
ASSISTANT_HELP = re.compile(
    r"\b(?:you (?:should|can|could|might want to)|i(?:'d| would) recommend|"
    r"make sure (?:you )?|your best bet|worth (?:checking|trying)|"
    r"hope (?:that |this )?helps|if you (?:want|need))\b",
    re.I,
)
PROFANITY = re.compile(
    r"\b(?:wtf|fuck(?:ing|ed)?|shit(?:ty)?|bullshit|crap|disgusting|garbage|trash)\b",
    re.I,
)
LAUGHTER = re.compile(r"\b(?:lol|lmao|rofl|ha(?:ha)+)\b", re.I)
BLUNT = re.compile(
    r"(?:^|[.!?]\s+)(?:no|nope|nah|wrong|stop)\b|\b(?:sucks?|ridiculous)\b",
    re.I,
)


def content_properties(real: list[str], generated: list[str], config: Any) -> list[dict[str, Any]]:
    technical = tuple(term.lower() for term in config.technical_terms if term.strip())
    vocabulary = tuple(
        term.lower()
        for term in (*config.technical_terms, *config.protected_entity_terms)
        if term.strip()
    )
    definitions: list[tuple[str, Callable[[list[str]], float], str]] = [
        ("repeated 4-gram share", lambda rows: repeated_ngram_share(rows, 4), "match target"),
        ("repeated 5-gram share", lambda rows: repeated_ngram_share(rows, 5), "match target"),
        ("distinct 3-word openers", distinct_openers, "share of comments"),
        ("distinct model designators", lambda rows: float(len(designators(rows))), "count"),
        ("top designator share", top_designator_share, "concentration"),
        ("has technical term", lambda rows: share(rows, lambda text: has_any(text, technical)), ""),
        ("has domain vocabulary", lambda rows: share(rows, lambda text: has_any(text, vocabulary)), ""),
        ("has a digit", lambda rows: share(rows, lambda text: bool(DIGIT.search(text))), ""),
        ("opens first person", lambda rows: share(rows, lambda text: bool(FIRST_PERSON.match(text))), ""),
        ("has a question mark", lambda rows: share(rows, lambda text: "?" in text), ""),
        ("uses a quote marker", lambda rows: share(rows, has_quote_marker), ""),
        ("uses a list shape", lambda rows: share(rows, lambda text: bool(LIST_SHAPE.search(text))), ""),
        ("no end punctuation", lambda rows: share(rows, lambda text: bool(text) and not text.endswith(END_PUNCT)), ""),
        ("median words", median_word_count, ""),
        ("mean words", mean_word_count, ""),
        ("word count CV", word_cv, "length spread"),
        ("longest comment words", lambda rows: float(max((word_count(text) for text in rows), default=0)), ""),
        ("comments under 10 words", lambda rows: share(rows, lambda text: word_count(text) < 10), ""),
        ("comments over 100 words", lambda rows: share(rows, lambda text: word_count(text) > 100), ""),
    ]
    return comparison_rows(real, generated, definitions)


def discourse_properties(
    real: dict[str, list[str]],
    generated: dict[str, list[str]],
    keys: list[str],
) -> list[dict[str, Any]]:
    rows = []
    for label, fn in (
        ("neighbour vocabulary overlap", adjacent_overlap),
        ("thread vocabulary breadth", vocab_breadth),
    ):
        real_values = [fn(real[key]) for key in keys if len(real[key]) > 2]
        generated_values = [fn(generated[key]) for key in keys if len(generated[key]) > 2]
        rows.append(
            comparison_row(
                label,
                statistics.mean(real_values) if real_values else 0.0,
                statistics.mean(generated_values) if generated_values else 0.0,
                "paired thread mean",
            )
        )
    return rows


def model_properties(
    real: dict[str, dict[str, dict[str, Any]]],
    generated: dict[str, dict[str, dict[str, Any]]],
) -> list[dict[str, Any]]:
    real_emotion = dominant_counts(real["emotion"].values())
    generated_emotion = dominant_counts(generated["emotion"].values())
    real_story = story_values(real["story"].values())
    generated_story = story_values(generated["story"].values())
    values: list[tuple[str, float, float, str]] = []
    if real_emotion and generated_emotion:
        values.extend(
            [
                ("dominant-emotion entropy", entropy(real_emotion), entropy(generated_emotion), "exact matched comments"),
                ("distinct dominant emotions", float(len(real_emotion)), float(len(generated_emotion)), "count"),
                ("neutral dominant share", label_share(real_emotion, "neutral"), label_share(generated_emotion, "neutral"), ""),
            ]
        )
    if real_story and generated_story:
        values.extend(
            [
                ("mean story probability", statistics.mean(real_story), statistics.mean(generated_story), "exact matched comments"),
                ("story probability > 0.5", above_share(real_story, 0.5), above_share(generated_story, 0.5), ""),
            ]
        )
    return [comparison_row(*row) for row in values]


def realization(
    records: list[dict[str, Any]],
    models: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    accepted = [row for row in records if isinstance(row.get("comment"), dict)]
    model_index = {name: comment_index(threads.values()) for name, threads in models.items()}
    tone_confusion: Counter[str] = Counter()
    affect_confusion: Counter[str] = Counter()
    tone_covered = tone_aligned = affect_covered = affect_aligned = 0
    story_groups: dict[str, list[float]] = {"planned_story": [], "planned_no_story": []}
    function_counts: Counter[str] = Counter()
    payload_counts: Counter[str] = Counter()
    role_counts: Counter[str] = Counter()

    for record in accepted:
        task = record.get("task") or {}
        comment = record.get("comment") or {}
        comment_id = normalize_id(comment.get("comment_id"))
        assigned_tone = normalize_label(task.get("tone_target"))
        predicted_tone = normalize_label((model_index["tone"].get(comment_id) or {}).get("pred_label"))
        if assigned_tone and predicted_tone:
            tone_covered += 1
            tone_aligned += assigned_tone == predicted_tone
            tone_confusion[f"{assigned_tone}->{predicted_tone}"] += 1

        assigned_affect = normalize_label(task.get("affect_role"))
        predicted_affect = normalize_label(
            (model_index["emotion"].get(comment_id) or {}).get("dominant_emotion")
        )
        if assigned_affect and predicted_affect:
            affect_covered += 1
            affect_aligned += assigned_affect == predicted_affect
            affect_confusion[f"{assigned_affect}->{predicted_affect}"] += 1

        probability = optional_float(
            (model_index["story"].get(comment_id) or {}).get("story_probability")
        )
        story_mode = normalize_label(task.get("story_mode"))
        group = "planned_no_story" if story_mode in {"", "none", "no_story"} else "planned_story"
        if probability is not None:
            story_groups[group].append(probability)
        function_counts[normalize_label(task.get("comment_function")) or "missing"] += 1
        payload_counts[normalize_label(task.get("payload_type")) or "missing"] += 1
        role_counts[normalize_label(task.get("speaker_role")) or "missing"] += 1

    total = len(accepted)
    return {
        "accepted_records": total,
        "tone": alignment_summary(tone_covered, tone_aligned, tone_confusion),
        "affect": alignment_summary(affect_covered, affect_aligned, affect_confusion),
        "story": {
            name: {
                "count": len(values),
                "mean_probability": statistics.mean(values) if values else None,
                "classified_story_share": above_share(values, 0.5) if values else None,
            }
            for name, values in story_groups.items()
        },
        "planned_surface": {
            "comment_function_counts": dict(function_counts.most_common()),
            "payload_type_counts": dict(payload_counts.most_common()),
            "speaker_role_counts": dict(role_counts.most_common()),
            "recommendation_advice_share": function_counts["recommendation_advice"] / total if total else 0.0,
            "soft_helpful_payload_share": payload_counts["soft_helpful"] / total if total else 0.0,
            "advisor_share": role_counts["advisor"] / total if total else 0.0,
        },
    }


def alignment_summary(covered: int, aligned: int, confusion: Counter[str]) -> dict[str, Any]:
    return {
        "covered": covered,
        "aligned": aligned,
        "exact_rate": aligned / covered if covered else None,
        "confusion": dict(confusion.most_common()),
    }


def surface_diagnostics(real: list[str], generated: list[str]) -> dict[str, Any]:
    definitions = [
        ("assistant-help phrase", lambda rows: share(rows, lambda text: bool(ASSISTANT_HELP.search(text))), "weak lexical probe"),
        ("profanity/strong disgust token", lambda rows: share(rows, lambda text: bool(PROFANITY.search(text))), "weak lexical probe"),
        ("laughter token", lambda rows: share(rows, lambda text: bool(LAUGHTER.search(text))), "weak lexical probe"),
        ("blunt surface marker", lambda rows: share(rows, lambda text: bool(BLUNT.search(text))), "weak lexical probe"),
    ]
    return {
        "provenance": "matched-side lexical regex; diagnostic only",
        "rows": comparison_rows(real, generated, definitions),
    }


def repetition_diagnostics(texts: list[str]) -> dict[str, Any]:
    normalized = Counter(" ".join(TOKEN.findall(text.lower())) for text in texts)
    duplicates = sum(count - 1 for text, count in normalized.items() if text and count > 1)
    return {
        "comment_count": len(texts),
        "exact_duplicate_excess": duplicates,
        "exact_unique_share": (len(normalized) / len(texts)) if texts else 0.0,
        "repeated_4gram_share": repeated_ngram_share(texts, 4),
        "repeated_5gram_share": repeated_ngram_share(texts, 5),
        "distinct_3word_opener_share": distinct_openers(texts),
        "top_repeated_ngrams": top_repeated_ngrams(texts),
    }


def examples(
    records: list[dict[str, Any]],
    real_texts: list[str],
    models: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    index = {name: comment_index(rows.values()) for name, rows in models.items()}
    tone_mismatches = []
    story_mismatches = []
    planned_helpful = []
    generated_colloquial = []
    for record in records:
        task = record.get("task") or {}
        comment = record.get("comment") or {}
        text = clean_text(comment.get("content"))
        if not text:
            continue
        comment_id = normalize_id(comment.get("comment_id"))
        assigned_tone = normalize_label(task.get("tone_target"))
        predicted_tone = normalize_label((index["tone"].get(comment_id) or {}).get("pred_label"))
        if assigned_tone and predicted_tone and assigned_tone != predicted_tone:
            tone_mismatches.append(
                {"comment_id": comment_id, "assigned": assigned_tone, "predicted": predicted_tone, "text": truncate(text)}
            )
        probability = optional_float((index["story"].get(comment_id) or {}).get("story_probability"))
        story_mode = normalize_label(task.get("story_mode"))
        planned_story = story_mode not in {"", "none", "no_story"}
        if probability is not None:
            story_mismatches.append(
                {
                    "comment_id": comment_id,
                    "planned": "story" if planned_story else "no_story",
                    "story_probability": round(probability, 4),
                    "_mismatch": (1.0 - probability) if planned_story else probability,
                    "text": truncate(text),
                }
            )
        if planned_helpful_task(task):
            planned_helpful.append(
                {
                    "comment_id": comment_id,
                    "function": normalize_label(task.get("comment_function")),
                    "payload": normalize_label(task.get("payload_type")),
                    "text": truncate(text),
                }
            )
        if colloquial_surface(text):
            generated_colloquial.append({"comment_id": comment_id, "text": truncate(text)})
    story_mismatches.sort(key=lambda row: row["_mismatch"], reverse=True)
    for row in story_mismatches:
        del row["_mismatch"]
    return {
        "story_mismatches": story_mismatches[:8],
        "tone_mismatches": tone_mismatches[:8],
        "planned_helpful": planned_helpful[:8],
        "generated_colloquial": generated_colloquial[:8],
        "real_colloquial": [
            {"text": truncate(text)} for text in real_texts if colloquial_surface(text)
        ][:8],
    }


def planned_helpful_task(task: dict[str, Any]) -> bool:
    return (
        normalize_label(task.get("comment_function")) == "recommendation_advice"
        or normalize_label(task.get("payload_type")) == "soft_helpful"
        or normalize_label(task.get("speaker_role")) == "advisor"
    )


def colloquial_surface(text: str) -> bool:
    return bool(PROFANITY.search(text) or LAUGHTER.search(text) or BLUNT.search(text))


def comparison_rows(
    real: list[str],
    generated: list[str],
    definitions: Iterable[tuple[str, Callable[[list[str]], float], str]],
) -> list[dict[str, Any]]:
    return [comparison_row(label, fn(real), fn(generated), note) for label, fn, note in definitions]


def comparison_row(label: str, real: float, generated: float, note: str) -> dict[str, Any]:
    return {
        "property": label,
        "real": real,
        "generated": generated,
        "gap": generated - real,
        "absolute_gap": abs(generated - real),
        "note": note,
    }


def comment_index(threads: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        normalize_id(comment.get("comment_id")): comment
        for thread in threads
        for comment in (thread.get("comments") or [])
        if isinstance(comment, dict) and normalize_id(comment.get("comment_id"))
    }


def dominant_counts(threads: Iterable[dict[str, Any]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for thread in threads:
        for comment in thread.get("comments") or []:
            label = normalize_label(comment.get("dominant_emotion") or comment.get("dominant_label"))
            if label:
                counts[label] += 1
    return counts


def story_values(threads: Iterable[dict[str, Any]]) -> list[float]:
    return [
        value
        for thread in threads
        for comment in (thread.get("comments") or [])
        if (value := optional_float(comment.get("story_probability"))) is not None
    ]


def repeated_ngram_share(texts: list[str], width: int) -> float:
    """Share of cross-comment n-gram document occurrences that repeat."""

    counts: Counter[tuple[str, ...]] = Counter()
    for text in texts:
        tokens = TOKEN.findall(text.lower())
        counts.update(
            {
                tuple(tokens[index : index + width])
                for index in range(max(0, len(tokens) - width + 1))
            }
        )
    total = sum(counts.values())
    return sum(count for count in counts.values() if count >= 2) / total if total else 0.0


def top_repeated_ngrams(texts: list[str], limit: int = 30) -> list[dict[str, Any]]:
    occurrences: Counter[tuple[str, ...]] = Counter()
    documents: Counter[tuple[str, ...]] = Counter()
    for text in texts:
        tokens = TOKEN.findall(text.lower())
        rows = [
            tuple(tokens[index : index + width])
            for width in (4, 5, 6)
            for index in range(max(0, len(tokens) - width + 1))
        ]
        occurrences.update(rows)
        documents.update(set(rows))
    repeated = [ngram for ngram, count in documents.items() if count >= 2]
    repeated.sort(
        key=lambda ngram: (documents[ngram], len(ngram), occurrences[ngram], ngram),
        reverse=True,
    )
    return [
        {
            "phrase": " ".join(ngram),
            "width": len(ngram),
            "comment_count": documents[ngram],
            "occurrences": occurrences[ngram],
            "comment_share": documents[ngram] / len(texts) if texts else 0.0,
        }
        for ngram in repeated[:limit]
    ]


def distinct_openers(texts: list[str]) -> float:
    return len({tuple(TOKEN.findall(text.lower())[:3]) for text in texts}) / len(texts) if texts else 0.0


def designators(texts: list[str]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for text in texts:
        counts.update(match.group().lower() for match in DESIGNATOR.finditer(text))
    return counts


def top_designator_share(texts: list[str]) -> float:
    counts = designators(texts)
    total = sum(counts.values())
    return max(counts.values()) / total if total else 0.0


def share(texts: list[str], predicate: Callable[[str], bool]) -> float:
    return sum(1 for text in texts if predicate(text)) / len(texts) if texts else 0.0


def word_cv(texts: list[str]) -> float:
    values = [word_count(text) for text in texts]
    mean = statistics.mean(values) if values else 0.0
    return statistics.pstdev(values) / mean if mean else 0.0


def mean_word_count(texts: list[str]) -> float:
    return statistics.mean(word_count(text) for text in texts) if texts else 0.0


def median_word_count(texts: list[str]) -> float:
    return statistics.median(word_count(text) for text in texts) if texts else 0.0


def adjacent_overlap(texts: list[str]) -> float:
    values = []
    for left, right in zip(texts, texts[1:]):
        left_words, right_words = content_words(left), content_words(right)
        if left_words and right_words:
            values.append(len(left_words & right_words) / min(len(left_words), len(right_words)))
    return statistics.mean(values) if values else 0.0


def vocab_breadth(texts: list[str]) -> float:
    vocabulary = set().union(*(content_words(text) for text in texts)) if texts else set()
    return len(vocabulary) / len(texts) if texts else 0.0


def content_words(text: str) -> set[str]:
    return {word for word in TOKEN.findall(text.lower()) if len(word) > 3}


def entropy(counts: Counter[str]) -> float:
    total = sum(counts.values())
    return -sum((count / total) * math.log(count / total) for count in counts.values() if count) if total else 0.0


def label_share(counts: Counter[str], label: str) -> float:
    total = sum(counts.values())
    return counts[label] / total if total else 0.0


def above_share(values: list[float], threshold: float) -> float:
    return sum(value > threshold for value in values) / len(values) if values else 0.0


def has_any(text: str, terms: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(term in lowered for term in terms)


def has_quote_marker(text: str) -> bool:
    return text.lstrip().startswith(">") or "&gt;" in text


def normalize_id(value: object) -> str:
    return re.sub(r"^t1_", "", str(value or "").strip())


def normalize_label(value: object) -> str:
    return str(value or "").strip().lower().replace(" ", "_").replace("-", "_")


def word_count(text: str) -> int:
    return len(text.split())


def truncate(text: str, limit: int = 260) -> str:
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"
