from __future__ import annotations

import re
from urllib.parse import urlparse


WORD_RE = re.compile(r"[a-z0-9]+")


def normalize_text(value: str) -> str:
    return " ".join(WORD_RE.findall(value.lower()))


def tokenize(value: str) -> set[str]:
    return set(WORD_RE.findall(value.lower()))


def score_match(card_name: str, aliases: list[str], anchor_text: str, href: str) -> float:
    candidate_text = normalize_text(f"{anchor_text} {href}")
    target_tokens = tokenize(card_name)
    for alias in aliases:
        target_tokens |= tokenize(alias)

    candidate_tokens = tokenize(anchor_text) | tokenize(href)
    if not target_tokens or not candidate_tokens:
        return 0.0

    best_phrase_score = 0.0
    for phrase in [card_name, *aliases]:
        normalized_phrase = normalize_text(phrase)
        if normalized_phrase and normalized_phrase in candidate_text:
            best_phrase_score = max(best_phrase_score, 1.0 + 0.1 * len(normalized_phrase.split()))

    overlap = len(target_tokens & candidate_tokens) / max(len(target_tokens), 1)

    generic_penalty = 0.0
    generic_markers = {
        "credit cards",
        "compare",
        "all cards",
        "travel credit cards",
        "cash back credit cards",
        "visa credit cards",
        "business credit cards",
    }
    if any(marker in candidate_text for marker in generic_markers):
        generic_penalty = 0.25

    return max(best_phrase_score, overlap) - generic_penalty


def aliases_from_pipe(value: str) -> list[str]:
    return [item.strip() for item in value.split("|") if item.strip()]


def text_mentions_alias(text: str, aliases: list[str]) -> bool:
    lowered = text.lower()
    return any(alias.lower() in lowered for alias in aliases)


def same_domain(url: str, domain: str) -> bool:
    return urlparse(url).netloc.endswith(domain)


def compact_text(*parts: str) -> str:
    combined = " ".join(part.strip() for part in parts if part and part.strip())
    return re.sub(r"\s+", " ", combined).strip()
