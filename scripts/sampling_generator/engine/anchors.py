from __future__ import annotations

from sampling_generator.engine.util import normalize_apostrophe_text
import re

GENERIC_ANCHOR_WORDS = {
    "advice",
    "answer",
    "better",
    "card",
    "cards",
    "cashback",
    "clean",
    "comment",
    "credit",
    "detail",
    "good",
    "helpful",
    "local",
    "point",
    "question",
    "reply",
    "setup",
    "thing",
    "worth",
}

def extract_url_anchor_labels(text: str) -> list[str]:
    labels: list[str] = []
    for match in re.finditer(r"https?://([^/\s)]+)(/[^\s)]*)?", text, flags=re.IGNORECASE):
        host = match.group(1).lower().removeprefix("www.")
        path = match.group(2) or ""
        if "americanexpress" in host:
            labels.append("American Express reference link")
        elif "barclaycard" in host or "barclays" in host:
            labels.append("Barclays card/prequal reference")
        elif "united.com" in host and "jetblue" in path.lower():
            labels.append("United/JetBlue partnership reference")
        elif "reddit.com" in host:
            labels.append("subreddit/sidebar reference")
        else:
            labels.append(f"{host} reference link")
    return labels

def extract_money_number_anchors(text: str) -> list[str]:
    anchors: list[str] = []
    patterns = (
        r"[$€£]\s?\d[\d,]*(?:\.\d+)?\s?(?:k|K)?(?:\s?(?:AF|annual fee|fee|credit|deposit|limit|spend|SUB|bonus))?",
        r"\b\d+(?:\.\d+)?\s?(?:x|%|pts?|points|miles|months?|years?|days?|weeks?|nights?|AF|annual fee)\b",
        r"\b\d+/\d+\b",
        r"\b\d+\s?k\s?(?:pts?|points|miles|spend)?\b",
    )
    for pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            anchor = clean_anchor_label(match.group(0))
            if anchor:
                anchors.append(anchor)
    return anchors

def extract_short_dp_anchors(text: str) -> list[str]:
    anchors: list[str] = []
    lowered = normalize_apostrophe_text(text)
    if re.search(r"\b(approved|approval|denied|declined)\b", lowered):
        anchors.append("approval/denial DP")
    if re.search(r"\b(called|phone rep|rep|customer service|huca)\b", lowered):
        anchors.append("phone rep / HUCA DP")
    if re.search(r"\b(grocer(?:y|ies)|gas|dining|travel|streaming|costco|target|walmart|warehouse)\b", lowered):
        anchors.append("category spend detail")
    if re.search(r"\b(first year|year one|after that|keeper card|downgrade|product change|pc)\b", lowered):
        anchors.append("first-year / product-change detail")
    return anchors

def clean_anchor_label(value: str) -> str:
    anchor = re.sub(r"\s+", " ", str(value or "")).strip(" .,:;[]")
    if not anchor:
        return ""
    if len(anchor) > 60:
        return ""
    lowered = anchor.lower()
    if lowered in GENERIC_ANCHOR_WORDS:
        return ""
    return anchor

def dedup_anchors(anchors: list[str], *, max_anchors: int) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for raw in anchors:
        anchor = clean_anchor_label(raw)
        if not anchor:
            continue
        key = concrete_anchor_key(anchor)
        if key in seen:
            continue
        seen.add(key)
        result.append(anchor)
        if len(result) >= max_anchors:
            break
    return result

def concrete_anchor_base_label(anchor: str) -> str:
    base = re.sub(r"\s+\((?:matched real|planner|local|seed)\)?$", "", str(anchor or "").strip(), flags=re.IGNORECASE)
    return clean_anchor_label(base)

def concrete_anchor_key(anchor: str) -> str:
    base = concrete_anchor_base_label(anchor)
    return re.sub(r"[^a-z0-9$%/]+", " ", base.lower()).strip()

def concrete_anchor_tokens(anchor: str) -> list[str]:
    key = concrete_anchor_key(anchor)
    tokens = [
        token
        for token in re.findall(r"[a-z0-9$%/]+", key)
        if len(token) >= 2 and token not in ANCHOR_OVERLAP_STOPWORDS and token not in GENERIC_ANCHOR_WORDS
    ]
    return tokens

ANCHOR_OVERLAP_STOPWORDS = {
    "about",
    "actually",
    "advice",
    "answer",
    "better",
    "card",
    "cards",
    "clean",
    "cleaner",
    "comment",
    "credit",
    "detail",
    "easier",
    "feels",
    "helpful",
    "local",
    "option",
    "point",
    "premium",
    "really",
    "reply",
    "setup",
    "simple",
    "stuff",
    "thing",
    "things",
    "whole",
    "would",
}
