from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import json
from pathlib import Path
import re
import time

import bs4
import requests

from .io_utils import read_csv_rows, write_csv_rows
from .platform_refresh import (
    REVIEW_MARKER as PLATFORM_REVIEW_MARKER,
    apply_platform_feature_refresh,
    choose_display_name,
    infer_issuer as infer_platform_issuer,
    slugify as platform_slugify,
)
from .text_utils import compact_text, normalize_text, tokenize


REVIEW_MARKER = "NEEDS_HUMAN_REVIEW"
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)

NERDWALLET_PLATFORM_FIELD = "nerdwallet_platform_description"
BANKRATE_PLATFORM_FIELD = "bankrate_platform_description"
CREDITKARMA_PLATFORM_FIELD = "creditkarma_platform_description"

DESCRIPTION_FIELDS = [
    NERDWALLET_PLATFORM_FIELD,
    BANKRATE_PLATFORM_FIELD,
    CREDITKARMA_PLATFORM_FIELD,
]

SOURCE_URL_FIELDS = {
    "nerdwallet": "nerdwallet_source_url",
    "bankrate": "bankrate_source_url",
    "creditkarma": "creditkarma_source_url",
}

NERDWALLET_ROOT_URL = "https://www.nerdwallet.com/credit-cards/best"
NERDWALLET_CATEGORY_URLS = (
    "https://www.nerdwallet.com/credit-cards/best/cash-back",
    "https://www.nerdwallet.com/credit-cards/best/balance-transfer",
    "https://www.nerdwallet.com/credit-cards/best/travel",
    "https://www.nerdwallet.com/credit-cards/best/bonus-offers",
    "https://www.nerdwallet.com/credit-cards/best/college-student",
    "https://www.nerdwallet.com/credit-cards/best/secured",
    "https://www.nerdwallet.com/credit-cards/best/airline",
    "https://www.nerdwallet.com/credit-cards/best/hotel",
    "https://www.nerdwallet.com/credit-cards/best/no-foreign-transaction-fee",
)

BANKRATE_BEST_URL = "https://www.bankrate.com/credit-cards/best-credit-cards/"
CREDITKARMA_SEARCH_URL = "https://www.creditkarma.com/credit-cards/search-cc"
CREDITKARMA_BOT_UA = "Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)"

GENERIC_CARD_TOKENS = {
    "american",
    "bank",
    "card",
    "cards",
    "credit",
    "express",
    "from",
    "mastercard",
    "the",
    "visa",
    "world",
    "signature",
    "elite",
}

ISSUER_ALIASES = {
    "american express": ("american express", "amex"),
    "bank of america": ("bank of america", "bofa"),
    "capital one": ("capital one",),
    "chase": ("chase",),
    "citi": ("citi", "citibank"),
    "discover": ("discover",),
    "wells fargo": ("wells fargo",),
    "barclays": ("barclays",),
    "fnbo": ("fnbo", "first national bank of omaha"),
    "navy federal": ("navy federal", "navy federal credit union"),
    "td bank": ("td bank",),
    "truist": ("truist",),
    "u.s. bank": ("u.s. bank", "us bank"),
}

NERDWALLET_DRIVER_LABELS = [
    "Annual fee",
    "Rewards rate",
    "Intro offer",
    "Recommended credit score",
    "Purchase intro APR",
    "Balance transfer intro APR",
    "Regular APR",
]

BANKRATE_METRIC_LABELS = [
    "Purchase intro APR",
    "Balance transfer intro APR",
    "Intro offer",
    "Rewards rate",
    "Annual fee",
    "Regular APR",
]

NOISE_PATTERNS = (
    r"\bAdd to compare\b",
    r"\bApply now Lock\b",
    r"\bInfo Hover to learn more\b",
    r"\bBankrate review\b",
    r"\bContinue\b",
    r"\bGet your free credit score\b",
    r"\bHover to learn more\b",
    r"\bRead full review\b",
    r"\bSee details, rates, & fees\b",
    r"\bShow more\b",
    r"\bView Rates & Fees\b",
)


@dataclass(frozen=True)
class PlatformEntry:
    name: str
    source_url: str
    description: str
    priority: int


@dataclass
class PlatformCard:
    canonical_name: str
    entries: dict[str, PlatformEntry]


def _build_session(user_agent: str = USER_AGENT) -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": user_agent,
            "Accept-Language": "en-US,en;q=0.9",
        }
    )
    return session


def _fetch_html(session: requests.Session, url: str) -> str:
    delay_seconds = 2.0
    for attempt in range(4):
        response = session.get(url, timeout=30)
        if response.status_code != 429:
            response.raise_for_status()
            response.encoding = response.encoding or response.apparent_encoding or "utf-8"
            return response.text
        if attempt == 3:
            response.raise_for_status()
        time.sleep(delay_seconds)
        delay_seconds *= 2
    raise RuntimeError(f"Failed to fetch {url}.")


def _repair_mojibake(value: str) -> str:
    if "â" not in value:
        return value
    try:
        return value.encode("latin1").decode("utf-8")
    except UnicodeError:
        return value


def _clean_text(value: str) -> str:
    cleaned = _repair_mojibake(value or "").replace("\xa0", " ")
    cleaned = compact_text(cleaned)
    cleaned = re.sub(r"\s+'s\b", "'s", cleaned)
    cleaned = re.sub(r"\s+([,.;:!?])", r"\1", cleaned)
    return cleaned.strip()


def _trim_text(value: str, max_chars: int = 3200) -> str:
    cleaned = _clean_text(value)
    if len(cleaned) <= max_chars:
        return cleaned
    trimmed = cleaned[: max_chars - 1].rsplit(" ", 1)[0].strip()
    return trimmed or cleaned[: max_chars - 1]


def _dedupe_parts(parts: list[str]) -> list[str]:
    kept: list[str] = []
    for part in parts:
        cleaned = _clean_text(part)
        if not cleaned:
            continue
        normalized = normalize_text(cleaned)
        if not normalized:
            continue
        if any(normalized == normalize_text(existing) for existing in kept):
            continue
        kept.append(cleaned)
    return kept


def _join_parts(parts: list[str]) -> str:
    cleaned_parts = _dedupe_parts(parts)
    if not cleaned_parts:
        return REVIEW_MARKER
    return _trim_text(" ".join(cleaned_parts))


def _strip_noise(value: str) -> str:
    cleaned = _clean_text(value)
    for pattern in NOISE_PATTERNS:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    cleaned = re.sub(r"\s+\|", " |", cleaned)
    return cleaned.strip(" -|;")


def _class_attr_contains(value: object, fragment: str) -> bool:
    if isinstance(value, str):
        classes = value.split()
    elif isinstance(value, list):
        classes = value
    else:
        classes = []
    return any(fragment in class_name for class_name in classes)


def _has_class_fragment(node: bs4.Tag, fragment: str) -> bool:
    return _class_attr_contains(node.get("class"), fragment)


def _canonical_card_name(value: str) -> str:
    cleaned = value.replace("®", " ").replace("™", " ")
    cleaned = re.sub(r"(?i)\bcredit card\b", " ", cleaned)
    cleaned = re.sub(r"(?i)\bcard from\b", " from ", cleaned)
    cleaned = re.sub(r"(?i)\bthe\b", " ", cleaned)
    cleaned = re.sub(r"(?i)tm\b", " ", cleaned)
    return _clean_text(cleaned)


def _significant_tokens(value: str) -> set[str]:
    normalized = _canonical_card_name(value)
    tokens = {token for token in tokenize(normalized) if token not in GENERIC_CARD_TOKENS}
    return tokens or tokenize(normalized)


def _format_spans(payload: object) -> str:
    if not isinstance(payload, dict):
        return ""
    spans = payload.get("spans")
    if not isinstance(spans, list):
        return ""
    return _clean_text("".join(str(span.get("text", "")) for span in spans if isinstance(span, dict)))


def _flatten_fb_content(content: object) -> list[str]:
    if isinstance(content, dict):
        items = content.get("items")
        if isinstance(items, list):
            return [text for text in (_format_spans(item) for item in items) if text]
        blocks = content.get("blocks")
        if isinstance(blocks, list):
            flattened: list[str] = []
            for block in blocks:
                if not isinstance(block, dict):
                    continue
                flattened.extend(_flatten_fb_content(block.get("content")))
            return flattened
        text = _format_spans(content)
        return [text] if text else []
    if isinstance(content, list):
        flattened: list[str] = []
        for item in content:
            flattened.extend(_flatten_fb_content(item))
        return flattened
    return []


def _extract_first_regex(text: str, pattern: str) -> str:
    match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
    return _clean_text(match.group(1)) if match else ""


def _extract_labelled_values(text: str, labels: list[str]) -> dict[str, str]:
    cleaned = _clean_text(text)
    lowered = cleaned.lower()
    positions: list[tuple[int, str]] = []
    for label in labels:
        index = lowered.find(label.lower())
        if index != -1:
            positions.append((index, label))
    positions.sort()

    values: dict[str, str] = {}
    for index, (start, label) in enumerate(positions):
        value_start = start + len(label)
        value_end = positions[index + 1][0] if index + 1 < len(positions) else len(cleaned)
        candidate = _clean_text(cleaned[value_start:value_end].strip(" :"))
        if candidate:
            values[label] = candidate
    return values


def _normalize_price(value: object) -> str:
    if value in (None, ""):
        return ""
    if isinstance(value, (int, float)):
        amount = float(value)
        return f"${int(amount)}" if amount.is_integer() else f"${amount}"
    return _clean_text(str(value))


def _extract_accordion_content(container: bs4.Tag, fragment: str) -> str:
    panel = container.find(id=lambda value: isinstance(value, str) and fragment in value)
    if panel is None:
        return ""
    return _strip_noise(panel.get_text(" ", strip=True))


def _issuer_terms(issuer: str) -> tuple[str, ...]:
    normalized = normalize_text(issuer)
    return ISSUER_ALIASES.get(normalized, (issuer,))


def _match_score(row: dict[str, str], entry: PlatformEntry) -> float:
    row_name = _canonical_card_name(row.get("card_name", ""))
    entry_name = _canonical_card_name(entry.name)
    row_norm = normalize_text(row_name)
    entry_norm = normalize_text(entry_name)
    if not row_norm or not entry_norm:
        return 0.0

    row_tokens = _significant_tokens(row_name)
    entry_tokens = _significant_tokens(entry_name)
    if not row_tokens or not entry_tokens:
        return 0.0

    shared_tokens = row_tokens & entry_tokens
    score = 0.0
    if row_norm == entry_norm:
        score += 10.0
    elif row_norm in entry_norm or entry_norm in row_norm:
        score += 4.0

    score += 4.0 * (len(shared_tokens) / max(len(row_tokens), 1))
    score -= 2.0 * len(row_tokens - entry_tokens)
    score -= 1.0 * len(entry_tokens - row_tokens)

    issuer_tokens = set()
    for issuer_term in _issuer_terms(row.get("issuer", "")):
        issuer_tokens |= tokenize(issuer_term)
    entry_context_tokens = tokenize(f"{entry.name} {entry.description}")
    if issuer_tokens and issuer_tokens & entry_context_tokens:
        score += 1.0

    if len(row_tokens) <= 1:
        score -= 5.0
    return score


def _select_entry(row: dict[str, str], entries: list[PlatformEntry]) -> PlatformEntry | None:
    best_entry: PlatformEntry | None = None
    best_key: tuple[float, int, int] | None = None
    for entry in entries:
        score = _match_score(row, entry)
        key = (score, entry.priority, len(entry.description))
        if best_key is None or key > best_key:
            best_entry = entry
            best_key = key
    if best_key is None or best_key[0] < 4.5:
        return None
    return best_entry


def _dedupe_entries(entries: list[PlatformEntry]) -> list[PlatformEntry]:
    best_by_name: dict[str, PlatformEntry] = {}
    for entry in entries:
        normalized_name = normalize_text(_canonical_card_name(entry.name))
        if not normalized_name or entry.description == REVIEW_MARKER:
            continue
        current = best_by_name.get(normalized_name)
        if current is None or (entry.priority, len(entry.description)) > (current.priority, len(current.description)):
            best_by_name[normalized_name] = entry
    return list(best_by_name.values())


def _load_json_ld_blocks(soup: bs4.BeautifulSoup) -> list[dict[str, object]]:
    blocks: list[dict[str, object]] = []
    for script in soup.find_all("script", type="application/ld+json"):
        raw_text = script.string or script.get_text() or ""
        if not raw_text.strip():
            continue
        try:
            data = json.loads(raw_text)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            blocks.append(data)
        elif isinstance(data, list):
            blocks.extend(item for item in data if isinstance(item, dict))
    return blocks


def _extract_nerdwallet_schema_map(soup: bs4.BeautifulSoup) -> dict[str, dict[str, str]]:
    schema_items: dict[str, dict[str, str]] = {}
    for block in _load_json_ld_blocks(soup):
        item_lists: list[dict[str, object]] = []
        if block.get("@type") == "ItemList":
            item_lists.append(block)
        main_entity = block.get("mainEntity")
        if isinstance(main_entity, dict) and main_entity.get("@type") == "ItemList":
            item_lists.append(main_entity)

        for item_list in item_lists:
            for list_item in item_list.get("itemListElement", []):
                if not isinstance(list_item, dict):
                    continue
                item = list_item.get("item")
                if not isinstance(item, dict):
                    continue
                name = _clean_text(str(item.get("name", "")))
                if not name:
                    continue
                review_rating = ""
                review = item.get("review")
                if isinstance(review, dict):
                    review_rating_payload = review.get("reviewRating")
                    if isinstance(review_rating_payload, dict):
                        review_rating = _clean_text(str(review_rating_payload.get("ratingValue", "")))
                annual_fee = ""
                offers = item.get("offers")
                if isinstance(offers, list) and offers:
                    offer = offers[0]
                    if isinstance(offer, dict):
                        price_spec = offer.get("priceSpecification")
                        if isinstance(price_spec, dict):
                            annual_fee = _normalize_price(price_spec.get("price"))
                schema_items[normalize_text(_canonical_card_name(name))] = {
                    "rating": review_rating,
                    "annual_fee": annual_fee,
                    "category": _clean_text(str(item.get("category", ""))),
                }
    return schema_items


def _extract_nerdwallet_root_entries(session: requests.Session) -> list[PlatformEntry]:
    soup = bs4.BeautifulSoup(_fetch_html(session, NERDWALLET_ROOT_URL), "html.parser")
    entries: list[PlatformEntry] = []

    for container in soup.find_all("div"):
        if not _has_class_fragment(container, "productCard"):
            continue
        heading = container.find("h3")
        if heading is None:
            continue
        name = _clean_text(heading.get_text(" ", strip=True))
        if not name:
            continue

        full_text = _strip_noise(container.get_text(" ", strip=True))
        rating = _extract_first_regex(full_text, r"NerdWallet rating\s*([0-5]\.\d)")
        our_pick = _extract_first_regex(full_text, r"Our pick for:\s*(.+?)\s+" + re.escape(name))

        drivers: dict[str, str] = {}
        for label in NERDWALLET_DRIVER_LABELS:
            candidate_values: list[str] = []
            for label_node in container.find_all(attrs={"data-testid": "product-card-driver-label"}):
                if _clean_text(label_node.get_text(" ", strip=True)) != label:
                    continue
                current: bs4.Tag | None = label_node.parent
                while current is not None:
                    candidate_text = _clean_text(current.get_text(" ", strip=True))
                    if candidate_text.startswith(label) and len(candidate_text) > len(label):
                        candidate_values.append(_clean_text(candidate_text[len(label) :].strip()))
                    if current == container:
                        break
                    current = current.parent if isinstance(current.parent, bs4.Tag) else None
            candidate_values = [value for value in candidate_values if value]
            if candidate_values:
                drivers[label] = min(candidate_values, key=len)

        rewards_breakdown = _extract_accordion_content(container, "card-panel-content-rewards-breakdown-")
        card_details = _extract_accordion_content(container, "card-panel-content-card-details-")
        nerdwallet_take = _extract_accordion_content(container, "card-panel-content-nw-take-")

        description_parts: list[str] = []
        if our_pick:
            description_parts.append(f"Our pick for: {our_pick}.")
        if rating:
            description_parts.append(f"NerdWallet rating: {rating}.")
        for label in NERDWALLET_DRIVER_LABELS:
            value = drivers.get(label, "")
            if value:
                description_parts.append(f"{label}: {value}.")
        if rewards_breakdown:
            description_parts.append(f"Rewards breakdown: {rewards_breakdown}.")
        if card_details:
            description_parts.append(f"Card details: {card_details}.")
        if nerdwallet_take:
            description_parts.append(f"NerdWallet's take: {nerdwallet_take}.")

        entries.append(
            PlatformEntry(
                name=name,
                source_url=NERDWALLET_ROOT_URL,
                description=_join_parts(description_parts),
                priority=3,
            )
        )

    return _dedupe_entries(entries)


def _extract_nerdwallet_category_entries(session: requests.Session, url: str) -> list[PlatformEntry]:
    soup = bs4.BeautifulSoup(_fetch_html(session, url), "html.parser")
    section = soup.find("section", class_=lambda value: _class_attr_contains(value, "table-of-contents-section"))
    if section is None:
        return []

    schema_map = _extract_nerdwallet_schema_map(soup)
    children = [child for child in section.find_all(recursive=False) if isinstance(child, bs4.Tag)]
    child_texts = [_clean_text(child.get_text(" ", strip=True)) for child in children]

    entries: list[PlatformEntry] = []
    for index in range(len(child_texts) - 2):
        name = child_texts[index]
        pick = child_texts[index + 1]
        summary = child_texts[index + 2]
        if not name or not pick.lower().startswith("our pick for:"):
            continue
        if len(summary.split()) < 20:
            continue
        if name.lower().startswith("star rating categories"):
            break

        summary = re.split(r"\bRead our review\b", summary, maxsplit=1, flags=re.IGNORECASE)[0]
        summary = re.split(r"»\s*For\s", summary, maxsplit=1)[0]
        summary = _strip_noise(summary)
        schema = schema_map.get(normalize_text(_canonical_card_name(name)), {})

        description_parts = [pick]
        if schema.get("rating"):
            description_parts.append(f"NerdWallet rating: {schema['rating']}.")
        if schema.get("annual_fee"):
            description_parts.append(f"Annual fee: {schema['annual_fee']}.")
        if summary:
            description_parts.append(summary)

        entries.append(
            PlatformEntry(
                name=name,
                source_url=url,
                description=_join_parts(description_parts),
                priority=2,
            )
        )

    return _dedupe_entries(entries)


def _extract_nerdwallet_entries(workers: int) -> list[PlatformEntry]:
    entries: list[PlatformEntry] = []
    entries.extend(_extract_nerdwallet_root_entries(_build_session()))

    with ThreadPoolExecutor(max_workers=max(1, min(workers, len(NERDWALLET_CATEGORY_URLS)))) as executor:
        future_map = {
            executor.submit(_extract_nerdwallet_category_entries, _build_session(), url): url
            for url in NERDWALLET_CATEGORY_URLS
        }
        for future in as_completed(future_map):
            entries.extend(future.result())
    return _dedupe_entries(entries)


def _extract_bankrate_entries(session: requests.Session) -> list[PlatformEntry]:
    soup = bs4.BeautifulSoup(_fetch_html(session, BANKRATE_BEST_URL), "html.parser")
    entries: list[PlatformEntry] = []

    for container in soup.find_all("div"):
        if not (_has_class_fragment(container, "rounded-lg") and _has_class_fragment(container, "mb-8")):
            continue
        heading = container.find("h2")
        if heading is None:
            continue
        name = _clean_text(heading.get_text(" ", strip=True))
        if not name:
            continue

        child_texts = [_strip_noise(child.get_text(" ", strip=True)) for child in container.find_all(recursive=False)]
        child_texts = [text for text in child_texts if text]

        header_text = next((text for text in child_texts if "Bankrate score" in text), "")
        metrics_text = next((text for text in child_texts if "Annual fee" in text and "Regular APR" in text), "")
        why_text = next((text for text in child_texts if text.startswith("Why you'll like this:")), "")
        reward_details = next((text for text in child_texts if text.startswith("Reward Details")), "")
        what_to_know = next((text for text in child_texts if text.startswith("What you should know")), "")
        card_details = next((text for text in child_texts if text.startswith("Card Details")), "")
        offer_banner = next((text for text in child_texts if text.lower().startswith("offer ends") or text.lower().startswith("limited time offer")), "")

        best_for = ""
        if name in header_text:
            prefix = _clean_text(header_text.split(name, 1)[0])
            match = re.search(r"Best for\s+(.+)$", prefix, flags=re.IGNORECASE)
            best_for = _clean_text(match.group(1)) if match else ""
        bankrate_score = _extract_first_regex(header_text, r"Bankrate score.*?([0-5]\.\d)")
        recommended_credit = _extract_first_regex(header_text, r"Recommended credit score:\s*(.+?)\s+Apply now\b")
        header_fields = _extract_labelled_values(
            header_text,
            ["Purchase intro APR", "Balance transfer intro APR"],
        )
        metric_fields = _extract_labelled_values(metrics_text, BANKRATE_METRIC_LABELS)

        description_parts: list[str] = []
        if best_for:
            description_parts.append(f"Best for: {best_for}.")
        if bankrate_score:
            description_parts.append(f"Bankrate score: {bankrate_score}.")
        if recommended_credit:
            description_parts.append(f"Recommended credit score: {recommended_credit}.")
        for label in ["Purchase intro APR", "Balance transfer intro APR", "Intro offer", "Rewards rate", "Annual fee", "Regular APR"]:
            value = header_fields.get(label, "") or metric_fields.get(label, "")
            if value:
                description_parts.append(f"{label}: {value}.")
        if offer_banner:
            description_parts.append(f"{offer_banner}.")
        if why_text:
            description_parts.append(why_text)
        if reward_details:
            description_parts.append(reward_details)
        if what_to_know:
            description_parts.append(what_to_know)
        if card_details:
            description_parts.append(card_details)

        entries.append(
            PlatformEntry(
                name=name,
                source_url=BANKRATE_BEST_URL,
                description=_join_parts(description_parts),
                priority=3,
            )
        )

    return _dedupe_entries(entries)


def _extract_creditkarma_entries(session: requests.Session) -> list[PlatformEntry]:
    soup = bs4.BeautifulSoup(_fetch_html(session, CREDITKARMA_SEARCH_URL), "html.parser")
    script = next(
        (tag for tag in soup.find_all("script") if "ccMarketplaceSearch" in (tag.string or tag.get_text() or "")),
        None,
    )
    if script is None:
        return []

    text = script.string or script.get_text() or ""
    start = text.find("push(")
    end = text.rfind(");")
    if start == -1 or end == -1:
        return []

    payload = json.loads(text[start + 5 : end])
    recommendations: list[dict[str, object]] = []
    for rehydrate_item in payload.get("rehydrate", {}).values():
        if not isinstance(rehydrate_item, dict):
            continue
        data = rehydrate_item.get("data")
        if not isinstance(data, dict):
            continue
        search_data = data.get("ccMarketplaceSearch")
        if not isinstance(search_data, dict):
            continue
        feed = search_data.get("feed")
        if not isinstance(feed, dict):
            continue
        items = feed.get("recommendations")
        if isinstance(items, list):
            recommendations.extend(item for item in items if isinstance(item, dict))

    entries: list[PlatformEntry] = []
    for recommendation in recommendations:
        name = _format_spans(recommendation.get("title"))
        if not name:
            continue

        headline = _format_spans(recommendation.get("headline"))
        reviews = recommendation.get("reviews")
        review_count = ""
        if isinstance(reviews, dict):
            review_count = _clean_text(str(reviews.get("count", "")))

        offer_description = recommendation.get("offerDescription")
        highlight_boxes: list[dict[str, object]] = []
        details_block: dict[str, object] | None = None
        if isinstance(offer_description, dict):
            highlight_container = offer_description.get("highlightBoxes")
            if isinstance(highlight_container, dict):
                boxes = highlight_container.get("boxes")
                if isinstance(boxes, list):
                    highlight_boxes = [box for box in boxes if isinstance(box, dict)]
            raw_details_block = offer_description.get("detailsBlock")
            if isinstance(raw_details_block, dict):
                details_block = raw_details_block

        box_parts: list[str] = []
        for box in highlight_boxes:
            title = _format_spans(box.get("title"))
            value = _format_spans(box.get("value"))
            subtext = _format_spans(box.get("subtext"))
            description = _format_spans(box.get("description"))
            lead = _clean_text(f"{title}: {compact_text(value, subtext)}") if title or value or subtext else ""
            if description:
                lead = _join_parts([lead, description]) if lead else description
            if lead and lead != REVIEW_MARKER:
                box_parts.append(lead)

        detail_items: list[str] = []
        if details_block is not None:
            detail_items = _flatten_fb_content(details_block.get("content"))
        card_details = _clean_text(" ".join(item for item in detail_items if item))

        description_parts: list[str] = []
        if headline:
            description_parts.append(headline)
        if review_count:
            description_parts.append(f"{review_count} reviews.")
        for part in box_parts:
            description_parts.append(f"{part}." if not part.endswith(".") else part)
        if card_details:
            description_parts.append(f"Card details: {card_details}.")

        entries.append(
            PlatformEntry(
                name=name,
                source_url=CREDITKARMA_SEARCH_URL,
                description=_join_parts(description_parts),
                priority=3,
            )
        )

    return _dedupe_entries(entries)


def _build_platform_registries(workers: int) -> dict[str, list[PlatformEntry]]:
    with ThreadPoolExecutor(max_workers=max(1, min(workers, 3))) as executor:
        futures = {
            executor.submit(_extract_nerdwallet_entries, workers): "nerdwallet",
            executor.submit(_extract_bankrate_entries, _build_session()): "bankrate",
            executor.submit(_extract_creditkarma_entries, _build_session(CREDITKARMA_BOT_UA)): "creditkarma",
        }
        registries: dict[str, list[PlatformEntry]] = {}
        for future in as_completed(futures):
            registries[futures[future]] = future.result()
    return registries


def _load_reference_rows(input_csv: Path) -> list[dict[str, str]]:
    candidates = [
        input_csv.with_name("product_descriptions_reviewed.csv"),
        input_csv.with_name("product_descriptions_raw_backup.csv"),
        input_csv,
    ]
    best_rows: list[dict[str, str]] = []
    for candidate in candidates:
        if not candidate.exists():
            continue
        rows = read_csv_rows(candidate)
        if len(rows) > len(best_rows):
            best_rows = rows
    return best_rows


def _looks_like_reference_card_row(row: dict[str, str]) -> bool:
    name = normalize_text(row.get("card_name", ""))
    if len(name.split()) < 2:
        return False
    generic_markers = (
        "overview",
        "site map",
        "compare",
        "benefits",
        "features",
        "homepage",
        "services",
        "solutions",
        "let s connect",
        "merchant services",
        "build credit",
        "find the card that fits",
        "business payment solutions",
        "explore credit card benefits",
    )
    return not any(marker in name for marker in generic_markers)


def _aggregate_platform_cards(registries: dict[str, list[PlatformEntry]]) -> list[PlatformCard]:
    cards: dict[str, PlatformCard] = {}
    for source_name, entries in registries.items():
        for entry in entries:
            canonical_name = normalize_text(_canonical_card_name(entry.name))
            if not canonical_name:
                continue
            card = cards.setdefault(canonical_name, PlatformCard(canonical_name=canonical_name, entries={}))
            current = card.entries.get(source_name)
            if current is None or (entry.priority, len(entry.description)) > (current.priority, len(current.description)):
                card.entries[source_name] = entry
    return sorted(cards.values(), key=lambda card: card.canonical_name)


def _platform_card_display_name(card: PlatformCard, fallback_row: dict[str, str] | None = None) -> str:
    platform_name = choose_display_name(*(entry.name for entry in card.entries.values()))
    if platform_name:
        return platform_name
    if fallback_row is None:
        return ""
    return choose_display_name(
        fallback_row.get("official_marketing_headline", ""),
        fallback_row.get("h1", ""),
        fallback_row.get("card_name", ""),
    )


def _combined_platform_description(card: PlatformCard) -> str:
    parts = [entry.description for entry in card.entries.values() if entry.description and entry.description != REVIEW_MARKER]
    return _join_parts(parts) if parts else REVIEW_MARKER


def _reference_row_penalty(row: dict[str, str], card: PlatformCard) -> float:
    penalty = 0.0
    entry_text = normalize_text(
        " ".join(
            [
                _platform_card_display_name(card),
                _combined_platform_description(card),
            ]
        )
    )
    row_text = normalize_text(f"{row.get('card_name', '')} {row.get('official_product_url', '')}")
    for token in ("business", "student", "secured"):
        if (token in row_text) != (token in entry_text):
            penalty += 1.5
    inferred_issuer = infer_platform_issuer(f"{_platform_card_display_name(card)} {_combined_platform_description(card)}")
    if inferred_issuer and normalize_text(row.get("issuer", "")) != normalize_text(inferred_issuer):
        penalty += 2.5
    if "/business-" in row.get("official_product_url", "") and "business" not in entry_text:
        penalty += 1.0
    return penalty


def _reference_match_score(row: dict[str, str], card: PlatformCard) -> float:
    display_name = _platform_card_display_name(card)
    row_context = compact_text(
        row.get("card_name", ""),
        row.get("official_marketing_headline", ""),
        row.get("h1", ""),
        row.get("official_product_url", ""),
        row.get("official_marketing_description", ""),
    )
    row_norm = normalize_text(row_context)
    display_norm = normalize_text(_canonical_card_name(display_name))
    row_tokens = _significant_tokens(row_context)
    display_tokens = _significant_tokens(display_name)
    if not display_norm or not row_tokens or not display_tokens:
        return 0.0
    if len(display_tokens) <= 1 and display_norm not in row_norm:
        return 0.0

    score = 0.0
    if display_norm in row_norm:
        score += 8.0
    overlap = len(display_tokens & row_tokens) / max(len(display_tokens), 1)
    score += 5.0 * overlap
    score -= 1.5 * len(display_tokens - row_tokens)

    inferred_issuer = infer_platform_issuer(f"{display_name} {_combined_platform_description(card)}")
    if inferred_issuer and normalize_text(inferred_issuer) == normalize_text(row.get("issuer", "")):
        score += 1.5

    return score


def _select_reference_row(card: PlatformCard, reference_rows: list[dict[str, str]]) -> dict[str, str] | None:
    scored: list[tuple[float, int, dict[str, str]]] = []
    for row in reference_rows:
        if not _looks_like_reference_card_row(row):
            continue
        score = _reference_match_score(row, card) - _reference_row_penalty(row, card)
        scored.append((score, len(row.get("official_input_text", "")), row))
    if not scored:
        return None
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    best_score, _, best_row = scored[0]
    second_score = scored[1][0] if len(scored) > 1 else -999.0
    if best_score < 5.0:
        return None
    if (best_score - second_score) < 0.35 and best_score < 8.0:
        return None
    return dict(best_row)


def _build_placeholder_row(fieldnames: list[str], card: PlatformCard) -> dict[str, str]:
    row = {field: "" for field in fieldnames}
    display_name = _platform_card_display_name(card)
    description_preview = " ".join(
        entry.description[:220]
        for entry in card.entries.values()
        if entry.description and entry.description != REVIEW_MARKER
    )
    issuer = infer_platform_issuer(f"{display_name} {description_preview}")
    row["card_name"] = display_name
    row["issuer"] = issuer
    row["card_id"] = platform_slugify(f"{issuer} {display_name}" if issuer else display_name)
    row["official_source_url"] = ""
    row["official_product_url"] = ""
    row["product_url_status"] = "needs_human_review"
    row["collection_method"] = "platform_first:no_official_match"
    row["page_title"] = ""
    row["meta_description"] = ""
    row["h1"] = ""
    row["official_marketing_headline"] = PLATFORM_REVIEW_MARKER
    row["official_marketing_description"] = PLATFORM_REVIEW_MARKER
    row["hero_bullets"] = ""
    row["official_input_text"] = PLATFORM_REVIEW_MARKER
    return row


def _attach_platform_fields(row: dict[str, str], card: PlatformCard) -> dict[str, str]:
    updated = dict(row)
    for source_name, description_field in (
        ("nerdwallet", NERDWALLET_PLATFORM_FIELD),
        ("bankrate", BANKRATE_PLATFORM_FIELD),
        ("creditkarma", CREDITKARMA_PLATFORM_FIELD),
    ):
        entry = card.entries.get(source_name)
        if entry is None:
            updated[description_field] = REVIEW_MARKER
            updated[SOURCE_URL_FIELDS[source_name]] = ""
            continue
        updated[description_field] = entry.description
        updated[SOURCE_URL_FIELDS[source_name]] = entry.source_url
    return updated


def enrich_editorial_descriptions(
    input_csv: Path,
    output_csv: Path,
    checkpoint_every: int = 10,
    row_limit: int | None = None,
    workers: int = 4,
) -> None:
    del checkpoint_every

    reference_rows = _load_reference_rows(input_csv)
    if not reference_rows:
        write_csv_rows(output_csv, [], [])
        return

    registries = _build_platform_registries(max(1, workers))
    platform_cards = _aggregate_platform_cards(registries)
    if row_limit is not None:
        platform_cards = platform_cards[:row_limit]
    if not platform_cards:
        write_csv_rows(output_csv, [], [])
        return

    fieldnames = list(reference_rows[0].keys())
    for field in DESCRIPTION_FIELDS:
        if field not in fieldnames:
            fieldnames.append(field)

    rebuilt_rows: list[dict[str, str]] = []
    for card in platform_cards:
        reference_row = _select_reference_row(card, reference_rows)
        if reference_row is None:
            updated_row = _build_placeholder_row(fieldnames, card)
        else:
            updated_row = dict(reference_row)
            updated_row["collection_method"] = "platform_first:matched_official"
            updated_row["product_url_status"] = updated_row.get("product_url_status", "collected") or "collected"

        updated_row = _attach_platform_fields(updated_row, card)
        display_name = _platform_card_display_name(card, updated_row)
        if display_name:
            updated_row["card_name"] = display_name
        if not updated_row.get("issuer"):
            updated_row["issuer"] = infer_platform_issuer(
                f"{updated_row.get('card_name', '')} "
                + " ".join(
                    entry.description[:220]
                    for entry in card.entries.values()
                    if entry.description and entry.description != REVIEW_MARKER
                )
            )
        updated_row["card_id"] = platform_slugify(
            f"{updated_row.get('issuer', '')} {updated_row.get('card_name', '')}".strip()
        )
        updated_row = apply_platform_feature_refresh(updated_row)
        rebuilt_rows.append(updated_row)

    rebuilt_rows.sort(key=lambda row: (normalize_text(row.get("issuer", "")), normalize_text(row.get("card_name", ""))))
    write_csv_rows(output_csv, rebuilt_rows, fieldnames)
