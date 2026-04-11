from __future__ import annotations

import re

from .text_utils import compact_text, normalize_text, tokenize


REVIEW_MARKER = "NEEDS_HUMAN_REVIEW"

ISSUER_ALIASES = {
    "american express": ("american express", "amex"),
    "bank of america": ("bank of america", "bofa"),
    "capital one": ("capital one",),
    "chase": ("chase",),
    "citi": ("citi", "citibank"),
    "discover": ("discover",),
    "wells fargo": ("wells fargo",),
    "barclays": ("barclays",),
    "u.s. bank": ("u.s. bank", "us bank"),
}

PARTNER_HINTS = {
    "aadvantage": "aadvantage",
    "aeroplan": "aeroplan",
    "american airlines": "american_airlines",
    "best western": "best_western",
    "choice privileges": "choice",
    "delta": "delta",
    "hilton": "hilton",
    "hyatt": "hyatt",
    "ihg": "ihg",
    "jetblue": "jetblue",
    "marriott": "marriott",
    "prime": "amazon",
    "southwest": "southwest",
    "united": "united",
    "wyndham": "wyndham",
}

PROGRAM_HINTS = {
    "membership rewards": "membership_rewards",
    "ultimate rewards": "ultimate_rewards",
    "thankyou": "thankyou_points",
    "aadvantage": "aadvantage",
    "aeroplan": "aeroplan",
    "delta skymiles": "delta_skymiles",
    "hilton honors": "hilton_honors",
    "marriott bonvoy": "marriott_bonvoy",
    "rapid rewards": "southwest_rapid_rewards",
    "trueblue": "jetblue_trueblue",
    "world of hyatt": "world_of_hyatt",
    "wyndham rewards": "wyndham_rewards",
    "choice privileges": "choice_privileges",
}

NETWORK_HINTS = {
    "american express": "american_express",
    "amex": "american_express",
    "discover": "discover",
    "mastercard": "mastercard",
    "visa": "visa",
}

LABEL_ALIASES = {
    "our pick for": "our_pick",
    "best for": "best_for",
    "nerdwallet rating": "rating",
    "bankrate score": "rating",
    "annual fee": "annual_fee",
    "rewards rate": "rewards_rate",
    "reward details": "reward_details",
    "rewards breakdown": "rewards_breakdown",
    "intro offer": "intro_offer",
    "purchase intro apr": "purchase_intro_apr",
    "balance transfer intro apr": "balance_transfer_intro_apr",
    "regular apr": "regular_apr",
    "apr": "regular_apr",
    "recommended credit score": "recommended_credit",
    "recommended credit": "recommended_credit",
    "card details": "card_details",
    "why you'll like this": "why_like",
    "what you should know": "what_to_know",
    "nerdwallet's take": "nerdwallet_take",
    "offer ends": "offer_banner",
    "limited time offer": "offer_banner",
}

KNOWN_LABELS = tuple(LABEL_ALIASES.values())


def _clean_text(value: str) -> str:
    cleaned = (value or "").replace("\xa0", " ")
    cleaned = compact_text(cleaned)
    cleaned = re.sub(r"\s+([,.;:!?])", r"\1", cleaned)
    return cleaned.strip(" |;")


def slugify(value: str) -> str:
    return normalize_text(value).replace(" ", "_")


def canonical_card_name(value: str) -> str:
    cleaned = value.replace("®", " ").replace("™", " ").replace("℠", " ")
    cleaned = re.sub(r"(?i)\bcredit card\b", " ", cleaned)
    cleaned = re.sub(r"(?i)\bthe\b", " ", cleaned)
    cleaned = re.sub(r"(?i)\bfrom\s+[a-z0-9 .&'/-]+$", " ", cleaned)
    return _clean_text(cleaned)


def is_review(value: str) -> bool:
    return _clean_text(value) == REVIEW_MARKER


def choose_display_name(*names: str) -> str:
    cleaned = [
        canonical_card_name(name)
        for name in names
        if _clean_text(name) and _clean_text(name) != REVIEW_MARKER
    ]
    if not cleaned:
        return ""
    return max(cleaned, key=lambda item: (len(tokenize(item)), len(item)))


def infer_issuer(text: str) -> str:
    normalized_text = normalize_text(text)
    special_names = {
        "bankamericard": "Bank Of America",
    }
    for needle, label in special_names.items():
        if needle in normalized_text:
            return label
    lowered = normalize_text(text)
    best_match = ""
    best_len = 0
    for issuer, aliases in ISSUER_ALIASES.items():
        for alias in aliases:
            alias_norm = normalize_text(alias)
            if alias_norm and alias_norm in lowered and len(alias_norm) > best_len:
                best_match = issuer
                best_len = len(alias_norm)
    return best_match.title().replace("Us", "U.S.") if best_match else ""


def infer_cobranded_partner(text: str) -> str:
    lowered = normalize_text(text)
    for needle, label in PARTNER_HINTS.items():
        if normalize_text(needle) in lowered:
            return label
    return ""


def infer_reward_program(text: str, issuer: str = "") -> str:
    lowered = normalize_text(text)
    for needle, label in PROGRAM_HINTS.items():
        if normalize_text(needle) in lowered:
            return label

    issuer_norm = normalize_text(issuer)
    if issuer_norm == "american express":
        if "gold" in lowered or "platinum" in lowered:
            if not any(partner in lowered for partner in ("delta", "hilton", "marriott")):
                return "membership_rewards"
    if issuer_norm == "chase":
        if "sapphire" in lowered or "ink business preferred" in lowered:
            return "ultimate_rewards"
    return ""


def infer_card_network(text: str) -> str:
    lowered = normalize_text(text)
    for needle, label in NETWORK_HINTS.items():
        if normalize_text(needle) in lowered:
            return label
    return ""


def infer_credit_hint(text: str) -> str:
    lowered = normalize_text(text)
    if "good to excellent" in lowered or "670 850" in lowered or "690 850" in lowered:
        return "good_excellent"
    if "740 850" in lowered or re.search(r"\bexcellent\b", lowered):
        return "excellent"
    if "fair to good" in lowered or "580 740" in lowered:
        return "fair_good"
    if "limited credit" in lowered or "no credit history" in lowered:
        return "limited_credit"
    if "bad credit" in lowered:
        return "bad_credit"
    return ""


def _normalize_label_token(token: str) -> tuple[str, str]:
    cleaned = _clean_text(token)
    if not cleaned:
        return "", ""

    if ":" in cleaned:
        prefix, remainder = cleaned.split(":", 1)
        normalized_prefix = normalize_text(prefix)
        label = LABEL_ALIASES.get(normalized_prefix, "")
        if label:
            return label, _clean_text(remainder)

    normalized = normalize_text(cleaned)
    return LABEL_ALIASES.get(normalized, ""), ""


def parse_platform_description(description: str) -> dict[str, list[str]]:
    if not description or is_review(description):
        return {}

    tokens = [_clean_text(part) for part in description.split("|")]
    tokens = [part for part in tokens if part]
    sections: dict[str, list[str]] = {}
    index = 0
    while index < len(tokens):
        label, inline_value = _normalize_label_token(tokens[index])
        if not label:
            index += 1
            continue
        if inline_value:
            sections.setdefault(label, []).append(inline_value)
            index += 1
            continue

        index += 1
        values: list[str] = []
        while index < len(tokens):
            next_label, inline_next_value = _normalize_label_token(tokens[index])
            if next_label:
                break
            values.append(tokens[index])
            index += 1
            if inline_next_value:
                break
        joined = _clean_text("; ".join(value for value in values if _clean_text(value)))
        if joined:
            sections.setdefault(label, []).append(joined)

    label_patterns = sorted(
        ((raw_label, canonical_label) for raw_label, canonical_label in LABEL_ALIASES.items()),
        key=lambda item: len(item[0]),
        reverse=True,
    )
    all_label_lookaheads = "|".join(re.escape(raw_label) + r"\s*:" for raw_label, _ in label_patterns)
    cleaned_description = _clean_text(description)
    for raw_label, canonical_label in label_patterns:
        pattern = rf"{re.escape(raw_label)}\s*:\s*(.+?)(?=(?:{all_label_lookaheads})|$)"
        match = re.search(pattern, cleaned_description, flags=re.IGNORECASE | re.DOTALL)
        if not match:
            continue
        value = _clean_text(match.group(1))
        if value:
            sections.setdefault(canonical_label, [])
            if value not in sections[canonical_label]:
                sections[canonical_label].append(value)
    return sections


def merge_platform_sections(row: dict[str, str]) -> dict[str, list[str]]:
    merged: dict[str, list[str]] = {}
    for field in (
        "nerdwallet_platform_description",
        "bankrate_platform_description",
        "creditkarma_platform_description",
    ):
        for label, values in parse_platform_description(row.get(field, "")).items():
            merged.setdefault(label, [])
            for value in values:
                cleaned = _clean_text(value)
                if cleaned and cleaned not in merged[label]:
                    merged[label].append(cleaned)
    return merged


def choose_consistent_value(values: list[str]) -> str:
    cleaned = [_clean_text(value) for value in values if _clean_text(value)]
    if not cleaned:
        return ""
    normalized = {normalize_text(value) for value in cleaned}
    if len(normalized) == 1:
        return max(cleaned, key=len)
    longest = max(cleaned, key=len)
    if all(normalize_text(value) in normalize_text(longest) for value in cleaned):
        return longest
    return REVIEW_MARKER


def _extract_dollar_amounts(text: str) -> list[str]:
    return [amount.replace(",", "") for amount in re.findall(r"\$([0-9][0-9,]*)", text)]


def parse_annual_fee_amount(text: str) -> str:
    cleaned = _clean_text(text)
    lowered = cleaned.lower()
    if not cleaned:
        return ""
    if "no annual fee" in lowered:
        return "0"
    dollar_amounts = _extract_dollar_amounts(cleaned)
    if not dollar_amounts:
        return ""
    if "then $" in lowered or "after the first year" in lowered:
        return dollar_amounts[-1]
    return dollar_amounts[0]


def choose_annual_fee(values: list[str], fallback_text: str = "", fallback_amount: str = "") -> tuple[str, str]:
    candidates = [_clean_text(value) for value in values if _clean_text(value)]
    if not candidates:
        return fallback_text, fallback_amount
    parsed = {parse_annual_fee_amount(value) for value in candidates if parse_annual_fee_amount(value)}
    if len(parsed) == 1:
        amount = next(iter(parsed))
        richest = max(
            (value for value in candidates if parse_annual_fee_amount(value) == amount),
            key=len,
        )
        lowered = richest.lower()
        intro_fee_pattern = bool(re.search(r"intro[^.]{0,40}annual fee|annual fee[^.]{0,40}first year|then\s+\$", lowered))
        if amount == "0" and not intro_fee_pattern:
            return "No annual fee", "0"
        if not intro_fee_pattern and amount:
            return f"${amount} annual fee", amount
        return richest, amount
    if len(parsed) > 1:
        return REVIEW_MARKER, REVIEW_MARKER
    chosen = choose_consistent_value(candidates)
    amount = parse_annual_fee_amount(chosen)
    return chosen or fallback_text, amount or fallback_amount


def infer_reward_structure(text: str) -> tuple[str, str]:
    lowered = normalize_text(text)
    if "cash back" in lowered or "cashback" in lowered:
        return "cash_back", "cash_back"
    if " miles " in f" {lowered} " or lowered.endswith(" miles") or lowered.startswith("miles "):
        return "miles", "miles"
    if " points " in f" {lowered} " or lowered.endswith(" points") or lowered.startswith("points "):
        return "points", "points"
    return "", ""


def infer_target_segment(text: str, source_urls: list[str], card_type: str, cobranded_partner: str) -> str:
    lowered = normalize_text(text)
    joined_urls = " ".join(source_urls)
    if card_type == "business":
        return "business"
    if "college student" in joined_urls or "student" in lowered:
        return "student"
    if "secured" in joined_urls or "secured" in lowered or "credit building" in lowered:
        return "credit_building"
    if "hotel" in joined_urls or cobranded_partner in {"hilton", "marriott", "hyatt", "choice", "wyndham", "best_western", "ihg"}:
        return "hotel"
    if "airline" in joined_urls or cobranded_partner in {"delta", "jetblue", "southwest", "united", "american_airlines", "aadvantage", "aeroplan"}:
        return "airline"
    if "cash back" in joined_urls or "cash back" in lowered or "cashback" in lowered:
        return "cash_back"
    if "travel" in joined_urls or "travel" in lowered or "no foreign transaction fee" in joined_urls or "points" in lowered or "miles" in lowered:
        return "travel"
    if "balance transfer" in joined_urls or "balance transfer" in lowered:
        return "balance_transfer"
    return "travel" if "points" in lowered or "miles" in lowered else ""


def _combine_values(values: list[str]) -> str:
    cleaned = [_clean_text(value) for value in values if _clean_text(value)]
    return " | ".join(cleaned)


def apply_platform_feature_refresh(row: dict[str, str]) -> dict[str, str]:
    updated = dict(row)
    sections = merge_platform_sections(updated)
    identity_text = _clean_text(
        " ".join(
            [
                updated.get("card_name", ""),
                updated.get("official_marketing_headline", ""),
                updated.get("h1", ""),
            ]
        )
    )
    platform_text = _clean_text(
        " ".join(
            [
                updated.get("nerdwallet_platform_description", ""),
                updated.get("bankrate_platform_description", ""),
                updated.get("creditkarma_platform_description", ""),
            ]
        )
    )

    evidence_parts = [
        updated.get("card_name", ""),
        updated.get("issuer", ""),
        updated.get("official_marketing_headline", ""),
        updated.get("official_marketing_description", ""),
        updated.get("official_input_text", ""),
        updated.get("nerdwallet_platform_description", ""),
        updated.get("bankrate_platform_description", ""),
        updated.get("creditkarma_platform_description", ""),
    ]
    evidence_text = _clean_text(" ".join(part for part in evidence_parts if part and not is_review(part)))

    annual_fee_text, annual_fee_amount = choose_annual_fee(
        sections.get("annual_fee", []),
        fallback_text=updated.get("annual_fee_text", ""),
        fallback_amount=updated.get("annual_fee_amount", ""),
    )
    updated["annual_fee_text"] = annual_fee_text or REVIEW_MARKER
    updated["annual_fee_amount"] = annual_fee_amount or REVIEW_MARKER

    rewards_text = _combine_values(
        sections.get("rewards_rate", [])
        + sections.get("rewards_breakdown", [])
        + sections.get("reward_details", [])
        + [updated.get("card_name", ""), updated.get("official_marketing_description", "")]
    )
    reward_structure, reward_currency = infer_reward_structure(rewards_text)
    updated["reward_structure_guess"] = reward_structure
    updated["reward_currency"] = reward_currency

    reward_program = infer_reward_program(evidence_text, updated.get("issuer", ""))
    updated["reward_program_guess"] = reward_program or updated.get("reward_program_guess", "")

    rates_summary_source = sections.get("rewards_breakdown", []) or sections.get("reward_details", [])
    if rates_summary_source:
        updated["earning_rates_summary"] = max(rates_summary_source, key=len)
    elif sections.get("rewards_rate"):
        updated["earning_rates_summary"] = choose_consistent_value(sections["rewards_rate"])
    elif is_review(updated.get("earning_rates_summary", "")):
        updated["earning_rates_summary"] = ""

    intro_offer = choose_consistent_value(sections.get("intro_offer", []))
    if intro_offer:
        updated["intro_bonus_text"] = intro_offer

    purchase_intro = choose_consistent_value(sections.get("purchase_intro_apr", []))
    balance_intro = choose_consistent_value(sections.get("balance_transfer_intro_apr", []))
    apr_parts = [part for part in [purchase_intro, balance_intro] if part]
    updated["intro_apr_text"] = " | ".join(apr_parts) if apr_parts else updated.get("intro_apr_text", "")

    regular_apr = choose_consistent_value(sections.get("regular_apr", []))
    if regular_apr:
        updated["regular_apr_text"] = regular_apr

    if not updated.get("foreign_transaction_fee_text") or is_review(updated.get("foreign_transaction_fee_text", "")):
        lowered = evidence_text.lower()
        if "no foreign transaction fee" in lowered or "no foreign transaction fees" in lowered:
            updated["foreign_transaction_fee_text"] = "No foreign transaction fees"

    card_type_text = normalize_text(f"{identity_text} {updated.get('official_product_url', '')}")
    card_type = "business" if "business" in card_type_text else "personal"
    updated["card_type"] = card_type

    partner = infer_cobranded_partner(identity_text)
    updated["cobranded_partner"] = partner

    target_segment = infer_target_segment(
        platform_text or evidence_text,
        [
            updated.get("nerdwallet_source_url", ""),
            updated.get("bankrate_source_url", ""),
            updated.get("creditkarma_source_url", ""),
        ],
        card_type,
        partner,
    )
    updated["target_customer_segment"] = target_segment or updated.get("target_customer_segment", "")

    network = infer_card_network(
        identity_text
    )
    if network:
        updated["card_network"] = network

    recommended_credit = choose_consistent_value(sections.get("recommended_credit", []))
    credit_hint = infer_credit_hint(recommended_credit)
    if credit_hint:
        updated["credit_needed_hint"] = credit_hint

    if updated.get("reward_program_guess") == REVIEW_MARKER:
        updated["reward_program_guess"] = ""
    if updated.get("reward_structure_guess") == REVIEW_MARKER:
        updated["reward_structure_guess"] = ""
    if updated.get("reward_currency") == REVIEW_MARKER:
        updated["reward_currency"] = ""

    return updated
