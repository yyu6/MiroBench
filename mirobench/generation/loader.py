"""Normalize any product JSON into a standard list of NormalizedProduct."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Optional
from urllib.parse import urlparse


@dataclass
class NormalizedProduct:
    title: str
    brand: Optional[str] = None
    price: Optional[float] = None
    description: Optional[str] = None
    rating: Optional[float] = None
    review_count: Optional[int] = None
    features: list[str] = field(default_factory=list)


def load_products(path: str) -> list[NormalizedProduct]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    raw_list = _detect_schema(data)
    return [_normalize(p) for p in raw_list]


def _detect_schema(data) -> list[dict]:
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        if "products" in data and isinstance(data["products"], list):
            return data["products"]
        for v in data.values():
            if isinstance(v, list) and v and isinstance(v[0], dict):
                return v
    raise ValueError(
        "Cannot detect product schema. Expected a JSON array or an object "
        "with a 'products' key. Got: " + str(type(data))
    )


def _normalize(raw: dict) -> NormalizedProduct:
    # Price: Best Buy stores {"currentPrice": float}, others store float directly
    price_raw = raw.get("price")
    if isinstance(price_raw, dict):
        current = price_raw.get("currentPrice")
        price = current if current is not None else price_raw.get("regularPrice")
    elif isinstance(price_raw, (int, float)):
        price = float(price_raw)
    else:
        price = None

    # Description: prefer richer fields
    description = (
        raw.get("page_description")
        or raw.get("full_description")
        or raw.get("description")
        or ""
    )
    if len(description) > 500:
        description = description[:500] + "..."

    # Features: Best Buy uses feature_entries list of dicts; others may use "features" list
    features: list[str] = []
    for entry in raw.get("feature_entries", []):
        if isinstance(entry, dict) and entry.get("feature"):
            features.append(entry["feature"])
        elif isinstance(entry, str):
            features.append(entry)
    if not features:
        raw_features = raw.get("features", [])
        if isinstance(raw_features, list):
            features = [str(f) for f in raw_features if f]

    return NormalizedProduct(
        title=_resolve_title(raw),
        brand=_resolve_brand(raw),
        price=price,
        description=description,
        rating=raw.get("rating"),
        review_count=raw.get("review_count"),
        features=features[:6],
    )


def _resolve_title(raw: dict) -> str:
    return (
        raw.get("title")
        or raw.get("card_name")
        or raw.get("product_name")
        or raw.get("name")
        or "Unknown Product"
    )


def _resolve_brand(raw: dict) -> Optional[str]:
    brand = raw.get("brand")
    if brand:
        return str(brand)

    url = raw.get("final_url") or raw.get("official_product_url")
    if not url:
        return None

    host = urlparse(str(url)).netloc.lower()
    if host.startswith("www."):
        host = host[4:]

    brand_map = {
        "americanexpress.com": "American Express",
        "chase.com": "Chase",
        "creditcards.chase.com": "Chase",
        "capitalone.com": "Capital One",
        "bankofamerica.com": "Bank of America",
        "citicards.citi.com": "Citi",
        "citi.com": "Citi",
        "discover.com": "Discover",
        "wellsfargo.com": "Wells Fargo",
        "creditcards.wellsfargo.com": "Wells Fargo",
        "usbank.com": "U.S. Bank",
        "barclaycardus.com": "Barclays",
        "synchrony.com": "Synchrony",
        "chime.com": "Chime",
        "openskycc.com": "OpenSky",
        "dcu.org": "DCU",
        "firstcard.app": "Firstcard",
        "current.com": "Current",
        "self.inc": "Self",
        "choicehotels.com": "Choice Privileges",
        "bestwestern.com": "Best Western",
        "penfed.org": "PenFed",
        "wyndhamhotels.com": "Wyndham",
        "hyatt.com": "World of Hyatt",
        "aeroplan.com": "Aeroplan",
        "aa.com": "American Airlines",
        "delta.com": "Delta",
        "united.com": "United",
        "hiltonhonors.com": "Hilton Honors",
        "southwest.com": "Southwest",
        "primevisa.com": "Prime Visa",
    }
    return brand_map.get(host)
