import json, os, pytest, tempfile
from mirobench.generation.loader import load_products, NormalizedProduct

BESTBUY_FIXTURE = {
    "meta": {"scraped_count": 2},
    "products": [
        {
            "title": "Sony WH-1000XM5",
            "brand": "Sony",
            "price": {"currentPrice": 349.99},
            "rating": 4.8,
            "review_count": 1200,
            "page_description": "Best ANC headphones.",
            "feature_entries": [
                {"feature": "Industry-leading noise cancellation"},
                {"feature": "30-hour battery life"},
            ],
        },
        {
            "title": "JLab GO Air POP",
            "brand": "JLab",
            "price": {"currentPrice": 24.99},
            "rating": 4.2,
            "review_count": 5000,
            "description": "Budget earbuds.",
            "feature_entries": [],
        },
    ],
}

ROOT_LIST_FIXTURE = [
    {"title": "Product A", "price": 99.99, "brand": "BrandX"},
]

CREDIT_CARD_FIXTURE = [
    {
        "row_index": 0,
        "card_name": "American Express Platinum Card®",
        "official_product_url": "https://www.americanexpress.com/us/credit-cards/card/platinum/",
        "final_url": "https://www.americanexpress.com/us/credit-cards/card/platinum/",
        "status": "ok",
        "description": "Long-form product description here.",
        "error": None,
    }
]

def _write(tmp, data):
    p = os.path.join(tmp, "products.json")
    with open(p, "w") as f:
        json.dump(data, f)
    return p


def test_load_bestbuy_format():
    with tempfile.TemporaryDirectory() as tmp:
        path = _write(tmp, BESTBUY_FIXTURE)
        products = load_products(path)
    assert len(products) == 2
    assert isinstance(products[0], NormalizedProduct)


def test_normalizes_nested_price():
    with tempfile.TemporaryDirectory() as tmp:
        path = _write(tmp, BESTBUY_FIXTURE)
        products = load_products(path)
    assert products[0].price == 349.99


def test_normalizes_flat_price():
    with tempfile.TemporaryDirectory() as tmp:
        path = _write(tmp, ROOT_LIST_FIXTURE)
        products = load_products(path)
    assert products[0].price == 99.99


def test_extracts_features():
    with tempfile.TemporaryDirectory() as tmp:
        path = _write(tmp, BESTBUY_FIXTURE)
        products = load_products(path)
    assert "Industry-leading noise cancellation" in products[0].features


def test_root_list_schema():
    with tempfile.TemporaryDirectory() as tmp:
        path = _write(tmp, ROOT_LIST_FIXTURE)
        products = load_products(path)
    assert len(products) == 1
    assert products[0].title == "Product A"


def test_unknown_schema_raises():
    with tempfile.TemporaryDirectory() as tmp:
        path = _write(tmp, {"foo": "bar", "baz": 42})
        with pytest.raises(ValueError, match="Cannot detect product schema"):
            load_products(path)


def test_description_fallback_order():
    """page_description preferred over description."""
    with tempfile.TemporaryDirectory() as tmp:
        path = _write(tmp, BESTBUY_FIXTURE)
        products = load_products(path)
    assert products[0].description == "Best ANC headphones."


def test_description_truncated_at_500():
    long_desc = "x" * 600
    fixture = {"products": [{"title": "T", "page_description": long_desc}]}
    with tempfile.TemporaryDirectory() as tmp:
        path = _write(tmp, fixture)
        products = load_products(path)
    assert len(products[0].description) <= 503  # 500 chars + "..."


def test_credit_card_raw_map_uses_card_name_as_title():
    with tempfile.TemporaryDirectory() as tmp:
        path = _write(tmp, CREDIT_CARD_FIXTURE)
        products = load_products(path)
    assert products[0].title == "American Express Platinum Card®"
    assert products[0].brand == "American Express"
    assert products[0].description == "Long-form product description here."
