import json
from unittest.mock import MagicMock, patch
from product_reddit_sim.loader import NormalizedProduct
from product_reddit_sim.analyzer import (
    analyze_products,
    ProductAnalysis,
    PersonaArchetype,
    _stratified_sample,
)

MOCK_RESPONSE = {
    "product_category": "wireless headphones",
    "key_themes": ["ANC quality", "battery life", "value"],
    "persona_archetypes": [
        {"name": "Audiophile", "description": "Cares about sound quality.", "weight": 0.30},
        {"name": "Commuter", "description": "Needs ANC for transit.", "weight": 0.40},
        {"name": "Budget Hunter", "description": "Seeks value.", "weight": 0.30},
    ],
    "discussion_seed_topics": ["Sony vs Bose ANC?", "Best under $100?"],
}

def _make_products(n=15):
    return [
        NormalizedProduct(title=f"Product {i}", price=float(i * 20), brand="BrandX")
        for i in range(1, n + 1)
    ]

def _mock_client(response_dict):
    client = MagicMock()
    completion = MagicMock()
    completion.choices[0].message.content = json.dumps(response_dict)
    client.chat.completions.create.return_value = completion
    return client


def test_returns_product_analysis():
    client = _mock_client(MOCK_RESPONSE)
    result = analyze_products(_make_products(), hint=None, client=client, model="gpt-4o-mini", seed=42)
    assert isinstance(result, ProductAnalysis)
    assert result.product_category == "wireless headphones"


def test_archetypes_parsed():
    client = _mock_client(MOCK_RESPONSE)
    result = analyze_products(_make_products(), hint=None, client=client, model="gpt-4o-mini", seed=42)
    assert len(result.persona_archetypes) == 3
    assert isinstance(result.persona_archetypes[0], PersonaArchetype)
    assert result.persona_archetypes[0].name == "Audiophile"


def test_hint_included_in_prompt():
    client = _mock_client(MOCK_RESPONSE)
    analyze_products(_make_products(), hint="for commuters", client=client, model="gpt-4o-mini", seed=42)
    call_args = client.chat.completions.create.call_args
    prompt = call_args[1]["messages"][0]["content"]
    assert "for commuters" in prompt


def test_prompt_saved_on_result():
    client = _mock_client(MOCK_RESPONSE)
    result = analyze_products(_make_products(), hint="test", client=client, model="gpt-4o-mini", seed=42)
    assert len(result._prompt) > 0
    assert len(result._raw_response) > 0


def test_stratified_sample_returns_at_most_n():
    products = _make_products(20)
    sample = _stratified_sample(products, n=10, rng=__import__("random").Random(42))
    assert len(sample) <= 10


def test_stratified_sample_spans_price_range():
    products = _make_products(20)  # prices 20..400
    sample = _stratified_sample(products, n=10, rng=__import__("random").Random(42))
    prices = [p.price for p in sample if p.price]
    assert min(prices) < 100
    assert max(prices) > 300


def test_weights_sum_validated():
    bad_response = dict(MOCK_RESPONSE)
    bad_response["persona_archetypes"] = [
        {"name": "A", "description": "x", "weight": 0.5},
        {"name": "B", "description": "y", "weight": 0.8},  # sum > 1
    ]
    client = _mock_client(bad_response)
    # Should not crash — normalize weights internally
    result = analyze_products(_make_products(), hint=None, client=client, model="gpt-4o-mini", seed=42)
    total = sum(a.weight for a in result.persona_archetypes)
    assert abs(total - 1.0) < 0.01
