import json
from unittest.mock import MagicMock
from product_reddit_sim.loader import NormalizedProduct
from product_reddit_sim.analyzer import PersonaArchetype, ProductAnalysis
from product_reddit_sim.persona_gen import generate_personas, _distribute_agents

ARCHETYPES = [
    PersonaArchetype("Audiophile", "Sound quality obsessed.", 0.30),
    PersonaArchetype("Commuter", "Needs ANC.", 0.40),
    PersonaArchetype("Budget Hunter", "Seeks value.", 0.30),
]

ANALYSIS = ProductAnalysis(
    product_category="wireless headphones",
    key_themes=["ANC", "battery"],
    persona_archetypes=ARCHETYPES,
    discussion_seed_topics=["Best ANC?"],
)

def _mock_client(n):
    profiles = [
        {
            "user_id": i + 1,
            "username": f"user_{i}",
            "name": f"Person {i}",
            "bio": "Short bio.",
            "persona": "Detailed persona description here.",
            "karma": 1000 * (i + 1),
            "age": 25 + i,
            "gender": "male",
            "mbti": "INTJ",
            "country": "USA",
            "profession": "Engineer",
            "interested_topics": ["headphones"],
            "archetype": ARCHETYPES[i % 3].name,
        }
        for i in range(n)
    ]
    client = MagicMock()
    completion = MagicMock()
    completion.choices[0].message.content = json.dumps({"personas": profiles})
    client.chat.completions.create.return_value = completion
    return client


def test_returns_correct_count():
    client = _mock_client(10)
    profiles, _, _ = generate_personas(ANALYSIS, n_agents=10,
                                        products=[], client=client,
                                        model="gpt-4o-mini", seed=42)
    assert len(profiles) == 10


def test_user_ids_are_sequential():
    client = _mock_client(5)
    profiles, _, _ = generate_personas(ANALYSIS, n_agents=5,
                                        products=[], client=client,
                                        model="gpt-4o-mini", seed=42)
    assert [p["user_id"] for p in profiles] == [1, 2, 3, 4, 5]


def test_required_oasis_fields_present():
    client = _mock_client(3)
    profiles, _, _ = generate_personas(ANALYSIS, n_agents=3,
                                        products=[], client=client,
                                        model="gpt-4o-mini", seed=42)
    required = {"user_id", "username", "name", "bio", "persona", "karma", "created_at"}
    for p in profiles:
        assert required.issubset(p.keys()), f"Missing fields in {p}"


def test_prompt_and_raw_returned():
    client = _mock_client(3)
    _, prompt, raw = generate_personas(ANALYSIS, n_agents=3,
                                        products=[], client=client,
                                        model="gpt-4o-mini", seed=42)
    assert len(prompt) > 0
    assert len(raw) > 0


def test_distribute_agents_sums_to_n():
    dist = _distribute_agents(ARCHETYPES, n=10)
    assert sum(dist.values()) == 10


def test_distribute_agents_all_archetypes_get_at_least_one():
    dist = _distribute_agents(ARCHETYPES, n=10)
    for arch in ARCHETYPES:
        assert dist[arch.name] >= 1


def test_product_list_included_in_prompt():
    products = [NormalizedProduct(title="Sony XM5", price=349.99, brand="Sony")]
    client = _mock_client(3)
    _, prompt, _ = generate_personas(ANALYSIS, n_agents=3,
                                      products=products, client=client,
                                      model="gpt-4o-mini", seed=42)
    assert "Sony XM5" in prompt
