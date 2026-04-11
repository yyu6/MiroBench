import json, os, tempfile
from unittest.mock import MagicMock
from product_reddit_sim.loader import NormalizedProduct
from product_reddit_sim.analyzer import PersonaArchetype, ProductAnalysis
from product_reddit_sim.config_builder import (
    build_config,
    _seed_post_count,
    _build_agent_configs,
)

ANALYSIS = ProductAnalysis(
    product_category="wireless headphones",
    key_themes=["ANC", "battery", "value"],
    persona_archetypes=[PersonaArchetype("Audiophile", "desc", 1.0)],
    discussion_seed_topics=["Best ANC under $200?"],
)

PROFILES = [
    {"user_id": 1, "username": "user1", "name": "User One", "bio": "bio",
     "persona": "persona text", "karma": 5000, "archetype": "Audiophile"},
    {"user_id": 2, "username": "user2", "name": "User Two", "bio": "bio",
     "persona": "persona text", "karma": 500, "archetype": "Audiophile"},
]

PRODUCTS = [NormalizedProduct(title=f"Product {i}", price=float(i*10), brand="B")
            for i in range(1, 21)]

SEED_POSTS_RESPONSE = {
    "seed_posts": [
        {"poster_agent_id": 1, "content": "What do you think about Product 5?", "post_type": "product_specific"},
        {"poster_agent_id": 2, "content": "Best budget headphones?", "post_type": "topic_based"},
    ]
}


def _mock_client():
    client = MagicMock()
    completion = MagicMock()
    completion.choices[0].message.content = json.dumps(SEED_POSTS_RESPONSE)
    client.chat.completions.create.return_value = completion
    return client


def test_writes_simulation_config_json():
    with tempfile.TemporaryDirectory() as tmp:
        build_config(ANALYSIS, PROFILES, PRODUCTS, tmp,
                     cli_args={"hours": 48, "rounds": 30, "model": "gpt-4o-mini", "base_url": ""},
                     client=_mock_client(), model="gpt-4o-mini", seed=42)
        assert os.path.exists(os.path.join(tmp, "simulation_config.json"))


def test_writes_reddit_profiles_json():
    with tempfile.TemporaryDirectory() as tmp:
        build_config(ANALYSIS, PROFILES, PRODUCTS, tmp,
                     cli_args={"hours": 48, "rounds": 30, "model": "gpt-4o-mini", "base_url": ""},
                     client=_mock_client(), model="gpt-4o-mini", seed=42)
        assert os.path.exists(os.path.join(tmp, "reddit_profiles.json"))


def test_config_contains_seed_posts():
    with tempfile.TemporaryDirectory() as tmp:
        build_config(ANALYSIS, PROFILES, PRODUCTS, tmp,
                     cli_args={"hours": 48, "rounds": 30, "model": "gpt-4o-mini", "base_url": ""},
                     client=_mock_client(), model="gpt-4o-mini", seed=42)
        with open(os.path.join(tmp, "simulation_config.json")) as f:
            cfg = json.load(f)
        assert len(cfg["event_config"]["initial_posts"]) >= 1


def test_agent_configs_count_matches_profiles():
    with tempfile.TemporaryDirectory() as tmp:
        build_config(ANALYSIS, PROFILES, PRODUCTS, tmp,
                     cli_args={"hours": 48, "rounds": 30, "model": "gpt-4o-mini", "base_url": ""},
                     client=_mock_client(), model="gpt-4o-mini", seed=42)
        with open(os.path.join(tmp, "simulation_config.json")) as f:
            cfg = json.load(f)
        assert len(cfg["agent_configs"]) == len(PROFILES)


def test_seed_post_count_scaling():
    assert _seed_post_count(20) == 3   # max(3, round(2.4)=2) = 3
    assert _seed_post_count(50) == 6   # max(3, round(50*0.12))=6
    assert _seed_post_count(200) == 10  # capped at 10


def test_agent_activity_scales_with_karma():
    configs = _build_agent_configs(PROFILES, rng=__import__("random").Random(42))
    high_karma_activity = next(c["activity_level"] for c in configs if c["agent_id"] == 1)
    low_karma_activity = next(c["activity_level"] for c in configs if c["agent_id"] == 2)
    assert high_karma_activity >= low_karma_activity


def test_archetype_key_stripped_from_profiles():
    with tempfile.TemporaryDirectory() as tmp:
        build_config(ANALYSIS, PROFILES, PRODUCTS, tmp,
                     cli_args={"hours": 48, "rounds": 30, "model": "gpt-4o-mini", "base_url": ""},
                     client=_mock_client(), model="gpt-4o-mini", seed=42)
        with open(os.path.join(tmp, "reddit_profiles.json")) as f:
            saved_profiles = json.load(f)
        for p in saved_profiles:
            assert "archetype" not in p, f"archetype key should be stripped but found in {p}"


def test_returns_seed_prompt_and_raw():
    with tempfile.TemporaryDirectory() as tmp:
        seed_prompt, seed_raw = build_config(
            ANALYSIS, PROFILES, PRODUCTS, tmp,
            cli_args={"hours": 48, "rounds": 30, "model": "gpt-4o-mini", "base_url": ""},
            client=_mock_client(), model="gpt-4o-mini", seed=42)
    assert len(seed_prompt) > 0
    assert len(seed_raw) > 0
