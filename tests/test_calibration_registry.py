"""
Tests for calibration/registry.py — KnobRegistry
"""
from __future__ import annotations

import pytest
from calibration.registry import KnobRegistry


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def registry():
    return KnobRegistry()


# ---------------------------------------------------------------------------
# TestLoadsDefaults
# ---------------------------------------------------------------------------

class TestLoadsDefaults:
    def test_has_twelve_knobs(self, registry):
        assert len(registry.knob_names()) == 12

    def test_has_persona_knobs(self, registry):
        persona_knobs = [n for n in registry.knob_names() if n.startswith("persona.")]
        assert len(persona_knobs) == 6

    def test_has_prompt_knobs(self, registry):
        prompt_knobs = [n for n in registry.knob_names() if n.startswith("prompt.")]
        assert len(prompt_knobs) == 6

    def test_specific_knob_names_present(self, registry):
        expected = {
            "persona.conflict_style_distribution",
            "persona.primary_motivation_distribution",
            "persona.knowledge_style_distribution",
            "persona.stance_distribution",
            "persona.sentiment_bias_min",
            "persona.sentiment_bias_max",
            "prompt.anti_paraphrase_instruction",
            "prompt.tone_guidance",
            "prompt.structure_preference_weight",
            "prompt.depth_soft_cap_instruction",
            "prompt.few_shot_style_anchor",
            "prompt.consensus_handling",
        }
        assert set(registry.knob_names()) == expected


# ---------------------------------------------------------------------------
# TestRequiredFields
# ---------------------------------------------------------------------------

class TestRequiredFields:
    REQUIRED_FIELDS = {"name", "layer", "domain", "type", "default", "description"}

    def test_all_knobs_have_required_fields(self, registry):
        for name in registry.knob_names():
            knob = registry.get(name)
            missing = self.REQUIRED_FIELDS - knob.keys()
            assert not missing, f"Knob '{name}' missing fields: {missing}"

    def test_distribution_knobs_have_keys_field(self, registry):
        for name in registry.knob_names():
            knob = registry.get(name)
            if knob["type"] == "distribution":
                assert "keys" in knob, f"Distribution knob '{name}' missing 'keys'"
                assert isinstance(knob["keys"], list)
                assert len(knob["keys"]) > 0

    def test_float_knobs_have_range(self, registry):
        for name in registry.knob_names():
            knob = registry.get(name)
            if knob["type"] == "float":
                assert "min" in knob, f"Float knob '{name}' missing 'min'"
                assert "max" in knob, f"Float knob '{name}' missing 'max'"

    def test_layer_values_are_valid(self, registry):
        valid_layers = {"persona", "prompt"}
        for name in registry.knob_names():
            knob = registry.get(name)
            assert knob["layer"] in valid_layers

    def test_type_values_are_valid(self, registry):
        valid_types = {"distribution", "float", "text"}
        for name in registry.knob_names():
            knob = registry.get(name)
            assert knob["type"] in valid_types


# ---------------------------------------------------------------------------
# TestGetDefaults
# ---------------------------------------------------------------------------

class TestGetDefaults:
    def test_defaults_returns_all_knobs(self, registry):
        defaults = registry.defaults()
        assert set(defaults.keys()) == set(registry.knob_names())

    def test_distribution_defaults_sum_to_one(self, registry):
        defaults = registry.defaults()
        for name in registry.knob_names():
            knob = registry.get(name)
            if knob["type"] == "distribution":
                dist = defaults[name]
                assert abs(sum(dist.values()) - 1.0) < 0.01, (
                    f"Default distribution '{name}' sums to {sum(dist.values())}"
                )

    def test_distribution_defaults_have_correct_keys(self, registry):
        defaults = registry.defaults()
        for name in registry.knob_names():
            knob = registry.get(name)
            if knob["type"] == "distribution":
                assert set(defaults[name].keys()) == set(knob["keys"])

    def test_float_defaults_in_range(self, registry):
        defaults = registry.defaults()
        for name in registry.knob_names():
            knob = registry.get(name)
            if knob["type"] == "float":
                val = defaults[name]
                assert knob["min"] <= val <= knob["max"], (
                    f"Default for '{name}' ({val}) outside [{knob['min']}, {knob['max']}]"
                )

    def test_sentiment_bias_min_default(self, registry):
        defaults = registry.defaults()
        assert defaults["persona.sentiment_bias_min"] == -0.3

    def test_sentiment_bias_max_default(self, registry):
        defaults = registry.defaults()
        assert defaults["persona.sentiment_bias_max"] == 0.4

    def test_few_shot_style_anchor_default_empty_string(self, registry):
        defaults = registry.defaults()
        assert defaults["prompt.few_shot_style_anchor"] == ""

    def test_text_defaults_are_strings(self, registry):
        defaults = registry.defaults()
        for name in registry.knob_names():
            knob = registry.get(name)
            if knob["type"] == "text":
                assert isinstance(defaults[name], str)


# ---------------------------------------------------------------------------
# TestValidateValidOverlays
# ---------------------------------------------------------------------------

class TestValidateValidOverlays:
    def test_empty_overlay_is_valid(self, registry):
        errors = registry.validate({})
        assert errors == []

    def test_full_defaults_overlay_is_valid(self, registry):
        errors = registry.validate(registry.defaults())
        assert errors == []

    def test_partial_overlay_is_valid(self, registry):
        overlay = {"persona.sentiment_bias_min": -0.5}
        errors = registry.validate(overlay)
        assert errors == []

    def test_valid_distribution_overlay(self, registry):
        knob = registry.get("persona.stance_distribution")
        keys = knob["keys"]
        n = len(keys)
        # uniform distribution
        dist = {k: 1.0 / n for k in keys}
        errors = registry.validate({"persona.stance_distribution": dist})
        assert errors == []

    def test_valid_float_at_boundary(self, registry):
        errors = registry.validate({"persona.sentiment_bias_min": -1.0})
        assert errors == []
        errors = registry.validate({"persona.sentiment_bias_min": 0.0})
        assert errors == []

    def test_valid_text_overlay(self, registry):
        errors = registry.validate({"prompt.tone_guidance": "Be concise."})
        assert errors == []

    def test_valid_empty_text_overlay(self, registry):
        errors = registry.validate({"prompt.few_shot_style_anchor": ""})
        assert errors == []


# ---------------------------------------------------------------------------
# TestValidateInvalidOverlays
# ---------------------------------------------------------------------------

class TestValidateInvalidOverlays:
    def test_unknown_knob_returns_error(self, registry):
        errors = registry.validate({"persona.nonexistent_knob": 42})
        assert len(errors) == 1
        assert "unknown" in errors[0].lower() or "nonexistent" in errors[0].lower()

    def test_distribution_not_dict_returns_error(self, registry):
        errors = registry.validate({"persona.stance_distribution": [0.25, 0.25, 0.25, 0.25]})
        assert len(errors) >= 1

    def test_distribution_wrong_sum_returns_error(self, registry):
        knob = registry.get("persona.stance_distribution")
        keys = knob["keys"]
        # values that sum to 0.5
        dist = {k: 0.5 / len(keys) for k in keys}
        errors = registry.validate({"persona.stance_distribution": dist})
        assert len(errors) >= 1
        assert any("sum" in e.lower() or "1.0" in e or "1" in e for e in errors)

    def test_distribution_negative_values_returns_error(self, registry):
        knob = registry.get("persona.stance_distribution")
        keys = knob["keys"]
        dist = {k: -0.1 for k in keys}
        errors = registry.validate({"persona.stance_distribution": dist})
        assert len(errors) >= 1

    def test_float_below_min_returns_error(self, registry):
        errors = registry.validate({"persona.sentiment_bias_min": -2.0})
        assert len(errors) >= 1
        assert any("min" in e.lower() or "range" in e.lower() or "bound" in e.lower() for e in errors)

    def test_float_above_max_returns_error(self, registry):
        errors = registry.validate({"persona.sentiment_bias_max": 2.0})
        assert len(errors) >= 1

    def test_text_not_string_returns_error(self, registry):
        errors = registry.validate({"prompt.tone_guidance": 42})
        assert len(errors) >= 1

    def test_multiple_errors_accumulate(self, registry):
        errors = registry.validate({
            "persona.sentiment_bias_min": -99.0,
            "persona.sentiment_bias_max": 99.0,
            "unknown.knob": "value",
        })
        assert len(errors) >= 3


# ---------------------------------------------------------------------------
# TestPersonaAndPromptOnly
# ---------------------------------------------------------------------------

class TestPersonaAndPromptOnly:
    def test_conflict_style_keys(self, registry):
        knob = registry.get("persona.conflict_style_distribution")
        assert set(knob["keys"]) == {"calm", "skeptical", "blunt", "sarcastic", "argumentative", "avoidant"}

    def test_primary_motivation_keys(self, registry):
        knob = registry.get("persona.primary_motivation_distribution")
        expected = {
            "helping", "venting", "showing expertise", "correcting people",
            "defending their own setup", "bargain-hunting", "complaining",
            "validation-seeking", "joking around"
        }
        assert set(knob["keys"]) == expected

    def test_knowledge_style_keys(self, registry):
        knob = registry.get("persona.knowledge_style_distribution")
        expected = {"beginner", "casual_user", "experienced_owner", "specialist", "overconfident_half_expert"}
        assert set(knob["keys"]) == expected

    def test_stance_keys(self, registry):
        knob = registry.get("persona.stance_distribution")
        assert set(knob["keys"]) == {"supportive", "neutral", "observer", "opposing"}

    def test_sentiment_bias_min_range(self, registry):
        knob = registry.get("persona.sentiment_bias_min")
        assert knob["min"] == -1.0
        assert knob["max"] == 0.0

    def test_sentiment_bias_max_range(self, registry):
        knob = registry.get("persona.sentiment_bias_max")
        assert knob["min"] == 0.0
        assert knob["max"] == 1.0

    def test_persona_knob_layers(self, registry):
        for name in registry.knob_names():
            if name.startswith("persona."):
                assert registry.get(name)["layer"] == "persona"

    def test_prompt_knob_layers(self, registry):
        for name in registry.knob_names():
            if name.startswith("prompt."):
                assert registry.get(name)["layer"] == "prompt"

    def test_prompt_knobs_are_text_type(self, registry):
        for name in registry.knob_names():
            if name.startswith("prompt."):
                assert registry.get(name)["type"] == "text"


# ---------------------------------------------------------------------------
# TestForLlmContext
# ---------------------------------------------------------------------------

class TestForLlmContext:
    def test_returns_string(self, registry):
        result = registry.for_llm_context()
        assert isinstance(result, str)

    def test_non_empty(self, registry):
        result = registry.for_llm_context()
        assert len(result.strip()) > 0

    def test_contains_persona_layer(self, registry):
        result = registry.for_llm_context()
        assert "persona" in result.lower()

    def test_contains_prompt_layer(self, registry):
        result = registry.for_llm_context()
        assert "prompt" in result.lower()

    def test_contains_all_knob_names(self, registry):
        result = registry.for_llm_context()
        for name in registry.knob_names():
            assert name in result, f"Knob '{name}' not found in for_llm_context() output"

    def test_contains_descriptions(self, registry):
        result = registry.for_llm_context()
        # At least some descriptions should appear
        for name in registry.knob_names():
            knob = registry.get(name)
            # Check a word from the description appears
            first_word = knob["description"].split()[0]
            assert first_word in result, f"Description for '{name}' not in context"
            break  # Just check one to avoid over-constraining format
