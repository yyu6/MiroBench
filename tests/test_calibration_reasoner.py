"""
Tests for calibration/reasoner.py
"""
from __future__ import annotations

import json
import pytest

from calibration.registry import KnobRegistry
from calibration.reasoner import (
    build_reasoner_prompt,
    parse_reasoner_response,
    generate_variants,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def registry():
    return KnobRegistry()


@pytest.fixture
def sample_overlay(registry):
    return registry.defaults()


@pytest.fixture
def sample_diagnostic():
    return {
        "fail_rate": 0.35,
        "mean_abs_delta": 0.12,
        "per_metric": {
            "sentiment": {"delta": 0.08, "fail": False},
            "toxicity": {"delta": 0.18, "fail": True},
            "diversity": {"delta": 0.14, "fail": True},
        },
    }


@pytest.fixture
def sample_real_baseline():
    return {
        "sentiment": 0.62,
        "toxicity": 0.05,
        "diversity": 0.73,
    }


@pytest.fixture
def sample_trajectory():
    return [
        {"iteration": 0, "fail_rate": 0.50, "strategy_label": "baseline"},
        {"iteration": 1, "fail_rate": 0.40, "strategy_label": "reduce_toxicity_v1"},
        {"iteration": 2, "fail_rate": 0.35, "strategy_label": "reduce_toxicity_v2"},
    ]


@pytest.fixture
def sample_failed_strategies():
    return ["reduce_toxicity_v1", "reduce_toxicity_v2"]


@pytest.fixture
def sample_metric_definitions():
    return (
        "sentiment: mean VADER compound score, range [-1, 1].\n"
        "toxicity: fraction of posts flagged by Detoxify (lower is better).\n"
        "diversity: token-level TTR across all posts."
    )


# ---------------------------------------------------------------------------
# TestBuildReasonerPrompt
# ---------------------------------------------------------------------------

class TestBuildReasonerPrompt:
    def test_contains_required_sections(
        self,
        registry,
        sample_overlay,
        sample_diagnostic,
        sample_real_baseline,
        sample_trajectory,
        sample_failed_strategies,
        sample_metric_definitions,
    ):
        prompt = build_reasoner_prompt(
            registry=registry,
            current_overlay=sample_overlay,
            current_diagnostic=sample_diagnostic,
            real_baseline=sample_real_baseline,
            trajectory=sample_trajectory,
            failed_strategies=sample_failed_strategies,
            metric_definitions=sample_metric_definitions,
        )

        assert "Tunable Knobs" in prompt
        assert sample_metric_definitions[:30] in prompt
        assert "fail_rate" in prompt

    def test_includes_failed_strategies(
        self,
        registry,
        sample_overlay,
        sample_diagnostic,
        sample_real_baseline,
        sample_trajectory,
        sample_failed_strategies,
        sample_metric_definitions,
    ):
        prompt = build_reasoner_prompt(
            registry=registry,
            current_overlay=sample_overlay,
            current_diagnostic=sample_diagnostic,
            real_baseline=sample_real_baseline,
            trajectory=sample_trajectory,
            failed_strategies=sample_failed_strategies,
            metric_definitions=sample_metric_definitions,
        )
        assert "reduce_toxicity_v1" in prompt
        assert "reduce_toxicity_v2" in prompt

    def test_includes_real_baseline(
        self,
        registry,
        sample_overlay,
        sample_diagnostic,
        sample_real_baseline,
        sample_trajectory,
        sample_failed_strategies,
        sample_metric_definitions,
    ):
        prompt = build_reasoner_prompt(
            registry=registry,
            current_overlay=sample_overlay,
            current_diagnostic=sample_diagnostic,
            real_baseline=sample_real_baseline,
            trajectory=sample_trajectory,
            failed_strategies=sample_failed_strategies,
            metric_definitions=sample_metric_definitions,
        )
        assert "toxicity" in prompt
        assert "0.05" in prompt

    def test_requests_json_response(
        self,
        registry,
        sample_overlay,
        sample_diagnostic,
        sample_real_baseline,
        sample_trajectory,
        sample_failed_strategies,
        sample_metric_definitions,
    ):
        prompt = build_reasoner_prompt(
            registry=registry,
            current_overlay=sample_overlay,
            current_diagnostic=sample_diagnostic,
            real_baseline=sample_real_baseline,
            trajectory=sample_trajectory,
            failed_strategies=sample_failed_strategies,
            metric_definitions=sample_metric_definitions,
        )
        # Should mention overlay_diff and strategy_label in instructions
        assert "overlay_diff" in prompt
        assert "strategy_label" in prompt


# ---------------------------------------------------------------------------
# TestParseReasonerResponse
# ---------------------------------------------------------------------------

class TestParseReasonerResponse:
    def test_parses_valid_json(self):
        raw = json.dumps({
            "diagnosis": "High toxicity in argumentative personas.",
            "strategy": "Reduce argumentative persona weight.",
            "strategy_label": "reduce_argumentative_v1",
            "overlay_diff": {
                "persona.conflict_style_distribution": {
                    "calm": 0.30,
                    "skeptical": 0.20,
                    "blunt": 0.15,
                    "sarcastic": 0.10,
                    "argumentative": 0.10,
                    "avoidant": 0.15,
                }
            },
            "prompt_alternatives": {},
            "constraints": [],
        })
        result = parse_reasoner_response(raw)
        assert result["diagnosis"] == "High toxicity in argumentative personas."
        assert result["strategy_label"] == "reduce_argumentative_v1"
        assert "persona.conflict_style_distribution" in result["overlay_diff"]

    def test_handles_missing_optional_fields(self):
        raw = json.dumps({
            "diagnosis": "Minor drift.",
            "strategy": "Tone adjustment.",
            "strategy_label": "tone_v1",
            "overlay_diff": {},
        })
        result = parse_reasoner_response(raw)
        # Optional fields default to empty
        assert result["prompt_alternatives"] == {}
        assert result["constraints"] == []

    def test_raises_on_invalid_json(self):
        with pytest.raises((json.JSONDecodeError, ValueError)):
            parse_reasoner_response("not valid json {{{")

    def test_raises_on_missing_required_fields(self):
        raw = json.dumps({"diagnosis": "something"})
        with pytest.raises((KeyError, ValueError)):
            parse_reasoner_response(raw)


# ---------------------------------------------------------------------------
# TestGenerateVariants
# ---------------------------------------------------------------------------

class TestGenerateVariants:
    def test_returns_5_overlays(self, registry, sample_overlay):
        base_diff = {
            "persona.sentiment_bias_min": -0.25,
            "persona.sentiment_bias_max": 0.35,
        }
        variants = generate_variants(
            current_overlay=sample_overlay,
            base_diff=base_diff,
            prompt_alternatives={},
            registry=registry,
            seed=42,
        )
        assert len(variants) == 5

    def test_candidate_0_is_exact(self, registry, sample_overlay):
        base_diff = {
            "persona.sentiment_bias_min": -0.25,
            "persona.sentiment_bias_max": 0.35,
        }
        variants = generate_variants(
            current_overlay=sample_overlay,
            base_diff=base_diff,
            prompt_alternatives={},
            registry=registry,
            seed=42,
        )
        # Candidate 0 should be exact merge of current_overlay + base_diff
        assert variants[0]["persona.sentiment_bias_min"] == -0.25
        assert variants[0]["persona.sentiment_bias_max"] == 0.35

    def test_candidates_1_2_are_perturbed(self, registry, sample_overlay):
        base_diff = {
            "persona.sentiment_bias_min": -0.25,
            "persona.sentiment_bias_max": 0.35,
        }
        variants = generate_variants(
            current_overlay=sample_overlay,
            base_diff=base_diff,
            prompt_alternatives={},
            registry=registry,
            seed=42,
        )
        # Candidates 1 and 2 should differ from candidate 0 (numeric perturbation)
        # At least one of them should differ in the float knobs
        c0_min = variants[0]["persona.sentiment_bias_min"]
        c0_max = variants[0]["persona.sentiment_bias_max"]
        c1_min = variants[1]["persona.sentiment_bias_min"]
        c1_max = variants[1]["persona.sentiment_bias_max"]
        c2_min = variants[2]["persona.sentiment_bias_min"]
        c2_max = variants[2]["persona.sentiment_bias_max"]

        # At least one perturbed candidate should differ
        assert (c1_min != c0_min or c1_max != c0_max or
                c2_min != c0_min or c2_max != c0_max)

    def test_prompt_alternatives_applied(self, registry, sample_overlay):
        base_diff = {}
        prompt_alternatives = {
            "prompt.tone_guidance": "Be direct and concise.",
            "prompt.consensus_handling": "Allow consensus to emerge naturally.",
        }
        variants = generate_variants(
            current_overlay=sample_overlay,
            base_diff=base_diff,
            prompt_alternatives=prompt_alternatives,
            registry=registry,
            seed=42,
        )
        # Candidates 3 and 4 should incorporate prompt alternatives
        assert variants[3]["prompt.tone_guidance"] == "Be direct and concise."
        assert variants[4]["prompt.consensus_handling"] == "Allow consensus to emerge naturally."

    def test_distribution_variants_sum_to_one(self, registry, sample_overlay):
        base_diff = {
            "persona.conflict_style_distribution": {
                "calm": 0.25,
                "skeptical": 0.20,
                "blunt": 0.15,
                "sarcastic": 0.15,
                "argumentative": 0.10,
                "avoidant": 0.15,
            }
        }
        variants = generate_variants(
            current_overlay=sample_overlay,
            base_diff=base_diff,
            prompt_alternatives={},
            registry=registry,
            seed=42,
        )
        for i, variant in enumerate(variants):
            dist = variant.get("persona.conflict_style_distribution")
            if dist and isinstance(dist, dict):
                total = sum(dist.values())
                assert abs(total - 1.0) < 0.02, (
                    f"Candidate {i} distribution sums to {total:.4f}"
                )
