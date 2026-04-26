"""
Knob Registry for the calibration system.

Defines all tunable knobs across the persona and prompt layers,
with validation, defaults, and LLM-context formatting.
"""
from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# Knob definitions
# ---------------------------------------------------------------------------

_KNOBS: list[dict] = [
    # ------------------------------------------------------------------
    # Persona layer
    # ------------------------------------------------------------------
    {
        "name": "persona.conflict_style_distribution",
        "layer": "persona",
        "domain": "persona.conflict_style",
        "type": "distribution",
        "keys": ["calm", "skeptical", "blunt", "sarcastic", "argumentative", "avoidant"],
        "default": {
            "calm": 0.20,
            "skeptical": 0.20,
            "blunt": 0.15,
            "sarcastic": 0.15,
            "argumentative": 0.15,
            "avoidant": 0.15,
        },
        "description": "Distribution over conflict-handling styles assigned to personas.",
    },
    {
        "name": "persona.primary_motivation_distribution",
        "layer": "persona",
        "domain": "persona.primary_motivation",
        "type": "distribution",
        "keys": [
            "helping",
            "venting",
            "showing expertise",
            "correcting people",
            "defending their own setup",
            "bargain-hunting",
            "complaining",
            "validation-seeking",
            "joking around",
        ],
        "default": {
            "helping": 0.15,
            "venting": 0.10,
            "showing expertise": 0.15,
            "correcting people": 0.10,
            "defending their own setup": 0.10,
            "bargain-hunting": 0.10,
            "complaining": 0.10,
            "validation-seeking": 0.10,
            "joking around": 0.10,
        },
        "description": "Distribution over primary motivations driving persona posting behaviour.",
    },
    {
        "name": "persona.knowledge_style_distribution",
        "layer": "persona",
        "domain": "persona.knowledge_style",
        "type": "distribution",
        "keys": [
            "beginner",
            "casual_user",
            "experienced_owner",
            "specialist",
            "overconfident_half_expert",
        ],
        "default": {
            "beginner": 0.20,
            "casual_user": 0.25,
            "experienced_owner": 0.25,
            "specialist": 0.15,
            "overconfident_half_expert": 0.15,
        },
        "description": "Distribution over self-assessed knowledge levels of simulated personas.",
    },
    {
        "name": "persona.stance_distribution",
        "layer": "persona",
        "domain": "persona.stance",
        "type": "distribution",
        "keys": ["supportive", "neutral", "observer", "opposing"],
        "default": {
            "supportive": 0.30,
            "neutral": 0.30,
            "observer": 0.20,
            "opposing": 0.20,
        },
        "description": "Distribution over stances personas take toward the thread topic.",
    },
    {
        "name": "persona.sentiment_bias_min",
        "layer": "persona",
        "domain": "persona.sentiment_bias",
        "type": "float",
        "min": -1.0,
        "max": 0.0,
        "default": -0.3,
        "description": "Lower bound of the uniform range from which persona sentiment bias is sampled.",
    },
    {
        "name": "persona.sentiment_bias_max",
        "layer": "persona",
        "domain": "persona.sentiment_bias",
        "type": "float",
        "min": 0.0,
        "max": 1.0,
        "default": 0.4,
        "description": "Upper bound of the uniform range from which persona sentiment bias is sampled.",
    },
    # ------------------------------------------------------------------
    # Prompt layer
    # ------------------------------------------------------------------
    {
        "name": "prompt.anti_paraphrase_instruction",
        "layer": "prompt",
        "domain": "prompt.anti_paraphrase",
        "type": "text",
        "default": (
            "Do not paraphrase or echo the previous messages. "
            "Use your own words and sentence structures."
        ),
        "description": "Instruction appended to prompts to discourage paraphrasing of prior turns.",
    },
    {
        "name": "prompt.tone_guidance",
        "layer": "prompt",
        "domain": "prompt.tone",
        "type": "text",
        "default": "Match the tone naturally found in real Reddit discussions.",
        "description": "High-level tone directive injected into generation prompts.",
    },
    {
        "name": "prompt.structure_preference_weight",
        "layer": "prompt",
        "domain": "prompt.structure",
        "type": "text",
        "default": (
            "Prefer flowing prose over bullet points or numbered lists "
            "unless the content naturally calls for them."
        ),
        "description": "Instruction weighting prose vs. structured output in generated posts.",
    },
    {
        "name": "prompt.depth_soft_cap_instruction",
        "layer": "prompt",
        "domain": "prompt.depth",
        "type": "text",
        "default": (
            "Keep responses concise — aim for 1–3 short paragraphs. "
            "Avoid padding or unnecessary elaboration."
        ),
        "description": "Soft-cap instruction governing response length and depth.",
    },
    {
        "name": "prompt.few_shot_style_anchor",
        "layer": "prompt",
        "domain": "prompt.few_shot",
        "type": "text",
        "default": "",
        "description": (
            "Optional few-shot example block prepended to the prompt to anchor writing style. "
            "Empty string disables it."
        ),
    },
    {
        "name": "prompt.consensus_handling",
        "layer": "prompt",
        "domain": "prompt.consensus",
        "type": "text",
        "default": (
            "Do not artificially manufacture consensus. "
            "Disagreement and uncertainty are natural and encouraged."
        ),
        "description": "Instruction governing how the model should treat emerging consensus in the thread.",
    },
]

# Build lookup index
_KNOB_INDEX: dict[str, dict] = {k["name"]: k for k in _KNOBS}


# ---------------------------------------------------------------------------
# KnobRegistry
# ---------------------------------------------------------------------------

class KnobRegistry:
    """Registry of all tunable calibration knobs."""

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def knob_names(self) -> list[str]:
        """Return an ordered list of all knob names."""
        return [k["name"] for k in _KNOBS]

    def get(self, name: str) -> dict:
        """Return the full knob definition dict for *name*.

        Raises KeyError for unknown knobs.
        """
        if name not in _KNOB_INDEX:
            raise KeyError(f"Unknown knob: '{name}'")
        return _KNOB_INDEX[name]

    def defaults(self) -> dict[str, Any]:
        """Return a mapping of knob name → default value for every knob."""
        return {k["name"]: k["default"] for k in _KNOBS}

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self, overlay: dict[str, Any]) -> list[str]:
        """Validate an overlay dict against the registry.

        Returns a (possibly empty) list of human-readable error strings.
        """
        errors: list[str] = []

        for name, value in overlay.items():
            if name not in _KNOB_INDEX:
                errors.append(f"Unknown knob: '{name}'.")
                continue

            knob = _KNOB_INDEX[name]
            knob_type = knob["type"]

            if knob_type == "distribution":
                errors.extend(self._validate_distribution(name, value, knob))
            elif knob_type == "float":
                errors.extend(self._validate_float(name, value, knob))
            elif knob_type == "text":
                errors.extend(self._validate_text(name, value))

        return errors

    # ------------------------------------------------------------------
    # LLM context
    # ------------------------------------------------------------------

    def for_llm_context(self) -> str:
        """Return a plain-text block describing all knobs for use in LLM prompts."""
        lines: list[str] = ["# Calibration Knob Registry\n"]

        current_layer: str | None = None
        for knob in _KNOBS:
            layer = knob["layer"]
            if layer != current_layer:
                lines.append(f"\n## Layer: {layer}\n")
                current_layer = layer

            lines.append(f"### {knob['name']}")
            lines.append(f"  type       : {knob['type']}")
            lines.append(f"  domain     : {knob['domain']}")
            lines.append(f"  description: {knob['description']}")

            if knob["type"] == "distribution":
                lines.append(f"  keys       : {', '.join(knob['keys'])}")
                lines.append(f"  default    : {knob['default']}")
            elif knob["type"] == "float":
                lines.append(f"  range      : [{knob['min']}, {knob['max']}]")
                lines.append(f"  default    : {knob['default']}")
            else:
                default_preview = knob["default"][:80] + "..." if len(knob["default"]) > 80 else knob["default"]
                lines.append(f"  default    : \"{default_preview}\"")

            lines.append("")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _validate_distribution(
        self, name: str, value: Any, knob: dict
    ) -> list[str]:
        errors: list[str] = []

        if not isinstance(value, dict):
            errors.append(
                f"Knob '{name}': expected a dict for distribution type, got {type(value).__name__}."
            )
            return errors

        # Check for negatives
        for k, v in value.items():
            if v < 0:
                errors.append(
                    f"Knob '{name}': negative value {v!r} for key '{k}' is not allowed."
                )

        # Check sum ≈ 1.0
        total = sum(value.values())
        if abs(total - 1.0) > 0.01:
            errors.append(
                f"Knob '{name}': distribution values must sum to 1.0 (±0.01), got {total:.4f}."
            )

        return errors

    def _validate_float(self, name: str, value: Any, knob: dict) -> list[str]:
        errors: list[str] = []

        if not isinstance(value, (int, float)):
            errors.append(
                f"Knob '{name}': expected a float, got {type(value).__name__}."
            )
            return errors

        if value < knob["min"]:
            errors.append(
                f"Knob '{name}': value {value} is below the minimum bound {knob['min']}."
            )
        if value > knob["max"]:
            errors.append(
                f"Knob '{name}': value {value} exceeds the maximum bound {knob['max']}."
            )

        return errors

    def _validate_text(self, name: str, value: Any) -> list[str]:
        if not isinstance(value, str):
            return [
                f"Knob '{name}': expected a str for text type, got {type(value).__name__}."
            ]
        return []
