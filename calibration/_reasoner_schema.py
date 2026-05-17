"""Response schemas and text-knob normalization for the calibration reasoner.

Extracted from reasoner.py.
"""
from __future__ import annotations

from typing import Any

from ._reasoner_constants import _REQUIRED_TEXT_KNOBS


def _required_text_overlay_schema() -> dict[str, Any]:
    """Return the strict schema for the two persisted text knobs."""
    return {
        "type": "object",
        "additionalProperties": False,
        "required": list(_REQUIRED_TEXT_KNOBS),
        "properties": {
            "persona.generation_guidance": {
                "type": "string",
                "minLength": 1,
            },
            "prompt.comment_style_guidance": {
                "type": "string",
                "minLength": 1,
            },
        },
    }


def materializer_response_format(expected_candidates: int) -> dict[str, Any]:
    """Return a strict schema requiring one materialized text block per candidate."""
    if expected_candidates < 1 or expected_candidates > 5:
        raise ValueError(
            f"expected_candidates must be between 1 and 5, got {expected_candidates}"
        )

    properties: dict[str, Any] = {}
    required: list[str] = []
    for idx in range(expected_candidates):
        key = f"candidate_{idx}"
        required.append(key)
        properties[key] = {
            "type": "object",
            "additionalProperties": False,
            "required": ["text_overlay_diff"],
            "properties": {
                "text_overlay_diff": _required_text_overlay_schema(),
            },
        }

    return {
        "type": "json_schema",
        "json_schema": {
            "name": "calibration_text_materializer_response",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "required": required,
                "properties": properties,
            },
        },
    }


def _response_format_for(schema_kind: str | None) -> dict[str, Any]:
    """Return the OpenAI response_format payload for a strict JSON schema."""
    if schema_kind == "strategist":
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "calibration_strategist_response",
                "strict": True,
                "schema": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["diagnosis", "candidates", "constraints"],
                    "properties": {
                        "diagnosis": {"type": "string", "minLength": 1},
                        "constraints": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "candidates": {
                            "type": "array",
                            "minItems": 5,
                            "maxItems": 5,
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "required": [
                                    "strategy_label",
                                    "strategy",
                                    "mechanism_family",
                                    "anti_incumbent",
                                    "primary_layer",
                                    "overlay_diff",
                                    "rationale",
                                ],
                                "properties": {
                                    "strategy_label": {"type": "string", "minLength": 1},
                                    "strategy": {"type": "string", "minLength": 1},
                                    "mechanism_family": {
                                        "type": "string",
                                        "enum": list(_MECHANISM_FAMILIES),
                                    },
                                    "anti_incumbent": {"type": "boolean"},
                                    "primary_layer": {
                                        "type": "string",
                                        "enum": ["persona", "prompt", "both"],
                                    },
                                    "rationale": {"type": "string", "minLength": 1},
                                    "overlay_diff": _required_text_overlay_schema(),
                                },
                            },
                        },
                    },
                },
            },
        }
    if schema_kind == "materializer":
        return materializer_response_format(5)
    return {"type": "json_object"}


def _normalize_text_knob_block(
    payload: Any,
    *,
    context: str,
) -> dict[str, str]:
    """Validate and normalize the required two text knobs."""
    if not isinstance(payload, dict):
        raise ValueError(f"{context} must be an object with the two text knobs.")
    normalized: dict[str, str] = {}
    missing: list[str] = []
    for key in _REQUIRED_TEXT_KNOBS:
        value = payload.get(key)
        if not isinstance(value, str) or not value.strip():
            missing.append(key)
        else:
            normalized[key] = value
    if missing:
        raise ValueError(f"{context} is missing required non-empty keys: {missing}")
    return normalized
