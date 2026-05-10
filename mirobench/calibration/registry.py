"""Registry for the two-slot calibration architecture.

The calibration loop no longer exposes a long list of fine-grained knobs.
Instead, the LLM edits only two persisted text slots:

- ``persona.generation_guidance``
- ``prompt.comment_style_guidance``

These are still stored in an overlay so the system can diff, log, resume, and
reuse them across iterations, but semantically they are full prompt patches
rather than tiny scalar knobs.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any

# ---------------------------------------------------------------------------
# Knob definitions
# ---------------------------------------------------------------------------

_KNOBS: list[dict] = [
    # ------------------------------------------------------------------
    # Persona layer
    # ------------------------------------------------------------------
    {
        "name": "persona.generation_guidance",
        "layer": "persona",
        "domain": "persona.generation",
        "type": "text",
        "default": "",
        "description": (
            "Long-form persona casting guidance injected into the persona-generation prompt. "
            "This is the PRIMARY knob for controlling what kinds of people get generated. "
            "Use this for concrete, example-rich instructions covering: conflict styles, "
            "motivations, knowledge levels, stances, sentiment ranges, and how their "
            "product experiences should show up in later comments. "
            "This single knob replaces all smaller persona-layer distribution/float knobs."
        ),
    },
    # ------------------------------------------------------------------
    # Prompt layer
    # ------------------------------------------------------------------
    {
        "name": "prompt.comment_style_guidance",
        "layer": "prompt",
        "domain": "prompt.comment_style",
        "type": "text",
        "default": "",
        "description": (
            "Long-form comment and reply writing guidance injected into the runtime prompts. "
            "This is the PRIMARY knob for controlling how agents write comments. "
            "Use this for concrete multi-sentence instructions covering: tone, reply behavior, "
            "comment structure, length variety, paraphrasing avoidance, consensus handling, "
            "depth/nesting preferences, and example comment shapes. "
            "This single knob replaces all smaller prompt-layer text knobs."
        ),
    },
]

# Build lookup index
_KNOB_INDEX: dict[str, dict] = {k["name"]: k for k in _KNOBS}


# ---------------------------------------------------------------------------
# KnobRegistry
# ---------------------------------------------------------------------------

class KnobRegistry:
    """Registry of the persisted calibration text slots."""

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

            errors.extend(self._validate_text(name, value))

        return errors

    def sanitize_overlay(self, overlay: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
        """Return a copy of *overlay* containing only known, valid knobs.

        Unknown knobs and values that fail validation are dropped from the
        returned overlay and reported in the error list.
        """
        cleaned: dict[str, Any] = {}
        errors: list[str] = []

        for name, value in overlay.items():
            if name not in _KNOB_INDEX:
                errors.append(f"Unknown knob: '{name}'.")
                continue

            knob_errors = self._validate_text(name, value)

            if knob_errors:
                errors.extend(knob_errors)
                continue

            cleaned[name] = deepcopy(value)

        return cleaned, errors

    # ------------------------------------------------------------------
    # LLM context
    # ------------------------------------------------------------------

    def for_llm_context(self) -> str:
        """Return a plain-text block describing all knobs for use in LLM prompts."""
        lines: list[str] = [
            "# Calibration Knob Registry\n",
            "There are exactly TWO knobs. Both are free-form text slots, but they are not tiny style toggles; "
            "they are the actual persisted prompt edits that will be injected into the simulator.\n",
            "Treat each knob value as a self-contained calibration patch. Every block you write "
            "must be specific enough that a downstream generator can follow it without guessing.\n",
            "Minimum quality bar for every block:\n"
            "- explain which failure pattern it is correcting\n"
            "- give a causal logic chain for why the behavior should change\n"
            "- say what to do and what to avoid\n"
            "- name the concrete details, anecdotes, disagreements, or reply shapes that should appear\n"
            "- include 2-4 short examples or mini-patterns\n"
            "- avoid empty slogans like 'be more direct' or 'be more realistic'\n",
        ]

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

            lines.append(f"  default    : \"{knob['default']}\"")

            lines.append("")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _validate_text(self, name: str, value: Any) -> list[str]:
        if not isinstance(value, str):
            return [
                f"Knob '{name}': expected a str for text type, got {type(value).__name__}."
            ]
        return []
