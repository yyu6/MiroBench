"""
Reasoner for the calibration system.

Responsible for:
- Assembling LLM prompts from registry state and diagnostics
- Calling the OpenAI API to get overlay diffs and reasoning
- Parsing structured JSON responses
- Generating candidate overlay variants for evaluation
"""
from __future__ import annotations

import json
import random
from typing import Any

from openai import OpenAI

from .overlay import merge_overlay
from .registry import KnobRegistry


# ---------------------------------------------------------------------------
# Prompt assembly
# ---------------------------------------------------------------------------

def build_reasoner_prompt(
    registry: KnobRegistry,
    current_overlay: dict,
    current_diagnostic: dict,
    real_baseline: dict,
    trajectory: list[dict],
    failed_strategies: list[str],
    metric_definitions: str,
) -> str:
    """Assemble a full reasoner prompt for the LLM.

    Parameters
    ----------
    registry : KnobRegistry
        The knob registry providing knob descriptions.
    current_overlay : dict
        Current active overlay (knob name → value).
    current_diagnostic : dict
        Diagnostic from the last evaluation run, containing keys:
        ``fail_rate``, ``mean_abs_delta``, ``per_metric``.
    real_baseline : dict
        Median values from real Reddit data (metric name → float).
    trajectory : list[dict]
        History of iterations, each with ``iteration``, ``fail_rate``,
        ``strategy_label``.
    failed_strategies : list[str]
        Strategy labels that have already been tried and did not improve.
    metric_definitions : str
        Plain-text description of each metric.

    Returns
    -------
    str
        The assembled prompt string.
    """
    sections: list[str] = []

    # --- Metric definitions ---
    sections.append("# Metric Definitions\n")
    sections.append(metric_definitions.strip())
    sections.append("")

    # --- Tunable Knobs ---
    sections.append("# Tunable Knobs\n")
    sections.append(registry.for_llm_context().strip())
    sections.append("")

    # --- Current Overlay ---
    sections.append("# Current Overlay\n")
    sections.append(json.dumps(current_overlay, indent=2))
    sections.append("")

    # --- Current Diagnostic ---
    sections.append("# Current Diagnostic\n")
    fail_rate = current_diagnostic.get("fail_rate", "N/A")
    mean_abs_delta = current_diagnostic.get("mean_abs_delta", "N/A")
    per_metric = current_diagnostic.get("per_metric", {})
    # Exclude threads key if present
    per_metric_filtered = {
        k: v for k, v in per_metric.items() if k != "threads"
    }
    sections.append(f"fail_rate: {fail_rate}")
    sections.append(f"mean_abs_delta: {mean_abs_delta}")
    sections.append("per_metric:")
    sections.append(json.dumps(per_metric_filtered, indent=2))
    sections.append("")

    # --- Real Baseline Medians ---
    sections.append("# Real Baseline Medians\n")
    sections.append(json.dumps(real_baseline, indent=2))
    sections.append("")

    # --- Trajectory ---
    sections.append("# Calibration Trajectory\n")
    if trajectory:
        for entry in trajectory:
            iter_num = entry.get("iteration", "?")
            fr = entry.get("fail_rate", "?")
            label = entry.get("strategy_label", "?")
            sections.append(f"  iteration {iter_num}: fail_rate={fr}, strategy={label}")
    else:
        sections.append("  (no history yet)")
    sections.append("")

    # --- Failed Strategies ---
    sections.append("# Failed Strategies (do not repeat these)\n")
    if failed_strategies:
        for s in failed_strategies:
            sections.append(f"  - {s}")
    else:
        sections.append("  (none yet)")
    sections.append("")

    # --- Instructions ---
    sections.append("# Instructions\n")
    sections.append(
        "You are a calibration reasoner for a Reddit discussion simulator. "
        "Your job is to close the gap between simulated and real Reddit metrics.\n"
        "\n"
        "Based on the information above, please:\n"
        "1. Diagnose the root cause of the current metric failures.\n"
        "2. Propose a concrete calibration strategy (different from any failed strategies above).\n"
        "3. Output an `overlay_diff` — a partial overlay containing only the knobs you want to change "
        "and their new values.\n"
        "4. Provide `prompt_alternatives` — a dict mapping prompt knob names to alternative text "
        "strings to try (may be empty).\n"
        "5. List any `constraints` you recommend the system enforce.\n"
        "\n"
        "Respond with a single JSON object with the following keys:\n"
        "  - `diagnosis` (str): root cause analysis\n"
        "  - `strategy` (str): what you propose to do and why\n"
        "  - `strategy_label` (str): a short snake_case identifier for this strategy\n"
        "  - `overlay_diff` (dict): the partial overlay diff to apply\n"
        "  - `prompt_alternatives` (dict): prompt knob name → alternative text (may be {})\n"
        "  - `constraints` (list[str]): any constraints to enforce (may be [])\n"
        "\n"
        "Return valid JSON only. Do not include any commentary outside the JSON object."
    )

    return "\n".join(sections)


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def call_reasoner(client: OpenAI, model: str, prompt: str) -> str:
    """Call the OpenAI API with the reasoner prompt.

    Parameters
    ----------
    client : OpenAI
        Authenticated OpenAI client.
    model : str
        Model identifier (e.g. ``"gpt-4o"``).
    prompt : str
        The assembled reasoner prompt.

    Returns
    -------
    str
        Raw response content string (expected to be JSON).
    """
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
        response_format={"type": "json_object"},
        timeout=120,
    )
    return response.choices[0].message.content


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def parse_reasoner_response(raw: str) -> dict:
    """Parse and validate the LLM's JSON response.

    Parameters
    ----------
    raw : str
        Raw JSON string from the LLM.

    Returns
    -------
    dict
        Parsed response with keys: ``diagnosis``, ``strategy``,
        ``strategy_label``, ``overlay_diff``, ``prompt_alternatives``,
        ``constraints``.

    Raises
    ------
    json.JSONDecodeError
        If ``raw`` is not valid JSON.
    KeyError
        If required fields are missing.
    """
    data = json.loads(raw)

    # Validate required fields — will raise KeyError if missing
    required = ("diagnosis", "strategy", "strategy_label", "overlay_diff")
    for field in required:
        if field not in data:
            raise KeyError(f"Missing required field in reasoner response: '{field}'")

    return {
        "diagnosis": data["diagnosis"],
        "strategy": data["strategy"],
        "strategy_label": data["strategy_label"],
        "overlay_diff": data["overlay_diff"],
        "prompt_alternatives": data.get("prompt_alternatives", {}),
        "constraints": data.get("constraints", []),
    }


# ---------------------------------------------------------------------------
# Variant generation
# ---------------------------------------------------------------------------

def _perturb_distribution(dist: dict, rng: random.Random) -> dict:
    """Jitter distribution values ±10% then renormalize to sum=1.

    Parameters
    ----------
    dist : dict
        Original distribution (str → float).
    rng : random.Random
        Random source for reproducibility.

    Returns
    -------
    dict
        Perturbed and renormalized distribution.
    """
    jittered = {
        k: max(0.0, v * (1.0 + rng.uniform(-0.10, 0.10)))
        for k, v in dist.items()
    }
    total = sum(jittered.values())
    if total == 0.0:
        # Fallback: uniform
        n = len(jittered)
        return {k: 1.0 / n for k in jittered}
    return {k: v / total for k, v in jittered.items()}


def _perturb_overlay(overlay: dict, registry: KnobRegistry, rng: random.Random) -> dict:
    """Return a copy of overlay with numeric/distribution knobs jittered.

    Parameters
    ----------
    overlay : dict
        Full overlay to perturb.
    registry : KnobRegistry
        Registry used to determine knob types.
    rng : random.Random
        Random source.

    Returns
    -------
    dict
        Perturbed overlay copy.
    """
    result = dict(overlay)
    for name, value in overlay.items():
        try:
            knob = registry.get(name)
        except KeyError:
            continue

        if knob["type"] == "float":
            jitter = rng.uniform(-0.05, 0.10)
            new_val = value * (1.0 + jitter)
            # Clamp to knob bounds
            new_val = max(knob["min"], min(knob["max"], new_val))
            result[name] = new_val

        elif knob["type"] == "distribution" and isinstance(value, dict):
            result[name] = _perturb_distribution(value, rng)

    return result


def generate_variants(
    current_overlay: dict,
    base_diff: dict,
    prompt_alternatives: dict,
    registry: KnobRegistry,
    seed: int = 42,
) -> list[dict]:
    """Generate 5 candidate overlays for evaluation.

    Candidate generation strategy:
      - 0: Exact merge of ``current_overlay`` + ``base_diff``
      - 1-2: Numeric/distribution perturbations (±5-10% jitter)
      - 3-4: Prompt alternatives (if provided), else more perturbations

    Parameters
    ----------
    current_overlay : dict
        The current active overlay.
    base_diff : dict
        The overlay diff proposed by the reasoner.
    prompt_alternatives : dict
        Prompt knob alternatives from the reasoner (may be empty).
    registry : KnobRegistry
        Registry for knob metadata.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    list[dict]
        Five candidate overlay dicts.
    """
    rng = random.Random(seed)

    # Candidate 0: exact
    candidate_0 = merge_overlay(current_overlay, base_diff)

    # Candidates 1-2: perturbations of candidate 0
    candidate_1 = _perturb_overlay(candidate_0, registry, rng)
    candidate_2 = _perturb_overlay(candidate_0, registry, rng)

    # Candidates 3-4: prompt alternatives or more perturbations
    alt_keys = list(prompt_alternatives.keys())

    if len(alt_keys) >= 1:
        # Candidate 3: apply first prompt alternative on top of candidate 0
        first_alt = {alt_keys[0]: prompt_alternatives[alt_keys[0]]}
        candidate_3 = merge_overlay(candidate_0, first_alt)
    else:
        candidate_3 = _perturb_overlay(candidate_0, registry, rng)

    if len(alt_keys) >= 2:
        # Candidate 4: apply second prompt alternative on top of candidate 0
        second_alt = {alt_keys[1]: prompt_alternatives[alt_keys[1]]}
        candidate_4 = merge_overlay(candidate_0, second_alt)
    elif len(alt_keys) == 1:
        # Only one alternative — apply it combined with a perturbation
        first_alt = {alt_keys[0]: prompt_alternatives[alt_keys[0]]}
        candidate_4 = _perturb_overlay(merge_overlay(candidate_0, first_alt), registry, rng)
    else:
        candidate_4 = _perturb_overlay(candidate_0, registry, rng)

    return [candidate_0, candidate_1, candidate_2, candidate_3, candidate_4]
