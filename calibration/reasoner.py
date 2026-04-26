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

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # type: ignore[assignment,misc]

from .overlay import merge_overlay
from .registry import KnobRegistry


# ---------------------------------------------------------------------------
# Static calibration knowledge injected into every reasoner prompt
# ---------------------------------------------------------------------------

METRIC_INTERPRETATION = """
## Metric Interpretation Guide

### P-value tier classification (apply to each metric's fail_rate and cliffs_delta)
- CRITICAL  : fail_rate > 0.50  OR  |cliffs_delta| >= 0.474  →  clearly deviating from real baseline
- SECONDARY : fail_rate 0.20–0.50  OR  0.33 <= |cliffs_delta| < 0.474  →  mildly suspicious
- ACCEPTABLE: fail_rate < 0.20  AND  |cliffs_delta| < 0.33  →  within normal range

### Content diversity metrics
- self_bleu_4:
  - generated_higher → comments are too lexically repetitive or templated; increase viewpoint variety
  - generated_lower  → comments are overly scattered; improve topical coherence
- self_bertscore_mean_f1:
  - generated_higher → comments are too semantically similar; add more distinct perspectives
  - generated_lower  → comments may be incoherent or off-topic; align to discussion context
- semantic_mean_cosine:
  - generated_higher → discussion is too semantically homogeneous; diversify viewpoints
  - generated_lower  → discussion lacks shared topic focus; improve topical anchoring

### Structural complexity metrics
- length_std / length_cv:
  - generated_higher → comment lengths vary too much; narrow the length distribution
  - generated_lower  → comments are mechanically uniform; allow more natural length variation
- avg_depth:
  - generated_higher → discussion is too deeply nested; reduce back-and-forth chains
  - generated_lower  → discussion is too flat; encourage reply chains and nested responses
- avg_branching_factor:
  - generated_higher → too many replies branch from single comments; reduce branching
  - generated_lower  → discussion lacks branching interactions; encourage more direct replies
- structural_virality:
  - generated_higher → reply tree is too chain-like; make thread more compact
  - generated_lower  → discussion is too flat/broadcast-like; add reply interactions

### Toxicity / aggression metrics
- toxicity_mean:
  - generated_higher → discussion is too toxic; reduce hostile language
  - generated_lower  → discussion may be too sanitized; add mild disagreement or skepticism
- insult_mean:
  - generated_higher → too many insults; reduce direct personal attacks
  - generated_lower  → too polite or conflict-avoidant; allow blunter pushback
- obscene_mean:
  - generated_higher → too much obscene language; reduce
  - generated_lower  → may lack casual harsh Reddit-style language
- aggression_score_mean:
  - generated_higher → discussion is too aggressive; soften tone
  - generated_lower  → discussion is too agreeable; add natural skepticism or bluntness
- threat_mean / severe_toxicity_mean / identity_attack_mean:
  - generated_higher → over-generated; reduce immediately
  - generated_lower  → ACCEPTABLE — do NOT optimize upward; never recommend adding threats,
                       hate speech, severe toxicity, or identity attacks as a strategy
""".strip()

CALIBRATION_PRINCIPLES = """
## Calibration Principles

1. Prioritize CRITICAL metrics first. Address SECONDARY metrics only if CRITICAL ones are resolved.
2. Do not overcorrect. Suggest moderate changes unless many metrics in the same dimension fail.
3. Identify the PRIMARY LAYER to change for each issue:
   - Persona layer (distributions, sentiment_bias): controls WHO speaks and with what attitude
   - Prompt layer (text instructions): controls HOW agents write their responses
   - Use persona layer for: fixing diversity/repetition, stance balance, aggression/politeness levels
   - Use prompt layer for: fixing structural patterns, length uniformity, paraphrasing, consensus artifacts
4. Content diversity too high → increase viewpoint diversity via stance/conflict_style/motivation distributions
5. Content diversity too low  → improve topical anchoring via tone_guidance and depth instructions
6. Structural metrics too low → encourage reply chains via structure_preference_weight and prompt instructions
7. Structural metrics too high → reduce excessive nesting via depth_soft_cap and structure_preference_weight
8. Toxicity/aggression too low  → shift conflict_style toward blunt/skeptical/argumentative; add pushback instructions
9. Toxicity/aggression too high → shift conflict_style toward calm/avoidant; soften tone_guidance
10. Never recommend adding identity_attack, threats, hate speech, or severe toxicity. If those are too low, mark as acceptable.
11. For the 5 candidates, explore a gradient: conservative → primary → aggressive, plus prompt variants.
""".strip()


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
    current_overlay : dict
    current_diagnostic : dict
        Output of score_candidate: {fail_rate, mean_abs_delta, per_metric}.
    real_baseline : dict
        Median values from real Reddit data {metric_name: float}.
    trajectory : list[dict]
        Per-iteration history.
    failed_strategies : list[str]
        Strategy labels already tried without improvement.
    metric_definitions : str
        Optional external metric reference (may be empty).

    Returns
    -------
    str
    """
    sections: list[str] = []

    # ── Metric interpretation and calibration principles ──────────────────
    sections.append(METRIC_INTERPRETATION)
    sections.append("")
    sections.append(CALIBRATION_PRINCIPLES)
    sections.append("")

    # ── Optional external metric definitions ──────────────────────────────
    if metric_definitions and metric_definitions.strip():
        sections.append("## Additional Metric Reference\n")
        sections.append(metric_definitions.strip())
        sections.append("")

    # ── Tunable knobs ─────────────────────────────────────────────────────
    sections.append("## Tunable Knobs\n")
    sections.append(registry.for_llm_context().strip())
    sections.append("")

    # ── Current overlay ───────────────────────────────────────────────────
    sections.append("## Current Overlay\n")
    sections.append(json.dumps(current_overlay, indent=2))
    sections.append("")

    # ── Current diagnostic (per-metric with tier classification) ──────────
    sections.append("## Current Diagnostic\n")
    fail_rate = current_diagnostic.get("fail_rate", "N/A")
    mean_abs_delta = current_diagnostic.get("mean_abs_delta", "N/A")
    sections.append(f"overall_fail_rate : {fail_rate}")
    sections.append(f"mean_abs_delta    : {mean_abs_delta}")
    sections.append("")

    per_metric = current_diagnostic.get("per_metric", {})
    if per_metric:
        sections.append("Per-metric breakdown (pre-classified by tier):")
        metric_rows: list[dict] = []
        for metric, info in per_metric.items():
            if metric == "threads":
                continue
            fr = info.get("fail_rate", 0.0)
            cd = info.get("cliffs_delta", 0.0)
            direction = info.get("direction", "")
            gen_median = info.get("generated_median", "")
            real_median = info.get("real_median", "")

            # Classify tier
            if fr > 0.50 or abs(cd) >= 0.474:
                tier = "CRITICAL"
            elif fr >= 0.20 or abs(cd) >= 0.33:
                tier = "SECONDARY"
            else:
                tier = "acceptable"

            metric_rows.append({
                "metric": metric,
                "tier": tier,
                "fail_rate": round(fr, 3),
                "cliffs_delta": round(cd, 3),
                "direction": direction,
                "generated_median": gen_median,
                "real_median": real_median,
            })

        # Sort: CRITICAL first, then SECONDARY, then acceptable
        tier_order = {"CRITICAL": 0, "SECONDARY": 1, "acceptable": 2}
        metric_rows.sort(key=lambda r: (tier_order.get(r["tier"], 3), -abs(r.get("cliffs_delta", 0))))
        sections.append(json.dumps(metric_rows, indent=2))
    sections.append("")

    # ── Real baseline statistics (from train split) ─────────────────────────
    sections.append("## Real Baseline Statistics (train split)\n")
    sections.append(
        "Each metric shows: median, mean, std, p25, p75, min, max, n (sample size).\n"
        "Use these to understand the real distribution shape, not just the center."
    )
    sections.append(json.dumps(real_baseline, indent=2))
    sections.append("")

    # ── Calibration trajectory ────────────────────────────────────────────
    sections.append("## Calibration Trajectory\n")
    if trajectory:
        for entry in trajectory:
            sections.append(
                f"  iteration {entry.get('iteration','?')}: "
                f"fail_rate={entry.get('fail_rate','?')}, "
                f"strategy={entry.get('strategy_label','?')}"
            )
    else:
        sections.append("  (no history yet)")
    sections.append("")

    # ── Failed strategies ─────────────────────────────────────────────────
    sections.append("## Failed Strategies (do not repeat these)\n")
    if failed_strategies:
        for s in failed_strategies:
            sections.append(f"  - {s}")
    else:
        sections.append("  (none yet)")
    sections.append("")

    # ── Task instructions ─────────────────────────────────────────────────
    sections.append("## Task\n")
    sections.append(
        "You are a calibration reasoner for a Reddit discussion simulator. "
        "Your job is to close the gap between simulated and real Reddit metrics.\n"
        "\n"
        "Step 1 — DIAGNOSE: For each CRITICAL and SECONDARY metric, explain why it is "
        "failing, referencing its tier, direction (too_high/too_low), and what that "
        "implies about the generated discussions.\n"
        "\n"
        "Step 2 — TARGET LAYER: Decide whether to change the persona layer, the prompt "
        "layer, or both. Explain your reasoning. Use persona for attitude/distribution "
        "problems; use prompt for structural/length/paraphrase problems.\n"
        "\n"
        "Step 3 — PRIMARY STRATEGY: Propose a concrete overlay_diff with specific knob "
        "changes. Values must be valid per the knob registry.\n"
        "\n"
        "Step 4 — CONSERVATIVE STRATEGY: Propose a conservative_diff — a smaller version "
        "of the same strategy (roughly half the magnitude of changes) for safer candidates.\n"
        "\n"
        "Step 5 — PROMPT ALTERNATIVES: Provide up to 2 alternative text values for prompt "
        "knobs (prompt_alternatives dict). Each alternative should differ meaningfully in "
        "approach, not just wording.\n"
        "\n"
        "Step 6 — CANDIDATE PLAN: Briefly describe how each of the 5 candidates will differ:\n"
        "  candidate_0: primary strategy (overlay_diff)\n"
        "  candidate_1: conservative strategy (conservative_diff)\n"
        "  candidate_2: aggressive version (primary strategy with larger perturbations)\n"
        "  candidate_3: primary strategy + first prompt alternative\n"
        "  candidate_4: primary strategy + second prompt alternative (or aggressive if none)\n"
        "\n"
        "Respond with a single JSON object with these keys:\n"
        "  - diagnosis        (str)         : root cause analysis with p-value tier references\n"
        "  - primary_layer    (str)         : 'persona', 'prompt', or 'both'\n"
        "  - strategy         (str)         : what you propose to change and why\n"
        "  - strategy_label   (str)         : short snake_case identifier\n"
        "  - overlay_diff     (dict)        : primary knob changes\n"
        "  - conservative_diff(dict)        : conservative version (smaller changes)\n"
        "  - prompt_alternatives (dict)     : prompt knob name → alternative text (may be {})\n"
        "  - candidate_rationale (list[str]): one sentence per candidate explaining its role\n"
        "  - constraints      (list[str])   : any hard constraints to enforce (may be [])\n"
        "\n"
        "Return valid JSON only. No commentary outside the JSON object."
    )

    return "\n".join(sections)


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def call_reasoner(client, model: str, prompt: str) -> str:
    """Call the OpenAI-compatible API with the reasoner prompt.

    Supports both openai >= 1.0 (client.chat.completions.create) and
    openai < 1.0 (openai.ChatCompletion.create).
    """
    if OpenAI is not None and isinstance(client, OpenAI):
        # openai >= 1.0
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            response_format={"type": "json_object"},
            timeout=120,
        )
        return response.choices[0].message.content
    else:
        # openai < 1.0 or compatible client
        import openai as _openai
        response = _openai.ChatCompletion.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            api_key=getattr(client, "api_key", None),
            api_base=getattr(client, "base_url", None),
        )
        return response["choices"][0]["message"]["content"]


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def parse_reasoner_response(raw: str) -> dict:
    """Parse and validate the LLM's JSON response.

    Returns
    -------
    dict with keys: diagnosis, primary_layer, strategy, strategy_label,
                    overlay_diff, conservative_diff, prompt_alternatives,
                    candidate_rationale, constraints.
    """
    data = json.loads(raw)

    required = ("diagnosis", "strategy", "strategy_label", "overlay_diff")
    for field in required:
        if field not in data:
            raise KeyError(f"Missing required field in reasoner response: '{field}'")

    return {
        "diagnosis": data["diagnosis"],
        "primary_layer": data.get("primary_layer", "both"),
        "strategy": data["strategy"],
        "strategy_label": data["strategy_label"],
        "overlay_diff": data["overlay_diff"],
        "conservative_diff": data.get("conservative_diff", {}),
        "prompt_alternatives": data.get("prompt_alternatives", {}),
        "candidate_rationale": data.get("candidate_rationale", []),
        "constraints": data.get("constraints", []),
    }


# ---------------------------------------------------------------------------
# Variant generation
# ---------------------------------------------------------------------------

def _perturb_distribution(dist: dict, rng: random.Random, scale: float = 0.10) -> dict:
    """Jitter distribution values ±scale then renormalize to sum=1."""
    jittered = {
        k: max(0.0, v * (1.0 + rng.uniform(-scale, scale)))
        for k, v in dist.items()
    }
    total = sum(jittered.values())
    if total == 0.0:
        n = len(jittered)
        return {k: 1.0 / n for k in jittered}
    return {k: v / total for k, v in jittered.items()}


def _perturb_overlay(
    overlay: dict,
    registry: KnobRegistry,
    rng: random.Random,
    scale: float = 0.15,
) -> dict:
    """Return a copy of overlay with numeric/distribution knobs jittered.

    Parameters
    ----------
    scale : float
        Maximum relative jitter (e.g. 0.15 = ±15%).
    """
    result = dict(overlay)
    for name, value in overlay.items():
        try:
            knob = registry.get(name)
        except KeyError:
            continue

        if knob["type"] == "float":
            jitter = rng.uniform(-scale * 0.5, scale)
            new_val = value * (1.0 + jitter)
            new_val = max(knob["min"], min(knob["max"], new_val))
            result[name] = new_val

        elif knob["type"] == "distribution" and isinstance(value, dict):
            result[name] = _perturb_distribution(value, rng, scale=scale)

    return result


def generate_variants(
    current_overlay: dict,
    base_diff: dict,
    prompt_alternatives: dict,
    registry: KnobRegistry,
    seed: int = 42,
    conservative_diff: dict | None = None,
) -> list[dict]:
    """Generate 5 candidate overlays for evaluation.

    Candidate layout:
      0 — Primary   : exact merge of current_overlay + base_diff
      1 — Conservative: merge of current_overlay + conservative_diff
                       (falls back to light jitter of candidate 0 if not provided)
      2 — Aggressive : heavy jitter (±20%) of candidate 0
      3 — Prompt alt 1: candidate 0 + first prompt alternative
                        (falls back to medium jitter if no alternatives)
      4 — Prompt alt 2: candidate 0 + second prompt alternative
                        (falls back to combo jitter if fewer than 2 alternatives)

    Parameters
    ----------
    conservative_diff : dict | None
        Smaller version of base_diff from the LLM. When None, light jitter is
        used for candidate 1.
    """
    rng = random.Random(seed)

    # Candidate 0: exact primary strategy
    candidate_0 = merge_overlay(current_overlay, base_diff)

    # Candidate 1: conservative — use LLM-provided conservative_diff if available
    if conservative_diff:
        candidate_1 = merge_overlay(current_overlay, conservative_diff)
    else:
        candidate_1 = _perturb_overlay(candidate_0, registry, rng, scale=0.07)

    # Candidate 2: aggressive — heavy jitter of candidate 0
    candidate_2 = _perturb_overlay(candidate_0, registry, rng, scale=0.20)

    # Candidates 3–4: prompt alternatives or fallback jitter
    alt_keys = list(prompt_alternatives.keys())

    if len(alt_keys) >= 1:
        first_alt = {alt_keys[0]: prompt_alternatives[alt_keys[0]]}
        candidate_3 = merge_overlay(candidate_0, first_alt)
    else:
        candidate_3 = _perturb_overlay(candidate_0, registry, rng, scale=0.12)

    if len(alt_keys) >= 2:
        second_alt = {alt_keys[1]: prompt_alternatives[alt_keys[1]]}
        candidate_4 = merge_overlay(candidate_0, second_alt)
    elif len(alt_keys) == 1:
        first_alt = {alt_keys[0]: prompt_alternatives[alt_keys[0]]}
        candidate_4 = _perturb_overlay(merge_overlay(candidate_0, first_alt), registry, rng, scale=0.12)
    else:
        candidate_4 = _perturb_overlay(candidate_0, registry, rng, scale=0.12)

    return [candidate_0, candidate_1, candidate_2, candidate_3, candidate_4]
