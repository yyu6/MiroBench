"""Shared constants for the leaderboard: families, domains, baseline, styling.

The 16 core metrics and their 5-family grouping mirror
``mirobench.cli.CORE_METRICS`` exactly; we re-declare the grouping here (the CLI
only stores a flat set) and assert agreement at import time so the two never
drift.
"""
from __future__ import annotations

import json
from pathlib import Path

# 5 families × (3+4+3+2+4) = 16 metrics. Order matches the leaderboard columns.
FAMILIES: dict[str, list[str]] = {
    "Diversity": ["self_bleu_4", "semantic_mean_cosine", "self_bertscore_mean_f1"],
    "Tone": ["hard_disagree_rate", "polite_rate", "impolite_rate", "neutral_rate"],
    "Structure": ["length_cv", "avg_depth", "structural_virality"],
    "Content": ["mean_story_probability", "emotion_entropy"],
    "Toxicity": ["toxicity_mean", "severe_toxicity_mean", "obscene_mean", "threat_mean"],
}
FAMILY_ORDER: list[str] = list(FAMILIES.keys())

# Flat set of the 16 core metrics — must equal mirobench.cli.CORE_METRICS.
CORE_METRICS: set[str] = {m for ms in FAMILIES.values() for m in ms}
METRIC_FAMILY: dict[str, str] = {m: fam for fam, ms in FAMILIES.items() for m in ms}

# Domains in board-column order, with display labels.
DOMAIN_ORDER: list[str] = [
    "credit_cards", "cameras", "cell_phones", "headphones", "laptops",
]
DOMAIN_LABELS: dict[str, str] = {
    "credit_cards": "Credit cards",
    "cameras": "Cameras",
    "cell_phones": "Cell phones",
    "headphones": "Headphones",
    "laptops": "Laptops",
}

# Total core metrics scored per (model, domain) cell. Fixed at 16 so every entry
# is ranked out of the same denominator; a core metric missing from a submission
# (e.g. hard_disagree_rate when the disagreement checkpoint is absent) counts as
# a fail rather than shrinking the denominator.
METRICS_PER_DOMAIN = 16

_BASELINE_PATH = Path(__file__).with_name("baseline.json")


def load_baseline() -> dict:
    """Load the real-vs-real noise-floor baseline (per-family W1 and |delta|)."""
    with open(_BASELINE_PATH) as f:
        return json.load(f)


def heat_class(count: int) -> str:
    """Map a 0..16 pass-count to one of the heat-N CSS classes defined in
    leaderboard.html. Falls back to the highest defined class <= count."""
    defined = (0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 15, 16)
    pick = 0
    for d in defined:
        if d <= count:
            pick = d
    return f"heat-{pick}"


def total_style(pct: float) -> tuple[str, str]:
    """Colour + weight for a 'pass / denom' total cell, from the pass percentage.

    Reproduces the hand-tuned styling currently in the HTML:
      >= 25%  -> blue (accent), font-medium   (best-so-far)
      <  5%   -> red, font-medium              (near floor)
      else    -> black (text), normal weight
    Nothing approaches the ~95% real-vs-real noise floor, so 'accent' marks the
    strongest current entry, not an objectively good score.
    """
    if pct >= 25.0:
        return "text-accent", " font-medium"
    if pct < 5.0:
        return "text-red", " font-medium"
    return "text-text", ""


# Fail loudly if the grouping ever drifts from the CLI's flat core set.
def _assert_matches_cli() -> None:
    try:
        from mirobench.cli import CORE_METRICS as CLI_CORE
    except Exception:
        return  # cli import is optional (e.g. minimal CI); skip the cross-check.
    if set(CLI_CORE) != CORE_METRICS:
        raise AssertionError(
            "leaderboard.FAMILIES is out of sync with mirobench.cli.CORE_METRICS:\n"
            f"  only in cli:        {sorted(set(CLI_CORE) - CORE_METRICS)}\n"
            f"  only in leaderboard:{sorted(CORE_METRICS - set(CLI_CORE))}"
        )


_assert_matches_cli()
