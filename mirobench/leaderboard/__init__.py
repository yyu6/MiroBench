"""MiroBench leaderboard tooling.

Turns per-submission scored CSVs under ``experiments/`` into the data-driven
leaderboard rendered in ``docs/leaderboard.html``.

Pipeline:
    experiments/**/thread_scores.csv  ──(build)──▶  docs/leaderboard.json
    docs/leaderboard.json             ──(render)─▶  docs/leaderboard.html

The numeric engine is :mod:`mirobench.compare` (pure numpy/scipy, no GPU), so
the whole pipeline runs in CI without model checkpoints.
"""

from .families import (
    CORE_METRICS,
    DOMAIN_LABELS,
    DOMAIN_ORDER,
    FAMILIES,
    FAMILY_ORDER,
    load_baseline,
)

__all__ = [
    "CORE_METRICS",
    "DOMAIN_LABELS",
    "DOMAIN_ORDER",
    "FAMILIES",
    "FAMILY_ORDER",
    "load_baseline",
]
