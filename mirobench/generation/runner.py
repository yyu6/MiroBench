"""Dispatch a Reddit simulation through the vanilla OASIS runner or the
GEO-patched runtime, in-process.

This module replaces the previous subprocess-based MiroFish runner. It now
runs everything inside the current Python process via
``mirobench.generation.oasis_runner``.

``discussion_backbone="vanilla_oasis"`` runs the upstream baseline. The
GEO patch layer is not imported and OASIS Reddit internals stay untouched.

``discussion_backbone="geo_patched"`` applies GEO's monkeypatches via
``mirobench.generation.oasis_runner_patch.apply_geo_runner_patch`` before
starting the simulation.
"""
from __future__ import annotations

from typing import Optional


def run_simulation(
    config_path: str,
    max_rounds: int,
    discussion_backbone: str = "vanilla_oasis",
) -> None:
    """Run one Reddit simulation.

    Args:
        config_path: Absolute path to ``simulation_config.json``.
        max_rounds: Hard cap on the number of simulation rounds.
        discussion_backbone: ``"vanilla_oasis"`` (default) for the upstream
            baseline, or ``"geo_patched"`` to enable GEO's runtime patches.
    """

    from .oasis_runner import run_simulation_in_process

    if discussion_backbone == "vanilla_oasis":
        banner = "OASIS Reddit simulation (vanilla baseline)"
        apply_patch = False
    elif discussion_backbone == "geo_patched":
        banner = "OASIS Reddit simulation (GEO-patched backbone)"
        apply_patch = True
    else:
        raise ValueError(
            "discussion_backbone must be one of: vanilla_oasis, geo_patched"
        )

    print("\n" + "=" * 60)
    print(banner)
    print(f"Config: {config_path}")
    print(f"Rounds: {max_rounds}")
    print("=" * 60 + "\n")

    run_simulation_in_process(
        config_path=config_path,
        max_rounds=max_rounds,
        apply_geo_patch=apply_patch,
    )
