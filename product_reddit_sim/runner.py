"""Invoke MiroFish run_reddit_simulation.py as a subprocess."""
from __future__ import annotations

import os
import subprocess
import sys

# Path to MiroFish script relative to this package (GEO/product_reddit_sim/../MiroFish/...)
_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
_GEO_ROOT = os.path.dirname(_PACKAGE_DIR)
MIROFISH_SCRIPT = os.path.join(
    _GEO_ROOT, "MiroFish", "backend", "scripts", "run_reddit_simulation.py"
)
MIROFISH_VENV_PYTHON = os.path.join(
    _GEO_ROOT, "MiroFish", "backend", ".venv", "bin", "python"
)


def run_simulation(config_path: str, max_rounds: int) -> None:
    """Run the OASIS Reddit simulation via MiroFish. Blocks until complete."""
    script = os.path.abspath(MIROFISH_SCRIPT)
    if not os.path.exists(script):
        raise FileNotFoundError(
            f"MiroFish simulation script not found at:\n  {script}\n"
            "Ensure MiroFish is cloned at GEO/MiroFish/ and backend "
            "dependencies installed (cd MiroFish/backend && uv sync)."
        )

    python = _find_python()

    cmd = [python, script, "--config", config_path,
           "--max-rounds", str(max_rounds), "--no-wait"]

    print(f"\n{'='*60}")
    print("STARTING OASIS REDDIT SIMULATION (MiroFish backbone)")
    print(f"Script:  {script}")
    print(f"Config:  {config_path}")
    print(f"Rounds:  {max_rounds}")
    print(f"{'='*60}\n")

    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError(
            f"MiroFish simulation exited with code {result.returncode}. "
            "Check logs in the simulation output directory."
        )


def _find_python() -> str:
    """Prefer MiroFish's own venv python; fall back to current interpreter."""
    if os.path.exists(MIROFISH_VENV_PYTHON):
        return MIROFISH_VENV_PYTHON
    return sys.executable
