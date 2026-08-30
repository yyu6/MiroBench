"""Invoke the vanilla OASIS/MiroFish simulation runner."""
from __future__ import annotations

import os
import subprocess
import sys

_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
_GEO_ROOT = os.path.dirname(_PACKAGE_DIR)
_MIROFISH_ROOT_CANDIDATES = [
    os.path.join(_GEO_ROOT, "third_party", "MiroFish"),
    os.path.join(_GEO_ROOT, "MiroFish"),
]


def _find_mirofish_root() -> str:
    for candidate in _MIROFISH_ROOT_CANDIDATES:
        if os.path.exists(os.path.join(candidate, "backend", "scripts", "run_reddit_simulation.py")):
            return candidate
    return _MIROFISH_ROOT_CANDIDATES[0]


def run_simulation(config_path: str, max_rounds: int) -> None:
    """Run one Reddit simulation via the restored vanilla OASIS runner."""
    mirofish_root = _find_mirofish_root()
    config_path = os.path.abspath(config_path)
    venv_python = os.path.join(mirofish_root, "backend", ".venv", "bin", "python")
    python = _find_python(venv_python)
    script = _vanilla_runner_script(mirofish_root)
    env = _build_simulation_env()
    _run_subprocess(
        python=python,
        script=script,
        config_path=config_path,
        max_rounds=max_rounds,
        env=env,
        banner="STARTING OASIS REDDIT SIMULATION (vanilla OASIS)",
    )


def _vanilla_runner_script(mirofish_root: str) -> str:
    """Return the restored upstream MiroFish runner path."""

    script = os.path.abspath(
        os.path.join(mirofish_root, "backend", "scripts", "run_reddit_simulation.py")
    )
    if not os.path.exists(script):
        raise FileNotFoundError(
            f"MiroFish simulation script not found at:\n  {script}\n"
            "Ensure MiroFish exists at GEO/third_party/MiroFish/ "
            "(or the legacy GEO/MiroFish/) and backend "
            "dependencies installed (cd third_party/MiroFish/backend && uv sync)."
        )
    return script


def _run_subprocess(
    python: str,
    script: str,
    config_path: str,
    max_rounds: int,
    env: dict[str, str],
    banner: str,
) -> None:
    """Execute one runner subprocess and surface non-zero exit codes."""

    cmd = [python, script, "--config", config_path, "--max-rounds", str(max_rounds), "--no-wait"]

    print(f"\n{'='*60}")
    print(banner)
    print(f"Script:  {script}")
    print(f"Config:  {config_path}")
    print(f"Rounds:  {max_rounds}")
    print(f"{'='*60}\n")

    result = subprocess.run(cmd, env=env, cwd=_GEO_ROOT)
    if result.returncode != 0:
        raise RuntimeError(
            f"MiroFish simulation exited with code {result.returncode}. "
            "Check logs in the simulation output directory."
        )


def _find_python(mirofish_venv_python: str) -> str:
    """Prefer MiroFish's own venv python; fall back to current interpreter."""
    if os.path.exists(mirofish_venv_python):
        return mirofish_venv_python
    return sys.executable


def _build_simulation_env() -> dict[str, str]:
    """Build the subprocess environment for the vendored runner."""
    env = os.environ.copy()
    scripts_path = os.path.join(_GEO_ROOT, "scripts")
    existing = env.get("PYTHONPATH", "")
    parts = [scripts_path]
    if existing:
        parts.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(parts)
    return env
