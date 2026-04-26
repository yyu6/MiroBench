"""
Runner for candidate simulations in the calibration module.

Provides ``run_candidates()``, which launches one ``run_discussion.py``
subprocess per candidate overlay, optionally in parallel via
``ProcessPoolExecutor``.
"""
from __future__ import annotations

import json
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from .overlay import save_overlay

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

# Fields from reference_run_config that map 1-to-1 to run_discussion.py flags.
_FIELD_TO_FLAG: dict[str, str] = {
    "agents": "--agents",
    "hours": "--hours",
    "rounds": "--rounds",
    "seed_posts": "--seed-posts",
    "seed": "--seed",
    "hint": "--hint",
    "few_shot_source": "--few-shot-source",
    "few_shot_count": "--few-shot-count",
    "few_shot_comments": "--few-shot-comments",
}


def _build_cmd(
    candidate_dir: Path,
    overlay_path: Path,
    reference_run_config: dict[str, Any],
    python: str,
    repo_root: Path,
) -> list[str]:
    """Build the subprocess command for one candidate."""
    script = repo_root / "run_discussion.py"
    output_dir = candidate_dir / "sim_output"

    cmd: list[str] = [python, str(script)]

    # Positional: input_file
    input_file = reference_run_config.get("input_file", "")
    cmd.append(str(input_file))

    # Named flags from reference config
    for field, flag in _FIELD_TO_FLAG.items():
        value = reference_run_config.get(field)
        if value is not None:
            cmd.extend([flag, str(value)])

    # Overlay and output
    cmd.extend(["--overlay", str(overlay_path)])
    cmd.extend(["--output-dir", str(output_dir)])

    return cmd


def _detect_sim_dir(output_dir: Path) -> Path | None:
    """Return the first subdirectory created inside *output_dir*, or None."""
    if not output_dir.exists():
        return None
    subdirs = [p for p in output_dir.iterdir() if p.is_dir()]
    if not subdirs:
        return None
    # Return the most-recently modified one (run_discussion creates a
    # timestamped directory).
    return max(subdirs, key=lambda p: p.stat().st_mtime)


def _run_one(
    candidate_id: int,
    overlay: dict[str, Any],
    iter_dir: Path,
    reference_run_config: dict[str, Any],
    python: str,
    repo_root: Path,
) -> dict[str, Any]:
    """Run a single candidate simulation and return the result dict."""
    candidate_dir = iter_dir / "candidates" / f"candidate_{candidate_id}"
    candidate_dir.mkdir(parents=True, exist_ok=True)

    overlay_path = candidate_dir / "overlay.json"
    save_overlay(overlay, overlay_path)

    cmd = _build_cmd(candidate_dir, overlay_path, reference_run_config, python, repo_root)
    output_dir = candidate_dir / "sim_output"

    proc = subprocess.run(cmd, capture_output=True)

    (candidate_dir / "stdout.log").write_bytes(proc.stdout or b"")
    (candidate_dir / "stderr.log").write_bytes(proc.stderr or b"")

    success = proc.returncode == 0
    sim_dir = _detect_sim_dir(output_dir) if success else None

    return {
        "candidate_id": candidate_id,
        "candidate_dir": str(candidate_dir),
        "sim_dir": str(sim_dir) if sim_dir is not None else None,
        "success": success,
        "returncode": proc.returncode,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_candidates(
    overlays: list[dict[str, Any]],
    iter_dir: Path,
    reference_run_config: dict[str, Any],
    parallel: int = 1,
    python: str = sys.executable,
    repo_root: Path | None = None,
) -> list[dict[str, Any]]:
    """Run candidate simulations, one per overlay.

    Parameters
    ----------
    overlays:
        List of overlay dicts; one subprocess is launched per entry.
    iter_dir:
        Root directory for this calibration iteration.  Candidate subdirs are
        created under ``iter_dir/candidates/candidate_N/``.
    reference_run_config:
        Dict with baseline run parameters forwarded to ``run_discussion.py``
        (keys: ``input_file``, ``agents``, ``hours``, ``rounds``,
        ``seed_posts``, ``seed``, ``hint``, ``few_shot_source``,
        ``few_shot_count``, ``few_shot_comments``).
    parallel:
        Number of workers.  Values > 1 use ``ProcessPoolExecutor``.
    python:
        Python executable to use (defaults to ``sys.executable``).
    repo_root:
        Repository root directory.  Defaults to the parent of this file's
        package (two levels up from ``calibration/runner.py``).

    Returns
    -------
    list[dict]
        One result dict per overlay, each containing:
        ``candidate_id``, ``candidate_dir``, ``sim_dir``,
        ``success``, ``returncode``.
    """
    if repo_root is None:
        # calibration/runner.py → calibration/ → repo root
        repo_root = Path(__file__).parent.parent

    iter_dir = Path(iter_dir)

    if parallel <= 1:
        results = []
        for i, overlay in enumerate(overlays):
            result = _run_one(
                candidate_id=i,
                overlay=overlay,
                iter_dir=iter_dir,
                reference_run_config=reference_run_config,
                python=python,
                repo_root=repo_root,
            )
            results.append(result)
        return results

    # Parallel execution
    futures: dict = {}
    results_map: dict[int, dict[str, Any]] = {}

    with ProcessPoolExecutor(max_workers=parallel) as executor:
        for i, overlay in enumerate(overlays):
            future = executor.submit(
                _run_one,
                candidate_id=i,
                overlay=overlay,
                iter_dir=iter_dir,
                reference_run_config=reference_run_config,
                python=python,
                repo_root=repo_root,
            )
            futures[future] = i

        for future in as_completed(futures):
            idx = futures[future]
            try:
                results_map[idx] = future.result()
            except Exception as exc:
                results_map[idx] = {
                    "candidate_id": idx,
                    "candidate_dir": str(
                        iter_dir / "candidates" / f"candidate_{idx}"
                    ),
                    "sim_dir": None,
                    "success": False,
                    "returncode": -1,
                    "error": str(exc),
                }

    # Return in original order
    return [results_map[i] for i in range(len(overlays))]
