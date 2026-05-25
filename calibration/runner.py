"""
Runner for candidate simulations in the calibration module.

Provides ``run_candidates()``, which launches one ``mirobench generate``
subprocess per candidate overlay, optionally in parallel via
``ProcessPoolExecutor``, then scores each sim dir with the full metric suite.
"""
from __future__ import annotations

import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from .overlay import save_overlay

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

# Fields from reference_run_config that map 1-to-1 to `mirobench generate` flags.
_FIELD_TO_FLAG: dict[str, str] = {
    "agents": "--agents",
    "hours": "--hours",
    "rounds": "--rounds",
    "seed_posts": "--seed-posts",
    "seed": "--seed",
    "hint": "--hint",
    "discussion_backbone": "--discussion-backbone",
    "few_shot_source": "--few-shot-source",
    "few_shot_count": "--few-shot-count",
    "few_shot_comments": "--few-shot-comments",
    "few_shot_thread_ids": "--few-shot-thread-ids",
}
_SUMMARY_MEAN_THREAD_ID = "__summary_mean__"


def _build_cmd(
    candidate_dir: Path,
    overlay_path: Path,
    reference_run_config: dict[str, Any],
    python: str,
    repo_root: Path,
) -> list[str]:
    """Build the subprocess command for one candidate.

    ``repo_root`` is kept in the signature for backward compatibility with
    callers but is no longer used — we invoke ``mirobench generate`` via the
    installed console script's module path, which works regardless of cwd.
    """
    output_dir = candidate_dir / "sim_output"

    cmd: list[str] = [python, "-m", "mirobench.cli", "generate"]

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
    # Return the most-recently modified one (mirobench generate creates a
    # timestamped directory).
    return max(subdirs, key=lambda p: p.stat().st_mtime)


def _sim_dir_looks_reusable(sim_dir: Path) -> bool:
    """Return True when *sim_dir* looks complete enough to resume from."""
    return (
        (sim_dir / "discussion.json").exists()
        and (
            (sim_dir / "thread_metrics_summary.csv").exists()
            or (sim_dir / "simulation_config.json").exists()
            or (sim_dir / "run_config.json").exists()
        )
    )


def _find_reusable_sim_dir(output_dir: Path) -> Path | None:
    """Return the most recent reusable simulation subdirectory, if any."""
    if not output_dir.exists():
        return None
    subdirs = sorted(
        (p for p in output_dir.iterdir() if p.is_dir()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for sim_dir in subdirs:
        if _sim_dir_looks_reusable(sim_dir):
            return sim_dir
    return None


def _reuse_existing_result(candidate_id: int, iter_dir: Path) -> dict[str, Any] | None:
    """Reuse an existing candidate simulation directory when resuming."""
    candidate_dir = iter_dir / "candidates" / f"candidate_{candidate_id}"
    sim_dir = _find_reusable_sim_dir(candidate_dir / "sim_output")
    if sim_dir is None:
        return None
    return {
        "candidate_id": candidate_id,
        "candidate_dir": str(candidate_dir),
        "sim_dir": str(sim_dir),
        "success": True,
        "returncode": 0,
        "reused_existing": True,
    }


def _score_sim_dir(
    sim_dir: Path,
    python: str,
    repo_root: Path,
    device: str = "cpu",
    metric_parallel: int = 2,
) -> None:
    """Run the full metric suite on *sim_dir* and produce thread_metrics_summary.csv.

    Mirrors the logic in scripts/evaluation/run_baseline_evaluation.py.
    Individual metric JSONs are skipped if they already exist (idempotent).
    The summarize step always runs so the CSV is up-to-date.

    Parameters
    ----------
    metric_parallel : int
        Maximum number of metric scripts to run concurrently (default 2).
    """
    script_dir = repo_root / "scripts" / "evaluation"
    summarize_script = script_dir / "summarize_thread_metrics.py"
    td = str(sim_dir)

    metric_commands: list[tuple[str, list[str]]] = [
        (
            "stance_disagreement_results.json",
            [python, str(script_dir / "score_thread_disagreement.py"), td,
             "--target-kind", "generated", "--device", device],
        ),
        (
            "self_bleu_results.json",
            [python, str(script_dir / "score_thread_self_bleu.py"), td,
             "--target-kind", "generated"],
        ),
        (
            "self_bertscore_results.json",
            [python, str(script_dir / "score_thread_self_bertscore.py"), td,
             "--target-kind", "generated", "--device", device],
        ),
        (
            "semantic_uniformity_results.json",
            [python, str(script_dir / "score_thread_semantic_uniformity.py"), td,
             "--target-kind", "generated", "--device", device],
        ),
        (
            "storyseeker_results.json",
            [python, str(script_dir / "score_thread_storyseeker.py"), td,
             "--target-kind", "generated", "--device", device],
        ),
        (
            "go_emotions_results.json",
            [python, str(script_dir / "score_thread_go_emotions.py"), td,
             "--target-kind", "generated", "--device", device],
        ),
        (
            "politeness_results.json",
            [python, str(script_dir / "score_thread_politeness.py"), td,
             "--target-kind", "generated", "--device", device],
        ),
        (
            "thread_structure_results.json",
            [python, str(script_dir / "score_thread_structure.py"), td,
             "--target-kind", "generated"],
        ),
        (
            "detoxify_results.json",
            [python, str(script_dir / "score_thread_detoxify.py"), td,
             "--target-kind", "generated", "--device", device],
        ),
    ]

    # Filter out already-computed metrics
    pending = [
        (name, cmd) for name, cmd in metric_commands
        if not (sim_dir / name).exists()
    ]

    if pending:
        _run_metric_commands_parallel(
            pending,
            sim_dir=sim_dir,
            max_workers=metric_parallel,
        )

    # Always re-summarize to produce/refresh thread_metrics_summary.csv.
    subprocess.run([python, str(summarize_script), td], check=True)


def _run_metric_commands_parallel(
    commands: list[tuple[str, list[str]]],
    sim_dir: Path,
    max_workers: int = 2,
) -> None:
    """Run metric scoring commands with bounded parallelism.

    Launches up to *max_workers* subprocesses concurrently. Raises
    ``subprocess.CalledProcessError`` on the first failure.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _run(item: tuple[str, list[str]]) -> tuple[str, subprocess.CompletedProcess]:
        name, cmd = item
        proc = subprocess.run(cmd, capture_output=True)
        return name, proc

    metric_log_dir = sim_dir / "metric_logs"
    metric_log_dir.mkdir(parents=True, exist_ok=True)

    failed: list[str] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_run, item): item[0] for item in commands}
        for future in as_completed(futures):
            name, proc = future.result()
            stem = name.removesuffix(".json")
            (metric_log_dir / f"{stem}.stdout.log").write_bytes(proc.stdout or b"")
            (metric_log_dir / f"{stem}.stderr.log").write_bytes(proc.stderr or b"")
            if proc.returncode != 0:
                failed.append(name)
                stderr_text = (proc.stderr or b"").decode("utf-8", errors="replace").strip()
                last_line = stderr_text.splitlines()[-1] if stderr_text else ""
                detail = f": {last_line}" if last_line else ""
                print(
                    f"  [WARN] metric {name} failed (rc={proc.returncode}){detail}, continuing"
                )
    if failed:
        print(
            f"  [WARN] {len(failed)}/{len(commands)} metric(s) failed: {', '.join(failed)}"
        )
        print(f"  [WARN] metric stderr/stdout logs saved under {metric_log_dir}")


def _run_sim_only(
    candidate_id: int,
    overlay: dict[str, Any],
    iter_dir: Path,
    reference_run_config: dict[str, Any],
    python: str,
    repo_root: Path,
    simulation_reasoning_effort: str | None = None,
) -> dict[str, Any]:
    """Run a single candidate simulation (no scoring). Returns result dict."""
    candidate_dir = iter_dir / "candidates" / f"candidate_{candidate_id}"
    candidate_dir.mkdir(parents=True, exist_ok=True)

    overlay_path = candidate_dir / "overlay.json"
    save_overlay(overlay, overlay_path)

    cmd = _build_cmd(candidate_dir, overlay_path, reference_run_config, python, repo_root)
    output_dir = candidate_dir / "sim_output"

    env = os.environ.copy()
    if simulation_reasoning_effort:
        env["LLM_REASONING_EFFORT"] = str(simulation_reasoning_effort)
    else:
        env.pop("LLM_REASONING_EFFORT", None)

    proc = subprocess.run(cmd, capture_output=True, env=env)

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


def _score_result(
    result: dict[str, Any],
    python: str,
    repo_root: Path,
    device: str = "cpu",
    metric_parallel: int = 2,
) -> dict[str, Any]:
    """Score a simulation result in-place. Mutates and returns *result*."""
    sim_dir = result.get("sim_dir")
    if sim_dir is None or not result.get("success"):
        return result
    try:
        _score_sim_dir(
            Path(sim_dir), python=python, repo_root=repo_root,
            device=device, metric_parallel=metric_parallel,
        )
    except subprocess.CalledProcessError as exc:
        candidate_dir = Path(result["candidate_dir"])
        (candidate_dir / "scoring_error.log").write_text(str(exc), encoding="utf-8")
        result["success"] = False
        result["sim_dir"] = None
    return result


def _run_one(
    candidate_id: int,
    overlay: dict[str, Any],
    iter_dir: Path,
    reference_run_config: dict[str, Any],
    python: str,
    repo_root: Path,
    device: str = "cpu",
    metric_parallel: int = 2,
    simulation_reasoning_effort: str | None = None,
) -> dict[str, Any]:
    """Run a single candidate simulation then score it. Convenience wrapper."""
    result = _run_sim_only(
        candidate_id=candidate_id, overlay=overlay, iter_dir=iter_dir,
        reference_run_config=reference_run_config, python=python, repo_root=repo_root,
        simulation_reasoning_effort=simulation_reasoning_effort,
    )
    return _score_result(
        result, python=python, repo_root=repo_root,
        device=device, metric_parallel=metric_parallel,
    )


def _count_threads(sim_dir: str | Path) -> int:
    """Count threads from a simulation's thread_metrics_summary.csv."""
    csv_path = Path(sim_dir) / "thread_metrics_summary.csv"
    if not csv_path.exists():
        return 0
    import pandas as _pd
    df = _pd.read_csv(csv_path)
    if "thread_id" in df.columns:
        df = df[df["thread_id"].astype(str) != _SUMMARY_MEAN_THREAD_ID]
    return len(df)


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
    device: str = "cpu",
    seed_offsets: list[int] | None = None,
    min_threads: int = 0,
    metric_parallel: int = 2,
    batch_schedule: list[int] | None = None,
    simulation_reasoning_effort: str | None = None,
    reuse_existing: bool = False,
) -> list[dict[str, Any]]:
    """Run candidate simulations, score each, one per overlay.

    Parameters
    ----------
    overlays:
        List of overlay dicts; one subprocess is launched per entry.
    iter_dir:
        Root directory for this calibration iteration.  Candidate subdirs are
        created under ``iter_dir/candidates/candidate_N/``.
    reference_run_config:
        Dict with baseline run parameters forwarded to ``mirobench generate``
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
    device:
        Torch device for metric scripts (``cpu``, ``cuda``, ``mps``).
    seed_offsets:
        Optional per-candidate seed offsets.  When provided, candidate *i*
        runs with ``base_seed + seed_offsets[i]`` instead of the base seed.
        Use this for multi-run evaluations of the same overlay so each run
        gets a different seed.
    min_threads:
        Early-stop threshold.  When > 0 and running sequentially, stop
        launching new candidates once the cumulative thread count across
        all successful runs reaches this value.  Ignored in parallel mode.

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

    def _config_for(i: int) -> dict[str, Any]:
        """Return reference_run_config, optionally with seed offset applied."""
        if seed_offsets is None:
            return reference_run_config
        cfg = dict(reference_run_config)
        base_seed = cfg.get("seed", 42)
        cfg["seed"] = base_seed + seed_offsets[i]
        return cfg

    # ── Batched execution: simulations parallel, scoring sequential ──
    #
    # Simulations are API-bound (network IO) → safe to parallelize.
    # Metric scoring is GPU/CPU-bound → run one-at-a-time per candidate
    # (within each scoring run, metric_parallel controls sub-parallelism).
    #
    # batch_schedule: how many sims to launch per round.
    #   Phase 1 (calibration): [5] — all candidates at once
    #   Phase 2 (post-cal):    [4, 3, 2, 1, ...] — decreasing, with early-stop
    #   Default (no schedule):  [1, 1, ...] or [parallel, parallel, ...]

    if parallel <= 1 and not batch_schedule:
        batch_schedule = [1] * len(overlays)
    elif not batch_schedule:
        batch_schedule = [parallel] * len(overlays)

    from concurrent.futures import ThreadPoolExecutor

    results: list[dict[str, Any]] = []
    cumulative_threads = 0
    idx = 0

    for batch_size in batch_schedule:
        if idx >= len(overlays):
            break
        if min_threads > 0 and cumulative_threads >= min_threads:
            print(f"  [early stop] {cumulative_threads} threads collected (>= {min_threads})")
            break

        batch_end = min(idx + batch_size, len(overlays))
        batch_indices = list(range(idx, batch_end))
        reused_results: dict[int, dict[str, Any]] = {}
        pending_indices: list[int] = []
        for i in batch_indices:
            reused = _reuse_existing_result(i, iter_dir) if reuse_existing else None
            if reused is not None:
                reused_results[i] = reused
            else:
                pending_indices.append(i)

        # ── Step 1: Run simulations in parallel (API calls) ──
        sim_results: dict[int, dict[str, Any]] = dict(reused_results)
        if len(pending_indices) == 1:
            i = pending_indices[0]
            sim_results[i] = _run_sim_only(
                candidate_id=i, overlay=overlays[i], iter_dir=iter_dir,
                reference_run_config=_config_for(i), python=python, repo_root=repo_root,
                simulation_reasoning_effort=simulation_reasoning_effort,
            )
        elif pending_indices:
            with ThreadPoolExecutor(max_workers=len(pending_indices)) as pool:
                futures = {}
                for i in pending_indices:
                    future = pool.submit(
                        _run_sim_only,
                        candidate_id=i, overlay=overlays[i], iter_dir=iter_dir,
                        reference_run_config=_config_for(i), python=python,
                        repo_root=repo_root,
                        simulation_reasoning_effort=simulation_reasoning_effort,
                    )
                    futures[future] = i
                for future in as_completed(futures):
                    i = futures[future]
                    try:
                        sim_results[i] = future.result()
                    except Exception as exc:
                        sim_results[i] = {
                            "candidate_id": i,
                            "candidate_dir": str(iter_dir / "candidates" / f"candidate_{i}"),
                            "sim_dir": None, "success": False,
                            "returncode": -1, "error": str(exc),
                        }

        # ── Step 2: Score sequentially (GPU/CPU bound) ──
        for i in batch_indices:
            result = _score_result(
                sim_results[i], python=python, repo_root=repo_root,
                device=device, metric_parallel=metric_parallel,
            )
            results.append(result)
            if result["success"] and result["sim_dir"]:
                cumulative_threads += _count_threads(result["sim_dir"])

        idx = batch_end

    return results
