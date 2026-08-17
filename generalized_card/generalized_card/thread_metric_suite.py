from __future__ import annotations

import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd


SUMMARY_FILENAME = "thread_metrics_summary.csv"
SUMMARY_MEAN_THREAD_ID = "__summary_mean__"
METRIC_COMMANDS = (
    ("stance_disagreement_results.json", "score_thread_disagreement.py", True),
    ("self_bleu_results.json", "score_thread_self_bleu.py", False),
    ("self_bertscore_results.json", "score_thread_self_bertscore.py", True),
    ("semantic_uniformity_results.json", "score_thread_semantic_uniformity.py", True),
    ("storyseeker_results.json", "score_thread_storyseeker.py", True),
    ("go_emotions_results.json", "score_thread_go_emotions.py", True),
    ("politeness_results.json", "score_thread_politeness.py", True),
    ("thread_structure_results.json", "score_thread_structure.py", False),
    ("detoxify_results.json", "score_thread_detoxify.py", True),
)


def score_thread_metric_suite(
    sim_dir: Path,
    *,
    python: str,
    repo_root: Path,
    device: str = "cpu",
    metric_parallel: int = 2,
) -> None:
    """Run every scorer once, validate its JSON, then rebuild the summary CSV."""

    scripts = repo_root / "scripts" / "evaluation"
    commands = metric_commands(
        scripts=scripts,
        sim_dir=sim_dir,
        python=python,
        device=device,
    )
    pending = [row for row in commands if not metric_output_is_valid(sim_dir / row[0])]
    if pending:
        run_metric_commands(pending, sim_dir=sim_dir, max_workers=metric_parallel)
    invalid = [
        name for name, _ in commands if not metric_output_is_valid(sim_dir / name)
    ]
    if invalid:
        raise RuntimeError(
            f"Metric scoring did not produce valid JSON under {sim_dir}: "
            + ", ".join(invalid)
        )
    subprocess.run(
        [python, str(scripts / "summarize_thread_metrics.py"), str(sim_dir)],
        check=True,
    )


def metric_commands(
    *,
    scripts: Path,
    sim_dir: Path,
    python: str,
    device: str,
) -> list[tuple[str, list[str]]]:
    commands = []
    for output, script, uses_device in METRIC_COMMANDS:
        command = [
            python,
            str(scripts / script),
            str(sim_dir),
            "--target-kind",
            "generated",
        ]
        if uses_device:
            command.extend(["--device", device])
        commands.append((output, command))
    return commands


def run_metric_commands(
    commands: list[tuple[str, list[str]]],
    *,
    sim_dir: Path,
    max_workers: int,
) -> None:
    """Run scorers concurrently, retrying failures once with one native thread."""

    parallel_env = os.environ.copy()
    parallel_env.setdefault("KMP_USE_SHM", "0")
    parallel_env.setdefault("TOKENIZERS_PARALLELISM", "false")
    log_dir = sim_dir / "metric_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    def execute(
        item: tuple[str, list[str]],
    ) -> tuple[str, subprocess.CompletedProcess[bytes]]:
        name, command = item
        return name, subprocess.run(command, capture_output=True, env=parallel_env)

    failed: list[tuple[str, list[str]]] = []
    with ThreadPoolExecutor(max_workers=max(1, max_workers)) as pool:
        futures = {pool.submit(execute, item): item for item in commands}
        for future in as_completed(futures):
            name, process = future.result()
            _, command = futures[future]
            write_process_logs(log_dir, name, process)
            if process.returncode or not metric_output_is_valid(sim_dir / name):
                failed.append((name, command))
    if failed:
        retry_failed_metrics(failed, sim_dir=sim_dir, log_dir=log_dir)


def retry_failed_metrics(
    failed: list[tuple[str, list[str]]],
    *,
    sim_dir: Path,
    log_dir: Path,
) -> None:
    retry_env = os.environ.copy()
    retry_env.update(
        {
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "KMP_USE_SHM": "0",
        }
    )
    unresolved = []
    for name, command in failed:
        output = sim_dir / name
        output.unlink(missing_ok=True)
        process = subprocess.run(command, capture_output=True, env=retry_env)
        write_process_logs(log_dir, name, process, prefix="retry.")
        if process.returncode or not metric_output_is_valid(output):
            unresolved.append(name)
    if unresolved:
        raise RuntimeError(
            "Metric scoring failed after serial retry: "
            f"{', '.join(unresolved)}; logs={log_dir}"
        )


def write_process_logs(
    log_dir: Path,
    output_name: str,
    process: subprocess.CompletedProcess[bytes],
    *,
    prefix: str = "",
) -> None:
    stem = output_name.removesuffix(".json")
    (log_dir / f"{stem}.{prefix}stdout.log").write_bytes(process.stdout or b"")
    (log_dir / f"{stem}.{prefix}stderr.log").write_bytes(process.stderr or b"")


def metric_output_is_valid(path: Path) -> bool:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return isinstance(payload, (dict, list)) and bool(payload)


def load_thread_metrics(directory: Path) -> pd.DataFrame:
    """Load a summary CSV, excluding the synthetic summary-mean row."""

    direct = directory / SUMMARY_FILENAME
    if direct.exists():
        return without_summary_mean(pd.read_csv(direct))
    frames = []
    for child in sorted(directory.iterdir()):
        path = child / SUMMARY_FILENAME
        if not child.is_dir() or not path.exists():
            continue
        frame = without_summary_mean(pd.read_csv(path))
        frame["_product_dir"] = child.name
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(
            f"No {SUMMARY_FILENAME} found in {directory} or its immediate children"
        )
    return pd.concat(frames, ignore_index=True)


def without_summary_mean(frame: pd.DataFrame) -> pd.DataFrame:
    if "thread_id" not in frame.columns:
        return frame
    return frame[frame["thread_id"].astype(str) != SUMMARY_MEAN_THREAD_ID]
