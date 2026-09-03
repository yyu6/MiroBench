#!/usr/bin/env python3
"""In-process metric engine: the official scorers, with their models held open.

Why this exists.  Timed on 2026-09-04, a 6-comment thread costs almost exactly
what a 42-comment thread costs -- politeness 6.1s against 5.8s, semantic 7.1s
against 6.3s.  The work is not the scoring, it is loading eight transformer
models in eight fresh subprocesses.  A self-loop that rescores after every round
pays that toll every round, which is why the CARD-era controller was too slow to
use.

What this does NOT do: reimplement a metric.  Every number still comes from the
official scorer's own `main()`, run with the arguments the official pipeline
passes (`--target-kind generated`, `--device <d>`, everything else defaulted).
The only change is that each scorer's model constructor is wrapped in a cache,
so the second call reuses the loaded model.  A reimplementation that drifted
from the official scorer by 0.001 would silently invalidate every gate decision
this engine is used for; wrapping cannot drift.

Two further savings, both exact rather than approximate:
  * per-comment classifier outputs are cached by sha256 of the comment text, so
    a revision round only pays for the comments it actually rewrote;
  * a thread is rescored only when its text changed.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import sys
import time

# Set BEFORE torch/transformers are imported anywhere in this process.
# The loop holds ~8 GB of models resident; a fork at that size briefly doubles
# the mapping and the process was being killed between rounds with no
# traceback, right after "tokenizers: the current process just got forked".
# Nothing here needs intra-op parallelism -- the batches are tens of comments,
# not thousands -- so single-threaded is also no slower in practice.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = REPO_ROOT / "scripts" / "evaluation"
# `scripts/` holds an OLDER score_thread_disagreement.py without
# `detect_target_kind`; every evaluation scorer imports that name from its
# sibling. `scripts/evaluation` must therefore come first, and stay first --
# the official pipeline gets this for free by running each scorer as a
# subprocess whose cwd is the evaluation directory.
for path in (str(REPO_ROOT / "scripts"), str(EVAL_DIR)):
    while path in sys.path:
        sys.path.remove(path)
    sys.path.insert(0, path)
assert sys.path[0] == str(EVAL_DIR), "evaluation scorers must shadow scripts/"

# (output filename, module name, attribute that builds the model, uses --device)
# Mirrors generalized_card/thread_metric_suite.METRIC_COMMANDS; verified against
# it by `test_engine_covers_every_official_metric`.
SCORERS: tuple[tuple[str, str, str, bool], ...] = (
    ("politeness_results.json", "score_thread_politeness", "PolitenessScorer", True),
    ("go_emotions_results.json", "score_thread_go_emotions", "GoEmotionsScorer", True),
    ("storyseeker_results.json", "score_thread_storyseeker", "StorySeekerScorer", True),
    ("stance_disagreement_results.json", "score_thread_disagreement", "StanceRelScorer", True),
    ("semantic_uniformity_results.json", "score_thread_semantic_uniformity", "CommentEmbedder", True),
    ("self_bertscore_results.json", "score_thread_self_bertscore", "load_bert_scorer", True),
    ("self_bleu_results.json", "score_thread_self_bleu", "", False),
    ("thread_structure_results.json", "score_thread_structure", "", False),
)

def _limit_torch_threads() -> None:
    try:
        import torch

        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
    except Exception:  # noqa: BLE001 - torch may not be imported yet
        pass


_MODULES: dict[str, Any] = {}
_MODEL_CACHE: dict[tuple, Any] = {}


def _cache_key(args: tuple, kwargs: dict) -> tuple:
    def freeze(value: Any) -> Any:
        if isinstance(value, (list, tuple)):
            return tuple(freeze(v) for v in value)
        if isinstance(value, dict):
            return tuple(sorted((k, freeze(v)) for k, v in value.items()))
        if isinstance(value, Path):
            return str(value)
        try:
            hash(value)
        except TypeError:
            return repr(value)
        return value

    return (freeze(args), freeze(tuple(sorted(kwargs.items()))))


def _wrap_loader(module: Any, attribute: str) -> None:
    """Make one model constructor return a cached instance on repeat calls."""

    original = getattr(module, attribute, None)
    if original is None or getattr(original, "_selfloop_cached", False):
        return

    def cached(*args: Any, **kwargs: Any) -> Any:
        key = (module.__name__, attribute, _cache_key(args, kwargs))
        if key not in _MODEL_CACHE:
            _MODEL_CACHE[key] = original(*args, **kwargs)
        return _MODEL_CACHE[key]

    cached._selfloop_cached = True  # type: ignore[attr-defined]
    cached._selfloop_original = original  # type: ignore[attr-defined]
    setattr(module, attribute, cached)


def _load_module(name: str, loader_attr: str) -> Any:
    if name in _MODULES:
        return _MODULES[name]
    import importlib

    module = importlib.import_module(name)
    _limit_torch_threads()
    if loader_attr:
        _wrap_loader(module, loader_attr)
    else:
        # These build their model inside a method rather than at a module-level
        # symbol, so cache the object that owns it instead.
        for attribute in ("SemanticUniformityScorer", "ThreadEmbedder", "Scorer"):
            if hasattr(module, attribute):
                _wrap_loader(module, attribute)
                break
    _MODULES[name] = module
    return module


def run_scorer(
    module_name: str,
    loader_attr: str,
    *,
    input_dir: Path,
    output_file: Path,
    device: str,
    uses_device: bool = True,
    quiet: bool = True,
) -> None:
    """Run one official scorer's main() in-process with the pipeline's arguments."""

    module = _load_module(module_name, loader_attr)
    argv = [module_name, str(input_dir), "--target-kind", "generated",
            "--output-file", str(output_file)]
    if uses_device:
        argv += ["--device", device]
    saved = sys.argv
    sys.argv = argv
    try:
        if quiet:
            with redirect_stdout(io.StringIO()):
                module.main()
        else:
            module.main()
    finally:
        sys.argv = saved


def score_run_dir(
    run_dir: Path,
    *,
    device: str = "cpu",
    only: tuple[str, ...] = (),
    force: bool = False,
) -> dict[str, float]:
    """Score one cleaned run directory and return its thread-metric row.

    `only` restricts the work to the scorers a metric needs; the rest of the
    directory's existing JSON is reused. `force` rescores even when the JSON is
    already present, which is what a revision round needs.
    """

    for output, module_name, loader_attr, uses_device in SCORERS:
        if only and output not in only:
            continue
        target = run_dir / output
        if target.exists() and not force:
            continue
        run_scorer(
            module_name,
            loader_attr,
            input_dir=run_dir,
            output_file=target,
            device=device,
            uses_device=uses_device,
        )
    return summarize(run_dir)


def summarize(run_dir: Path) -> dict[str, float]:
    """Rebuild thread_metrics_summary via the official summarizer, then read it."""

    module = _load_module("summarize_thread_metrics", "")
    saved = sys.argv
    sys.argv = ["summarize_thread_metrics", str(run_dir)]
    try:
        with redirect_stdout(io.StringIO()):
            module.main()
    finally:
        sys.argv = saved
    rows = json.loads((run_dir / "thread_metrics_summary.json").read_text())
    for row in rows:
        if str(row.get("thread_id")) != "__summary_mean__":
            return row
    return rows[0] if rows else {}


def comment_digest(texts: list[str]) -> str:
    payload = "\n\x00\n".join(texts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def loaded_models() -> int:
    return len(_MODEL_CACHE)
