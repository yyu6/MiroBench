"""Validation for leaderboard submissions.

A submission lives at ``experiments/<model-slug>/<domain>/thread_scores.csv``
with a per-model ``experiments/<model-slug>/meta.json``. These checks run in CI
on every PR that touches ``experiments/`` so malformed entries fail fast with a
human-readable reason instead of silently skewing the board.
"""
from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

from .families import CORE_METRICS, DOMAIN_ORDER

# Submissions below this many usable threads are too noisy to compare; the
# README documents the >= 50 floor.
MIN_THREADS = 50

META_REQUIRED = ("display_name", "engine", "submitter")
META_OPTIONAL = ("date", "link", "tier", "notes")


@dataclass
class Issue:
    level: str  # "error" | "warning"
    msg: str


@dataclass
class SubmissionReport:
    path: Path
    model: str
    domain: str
    n_threads: int = 0
    core_present: set[str] = field(default_factory=set)
    issues: list[Issue] = field(default_factory=list)

    @property
    def errors(self) -> list[str]:
        return [i.msg for i in self.issues if i.level == "error"]

    @property
    def warnings(self) -> list[str]:
        return [i.msg for i in self.issues if i.level == "warning"]

    @property
    def ok(self) -> bool:
        return not self.errors

    def error(self, msg: str) -> None:
        self.issues.append(Issue("error", msg))

    def warn(self, msg: str) -> None:
        self.issues.append(Issue("warning", msg))


def _count_usable_threads(csv_path: Path) -> int:
    n = 0
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            tid = str(row.get("thread_id", ""))
            if tid and tid != "__summary_mean__":
                n += 1
    return n


def _numeric_core_columns(csv_path: Path) -> set[str]:
    """Core metrics that have at least one numeric (non-empty, non-NaN) value."""
    have: set[str] = set()
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            if str(row.get("thread_id", "")) == "__summary_mean__":
                continue
            for m in CORE_METRICS - have:
                v = row.get(m, "")
                if v in (None, ""):
                    continue
                try:
                    if not math.isnan(float(v)):
                        have.add(m)
                except (ValueError, TypeError):
                    pass
    return have


def validate_submission(csv_path: Path, domain: str, model: str) -> SubmissionReport:
    """Validate a single ``thread_scores.csv`` for one (model, domain)."""
    rep = SubmissionReport(path=csv_path, model=model, domain=domain)

    if domain not in DOMAIN_ORDER:
        rep.error(f"unknown domain '{domain}' (expected one of {DOMAIN_ORDER})")
        return rep
    if not csv_path.exists():
        rep.error(f"file not found: {csv_path}")
        return rep

    with open(csv_path) as f:
        header = next(csv.reader(f), [])
    if "thread_id" not in header:
        rep.error("CSV has no 'thread_id' column — not a thread_scores.csv?")
        return rep

    rep.n_threads = _count_usable_threads(csv_path)
    if rep.n_threads < MIN_THREADS:
        rep.error(
            f"only {rep.n_threads} usable threads (< {MIN_THREADS} floor); "
            "generate more before submitting"
        )

    rep.core_present = _numeric_core_columns(csv_path)
    missing = CORE_METRICS - rep.core_present
    if missing:
        rep.warn(
            f"{len(missing)} core metric(s) missing/empty and will count as FAIL "
            f"(out of {len(CORE_METRICS)}): {sorted(missing)}"
        )
    return rep


def validate_meta(meta_path: Path) -> tuple[dict, list[Issue]]:
    """Validate a per-model meta.json; returns (data, issues)."""
    issues: list[Issue] = []
    if not meta_path.exists():
        return {}, [Issue("error", f"missing meta.json: {meta_path}")]
    try:
        data = json.loads(meta_path.read_text())
    except json.JSONDecodeError as e:
        return {}, [Issue("error", f"invalid JSON in {meta_path}: {e}")]
    for key in META_REQUIRED:
        if not data.get(key):
            issues.append(Issue("error", f"meta.json missing required field '{key}'"))
    tier = data.get("tier", "community")
    if tier not in ("paper", "community"):
        issues.append(Issue("warning", f"tier '{tier}' not in (paper, community)"))
    return data, issues
