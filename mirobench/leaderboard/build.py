"""Build ``docs/leaderboard.json`` from the paper seed + ``experiments/`` CSVs.

For every ``experiments/<model>/<domain>/thread_scores.csv`` we re-run the
statistical comparison ourselves (never trusting a submitter-supplied
comparison file), filter to the 16 core metrics, and derive:

  * per (model, domain): how many core metrics have MWU p > 0.05  (out of 16)
  * per family: mean Wasserstein W1 and mean |Cliff's delta| across the
    family's metrics over the domains the entry covers

Paper entries (200 threads x 5 domains) whose raw CSVs are not in the repo are
read from ``experiments/_seed/paper.json`` at the aggregated level and merged in.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import mirobench
from mirobench.compare import compare_against_reference

from .families import (
    CORE_METRICS,
    DOMAIN_ORDER,
    FAMILIES,
    FAMILY_ORDER,
    METRIC_FAMILY,
    METRICS_PER_DOMAIN,
    load_baseline,
)
from .schema import validate_meta, validate_submission

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS_DIR = REPO_ROOT / "experiments"
SEED_PATH = EXPERIMENTS_DIR / "_seed" / "paper.json"
OUTPUT_PATH = REPO_ROOT / "docs" / "leaderboard.json"
_DATA_DIR = Path(mirobench.__file__).resolve().parent / "data"


def _reference_csv(domain: str) -> Path:
    return _DATA_DIR / domain / "reference_scores" / "thread_scores.csv"


def _family_means(rows: list[dict[str, Any]]) -> dict[str, dict[str, float] | None]:
    """Mean W1 and mean |delta| per family over the core-metric rows given."""
    bucket: dict[str, list[tuple[float, float]]] = {f: [] for f in FAMILY_ORDER}
    for r in rows:
        m = r.get("metric")
        if m not in CORE_METRICS:
            continue
        fam = METRIC_FAMILY[m]
        bucket[fam].append(
            (float(r.get("wasserstein", 0.0)), float(r.get("abs_cliffs_delta", 0.0)))
        )
    out: dict[str, dict[str, float] | None] = {}
    for fam in FAMILY_ORDER:
        vals = bucket[fam]
        if not vals:
            out[fam] = None
            continue
        w1 = sum(v[0] for v in vals) / len(vals)
        cd = sum(v[1] for v in vals) / len(vals)
        out[fam] = {"w1": w1, "cliffs": cd}
    return out


def _pass_count(rows: list[dict[str, Any]]) -> int:
    """Core metrics with MWU p > 0.05 (out of METRICS_PER_DOMAIN)."""
    return sum(
        1 for r in rows
        if r.get("metric") in CORE_METRICS
        and isinstance(r.get("mwu_p_value"), float)
        and r["mwu_p_value"] > 0.05
    )


def _discover_submission_dirs() -> list[Path]:
    """Model directories under experiments/, skipping _seed and dotfiles."""
    if not EXPERIMENTS_DIR.exists():
        return []
    return sorted(
        d for d in EXPERIMENTS_DIR.iterdir()
        if d.is_dir() and not d.name.startswith("_") and not d.name.startswith(".")
    )


def _build_computed_entry(model_dir: Path, log) -> dict[str, Any] | None:
    meta, meta_issues = validate_meta(model_dir / "meta.json")
    for i in meta_issues:
        log(f"  [{i.level}] {model_dir.name}/meta.json: {i.msg}")
    if any(i.level == "error" for i in meta_issues):
        return None

    domain_dirs = sorted(
        d for d in model_dir.iterdir()
        if d.is_dir() and (d / "thread_scores.csv").exists()
    )
    if not domain_dirs:
        log(f"  [warning] {model_dir.name}: no <domain>/thread_scores.csv found")
        return None

    domain_pass: dict[str, int] = {}
    domain_n: dict[str, int] = {}
    all_rows: list[dict[str, Any]] = []

    for dd in domain_dirs:
        domain = dd.name
        csv_path = dd / "thread_scores.csv"
        rep = validate_submission(csv_path, domain, model_dir.name)
        for w in rep.warnings:
            log(f"  [warning] {model_dir.name}/{domain}: {w}")
        if not rep.ok:
            for e in rep.errors:
                log(f"  [error] {model_dir.name}/{domain}: {e}")
            return None
        ref = _reference_csv(domain)
        if not ref.exists():
            log(f"  [error] {model_dir.name}/{domain}: reference not found {ref}")
            return None
        rows = compare_against_reference(csv_path, ref, domain=domain,
                                         model=meta["display_name"])
        rows = [r for r in rows if r.get("metric") in CORE_METRICS]
        all_rows.extend(rows)
        domain_pass[domain] = _pass_count(rows)
        domain_n[domain] = rep.n_threads
        log(f"  {model_dir.name}/{domain}: {domain_pass[domain]}/"
            f"{METRICS_PER_DOMAIN} pass  (n={rep.n_threads})")

    return {
        "model": meta["display_name"],
        "engine": meta["engine"],
        "tier": meta.get("tier", "community"),
        "submitter": meta.get("submitter"),
        "link": meta.get("link"),
        "date": meta.get("date"),
        "domains": [d.name for d in domain_dirs],
        "domain_pass": domain_pass,
        "domain_n": domain_n,
        "families": _family_means(all_rows),
    }


def _load_seed() -> dict[str, Any]:
    if not SEED_PATH.exists():
        return {"entries": [], "pending": []}
    return json.loads(SEED_PATH.read_text())


def build(log=print) -> dict[str, Any]:
    """Assemble the full leaderboard data object and return it."""
    seed = _load_seed()
    entries: list[dict[str, Any]] = list(seed.get("entries", []))
    seen = {(e["model"], e.get("tier", "paper")) for e in entries}

    log(f"Seed: {len(entries)} paper entr(y/ies).")
    log(f"Scanning {EXPERIMENTS_DIR} ...")
    for model_dir in _discover_submission_dirs():
        entry = _build_computed_entry(model_dir, log)
        if entry is None:
            continue
        key = (entry["model"], entry.get("tier", "community"))
        if key in seen:
            # A computed entry overrides a seed entry of the same name+tier.
            entries = [e for e in entries
                       if (e["model"], e.get("tier", "paper")) != key]
        entries.append(entry)
        seen.add(key)

    data = {
        "metrics_per_domain": METRICS_PER_DOMAIN,
        "domains": DOMAIN_ORDER,
        "family_order": FAMILY_ORDER,
        "families": {f: FAMILIES[f] for f in FAMILY_ORDER},
        "baseline": load_baseline(),
        "entries": entries,
        "pending": seed.get("pending", []),
    }
    return data


def write(data: dict[str, Any], path: Path = OUTPUT_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


def main(argv: list[str] | None = None) -> int:
    data = build()
    write(data)
    print(f"\nWrote {OUTPUT_PATH}  ({len(data['entries'])} entries, "
          f"{len(data['pending'])} pending)")
    return 0
