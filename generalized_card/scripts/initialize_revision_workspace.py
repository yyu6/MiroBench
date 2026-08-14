#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from generalized_card.audit import audit_generated_root  # noqa: E402
from generalized_card.core_contract import (  # noqa: E402
    REVISION_CORE_POLICY_VERSION,
    verify_core_contract,
)
from generalized_card.domain import REPO_ROOT, load_domain_config  # noqa: E402


REVISION_CORE_NAMES = (
    "revision_orchestrator",
    "revision_stage_runner",
    "reviser_adapter",
    "domain_prompt_adapter",
    "selfbleu_controller",
    "generalized_selfbleu_controller",
    "selfbleu_reviser",
    "selfbleu_backend",
    "selfbert_controller",
    "selfbert_reviser",
    "selfbert_backend",
    "text_metric_controller",
    "text_metric_reviser",
    "tone_controller",
    "tone_reviser",
    "tone_backend",
    "story_structure_controller",
    "story_reviser",
    "story_backend",
    "structure_reviser",
    "structure_backend",
    "revision_memory",
    "distribution_diagnostics",
    "output_audit",
    "cleanup",
    "score_runner",
    "matched_evaluator",
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a current-reviser workspace from an audited existing generation run."
    )
    parser.add_argument("--source-tag", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    source_root = REPO_ROOT / "artifacts" / "generalized_card" / "runs" / args.source_tag
    target_root = REPO_ROOT / "artifacts" / "generalized_card" / "runs" / args.tag
    if source_root.resolve() == target_root.resolve():
        raise SystemExit("--tag must differ from --source-tag")
    if target_root.exists():
        if args.resume and _same_source(target_root, source_root):
            source_summary = _load_json(source_root / "logs" / "token_usage_summary.json")
            imported = _synchronize_usage(
                source_root=source_root,
                target_root=target_root,
                target_tag=args.tag,
                source_summary=source_summary,
            )
            target_config_path = target_root / "run_config.json"
            target_config = _load_json(target_config_path)
            target_config["source_generation_usage_records"] = imported
            target_config["source_generation_usage_components"] = sorted(
                (source_summary.get("by_component") or {}).keys()
            )
            _write_json(target_config_path, target_config)
            elapsed = float(_load_json(target_root / "run_state.json").get("elapsed_seconds") or 0.0)
            _summarize_usage(target_root, elapsed)
            print(f"[revision-workspace-resume] {target_root}", flush=True)
            print(f"[revision-workspace-usage] imported_generation_records={imported}", flush=True)
            return
        raise SystemExit(f"Target workspace already exists: {target_root}")

    source_config = _load_json(source_root / "run_config.json")
    source_artifact = _load_json(source_root / "current_artifact.json")
    if not source_config or not source_artifact:
        raise SystemExit("Source run requires run_config.json and current_artifact.json")
    for key in ("root", "scores", "matched"):
        path = Path(str(source_artifact.get(key) or ""))
        if not path.exists():
            raise SystemExit(f"Source artifact is missing {key}: {path}")

    expected = int(source_config.get("max_posts") or 0)
    scores = Path(str(source_artifact["scores"]))
    matched = Path(str(source_artifact["matched"]))
    if _csv_rows(scores) != expected:
        raise SystemExit(f"Source score rows do not match expected posts: {_csv_rows(scores)}/{expected}")
    for name in ("matched_generated_thread_scores.csv", "matched_real_thread_scores.csv"):
        path = matched / name
        if _csv_rows(path) != expected:
            raise SystemExit(f"Source matched rows do not match expected posts: {path}")
    matched_eval = matched / "matched_seed_group_eval.json"
    if not matched_eval.exists():
        raise SystemExit(f"Source matched evaluation is missing: {matched_eval}")

    domain_name = str(source_config.get("domain_config") or source_config["domain"]["domain_id"])
    domain = load_domain_config(domain_name)
    generated_root = Path(str(source_config["generated_root"]))
    seed_pool = Path(str(source_config["seed_pool"]))
    audit = audit_generated_root(
        generated_root,
        config=domain,
        seed_pool=seed_pool,
        domain_profile=(
            Path(str(source_config["domain_profile"]))
            if source_config.get("domain_profile")
            else None
        ),
    )
    if not audit["healthy"]:
        raise SystemExit("Source generation failed the output audit: " + json.dumps(audit, ensure_ascii=False))

    revision_provenance = verify_core_contract(REVISION_CORE_NAMES)
    source_summary = _load_json(source_root / "logs" / "token_usage_summary.json")
    source_elapsed = float(source_summary.get("elapsed_seconds") or 0.0)
    source_usage = source_root / "logs" / "token_usage.jsonl"

    target_root.mkdir(parents=True)
    (target_root / "logs").mkdir()
    imported_usage_records = _synchronize_usage(
        source_root=source_root,
        target_root=target_root,
        target_tag=args.tag,
        source_summary=source_summary,
    )
    if source_summary:
        _write_json(target_root / "logs" / "source_generation_token_usage_summary.json", source_summary)

    source_policy = str(
        source_config.get("generator_policy_version")
        or source_config.get("card_core_policy_version")
        or "unversioned-legacy-generalized-generator"
    )
    config = {
        **source_config,
        "tag": args.tag,
        "run_kind": "revision_workspace",
        "source_generation": {
            "tag": args.source_tag,
            "root": str(source_root),
            "generator_profile": str(source_config.get("generator_profile") or "legacy-generalized-v2"),
            "generator_policy_version": source_policy,
            "run_config_sha256": _sha256(source_root / "run_config.json"),
            "generated_root": str(generated_root),
            "initial_artifact": source_artifact,
        },
        "source_generation_policy_version": source_policy,
        "revision_core_policy_version": REVISION_CORE_POLICY_VERSION,
        "revision_core_provenance": revision_provenance,
        "initialized_at_epoch": time.time(),
        "source_generation_usage_records": imported_usage_records,
        "source_generation_usage_components": sorted(
            (source_summary.get("by_component") or {}).keys()
        ),
    }
    # This workspace cannot be resumed as a generator run. Preserve the
    # generation command only under source_generation for provenance.
    config.pop("card_core_policy_version", None)
    config["source_generation"]["command"] = source_config.get("command")
    _write_json(target_root / "run_config.json", config)

    artifact = {
        **source_artifact,
        "stage": "imported_initial_evaluation",
        "revision_attempts": {},
        "source_generation_tag": args.source_tag,
        "revision_core_policy_version": REVISION_CORE_POLICY_VERSION,
        "updated_at_epoch": time.time(),
    }
    _write_json(target_root / "current_artifact.json", artifact)
    _write_json(target_root / "source_output_audit.json", audit)
    _write_json(
        target_root / "source_artifact_manifest.json",
        {
            "source_tag": args.source_tag,
            "source_root": str(source_root),
            "files": {
                "run_config": _file_record(source_root / "run_config.json"),
                "scores": _file_record(scores),
                "matched_eval": _file_record(matched_eval),
                "token_usage": _file_record(source_usage),
            },
            "revision_core_policy_version": REVISION_CORE_POLICY_VERSION,
            "revision_core_provenance": revision_provenance,
            "created_at_epoch": time.time(),
        },
    )
    _write_json(
        target_root / "run_state.json",
        {
            "status": "revision_workspace_initialized",
            "return_code": 0,
            "elapsed_seconds": source_elapsed,
            "source_generation_elapsed_seconds": source_elapsed,
            "updated_at_epoch": time.time(),
        },
    )
    _summarize_usage(target_root, source_elapsed)
    print(f"[revision-workspace] source={source_root}", flush=True)
    print(f"[revision-workspace] target={target_root}", flush=True)
    print(f"[revision-workspace] posts={expected} comments={audit['comments']}", flush=True)
    print(f"[revision-workspace] imported_generation_records={imported_usage_records}", flush=True)
    print(f"[revision-workspace] revision_policy={REVISION_CORE_POLICY_VERSION}", flush=True)


def _same_source(target_root: Path, source_root: Path) -> bool:
    config = _load_json(target_root / "run_config.json")
    return str((config.get("source_generation") or {}).get("root") or "") == str(source_root)


def _synchronize_usage(
    *,
    source_root: Path,
    target_root: Path,
    target_tag: str,
    source_summary: dict[str, Any],
) -> int:
    """Import generation usage while excluding historical source revisers."""

    allowed_source_components = set((source_summary.get("by_component") or {}).keys())
    source_rows = [
        row
        for row in _load_jsonl(source_root / "logs" / "token_usage.jsonl")
        if str(row.get("component") or "") in allowed_source_components
    ]
    target_rows = [
        row
        for row in _load_jsonl(target_root / "logs" / "token_usage.jsonl")
        if str(row.get("run_tag") or "") == target_tag
    ]
    path = target_root / "logs" / "token_usage.jsonl"
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in (*source_rows, *target_rows):
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)
    return len(source_rows)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _summarize_usage(target_root: Path, elapsed: float) -> None:
    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "summarize_token_usage.py"),
            str(target_root / "logs" / "token_usage.jsonl"),
            "--output",
            str(target_root / "logs" / "token_usage_summary.json"),
            "--elapsed-seconds",
            str(elapsed),
        ],
        cwd=REPO_ROOT,
        check=False,
    )


def _csv_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def _file_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.exists(),
        "sha256": _sha256(path) if path.exists() and path.is_file() else None,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


if __name__ == "__main__":
    main()
