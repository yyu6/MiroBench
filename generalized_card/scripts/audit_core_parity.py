#!/usr/bin/env python3
"""Audit generalized CARD against its pinned CARD core without API calls."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from generalized_card.backend import (  # noqa: E402
    CARD_SNAPSHOT_BACKEND,
    CORE_ALGORITHM_SYMBOLS,
    GENERATOR_PROFILES,
    GENERALIZED_V2_BACKEND,
    GENERALIZED_V2_PROFILE,
    configure_generator_backend,
    load_generator_backend,
)
from generalized_card.core_contract import (  # noqa: E402
    CORE_FILES,
    CORE_POLICY_VERSION,
    CURRENT_ACTIVE_CORE_NAMES,
    REVISION_CORE_POLICY_VERSION,
    verify_core_contract,
)
from generalized_card.domain import load_domain_config  # noqa: E402
from generalized_card.reviser_backend import (  # noqa: E402
    configure_reviser_backend,
    load_reviser_backend,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", default="camera")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--include-legacy",
        action="store_true",
        help="Also load the historical CARD snapshot and reviser adapters.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_domain_config(args.domain)
    core_names = tuple(CORE_FILES) if args.include_legacy else CURRENT_ACTIVE_CORE_NAMES
    core = verify_core_contract(core_names)
    profiles = GENERATOR_PROFILES if args.include_legacy else (GENERALIZED_V2_PROFILE,)
    generators = {
        profile: configure_generator_backend(
            load_generator_backend(profile=profile),
            config,
            profile=profile,
        ).GENERALIZED_CARD_PARITY
        for profile in profiles
    }
    revisers: dict[str, dict[str, Any]] = {}
    if args.include_legacy:
        for kind in ("selfbleu", "selfbert", "tone", "story", "structure"):
            module = configure_reviser_backend(
                load_reviser_backend(kind),
                kind=kind,
                config=config,
            )
            revisers[kind] = module.GENERALIZED_CARD_REVISER_PARITY
    report = {
        "healthy": (
            all(not row["unexpected_backend_functions"] for row in generators.values())
            and all(
                not row["unexpected_backend_functions"] for row in revisers.values()
            )
        ),
        "policy_version": CORE_POLICY_VERSION,
        "revision_policy_version": REVISION_CORE_POLICY_VERSION,
        "domain": config.to_public_dict(),
        "scope": "active_generation_and_evaluation"
        if not args.include_legacy
        else "active_plus_legacy",
        "core_files": core,
        "generator_adapters": generators,
        "historical_snapshot_comparison": (
            _compare_generator_functions()
            if args.include_legacy
            else {"status": "not_run; pass --include-legacy"}
        ),
        "reviser_adapters": revisers,
        "legacy_revision_profiles": (
            {
                "card-core": ["diversity", "tone"],
                "extended": [
                    "diversity",
                    "selfbert",
                    "semantic",
                    "tone",
                    "emotion",
                    "length",
                    "story",
                    "structure",
                ],
            }
            if args.include_legacy
            else {"status": "not_run; pass --include-legacy"}
        ),
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
        print(f"[parity-audit] wrote={args.output}", flush=True)
    print(rendered, end="")
    if not report["healthy"]:
        raise SystemExit(2)


def _compare_generator_functions() -> dict[str, Any]:
    snapshot = _function_ast_hashes(CARD_SNAPSHOT_BACKEND)
    generalized = _function_ast_hashes(GENERALIZED_V2_BACKEND)
    same: list[str] = []
    changed: list[str] = []
    missing: list[str] = []
    hashes: dict[str, dict[str, str]] = {}
    for name in CORE_ALGORITHM_SYMBOLS:
        snapshot_hash = snapshot.get(name, "missing")
        generalized_hash = generalized.get(name, "missing")
        hashes[name] = {
            "card_snapshot_ast_sha256": snapshot_hash,
            "generalized_v2_ast_sha256": generalized_hash,
        }
        if "missing" in {snapshot_hash, generalized_hash}:
            missing.append(name)
        elif snapshot_hash == generalized_hash:
            same.append(name)
        else:
            changed.append(name)
    return {
        "card_snapshot": str(CARD_SNAPSHOT_BACKEND),
        "generalized_v2": str(GENERALIZED_V2_BACKEND),
        "ast_identical_count": len(same),
        "ast_different_count": len(changed),
        "missing_count": len(missing),
        "ast_identical_functions": same,
        "ast_different_functions": changed,
        "missing_functions": missing,
        "function_hashes": hashes,
    }


def _function_ast_hashes(path: Path) -> dict[str, str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    hashes: dict[str, str] = {}
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        payload = ast.dump(node, include_attributes=False).encode("utf-8")
        hashes[node.name] = hashlib.sha256(payload).hexdigest()
    return hashes


if __name__ == "__main__":
    main()
