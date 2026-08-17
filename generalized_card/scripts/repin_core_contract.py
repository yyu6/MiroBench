#!/usr/bin/env python3
"""Recompute every pinned hash in `core_contract.CORE_FILES` and report drift.

Re-pinning by hand means listing the files you *believe* you changed. That is
the same shape as the config-diff script that silently skipped
`plan_quality.repair_rounds` and cost a confounded run: a check that depends on
the author remembering the whole set is not a check.

This walks the entire `CORE_FILES` table instead, so a file that changed without
being noticed still shows up.

    python3 generalized_card/scripts/repin_core_contract.py           # report only
    python3 generalized_card/scripts/repin_core_contract.py --write   # rewrite
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import re
import subprocess
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from generalized_card.core_contract import (  # noqa: E402
    CORE_FILES,
    CURRENT_ACTIVE_CORE_NAMES,
)
from generalized_card.domain import REPO_ROOT  # noqa: E402

CONTRACT = PACKAGE_ROOT / "generalized_card" / "core_contract.py"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    text = CONTRACT.read_text()
    drifted: list[tuple[str, str, str, str]] = []
    missing: list[str] = []
    tracked = set(
        subprocess.run(
            ["git", "ls-files"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
    )
    untracked_active = [
        CORE_FILES[name][0]
        for name in CURRENT_ACTIVE_CORE_NAMES
        if CORE_FILES[name][0] not in tracked
    ]
    unpinned_imports = unpinned_local_imports()

    for name, (relative, expected) in sorted(CORE_FILES.items()):
        path = REPO_ROOT / relative
        if not path.exists():
            missing.append(f"{name}: {relative}")
            continue
        actual = sha256(path)
        if actual != expected:
            drifted.append((name, relative, expected, actual))

    print(f"pinned files checked : {len(CORE_FILES)}")
    print(f"missing on disk      : {len(missing)}")
    for row in missing:
        print(f"  MISSING {row}")
    print(f"untracked active     : {len(untracked_active)}")
    for path in untracked_active:
        print(f"  UNTRACKED {path}")
    print(f"unpinned local imports: {len(unpinned_imports)}")
    for path in unpinned_imports:
        print(f"  UNPINNED {path}")
    print(f"drifted              : {len(drifted)}")
    for name, relative, expected, actual in drifted:
        marker = "NEW " if expected == "PENDING" else "CHG "
        print(f"  {marker}{name:<26} {relative}")
        print(f"       {expected[:16]}... -> {actual[:16]}...")

    if untracked_active or unpinned_imports:
        print(
            "\nactive sources must be recoverable from git and their local import "
            "closure must be pinned."
        )
        return 2
    if not drifted:
        print("\nnothing to re-pin.")
        return 0
    if not args.write:
        print("\nreport only; pass --write to re-pin.")
        return 1

    for name, relative, expected, actual in drifted:
        pattern = re.compile(
            r'("%s":\s*\(\s*\n\s*"%s",\s*\n\s*)"[^"]*"'
            % (re.escape(name), re.escape(relative))
        )
        text, count = pattern.subn(lambda m: m.group(1) + f'"{actual}"', text)
        if count != 1:
            print(f"  FAILED to rewrite {name} (matches={count})")
            return 2
    CONTRACT.write_text(text)
    print(f"\nre-pinned {len(drifted)} entries in {CONTRACT}")
    return 0


def unpinned_local_imports() -> list[str]:
    """Walk local imports from every active source, including sibling scripts."""

    pinned_paths = {CORE_FILES[name][0] for name in CURRENT_ACTIVE_CORE_NAMES}
    pending = [REPO_ROOT / CORE_FILES[name][0] for name in CURRENT_ACTIVE_CORE_NAMES]
    visited: set[Path] = set()
    while pending:
        path = pending.pop().resolve()
        if path in visited or not path.is_file():
            continue
        visited.add(path)
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            pending.extend(resolve_local_imports(path, node))
    return sorted(
        str(path.relative_to(REPO_ROOT))
        for path in visited
        if path != CONTRACT.resolve()
        and str(path.relative_to(REPO_ROOT)) not in pinned_paths
    )


def resolve_local_imports(path: Path, node: ast.AST) -> list[Path]:
    modules: list[tuple[str, int]] = []
    if isinstance(node, ast.ImportFrom):
        if node.module:
            modules.append((node.module, node.level))
        else:
            modules.extend((alias.name, node.level) for alias in node.names)
    elif isinstance(node, ast.Import):
        modules.extend((alias.name, 0) for alias in node.names)
    resolved: list[Path] = []
    for module, level in modules:
        parts = module.split(".")
        roots: list[Path]
        if level:
            root = path.parent
            for _ in range(level - 1):
                root = root.parent
            roots = [root]
        else:
            roots = [path.parent, REPO_ROOT, PACKAGE_ROOT]
        for root in roots:
            candidate = root.joinpath(*parts).with_suffix(".py")
            if candidate.is_file() and candidate.resolve().is_relative_to(REPO_ROOT):
                resolved.append(candidate)
                break
    return resolved


if __name__ == "__main__":
    raise SystemExit(main())
