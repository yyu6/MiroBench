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
import hashlib
import re
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from generalized_card.core_contract import CORE_FILES  # noqa: E402
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
    print(f"drifted              : {len(drifted)}")
    for name, relative, expected, actual in drifted:
        marker = "NEW " if expected == "PENDING" else "CHG "
        print(f"  {marker}{name:<26} {relative}")
        print(f"       {expected[:16]}... -> {actual[:16]}...")

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


if __name__ == "__main__":
    raise SystemExit(main())
