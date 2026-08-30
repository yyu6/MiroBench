#!/usr/bin/env python3
"""Clone pinned baseline repositories and apply committed runtime overrides."""
from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

from common import EXPERIMENT_ROOT, REPO_ROOT, read_json


LOCK_PATH = EXPERIMENT_ROOT / "config" / "external_repositories.json"
OVERRIDE_ROOT = EXPERIMENT_ROOT / "vendor_overrides"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-synthpai", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    specs = read_json(LOCK_PATH)
    selected = ["MiroFish"] if args.skip_synthpai else ["MiroFish", "SynthPAI"]
    for name in selected:
        spec = dict(specs[name])
        target = REPO_ROOT / str(spec["path"])
        ensure_checkout(
            name=name,
            target=target,
            url=str(spec["url"]),
            commit=str(spec["commit"]),
            verify_only=args.verify_only,
        )
        if args.verify_only:
            verify_overrides(name=name, target=target)
        else:
            apply_overrides(name=name, target=target)
    print(f"[external-ready] {', '.join(selected)}")


def ensure_checkout(
    *, name: str, target: Path, url: str, commit: str, verify_only: bool
) -> None:
    if not (target / ".git").exists():
        if verify_only:
            raise SystemExit(f"{name} checkout missing: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        run(["git", "clone", "--filter=blob:none", url, str(target)])

    head = capture(["git", "-C", str(target), "rev-parse", "HEAD"])
    if head == commit:
        return
    if verify_only:
        raise SystemExit(f"{name} commit mismatch: expected={commit} actual={head}")
    dirty = capture(["git", "-C", str(target), "status", "--porcelain"])
    if dirty:
        raise SystemExit(
            f"Refusing to change dirty {name} checkout at {target}. "
            "Move it aside and rerun setup.sh."
        )
    run(["git", "-C", str(target), "fetch", "--depth", "1", "origin", commit])
    run(["git", "-C", str(target), "checkout", "--detach", commit])


def apply_overrides(*, name: str, target: Path) -> None:
    source = OVERRIDE_ROOT / name
    if not source.exists():
        return
    for path in sorted(item for item in source.rglob("*") if item.is_file()):
        destination = target / path.relative_to(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    print(f"[overrides] {name} <- {source}")


def verify_overrides(*, name: str, target: Path) -> None:
    source = OVERRIDE_ROOT / name
    failures = []
    for path in sorted(item for item in source.rglob("*") if item.is_file()):
        destination = target / path.relative_to(source)
        if not destination.is_file() or destination.read_bytes() != path.read_bytes():
            failures.append(str(path.relative_to(source)))
    if failures:
        raise SystemExit(
            f"{name} compatibility overrides are missing or changed: "
            + ", ".join(failures[:10])
        )


def run(command: list[str]) -> None:
    print("[run] " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def capture(command: list[str]) -> str:
    return subprocess.check_output(command, text=True).strip()


if __name__ == "__main__":
    main()
