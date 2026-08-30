#!/usr/bin/env python3
"""Install the committed matched-input bundle into the artifact workspace."""
from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path

from common import EXPERIMENT_ROOT, REPO_ROOT, read_json


DEFAULT_SOURCE = EXPERIMENT_ROOT / "portable_inputs"
DEFAULT_OUTPUT = REPO_ROOT / "artifacts" / "reddit_multidomain_baselines" / "inputs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--domains", nargs="*")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.source.expanduser().resolve()
    output = args.output.expanduser().resolve()
    manifest_path = source / "portable_inputs_manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"Portable input manifest not found: {manifest_path}")
    manifest = read_json(manifest_path)
    verify_manifest(source, manifest)
    available = list(manifest.get("domains") or [])
    domains = args.domains or available
    unknown = sorted(set(domains) - set(available))
    if unknown:
        raise SystemExit(f"Domains absent from portable bundle: {', '.join(unknown)}")
    if args.verify_only:
        verify_installation(source=source, output=output, domains=domains)
        print(f"[verified] domains={len(domains)} source={source} output={output}")
        return

    for domain in domains:
        install_file(
            source / "seed_pools" / f"{domain}.json",
            output / "seed_pools" / f"{domain}.json",
            force=args.force,
        )
        install_tree(
            source / "real_reference" / domain,
            output / "real_reference" / domain,
            force=args.force,
        )
    print(f"[installed] domains={len(domains)} output={output}")


def verify_manifest(source: Path, manifest: dict) -> None:
    failures: list[str] = []
    for record in manifest.get("files") or []:
        path = source / str(record["path"])
        if not path.is_file():
            failures.append(f"missing:{record['path']}")
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != record.get("sha256"):
            failures.append(f"checksum:{record['path']}")
    if failures:
        raise SystemExit("Portable input verification failed: " + ", ".join(failures[:10]))


def verify_installation(*, source: Path, output: Path, domains: list[str]) -> None:
    failures: list[str] = []
    for domain in domains:
        seed_relative = Path("seed_pools") / f"{domain}.json"
        installed_seed_path = output / seed_relative
        if not installed_seed_path.is_file():
            failures.append(f"missing:{seed_relative}")
            continue
        source_seed_count = len((read_json(source / seed_relative).get("seed_posts") or []))
        installed_seed_count = len((read_json(installed_seed_path).get("seed_posts") or []))
        if installed_seed_count != source_seed_count:
            failures.append(
                f"seed-count:{domain}:expected={source_seed_count}:actual={installed_seed_count}"
            )

        source_runs = list((source / "real_reference" / domain).glob("run_*/discussion.json"))
        installed_runs = list((output / "real_reference" / domain).glob("run_*/discussion.json"))
        if len(installed_runs) != len(source_runs):
            failures.append(
                f"reference-runs:{domain}:expected={len(source_runs)}:actual={len(installed_runs)}"
            )
    if failures:
        raise SystemExit(
            "Installed portable inputs are incomplete: "
            + ", ".join(failures[:10])
            + ". Rerun setup.sh or install_portable_inputs.py."
        )


def install_file(source: Path, target: Path, *, force: bool) -> None:
    if target.exists() and not force:
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def install_tree(source: Path, target: Path, *, force: bool) -> None:
    if target.exists() and not force:
        return
    if target.exists():
        shutil.rmtree(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, target)


if __name__ == "__main__":
    main()
