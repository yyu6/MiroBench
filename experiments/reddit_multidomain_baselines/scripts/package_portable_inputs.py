#!/usr/bin/env python3
"""Create the committed, privacy-preserving matched-input bundle.

The raw crawler export is intentionally not committed.  This script packages
only the selected seed posts and their matched real-reference discussions,
replaces Reddit author names with deterministic opaque ids, and removes local
absolute paths.  It is a maintainer tool; fresh machines consume the resulting
``portable_inputs`` directory through ``install_portable_inputs.py``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import Any

from common import EXPERIMENT_ROOT, read_json, write_json


DEFAULT_SOURCE = (
    EXPERIMENT_ROOT.parents[1]
    / "artifacts"
    / "reddit_multidomain_baselines"
    / "inputs"
)
DEFAULT_OUTPUT = EXPERIMENT_ROOT / "portable_inputs"
USER_MENTION_RE = re.compile(r"(?<![A-Za-z0-9_])/?u/[A-Za-z0-9_-]{2,32}", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.source.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists():
        if not args.force:
            raise SystemExit(f"Output exists; pass --force to replace it: {output}")
        shutil.rmtree(output)

    pool_dir = source / "seed_pools"
    domains = sorted(path.stem for path in pool_dir.glob("*.json"))
    if not domains:
        raise SystemExit(f"No seed pools found under {pool_dir}")

    for domain in domains:
        package_domain(source=source, output=output, domain=domain)

    files = sorted(path for path in output.rglob("*") if path.is_file())
    manifest = {
        "format_version": 1,
        "domains": domains,
        "privacy": {
            "author_names": "deterministically replaced with reddit_user_<hash>",
            "reddit_user_mentions": "replaced with [REDDIT_USER]",
            "local_absolute_paths": "removed from seed metadata and manifests",
        },
        "files": [
            {
                "path": str(path.relative_to(output)),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "bytes": path.stat().st_size,
            }
            for path in files
        ],
    }
    write_json(output / "portable_inputs_manifest.json", manifest)
    print(f"[done] domains={len(domains)} files={len(files)} output={output}")


def package_domain(*, source: Path, output: Path, domain: str) -> None:
    pool = read_json(source / "seed_pools" / f"{domain}.json")
    pool_meta = dict(pool.get("meta") or {})
    for key in (
        "data_root",
        "posts_jsonl",
        "comments_jsonl",
        "raw_root",
        "source_seed_pool_json",
    ):
        pool_meta.pop(key, None)
    pool_meta["portable_source"] = True
    pool["meta"] = pool_meta
    pool = sanitize_text_values(pool)
    write_json(output / "seed_pools" / f"{domain}.json", pool)

    source_reference = source / "real_reference" / domain
    target_reference = output / "real_reference" / domain
    author_map: dict[str, str] = {}
    for discussion_path in sorted(source_reference.glob("run_*_sampled_reddit/discussion.json")):
        discussion = read_json(discussion_path)
        sanitize_authors(discussion, domain=domain, author_map=author_map)
        discussion = sanitize_text_values(discussion)
        relative = discussion_path.relative_to(source_reference)
        write_json(target_reference / relative, discussion)

    reference_manifest = read_json(source_reference / "reference_manifest.json")
    reference_manifest["seed_pool"] = f"seed_pools/{domain}.json"
    reference_manifest["portable_source"] = True
    write_json(target_reference / "reference_manifest.json", reference_manifest)


def sanitize_authors(value: Any, *, domain: str, author_map: dict[str, str]) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if key == "author" and isinstance(child, str):
                value[key] = pseudonym(domain, child, author_map)
            else:
                sanitize_authors(child, domain=domain, author_map=author_map)
    elif isinstance(value, list):
        for child in value:
            sanitize_authors(child, domain=domain, author_map=author_map)


def pseudonym(domain: str, author: str, author_map: dict[str, str]) -> str:
    normalized = author.strip()
    if not normalized or normalized.casefold() in {"[deleted]", "deleted", "none"}:
        return "[deleted]"
    if normalized not in author_map:
        digest = hashlib.sha256(f"{domain}:{normalized}".encode("utf-8")).hexdigest()[:12]
        author_map[normalized] = f"reddit_user_{digest}"
    return author_map[normalized]


def sanitize_text_values(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: sanitize_text_values(child) for key, child in value.items()}
    if isinstance(value, list):
        return [sanitize_text_values(child) for child in value]
    if isinstance(value, str):
        return USER_MENTION_RE.sub("[REDDIT_USER]", value)
    return value


if __name__ == "__main__":
    main()
