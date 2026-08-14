#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from generalized_card.domain import REPO_ROOT, load_domain_config  # noqa: E402
from generalized_card.domain_profile import build_domain_profile  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a frozen non-test domain profile.")
    parser.add_argument("--domain", default="camera")
    parser.add_argument("--seed-pool", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-perspectives", type=int, default=32)
    args = parser.parse_args()
    config = load_domain_config(args.domain)
    output = args.output or (
        REPO_ROOT / "artifacts" / "generalized_card" / "domain_profiles" / f"{config.domain_id}.json"
    )
    payload = build_domain_profile(
        config,
        seed_pool_path=args.seed_pool.expanduser().resolve(),
        output_path=output.expanduser().resolve(),
        max_perspectives=max(8, args.max_perspectives),
    )
    print("[domain-profile] " + json.dumps({
        "output": str(output.expanduser().resolve()),
        "sha256": payload["profile_sha256"],
        "reference_threads": payload["source"]["reference_thread_count"],
        "perspectives": len(payload["perspectives"]),
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
