#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from generalized_card.data import audit_domain, build_seed_pool  # noqa: E402
from generalized_card.domain import REPO_ROOT, load_domain_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a deterministic matched-real seed pool.")
    parser.add_argument("--domain", default="camera")
    parser.add_argument("--count", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-comments", type=int, default=0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()

    config = load_domain_config(args.domain)
    audit = audit_domain(config)
    print("[domain-audit] " + json.dumps(audit, ensure_ascii=False, sort_keys=True), flush=True)
    if args.audit_only:
        return
    output = args.output or (
        REPO_ROOT
        / "artifacts"
        / "generalized_card"
        / "seed_pools"
        / f"{config.domain_id}_{args.count}_seed{args.seed}.json"
    )
    payload = build_seed_pool(
        config,
        output,
        count=args.count,
        seed=args.seed,
        min_comments=args.min_comments or None,
    )
    print(
        f"[seed-pool] domain={config.domain_id} rows={len(payload['seed_posts'])} "
        f"output={output.expanduser().resolve()}",
        flush=True,
    )


if __name__ == "__main__":
    main()
