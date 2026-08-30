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
    parser.add_argument(
        "--exclude-pool",
        type=Path,
        nargs="*",
        default=(),
        help=(
            "Existing seed-pool JSON files whose threads must NOT appear in this "
            "pool. Use it to build a calibration pool disjoint from every "
            "evaluation pool, which is what lets a realization matrix be "
            "measured without touching a thread it will later be judged on."
        ),
    )
    args = parser.parse_args()

    exclude: set[tuple[str, str]] = set()
    for path in args.exclude_pool or ():
        data = json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))
        for row in data.get("seed_posts") or ():
            exclude.add((str(row.get("source_product_dir")), str(row.get("source_raw_post_id"))))
    if exclude:
        print(f"[exclude] {len(exclude)} threads held out from {len(args.exclude_pool)} pool(s)", flush=True)

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
        exclude_keys=exclude or None,
    )
    print(
        f"[seed-pool] domain={config.domain_id} rows={len(payload['seed_posts'])} "
        f"output={output.expanduser().resolve()}",
        flush=True,
    )


if __name__ == "__main__":
    main()
