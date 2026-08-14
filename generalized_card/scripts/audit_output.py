#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from generalized_card.audit import audit_generated_root  # noqa: E402
from generalized_card.domain import load_domain_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit generated discussions for unusable output.")
    parser.add_argument("generated_root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--domain", default="")
    parser.add_argument("--seed-pool", type=Path)
    parser.add_argument("--domain-profile", type=Path)
    parser.add_argument("--min-accepted-share", type=float, default=0.50)
    parser.add_argument("--min-unique-share", type=float, default=0.80)
    parser.add_argument("--min-mean-words", type=float, default=5.0)
    parser.add_argument("--max-plan-collision-rate", type=float, default=0.10)
    parser.add_argument("--max-perspective-share", type=float, default=0.34)
    parser.add_argument("--require-healthy", action="store_true")
    args = parser.parse_args()
    config = load_domain_config(args.domain) if args.domain else None
    report = audit_generated_root(
        args.generated_root,
        config=config,
        seed_pool=args.seed_pool,
        domain_profile=args.domain_profile,
        min_accepted_share=args.min_accepted_share,
        min_unique_share=args.min_unique_share,
        min_mean_words=args.min_mean_words,
        max_plan_collision_rate=args.max_plan_collision_rate,
        max_perspective_share=args.max_perspective_share,
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("[output-audit] " + json.dumps(report, ensure_ascii=False, sort_keys=True), flush=True)
    if args.require_healthy and not report["healthy"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
