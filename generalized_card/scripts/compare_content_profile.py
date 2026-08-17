#!/usr/bin/env python3
"""Write a matched, read-only content and Planner→Writer realization audit."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_ROOT.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from generalized_card.content_profile import (  # noqa: E402
    build_content_profile,
    render_markdown,
    write_report,
)
from generalized_card.domain import load_domain_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--domain", default="camera")
    parser.add_argument(
        "--runs-root",
        default=str(REPO_ROOT / "artifacts/generalized_card/runs"),
    )
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-md", type=Path)
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()

    run_dir = Path(args.runs_root).expanduser().resolve() / args.tag
    report = build_content_profile(run_dir, load_domain_config(args.domain))
    markdown = render_markdown(report)
    if not args.no_write:
        json_path = args.output_json or run_dir / "content_profile_audit.json"
        markdown_path = args.output_md or run_dir / "content_profile_audit.md"
        write_report(report, json_path=json_path, markdown_path=markdown_path)
        print(f"[content-profile] json={json_path}")
        print(f"[content-profile] markdown={markdown_path}")
    print(markdown, end="")


if __name__ == "__main__":
    main()
