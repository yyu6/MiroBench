#!/usr/bin/env python3
"""Thin domain-neutral entry point for CARD's Self-BLEU controller.

The generalized pipeline adapts prompts and factual anchors inside the reviser
backend.  Distribution diagnosis, strategy selection, round acceptance,
protected-metric checks, rollback, history, and resume are delegated unchanged
to the operational CARD controller.
"""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_ROOT = REPO_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import run_metric_revision_controller as card_controller  # noqa: E402


def main() -> None:
    if "--self-test" in sys.argv:
        card_controller.run_self_test()
        print("generalized Self-BLEU controller parity test passed", flush=True)
        return
    card_controller.main()


if __name__ == "__main__":
    main()
