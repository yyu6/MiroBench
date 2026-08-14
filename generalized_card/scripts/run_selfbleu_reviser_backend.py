#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from generalized_card.domain import load_domain_from_env  # noqa: E402
from generalized_card.reviser_backend import (  # noqa: E402
    configure_reviser_backend,
    load_reviser_backend,
    run_adapter_self_test,
)


def main() -> None:
    config = load_domain_from_env()
    if "--self-test" in sys.argv:
        run_adapter_self_test("selfbleu", config)
        return
    backend = configure_reviser_backend(load_reviser_backend("selfbleu"), kind="selfbleu", config=config)
    backend.main()


if __name__ == "__main__":
    main()
