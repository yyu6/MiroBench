#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from generalized_card.backend import (  # noqa: E402
    DEFAULT_GENERATOR_PROFILE,
    configure_generator_backend,
    load_generator_backend,
)
from generalized_card.domain import load_domain_from_env  # noqa: E402


def main() -> None:
    config = load_domain_from_env()
    profile = os.environ.get(
        "GENERALIZED_CARD_GENERATOR_PROFILE", DEFAULT_GENERATOR_PROFILE
    )
    backend = configure_generator_backend(
        load_generator_backend(profile=profile),
        config,
        profile=profile,
    )
    backend.main()


if __name__ == "__main__":
    main()
