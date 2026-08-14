"""Transport helpers for structured Comment Planner responses.

Providers commonly render the prompt's displayed anonymous slot label as
``S12`` even when the JSON schema illustrates a numeric ``sample_id``. The
label is transport syntax, not a semantic control, so every Planner consumer
must normalize it consistently before enriching a plan.
"""

from __future__ import annotations

import re
from typing import Any


_SAMPLE_ID_RE = re.compile(r"[sS]\s*(\d+)")


def parse_sample_id(value: Any, *, default: int = 0) -> int:
    """Return a positive anonymous sample ID from ``12`` or ``S12``."""

    text = str(value or "").strip()
    matched = _SAMPLE_ID_RE.fullmatch(text)
    if matched is not None:
        return int(matched.group(1))
    try:
        parsed = int(float(text))
    except (TypeError, ValueError):
        return int(default)
    return parsed if parsed > 0 else int(default)
