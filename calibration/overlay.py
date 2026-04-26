"""
Overlay system for the calibration module.

Functions
---------
load_overlay   : Load a JSON overlay file, returning empty dict if missing.
save_overlay   : Save an overlay dict as formatted JSON, creating parent dirs.
merge_overlay  : Merge overlay on top of defaults; overlay values take precedence.
diff_overlay   : Return knobs that differ between two dicts (values from b).
"""
from __future__ import annotations

import json
from pathlib import Path


def load_overlay(path: Path) -> dict:
    """Load a JSON overlay file.

    Parameters
    ----------
    path : Path
        File path to load.

    Returns
    -------
    dict
        Parsed JSON content, or an empty dict if the file does not exist.
    """
    path = Path(path)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_overlay(overlay: dict, path: Path) -> None:
    """Save an overlay dict as formatted JSON.

    Creates parent directories if they do not exist.

    Parameters
    ----------
    overlay : dict
        The overlay data to serialise.
    path : Path
        Destination file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(overlay, indent=2), encoding="utf-8")


def merge_overlay(defaults: dict, overlay: dict) -> dict:
    """Merge overlay on top of defaults.

    Overlay values take precedence over defaults. Neither input dict is
    mutated.

    Parameters
    ----------
    defaults : dict
        Base knob values.
    overlay : dict
        Overriding knob values.

    Returns
    -------
    dict
        Merged result with overlay values winning on conflict.
    """
    result = dict(defaults)
    result.update(overlay)
    return result


def diff_overlay(a: dict, b: dict) -> dict:
    """Return knobs that differ between a and b.

    Only keys present in b are considered. Values come from b. Keys whose
    value in b is None are excluded from the result.

    Parameters
    ----------
    a : dict
        Reference dict (e.g. previous state or defaults).
    b : dict
        Candidate dict (e.g. new overlay).

    Returns
    -------
    dict
        Keys where b differs from a, with values from b. None values in b
        are skipped.
    """
    diff: dict = {}
    for key, val_b in b.items():
        if val_b is None:
            continue
        if a.get(key) != val_b:
            diff[key] = val_b
    return diff
