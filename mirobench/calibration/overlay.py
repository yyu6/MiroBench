"""
Overlay system for the calibration module.

Functions
---------
load_overlay   : Load a JSON overlay file, returning empty dict if missing.
save_overlay   : Save an overlay dict as formatted JSON, creating parent dirs.
merge_overlay  : Merge overlay on top of defaults; overlay values take precedence.
append_text_overlay : Merge overlay by appending text-valued patches to existing text slots.
apply_structured_phase_overlay : Append patches into named manual-phase blocks and
                                 render them back into the two runtime text knobs.
render_structured_overlay : Re-render stored phase blocks into the two runtime text knobs.
diff_overlay   : Return knobs that differ between two dicts (values from b).
"""
from __future__ import annotations

import json
from pathlib import Path

TEXT_OVERLAY_KEYS = (
    "persona.generation_guidance",
    "prompt.comment_style_guidance",
)
STRUCTURED_PHASE_BLOCKS_KEY = "_manual_phase_blocks"


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


def append_text_overlay(
    defaults: dict,
    overlay: dict,
    *,
    text_keys: tuple[str, ...] = TEXT_OVERLAY_KEYS,
) -> dict:
    """Merge overlay on top of defaults, appending the known text slots.

    Non-text keys use normal overwrite semantics. For keys in ``text_keys``,
    the new text patch is appended to the existing text instead of replacing it.
    If the patch text already appears verbatim in the base text, the base text
    is kept unchanged.
    """
    result = dict(defaults)
    for key, value in overlay.items():
        if key not in text_keys or not isinstance(value, str):
            result[key] = value
            continue
        result[key] = _append_unique_text(str(result.get(key, "") or ""), value)
    return result


def _append_unique_text(base_text: str, patch_text: str) -> str:
    """Append *patch_text* to *base_text* unless it is empty or already present."""
    patch_text = str(patch_text or "").strip()
    base_text = str(base_text or "").strip()

    if not patch_text:
        return base_text
    if not base_text:
        return patch_text
    if patch_text in base_text:
        return base_text
    return f"{base_text}\n\n{patch_text}"


def _normalize_phase_blocks(raw_blocks: object) -> dict[str, dict]:
    """Return a normalized mapping of structured manual-phase blocks."""
    if not isinstance(raw_blocks, dict):
        return {}

    normalized: dict[str, dict] = {}
    for phase_name, payload in raw_blocks.items():
        if not isinstance(phase_name, str) or not isinstance(payload, dict):
            continue
        normalized[phase_name] = {
            "phase_label": str(payload.get("phase_label", phase_name)).strip() or phase_name,
            "phase_order": int(payload.get("phase_order", 9999)),
            "persona.generation_guidance": str(
                payload.get("persona.generation_guidance", "") or ""
            ).strip(),
            "prompt.comment_style_guidance": str(
                payload.get("prompt.comment_style_guidance", "") or ""
            ).strip(),
        }
    return normalized


def render_structured_overlay(
    overlay: dict,
    *,
    text_keys: tuple[str, ...] = TEXT_OVERLAY_KEYS,
) -> dict:
    """Render structured manual-phase blocks back into runtime text knobs."""
    result = dict(overlay)
    blocks = _normalize_phase_blocks(result.get(STRUCTURED_PHASE_BLOCKS_KEY, {}))
    if not blocks:
        return result

    ordered_blocks = sorted(
        blocks.items(),
        key=lambda item: (
            int(item[1].get("phase_order", 9999)),
            str(item[1].get("phase_label", item[0])),
            item[0],
        ),
    )
    for key in text_keys:
        sections: list[str] = []
        for phase_name, block in ordered_blocks:
            text = str(block.get(key, "") or "").strip()
            if not text:
                continue
            label = str(block.get("phase_label", phase_name)).strip() or phase_name
            sections.append(f"[{label}]\n{text}")
        if sections:
            result[key] = "\n\n".join(sections)
    result[STRUCTURED_PHASE_BLOCKS_KEY] = blocks
    return result


def apply_structured_phase_overlay(
    defaults: dict,
    overlay: dict,
    *,
    phase_name: str,
    phase_label: str | None = None,
    phase_order: int | None = None,
    text_keys: tuple[str, ...] = TEXT_OVERLAY_KEYS,
) -> dict:
    """Append text patches into a named manual-phase block and re-render text knobs.

    The structured representation is stored under ``_manual_phase_blocks`` and the
    two runtime text knobs are rendered from that structure so later blocks see a
    stable, labeled prompt layout instead of one long undifferentiated append-only blob.
    """
    if not phase_name:
        return append_text_overlay(defaults, overlay, text_keys=text_keys)

    result = dict(defaults)
    blocks = _normalize_phase_blocks(result.get(STRUCTURED_PHASE_BLOCKS_KEY, {}))
    current_block = dict(blocks.get(phase_name, {}))
    current_block["phase_label"] = str(
        phase_label or current_block.get("phase_label") or phase_name
    ).strip() or phase_name
    current_block["phase_order"] = int(
        phase_order if phase_order is not None else current_block.get("phase_order", 9999)
    )

    for key, value in overlay.items():
        if key in text_keys and isinstance(value, str):
            current_block[key] = _append_unique_text(current_block.get(key, ""), value)
        else:
            result[key] = value

    blocks[phase_name] = current_block
    result[STRUCTURED_PHASE_BLOCKS_KEY] = blocks
    return render_structured_overlay(result, text_keys=text_keys)


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
