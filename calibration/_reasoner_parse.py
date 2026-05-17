"""Response parsing for reasoner/materializer LLM outputs.

Extracted from reasoner.py to keep file size manageable.
Logic unchanged — same functions, same behavior.
"""
from __future__ import annotations

import json
import re
from typing import Any


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def _parse_jsonish_response(raw: str) -> dict[str, Any]:
    """Parse a JSON-like LLM response, tolerating common wrapper noise."""
    if not raw or not raw.strip():
        raise ValueError("LLM returned empty response")

    cleaned = raw.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        cleaned = "\n".join(lines)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        import re

        fixed = re.sub(r",\s*([}\]])", r"\1", cleaned)
        try:
            return json.loads(fixed)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", fixed, re.DOTALL)
            if match:
                return json.loads(match.group())
            raise

def _fallback_candidate(data: dict) -> dict:
    """Build a best-effort candidate dict from an unexpected LLM response shape.

    Searches the top-level dict for anything resembling an overlay_diff
    (keys containing 'persona.' or 'prompt.'), strategy metadata, etc.

    Raises ``ValueError`` if no overlay_diff content can be found — an empty
    fallback would silently waste an entire calibration iteration.
    """
    overlay_diff = data.get("overlay_diff", {})

    # If no explicit overlay_diff, check for text-knob keys at top level
    if not overlay_diff:
        for key in list(data.keys()):
            if key.startswith("persona.") or key.startswith("prompt."):
                overlay_diff[key] = data[key]

    # Try to find overlay_diff nested in any dict value
    if not overlay_diff:
        for _v in data.values():
            if isinstance(_v, dict) and any(
                k.startswith("persona.") or k.startswith("prompt.") for k in _v
            ):
                overlay_diff = _v
                break

    if not overlay_diff:
        raise ValueError(
            "LLM response contains no overlay_diff and no recognizable "
            f"persona./prompt. knob keys. Keys present: {list(data.keys())}"
        )
    overlay_diff = _normalize_text_knob_block(
        overlay_diff,
        context="fallback overlay_diff",
    )

    return {
        "strategy_label": data.get("strategy_label", "fallback_strategy"),
        "strategy": data.get("strategy", data.get("rationale", data.get("description", ""))),
        "mechanism_family": str(data.get("mechanism_family", "semantic_diversity")).strip().lower(),
        "anti_incumbent": bool(data.get("anti_incumbent", False)),
        "primary_layer": str(data.get("primary_layer", "both")).strip().lower(),
        "overlay_diff": overlay_diff,
        "rationale": data.get("rationale", "auto-extracted from unexpected response format"),
    }


def parse_reasoner_response(raw: str) -> dict:
    """Parse and validate the LLM's JSON response.

    Supports two formats:
    - **New (5 independent strategies):** ``{diagnosis, candidates: [{strategy_label,
      strategy, primary_layer, overlay_diff, rationale}, ...]}``
    - **Legacy (single strategy):** ``{diagnosis, strategy_label, overlay_diff, ...}``

    Returns
    -------
    dict with keys: diagnosis, candidates (list of 5 dicts), constraints.
    Each candidate has: strategy_label, strategy, primary_layer, overlay_diff, rationale.
    """
    data = _parse_jsonish_response(raw)

    # Ensure diagnosis is always a string (LLM may return dict/list)
    raw_diag = data.get("diagnosis", "")
    if not isinstance(raw_diag, str):
        raw_diag = json.dumps(raw_diag, ensure_ascii=False)

    def _normalize_primary_layer(value: Any) -> str:
        layer = str(value or "both").strip().lower()
        if layer not in {"persona", "prompt", "both"}:
            return "both"
        return layer

    def _normalize_mechanism_family(value: Any) -> str:
        family = str(value or "semantic_diversity").strip().lower()
        if family not in _MECHANISM_FAMILIES:
            return "semantic_diversity"
        return family

    if "candidates" in data and isinstance(data["candidates"], list) and data["candidates"]:
        # ── New format: 5 independent strategies ──
        candidates = []
        for i, c in enumerate(data["candidates"][:5]):
            if not isinstance(c, dict):
                continue
            overlay_diff = _normalize_text_knob_block(
                c.get("overlay_diff", {}),
                context=f"candidate[{i}].overlay_diff",
            )
            candidates.append({
                "strategy_label": c.get("strategy_label", f"strategy_{i}"),
                "strategy": c.get("strategy", c.get("rationale", "")),
                "mechanism_family": _normalize_mechanism_family(c.get("mechanism_family")),
                "anti_incumbent": bool(c.get("anti_incumbent", False)),
                "primary_layer": _normalize_primary_layer(c.get("primary_layer", "both")),
                "overlay_diff": overlay_diff,
                "rationale": c.get("rationale", ""),
            })
        if not candidates:
            # candidates list existed but contained no valid dicts — fall through
            candidates = [_fallback_candidate(data)]
        if len(candidates) != 5:
            raise ValueError(
                f"Reasoner returned {len(candidates)} valid candidates; expected exactly 5."
            )
        return {
            "diagnosis": raw_diag,
            "candidates": candidates,
            "constraints": data.get("constraints", []),
            # Back-compat fields for log/trajectory
            "strategy_label": candidates[0]["strategy_label"],
            "mechanism_family": candidates[0]["mechanism_family"],
            "anti_incumbent": candidates[0]["anti_incumbent"],
            "primary_layer": candidates[0]["primary_layer"],
        }
    else:
        # ── Legacy / unexpected format → extract whatever is available ──
        base_candidate = _fallback_candidate(data)
        return {
            "diagnosis": raw_diag,
            "candidates": [base_candidate],  # generate_variants will expand
            "constraints": data.get("constraints", []),
            "strategy_label": base_candidate["strategy_label"],
            "mechanism_family": base_candidate["mechanism_family"],
            "anti_incumbent": base_candidate["anti_incumbent"],
            "primary_layer": base_candidate["primary_layer"],
            # Legacy fields for fallback
            "overlay_diff": base_candidate["overlay_diff"],
            "conservative_diff": data.get("conservative_diff", {}),
            "prompt_alternatives": data.get("prompt_alternatives", {}),
            "candidate_rationale": data.get("candidate_rationale", []),
        }


def parse_text_materializer_response(raw: str, expected_candidates: int = 5) -> dict[int, dict[str, Any]]:
    """Parse the second-stage text materializer response.

    Returns a mapping: candidate_id -> text_overlay_diff.
    """
    data = _parse_jsonish_response(raw)

    parsed: dict[int, dict[str, Any]] = {}

    # Preferred strict shape: top-level fixed keys candidate_0 ... candidate_N.
    for idx in range(expected_candidates):
        key = f"candidate_{idx}"
        if key not in data:
            continue
        item = data[key]
        if not isinstance(item, dict):
            raise ValueError(
                f"Text materializer field '{key}' must be an object, got {type(item).__name__}"
            )
        diff = item.get("text_overlay_diff")
        if not isinstance(diff, dict):
            diff = _extract_text_overlay_from_dict(item)
        parsed[idx] = _normalize_text_knob_block(
            diff,
            context=f"materializer.{key}.text_overlay_diff",
        )

    if parsed:
        missing_ids = [idx for idx in range(expected_candidates) if idx not in parsed]
        if missing_ids:
            raise ValueError(
                f"Text materializer missing candidate outputs for ids: {missing_ids}"
            )
        return parsed

    raw_candidates = data.get("candidates", [])

    # If candidates is not a list, try to recover
    if not isinstance(raw_candidates, list):
        # LLM might have returned a single candidate dict at top level
        if isinstance(raw_candidates, dict):
            raw_candidates = [raw_candidates]
        else:
            # Try to treat the whole response as a single candidate
            text_diff = _extract_text_overlay_from_dict(data)
            if text_diff:
                return {0: text_diff}
            raise ValueError(
                "Text materializer response missing 'candidates' list and no "
                f"knob keys found. Keys present: {list(data.keys())}"
            )

    parsed = {}
    for idx, item in enumerate(raw_candidates):
        if not isinstance(item, dict):
            continue
        candidate_id = item.get("candidate_id", idx)
        try:
            candidate_id = int(candidate_id)
        except (TypeError, ValueError):
            candidate_id = idx
        if candidate_id < 0 or candidate_id >= expected_candidates:
            continue
        diff = item.get("text_overlay_diff")
        if not isinstance(diff, dict):
            # LLM may have put knob keys directly in the candidate dict
            diff = _extract_text_overlay_from_dict(item)
        parsed[candidate_id] = _normalize_text_knob_block(
            diff,
            context=f"materializer.candidates[{candidate_id}].text_overlay_diff",
        )
    missing_ids = [idx for idx in range(expected_candidates) if idx not in parsed]
    if missing_ids:
        raise ValueError(
            f"Text materializer missing candidate outputs for ids: {missing_ids}"
        )
    return parsed


def _extract_text_overlay_from_dict(data: dict) -> dict[str, Any]:
    """Extract persona./prompt. knob values from an arbitrary dict."""
    result: dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(key, str) and (key.startswith("persona.") or key.startswith("prompt.")):
            result[key] = value
    if not result:
        # Check nested dicts for knob keys
        for value in data.values():
            if isinstance(value, dict):
                for k, v in value.items():
                    if isinstance(k, str) and (k.startswith("persona.") or k.startswith("prompt.")):
                        result[k] = v
                if result:
                    break
    return result

