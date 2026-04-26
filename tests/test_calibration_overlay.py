"""
Tests for calibration/overlay.py — load/save/merge/diff functions.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from calibration.overlay import (
    diff_overlay,
    load_overlay,
    merge_overlay,
    save_overlay,
)

# ---------------------------------------------------------------------------
# Optional registry import for defaults-based tests
# ---------------------------------------------------------------------------
try:
    from calibration.registry import KnobRegistry
    HAS_REGISTRY = True
except ImportError:
    HAS_REGISTRY = False


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_DEFAULTS = {
    "temperature": 0.7,
    "top_p": 0.9,
    "max_tokens": 512,
    "presence_penalty": 0.0,
}


# ---------------------------------------------------------------------------
# merge_overlay
# ---------------------------------------------------------------------------

class TestMergeOverlay:
    def test_merge_with_empty_overlay(self):
        """Merging an empty overlay leaves defaults unchanged."""
        result = merge_overlay(SAMPLE_DEFAULTS, {})
        assert result == SAMPLE_DEFAULTS

    def test_merge_overrides_single_knob(self):
        """A single overlay value overrides the corresponding default."""
        overlay = {"temperature": 1.2}
        result = merge_overlay(SAMPLE_DEFAULTS, overlay)
        assert result["temperature"] == 1.2
        # Other keys should remain unchanged
        assert result["top_p"] == SAMPLE_DEFAULTS["top_p"]
        assert result["max_tokens"] == SAMPLE_DEFAULTS["max_tokens"]
        assert result["presence_penalty"] == SAMPLE_DEFAULTS["presence_penalty"]

    def test_merge_overrides_multiple_knobs(self):
        """Multiple overlay values are all applied."""
        overlay = {"temperature": 0.5, "max_tokens": 1024}
        result = merge_overlay(SAMPLE_DEFAULTS, overlay)
        assert result["temperature"] == 0.5
        assert result["max_tokens"] == 1024
        assert result["top_p"] == SAMPLE_DEFAULTS["top_p"]

    def test_merge_does_not_mutate_defaults(self):
        """merge_overlay must not modify the original defaults dict."""
        defaults_copy = dict(SAMPLE_DEFAULTS)
        overlay = {"temperature": 99.0}
        merge_overlay(SAMPLE_DEFAULTS, overlay)
        assert SAMPLE_DEFAULTS == defaults_copy

    def test_merge_does_not_mutate_overlay(self):
        """merge_overlay must not modify the overlay dict."""
        overlay = {"temperature": 99.0}
        overlay_copy = dict(overlay)
        merge_overlay(SAMPLE_DEFAULTS, overlay)
        assert overlay == overlay_copy

    def test_merge_overlay_adds_new_keys(self):
        """Overlay keys not in defaults are added to the result."""
        overlay = {"new_knob": 42}
        result = merge_overlay(SAMPLE_DEFAULTS, overlay)
        assert result["new_knob"] == 42

    @pytest.mark.skipif(not HAS_REGISTRY, reason="KnobRegistry not yet available")
    def test_merge_with_registry_defaults(self):
        """Works correctly when defaults come from KnobRegistry."""
        registry = KnobRegistry()
        defaults = registry.defaults()
        overlay = {}
        result = merge_overlay(defaults, overlay)
        assert result == defaults


# ---------------------------------------------------------------------------
# save_overlay / load_overlay
# ---------------------------------------------------------------------------

class TestSaveLoadRoundtrip:
    def test_save_and_load_roundtrip(self, tmp_path: Path):
        """Saving then loading an overlay returns the identical dict."""
        overlay = {"temperature": 0.5, "top_p": 0.8}
        dest = tmp_path / "overlay.json"
        save_overlay(overlay, dest)
        loaded = load_overlay(dest)
        assert loaded == overlay

    def test_save_creates_parent_dirs(self, tmp_path: Path):
        """save_overlay creates intermediate directories if needed."""
        dest = tmp_path / "nested" / "deep" / "overlay.json"
        save_overlay({"temperature": 0.3}, dest)
        assert dest.exists()

    def test_save_writes_valid_json(self, tmp_path: Path):
        """The saved file is valid, human-readable JSON (indented)."""
        overlay = {"temperature": 0.3, "max_tokens": 256}
        dest = tmp_path / "overlay.json"
        save_overlay(overlay, dest)
        raw = dest.read_text()
        parsed = json.loads(raw)
        assert parsed == overlay
        # Check formatting: indented JSON has newlines
        assert "\n" in raw

    def test_load_nonexistent_returns_empty(self, tmp_path: Path):
        """Loading a path that does not exist returns an empty dict."""
        missing = tmp_path / "nonexistent.json"
        result = load_overlay(missing)
        assert result == {}

    def test_load_empty_dict(self, tmp_path: Path):
        """A file containing {} loads as an empty dict."""
        dest = tmp_path / "empty.json"
        dest.write_text("{}")
        result = load_overlay(dest)
        assert result == {}


# ---------------------------------------------------------------------------
# diff_overlay
# ---------------------------------------------------------------------------

class TestDiffOverlay:
    def test_diff_no_changes(self):
        """Identical dicts produce an empty diff."""
        result = diff_overlay(SAMPLE_DEFAULTS, SAMPLE_DEFAULTS)
        assert result == {}

    def test_diff_detects_changes(self):
        """Changed values appear in the diff with b's value."""
        a = {"temperature": 0.7, "top_p": 0.9}
        b = {"temperature": 1.2, "top_p": 0.9}
        result = diff_overlay(a, b)
        assert result == {"temperature": 1.2}

    def test_diff_detects_additions(self):
        """Keys present in b but not a appear in the diff."""
        a = {"temperature": 0.7}
        b = {"temperature": 0.7, "new_knob": 42}
        result = diff_overlay(a, b)
        assert result == {"new_knob": 42}

    def test_diff_skips_none_values_in_b(self):
        """Keys in b with None values are excluded from the diff."""
        a = {"temperature": 0.7, "top_p": 0.9}
        b = {"temperature": 0.7, "top_p": None}
        result = diff_overlay(a, b)
        assert "top_p" not in result

    def test_diff_values_come_from_b(self):
        """Diff values are always taken from b, not a."""
        a = {"temperature": 0.7}
        b = {"temperature": 1.5}
        result = diff_overlay(a, b)
        assert result["temperature"] == 1.5

    def test_diff_ignores_deletions(self):
        """Keys in a but missing from b are not reported as changes."""
        a = {"temperature": 0.7, "top_p": 0.9}
        b = {"temperature": 0.7}
        result = diff_overlay(a, b)
        assert result == {}

    def test_diff_multiple_changes(self):
        """Multiple changed keys are all reported."""
        a = {"temperature": 0.7, "top_p": 0.9, "max_tokens": 512}
        b = {"temperature": 1.0, "top_p": 0.95, "max_tokens": 512}
        result = diff_overlay(a, b)
        assert result == {"temperature": 1.0, "top_p": 0.95}
