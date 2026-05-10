"""
Calibration iteration log — append-only, persisted as JSON.

CalibrationLog stores one record per calibration iteration and
provides lightweight query helpers for strategy tracking.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import List


class CalibrationLog:
    """Append-only log of calibration iteration records.

    Each entry is an arbitrary dict.  The expected schema (not enforced
    here) is:

        {
            "iteration":      int,
            "strategy_label": str,
            "best_fail_rate": float,
            "selection":      {"beat_current_best": bool, ...},
            "candidates":     [...],   # large — omitted in trajectory()
        }

    Persistence format
    ------------------
    JSON file: {"entries": [<entry>, ...]}
    """

    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        self._entries: list[dict] = []

        if self._path.exists():
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            self._entries = raw.get("entries", [])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def append(self, entry: dict) -> None:
        """Append *entry* and persist immediately to disk."""
        self._entries.append(entry)
        self._save()

    def upsert_iteration(self, entry: dict) -> None:
        """Replace an existing entry for *entry["iteration"]*, or append it."""
        iteration = entry.get("iteration")
        if iteration is None:
            self.append(entry)
            return
        for idx, existing in enumerate(self._entries):
            if existing.get("iteration") == iteration:
                self._entries[idx] = entry
                self._save()
                return
        self._entries.append(entry)
        self._save()

    def entries(self) -> List[dict]:
        """Return a shallow copy of all entries."""
        return list(self._entries)

    def failed_strategies(self) -> List[str]:
        """Return strategy labels from iterations that did not beat current best.

        For new-format entries with ``candidate_strategies``, includes each
        individual candidate's strategy_label.  For legacy entries, uses the
        top-level ``strategy_label``.
        """
        labels: list[str] = []
        for e in self._entries:
            if e.get("selection", {}).get("beat_current_best", True):
                continue
            cand_strats = e.get("candidate_strategies", [])
            if cand_strats:
                labels.extend(cs.get("strategy_label", "") for cs in cand_strats)
            else:
                labels.append(e.get("strategy_label", ""))
        return [l for l in labels if l]

    def trajectory(self) -> List[dict]:
        """Return entries without the 'candidates' key (lightweight summary)."""
        result = []
        for e in self._entries:
            slim = copy.copy(e)
            slim.pop("candidates", None)
            result.append(slim)
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps({"entries": self._entries}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
