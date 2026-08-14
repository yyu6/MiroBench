from __future__ import annotations

from datetime import datetime
from datetime import timezone
from pathlib import Path
from typing import Any
import inspect
import json
import random
import subprocess

def is_gratitude_text(text: str) -> bool:
    lowered = str(text or "").lower()
    return any(marker in lowered for marker in ("thank you", "thanks", "appreciate", "good to know", "best of luck"))

def normalize_apostrophe_text(text: str) -> str:
    return (
        str(text or "")
        .replace("’", "'")
        .replace("‘", "'")
        .replace("`", "'")
        .lower()
    )

def safe_getsource(obj: Any) -> str:
    try:
        return inspect.getsource(obj)
    except (OSError, TypeError):
        return ""

def run_git_text(cmd: list[str]) -> str:
    try:
        completed = subprocess.run(cmd, check=False, capture_output=True, text=True, cwd=Path.cwd())
    except Exception as exc:  # noqa: BLE001 - snapshot should not block generation
        return f"(failed to run {' '.join(cmd)}: {exc})\n"
    text = (completed.stdout or "") + (completed.stderr or "")
    return text if text else "\n"

def normalize_vocab_list(value: Any, vocabulary: tuple[str, ...], default: tuple[str, ...]) -> tuple[str, ...]:
    if not isinstance(value, list):
        return default
    vocab = set(vocabulary)
    cleaned: list[str] = []
    for item in value:
        text = str(item).strip().lower()
        if text in vocab and text not in cleaned:
            cleaned.append(text)
    return tuple(cleaned) if cleaned else default

def normalize_vocab_value(value: Any, vocabulary: tuple[str, ...], default: str) -> str:
    text = str(value or "").strip().lower()
    return text if text in set(vocabulary) else default

def normalize_claim_key(value: Any) -> str:
    text = " ".join(str(value or "").strip().lower().split())
    if not text:
        return ""
    return compact(text, 90)

def weighted_choice(rng: random.Random, choices: tuple[tuple[Any, float], ...]) -> Any:
    total = sum(max(0.0, float(weight)) for _, weight in choices)
    if total <= 0:
        return choices[0][0]
    point = rng.random() * total
    acc = 0.0
    for value, weight in choices:
        acc += max(0.0, float(weight))
        if point <= acc:
            return value
    return choices[-1][0]

def safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default

def nonempty(value: Any, default: str) -> str:
    text = str(value or "").strip()
    return text if text else default

def compact(text: str, max_chars: int) -> str:
    cleaned = " ".join(str(text or "").replace("\r", "\n").split())
    if len(cleaned) <= max_chars:
        return cleaned
    cut = cleaned[: max(0, max_chars - 1)]
    if " " in cut:
        cut = cut.rsplit(" ", 1)[0]
    return cut.rstrip(" ,.;:") + "..."

def first_line(text: str) -> str:
    for line in str(text or "").splitlines():
        if line.strip():
            return line.strip()
    return ""

def increment(counter: dict[str, int], key: str) -> None:
    counter[key] = counter.get(key, 0) + 1

def normalize_exact(text: str) -> str:
    return " ".join(text.lower().split())

def median_int(values: list[int]) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    return int(ordered[len(ordered) // 2])

def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default

def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
