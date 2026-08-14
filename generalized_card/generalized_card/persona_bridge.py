from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Iterable


MODE_NONE = "none"
MODE_PROJECTED = "matraix-projected"
MODE_FULL = "matraix-full"
PERSONA_MODES = (MODE_NONE, MODE_PROJECTED, MODE_FULL)
AUDITED_MATRAIX_COMMIT = "e85c8772fc8a769ff70662c5368066024b6e15b8"

_MARKER_RE = re.compile(
    r"^\s*<generalized-card-matraix\s+"
    r'persona-id="(?P<persona_id>[^"]+)"\s+'
    r'seed-index="(?P<seed_index>\d+)"\s+'
    r'task-id="(?P<task_id>\d+)"\s*/>\s*',
    flags=re.MULTILINE,
)
_NULLISH = {"", "none", "null", "n/a", "na", "not applicable", "unknown"}
_MINOR_AGES = {"Under 5", "5-12", "13-17"}
_ENGLISH_OK = {"Native", "Fluent (C1-C2)", "Intermediate (B1-B2)"}

_MAX_PROJECTED_DIMENSIONS = 10
_IDENTITY_BOUNDARY = (
    "Express this identity only through word choice, confidence, attention, and "
    "interaction style. Never state the profile or invent biography, expertise, "
    "personal experience, or facts from it. The Reddit task and visible discussion "
    "below are the only sources of factual content."
)


@dataclass(frozen=True)
class PersonaAssignment:
    persona_id: str
    source_path: Path
    system_prompt: str
    selected_dimensions: dict[str, str]

    @property
    def system_sha256(self) -> str:
        return hashlib.sha256(self.system_prompt.encode("utf-8")).hexdigest()


class MatraixPersonaRuntime:
    """CARD adapter over MatrAIx's official loader and system renderer.

    The MatrAIx Harbor user-simulation loop targets one user chatting with an
    application. CARD instead needs many independent Reddit authors. This
    adapter therefore reuses MatrAIx's persona records, loader, Jinja renderer,
    and identity-first system channel while leaving CARD's thread planner,
    task controls, reply tree, writer guards, and memory unchanged.
    """

    def __init__(
        self,
        *,
        mode: str,
        matraix_root: Path,
        dataset_dir: Path,
        assignment_seed: int,
        expertise_dimensions: Iterable[str] = (),
    ) -> None:
        if mode not in PERSONA_MODES:
            raise ValueError(f"Unknown persona-conditioning mode: {mode}")
        self.mode = mode
        self.matraix_root = matraix_root.expanduser().resolve()
        self.dataset_dir = dataset_dir.expanduser().resolve()
        self.assignment_seed = int(assignment_seed)
        self.expertise_dimensions = tuple(
            item.strip() for item in expertise_dimensions if item.strip()
        )
        self._personas_by_id: dict[str, Any] = {}
        self._eligible: list[Any] = []
        self._system_cache: dict[str, PersonaAssignment] = {}
        self.commit = "disabled"
        self.template_path: Path | None = None
        self._official = None
        if self.mode != MODE_NONE:
            self._initialize()

    @property
    def enabled(self) -> bool:
        return self.mode != MODE_NONE

    def _initialize(self) -> None:
        if not self.matraix_root.is_dir():
            raise FileNotFoundError(
                f"MatrAIx repository not found: {self.matraix_root}. Clone "
                "https://github.com/MatrAIx-ai/MatrAIx-Persona-8B first."
            )
        if not self.dataset_dir.is_dir():
            raise FileNotFoundError(f"MatrAIx persona dataset not found: {self.dataset_dir}")
        self._official = _load_official_modules(self.matraix_root)
        self.commit = _git_commit(self.matraix_root)
        if (
            self.commit != AUDITED_MATRAIX_COMMIT
            and os.environ.get("GENERALIZED_CARD_ALLOW_MATRAIX_DRIFT") != "1"
        ):
            raise RuntimeError(
                f"MatrAIx commit is {self.commit}, expected audited commit "
                f"{AUDITED_MATRAIX_COMMIT}. Check out the audited commit, or set "
                "GENERALIZED_CARD_ALLOW_MATRAIX_DRIFT=1 only for a non-paper experiment."
            )
        load_persona = self._official["load_persona"]
        for path in sorted(self.dataset_dir.glob("persona_*.yaml")):
            persona = load_persona(path)
            if not persona.persona_id:
                continue
            self._personas_by_id[persona.persona_id] = persona
            if _eligible_for_english_reddit(persona.dimensions):
                self._eligible.append(persona)
        if not self._eligible:
            raise RuntimeError(
                f"No English-capable adult personas found in {self.dataset_dir}"
            )
        self.template_path = self._official["resolve_persona_template"](
            self._eligible[0],
            None,
            self._official["PERSONA_SYSTEM_TEMPLATE"],
        )

    def assign(self, *, seed_index: int, task: Any) -> PersonaAssignment | None:
        if not self.enabled:
            return None
        scored = [
            (_compatibility_score(persona.dimensions, task, self.expertise_dimensions), persona)
            for persona in self._eligible
        ]
        best = max(score for score, _persona in scored)
        # Keep a broad near-best set so role compatibility does not collapse the
        # population to a few repeated profiles.
        candidates = [persona for score, persona in scored if score >= best - 1]
        candidates.sort(
            key=lambda persona: _stable_rank(
                self.assignment_seed,
                seed_index,
                _task_value(task, "local_task_id", _task_value(task, "comment_id", 0)),
                persona.persona_id,
            )
        )
        return self.assignment_for_id(candidates[0].persona_id)

    def assignment_for_id(self, persona_id: str) -> PersonaAssignment:
        cached = self._system_cache.get(persona_id)
        if cached is not None:
            return cached
        persona = self._personas_by_id.get(persona_id)
        if persona is None:
            raise KeyError(f"Unknown MatrAIx persona_id: {persona_id}")
        selected = _project_dimensions(
            persona.dimensions,
            expertise_dimensions=self.expertise_dimensions,
        )
        rendered_persona = persona
        if self.mode == MODE_PROJECTED:
            Persona = self._official["Persona"]
            rendered_persona = Persona(
                persona_path=persona.persona_path,
                schema_version=persona.schema_version,
                data={"dimensions": selected},
                persona_id=persona.persona_id,
                version=persona.version,
                display_name=None,
                summary=None,
                system_prompt=None,
            )
        identity = self._official["render_persona_template"](
            self.template_path,
            rendered_persona,
        )
        system = f"{identity.strip()}\n\n{_IDENTITY_BOUNDARY}"
        assignment = PersonaAssignment(
            persona_id=persona_id,
            source_path=persona.persona_path,
            system_prompt=system,
            selected_dimensions=selected,
        )
        self._system_cache[persona_id] = assignment
        return assignment

    def marker(self, *, seed_index: int, task: Any) -> str:
        assignment = self.assign(seed_index=seed_index, task=task)
        if assignment is None:
            return ""
        task_id = int(
            _task_value(task, "local_task_id", _task_value(task, "comment_id", 0)) or 0
        )
        return (
            f'<generalized-card-matraix persona-id="{assignment.persona_id}" '
            f'seed-index="{int(seed_index)}" task-id="{task_id}"/>'
        )

    def public_config(self) -> dict[str, Any]:
        if not self.enabled:
            return {"mode": MODE_NONE}
        rendered_lengths = [
            len(self.assignment_for_id(persona.persona_id).system_prompt)
            for persona in self._eligible
        ]
        return {
            "mode": self.mode,
            "matraix_root": str(self.matraix_root),
            "matraix_commit": self.commit,
            "dataset_dir": str(self.dataset_dir),
            "dataset_personas": len(self._personas_by_id),
            "eligible_personas": len(self._eligible),
            "assignment_seed": self.assignment_seed,
            "expertise_dimensions": list(self.expertise_dimensions),
            "template_path": str(self.template_path),
            "template_source": "official-matraix-persona-system",
            "system_chars_min": min(rendered_lengths),
            "system_chars_max": max(rendered_lengths),
            "system_chars_mean": round(sum(rendered_lengths) / len(rendered_lengths), 2),
        }


def build_runtime(
    *,
    mode: str,
    matraix_root: Path,
    dataset_dir: Path,
    assignment_seed: int,
    expertise_dimensions: Iterable[str] = (),
) -> MatraixPersonaRuntime:
    return MatraixPersonaRuntime(
        mode=mode,
        matraix_root=matraix_root,
        dataset_dir=dataset_dir,
        assignment_seed=assignment_seed,
        expertise_dimensions=expertise_dimensions,
    )


@lru_cache(maxsize=1)
def runtime_from_env() -> MatraixPersonaRuntime:
    mode = os.environ.get("GENERALIZED_CARD_PERSONA_MODE", MODE_NONE).strip() or MODE_NONE
    repo_root = Path(__file__).resolve().parents[2]
    matraix_root = Path(
        os.environ.get(
            "GENERALIZED_CARD_MATRAIX_ROOT",
            str(repo_root / "third_party" / "MatrAIx-Persona-8B"),
        )
    )
    dataset = Path(
        os.environ.get(
            "GENERALIZED_CARD_PERSONA_DATASET",
            str(matraix_root / "persona" / "datasets" / "matraix-persona-dev-sample"),
        )
    )
    dimensions = _csv_values(
        os.environ.get("GENERALIZED_CARD_PERSONA_EXPERTISE_DIMENSIONS", "")
    )
    return build_runtime(
        mode=mode,
        matraix_root=matraix_root,
        dataset_dir=dataset,
        assignment_seed=int(os.environ.get("GENERALIZED_CARD_PERSONA_SEED", "42")),
        expertise_dimensions=dimensions,
    )


def reset_runtime_cache() -> None:
    runtime_from_env.cache_clear()


def persona_marker_for_task(seed_post: Any, task: Any) -> str:
    return runtime_from_env().marker(seed_index=int(seed_post.index), task=task)


def inject_persona_system(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    """Move the internal persona marker into a dedicated system channel."""

    runtime = runtime_from_env()
    if not runtime.enabled:
        return messages
    revised = [dict(message) for message in messages]
    persona_id = ""
    for message in revised:
        if message.get("role") != "user":
            continue
        content = str(message.get("content") or "")
        match = _MARKER_RE.search(content)
        if match is None:
            continue
        persona_id = match.group("persona_id")
        message["content"] = _MARKER_RE.sub("", content, count=1).lstrip()
        break
    if not persona_id:
        return revised
    identity = runtime.assignment_for_id(persona_id).system_prompt
    for message in revised:
        if message.get("role") == "system":
            message["content"] = f"{identity}\n\n{message.get('content', '')}".strip()
            break
    else:
        revised.insert(0, {"role": "system", "content": identity})
    return revised


def annotate_generated_outputs(
    generated_root: Path,
    runtime: MatraixPersonaRuntime,
) -> dict[str, Any]:
    """Persist deterministic per-comment persona provenance after generation."""

    if not runtime.enabled or not generated_root.exists():
        return {"mode": runtime.mode, "comments": 0}
    persona_counts: Counter[str] = Counter()
    prompt_lengths: list[int] = []
    comment_count = 0
    file_count = 0
    for path in sorted(generated_root.glob("run_*_sampled_reddit/discussion.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        changed = False
        by_comment_id: dict[int, dict[str, Any]] = {}
        for post in payload.get("posts") or []:
            seed_index = int(post.get("seed_index") or 0)
            for comment in _walk_comments(post.get("comments") or []):
                comment_id = int(comment.get("comment_id") or 0)
                task_id = comment_id % 10000
                proxy = dict(comment)
                proxy["local_task_id"] = task_id
                assignment = runtime.assign(seed_index=seed_index, task=proxy)
                if assignment is None:
                    continue
                meta = _assignment_meta(runtime, assignment)
                if comment.get("persona_conditioning") != meta:
                    comment["persona_conditioning"] = meta
                    changed = True
                by_comment_id[comment_id] = meta
                persona_counts[assignment.persona_id] += 1
                prompt_lengths.append(len(assignment.system_prompt))
                comment_count += 1
            for record in post.get("generation_records") or []:
                record_comment = record.get("comment")
                if not isinstance(record_comment, dict):
                    continue
                comment_id = int(record_comment.get("comment_id") or 0)
                meta = by_comment_id.get(comment_id)
                if meta is not None and record_comment.get("persona_conditioning") != meta:
                    record_comment["persona_conditioning"] = meta
                    changed = True
        if changed:
            _atomic_json(path, payload)
        file_count += 1
    manifest = {
        **runtime.public_config(),
        "discussion_files": file_count,
        "comments": comment_count,
        "unique_personas_used": len(persona_counts),
        "persona_comment_counts": dict(sorted(persona_counts.items())),
        "assigned_system_chars_min": min(prompt_lengths) if prompt_lengths else 0,
        "assigned_system_chars_max": max(prompt_lengths) if prompt_lengths else 0,
        "assigned_system_chars_mean": (
            round(sum(prompt_lengths) / len(prompt_lengths), 2) if prompt_lengths else 0.0
        ),
    }
    _atomic_json(generated_root.parent / "persona_assignment_manifest.json", manifest)
    return manifest


def _assignment_meta(
    runtime: MatraixPersonaRuntime,
    assignment: PersonaAssignment,
) -> dict[str, Any]:
    return {
        "provider": "MatrAIx-Persona-8B",
        "mode": runtime.mode,
        "persona_id": assignment.persona_id,
        "matraix_commit": runtime.commit,
        "dataset": runtime.dataset_dir.name,
        "system_sha256": assignment.system_sha256,
        "system_chars": len(assignment.system_prompt),
        "selected_dimensions": assignment.selected_dimensions,
    }


def _load_official_modules(root: Path) -> dict[str, Any]:
    paths = (root / "environment" / "agents", root / "src")
    for path in reversed(paths):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)
    from matraix.agents.persona.loader import Persona, load_persona
    from matraix.agents.persona.templating import (
        PERSONA_SYSTEM_TEMPLATE,
        render_persona_template,
        resolve_persona_template,
    )

    return {
        "Persona": Persona,
        "load_persona": load_persona,
        "PERSONA_SYSTEM_TEMPLATE": PERSONA_SYSTEM_TEMPLATE,
        "render_persona_template": render_persona_template,
        "resolve_persona_template": resolve_persona_template,
    }


def _eligible_for_english_reddit(dimensions: dict[str, Any]) -> bool:
    age = dimensions.get("age_bracket")
    if age in _MINOR_AGES:
        return False
    english = dimensions.get("english_proficiency", "__missing__")
    primary = dimensions.get("primary_language", "__missing__")
    if english in _ENGLISH_OK:
        return True
    # Missing evidence is allowed only when no conflicting language is known.
    return english == "__missing__" and primary in {"__missing__", "English"}


def _project_dimensions(
    dimensions: dict[str, Any],
    *,
    expertise_dimensions: Iterable[str],
) -> dict[str, str]:
    ordered = (
        "register",
        "tone_expected",
        "expertise_gap",
        *tuple(expertise_dimensions),
        "decision_style",
        "trust_level",
        "risk_tolerance",
        "emotional_state",
        "tech_savviness",
        "time_pressure",
        "skill_writing",
        "skill_storytelling",
        "dominant_trait",
        "intent",
        "query_complexity",
    )
    selected: dict[str, str] = {}
    for key in ordered:
        if key in selected:
            continue
        value = dimensions.get(key)
        text = str(value or "").strip()
        if text.lower() in _NULLISH:
            continue
        if key == "dominant_trait" and text == "Balanced":
            continue
        selected[key] = text
        if len(selected) >= _MAX_PROJECTED_DIMENSIONS:
            break
    return selected


def _compatibility_score(
    dimensions: dict[str, Any],
    task: Any,
    expertise_dimensions: Iterable[str],
) -> int:
    role = str(_task_value(task, "speaker_role", ""))
    voice = str(_task_value(task, "voice", ""))
    tone = str(_task_value(task, "tone_shape", _task_value(task, "tone_target", "")))
    expected = str(dimensions.get("tone_expected") or "")
    gap = str(dimensions.get("expertise_gap") or "")
    trust = str(dimensions.get("trust_level") or "")
    emotion = str(dimensions.get("emotional_state") or "")
    score = 0

    expertise_values = [str(dimensions.get(key) or "") for key in expertise_dimensions]
    has_domain_expertise = any(
        value in {"Aware", "Familiar", "Proficient", "Expert", "Some exposure", "Experienced", "Veteran"}
        for value in expertise_values
    )
    if role in {"advisor", "datapoint_only", "contrarian"}:
        score += int(has_domain_expertise)
        score += int(gap in {"Peer-level", "Expert testing the system", "Teaching the model"})
    if role == "confused_asker":
        score += 2 * int(gap == "Novice asking expert")
    if role == "gratitude_reply":
        score += int(expected == "Warm / empathetic") + int(trust == "Trusting")
    if role == "jokester":
        score += 2 * int(expected == "Playful")
    if role == "ranter":
        score += int(expected == "Blunt") + int(emotion == "Frustrated")
    if role == "contrarian" or "disagree" in tone or voice in {"blunt", "sarcastic"}:
        score += int(trust in {"Verifying", "Skeptical"}) + int(expected == "Blunt")
    if voice in {"polite_soft", "grateful"} or "polite" in tone:
        score += int(expected == "Warm / empathetic")
    if voice == "uncertain" or "uncertain" in tone:
        score += int(gap == "Novice asking expert")
    return score


def _task_value(task: Any, key: str, default: Any = None) -> Any:
    if isinstance(task, dict):
        return task.get(key, default)
    return getattr(task, key, default)


def _stable_rank(seed: int, seed_index: int, task_id: int, persona_id: str) -> str:
    value = f"{seed}:{seed_index}:{task_id}:{persona_id}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _git_commit(root: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _walk_comments(comments: list[dict[str, Any]]):
    for comment in comments:
        yield comment
        yield from _walk_comments(comment.get("replies") or [])


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _csv_values(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())
