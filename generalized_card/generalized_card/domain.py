from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_ROOT.parent


@dataclass(frozen=True)
class DomainConfig:
    domain_id: str
    display_name: str
    community_context: str
    raw_discussions_dir: Path
    real_scores_csv: Path
    topic_facets: tuple[str, ...]
    technical_terms: tuple[str, ...]
    protected_entity_terms: tuple[str, ...]
    persona_expertise_dimensions: tuple[str, ...] = ()
    min_comments: int = 5

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DomainConfig":
        required = (
            "domain_id",
            "display_name",
            "community_context",
            "raw_discussions_dir",
            "real_scores_csv",
        )
        missing = [key for key in required if not str(payload.get(key) or "").strip()]
        if missing:
            raise ValueError("Domain config missing fields: " + ", ".join(missing))
        return cls(
            domain_id=str(payload["domain_id"]),
            display_name=str(payload["display_name"]),
            community_context=str(payload["community_context"]),
            raw_discussions_dir=_repo_path(payload["raw_discussions_dir"]),
            real_scores_csv=_repo_path(payload["real_scores_csv"]),
            topic_facets=_strings(payload.get("topic_facets")),
            technical_terms=_strings(payload.get("technical_terms")),
            protected_entity_terms=_strings(payload.get("protected_entity_terms")),
            persona_expertise_dimensions=_strings(
                payload.get("persona_expertise_dimensions")
            ),
            min_comments=max(1, int(payload.get("min_comments") or 5)),
        )

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "domain_id": self.domain_id,
            "display_name": self.display_name,
            "community_context": self.community_context,
            "raw_discussions_dir": str(self.raw_discussions_dir),
            "real_scores_csv": str(self.real_scores_csv),
            "topic_facets": list(self.topic_facets),
            "technical_terms": list(self.technical_terms),
            "protected_entity_terms": list(self.protected_entity_terms),
            "persona_expertise_dimensions": list(self.persona_expertise_dimensions),
            "min_comments": self.min_comments,
        }


def load_domain_config(value: str | Path) -> DomainConfig:
    path = resolve_domain_config_path(value)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Domain config must be a JSON object: {path}")
    config = DomainConfig.from_dict(payload)
    if not config.raw_discussions_dir.exists():
        raise FileNotFoundError(f"Raw discussion directory not found: {config.raw_discussions_dir}")
    return config


def load_domain_from_env() -> DomainConfig:
    value = os.environ.get("GENERALIZED_CARD_DOMAIN", "").strip()
    if not value:
        raise RuntimeError(
            "GENERALIZED_CARD_DOMAIN is required; domain adapters must never "
            "silently fall back to a specific dataset."
        )
    return load_domain_config(value)


def resolve_domain_config_path(value: str | Path) -> Path:
    raw = Path(value).expanduser()
    candidates = [raw]
    if raw.suffix != ".json":
        short_name = raw.name.removesuffix("_product")
        candidates.extend(
            [
                PACKAGE_ROOT / "configs" / "domains" / f"{raw.name}.json",
                PACKAGE_ROOT / "configs" / "domains" / f"{raw.name.replace('-', '_')}.json",
                PACKAGE_ROOT / "configs" / "domains" / f"{short_name}.json",
            ]
        )
    for candidate in candidates:
        resolved = candidate if candidate.is_absolute() else (REPO_ROOT / candidate)
        if resolved.exists():
            return resolved.resolve()
    raise FileNotFoundError(f"Domain config not found: {value}")


def _repo_path(value: Any) -> Path:
    path = Path(str(value)).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def _strings(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(str(item).strip() for item in value if str(item).strip())
