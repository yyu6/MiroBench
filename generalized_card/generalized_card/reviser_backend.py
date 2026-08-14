from __future__ import annotations

import importlib.util
import inspect
import json
import re
import sys
from types import ModuleType
from typing import Any

from . import prompts
from .core_contract import verify_core_contract
from .domain import DomainConfig, REPO_ROOT


LEGACY_SCRIPTS = {
    "selfbleu": REPO_ROOT / "scripts" / "postprocess_selfbleu_lexical_reviser.py",
    "selfbert": REPO_ROOT / "scripts" / "postprocess_selfbert_discourse_diversify.py",
    "tone": REPO_ROOT / "scripts" / "postprocess_tone_calibrated_reviser.py",
    "story": REPO_ROOT / "scripts" / "postprocess_story_probability_reviser.py",
    "structure": REPO_ROOT / "scripts" / "postprocess_structure_reviser.py",
}


REVISER_DOMAIN_BOUNDARIES = {
    "selfbleu": {
        "build_reviser_prompt",
        "filtered_named_entities",
        "preserves_numbers",
        "parse_reviser_response",
    },
    "selfbert": {
        "build_rewrite_prompt",
        "parse_response",
        "named_entities",
        "validate_rewrite",
        "chat_completion_text",
    },
    "tone": {
        "build_tone_prompt",
        "build_stance_prompt",
        "validate_candidate",
        "parse_reviser_response",
    },
    "story": {
        "build_prompt",
        "protected_entities",
        "protected_numbers",
        "parse_candidates",
    },
    "structure": {"named_entities"},
}


def load_reviser_backend(kind: str) -> ModuleType:
    path = LEGACY_SCRIPTS.get(kind)
    if path is None:
        raise ValueError(f"Unsupported reviser kind: {kind}")
    verify_core_contract((f"{kind}_reviser",))
    for import_path in (REPO_ROOT / "scripts", REPO_ROOT / "scripts" / "evaluation"):
        if str(import_path) not in sys.path:
            sys.path.insert(0, str(import_path))
    name = f"generalized_card_legacy_{kind}_reviser"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load reviser backend: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def configure_reviser_backend(
    module: ModuleType,
    *,
    kind: str,
    config: DomainConfig,
) -> ModuleType:
    """Adapt only CARD's domain boundary; preserve its reviser algorithms."""

    original_functions = {
        name: value
        for name, value in vars(module).items()
        if inspect.isfunction(value)
    }

    if kind == "selfbleu":
        original_prompt = module.build_reviser_prompt

        def build_selfbleu_prompt(**kwargs: Any) -> str:
            rendered = prompts.adapt_card_reviser_prompt(
                config,
                original_prompt(**kwargs),
                kind="selfbleu",
            )
            diagnostic = prompts.selfbleu_ngram_diagnostic(
                config,
                comments=kwargs["comments"],
                target=kwargs["target"],
            )
            return prompts.insert_reviser_guidance(rendered, diagnostic)

        module.build_reviser_prompt = build_selfbleu_prompt
        module.filtered_named_entities = lambda text: prompts.protected_entities(config, text)
        module.preserves_numbers = _preserves_numbers
        module.parse_reviser_response = parse_candidate_response
    elif kind == "selfbert":
        original_prompt = module.build_rewrite_prompt
        module.build_rewrite_prompt = lambda **kwargs: prompts.adapt_card_reviser_prompt(
            config,
            original_prompt(**kwargs),
            kind="selfbert",
        )
        module.parse_response = parse_selfbert_candidate_response
        module.named_entities = lambda text: prompts.protected_entities(config, text)
        module.validate_rewrite = _selfbert_candidate_validator(module, config)
        module.chat_completion_text = _retrying_chat_completion(module.chat_completion_text)
    elif kind == "tone":
        original_tone_prompt = module.build_tone_prompt
        original_stance_prompt = module.build_stance_prompt
        module.build_tone_prompt = lambda **kwargs: prompts.adapt_card_reviser_prompt(
            config,
            original_tone_prompt(**kwargs),
            kind="tone",
        )
        module.build_stance_prompt = lambda **kwargs: prompts.adapt_card_reviser_prompt(
            config,
            original_stance_prompt(**kwargs),
            kind="tone",
        )
        module.validate_candidate = _candidate_validator(module, config)
        module.parse_reviser_response = parse_candidate_response
    elif kind == "story":
        original_prompt = module.build_prompt
        module.build_prompt = lambda **kwargs: prompts.adapt_card_reviser_prompt(
            config,
            original_prompt(**kwargs),
            kind="story",
        )
        module.protected_entities = lambda text: prompts.protected_entities(config, text)
        module.protected_numbers = prompts.protected_numbers
        module.parse_candidates = parse_candidate_response
    elif kind == "structure":
        module.named_entities = lambda text: prompts.protected_entities(config, text)
    else:
        raise ValueError(f"Unsupported reviser kind: {kind}")
    changed_functions = sorted(
        name
        for name, original in original_functions.items()
        if getattr(module, name) is not original
    )
    unexpected = sorted(
        set(changed_functions) - REVISER_DOMAIN_BOUNDARIES[kind]
    )
    if unexpected:
        raise RuntimeError(
            f"Generalized {kind} adapter changed functions outside the declared "
            "domain boundary: " + ", ".join(unexpected)
        )
    module.GENERALIZED_CARD_REVISER_PARITY = {
        "kind": kind,
        "changed_backend_functions": changed_functions,
        "unexpected_backend_functions": unexpected,
        "domain_adaptation_boundaries": sorted(REVISER_DOMAIN_BOUNDARIES[kind]),
    }
    return module


def run_adapter_self_test(kind: str, config: DomainConfig) -> None:
    module = configure_reviser_backend(load_reviser_backend(kind), kind=kind, config=config)
    if kind == "selfbleu":
        assert callable(module.build_reviser_prompt)
        assert module.preserves_numbers("Sony A7 IV costs $2,000", "Sony A7 IV costs $2,000")
        assert not module.preserves_numbers("24mm at $500", "35mm at $500")
    elif kind == "selfbert":
        assert callable(module.build_rewrite_prompt)
        rows = module.parse_response(
            '{"candidates":[{"style":"x","discourse_job":"minor_tangent",'
            '"preserved_tone":"yes","preserved_story_mode":"yes",'
            '"preserved_stance":"yes","preserved_reply_relation":"yes",'
            '"text":"The lens cost is still the awkward part."}]}'
        )
        assert rows[0]["discourse_job"] == "minor_tangent"
    elif kind == "tone":
        assert callable(module.build_tone_prompt)
        assert callable(module.build_stance_prompt)
    elif kind == "story":
        assert callable(module.build_prompt)
        assert module.protected_numbers("24mm at $500") == {"24mm", "$500"}
    elif kind == "structure":
        assert "sony" in module.named_entities("Sony A7 IV")
    salvaged = parse_candidate_response(
        '{"candidates":[{"style":"a","text":"complete"},{"style":"b","text":"cut'
    )
    assert salvaged == [{"style": "a", "text": "complete"}]
    print(f"[adapter-self-test] kind={kind} domain={config.domain_id} PASS", flush=True)


def _candidate_validator(module: ModuleType, config: DomainConfig):
    from postprocess_metric_gated_candidate_replacement import (  # type: ignore
        PLANNER_LABEL_TOKENS,
        contains_placeholder_token,
        content_tokens,
        normalize_text,
    )
    from postprocess_selfbleu_lexical_reviser import claim_overlap  # type: ignore

    def validate(
        *,
        old: str,
        candidate: str,
        visible_context: str,
        min_claim_overlap: float,
        min_word_ratio: float,
        max_word_ratio: float,
    ) -> tuple[bool, str, float, float]:
        if not candidate:
            return False, "empty", 0.0, 0.0
        lower = candidate.lower()
        if any(token in lower for token in PLANNER_LABEL_TOKENS):
            return False, "planner_label_leakage", 0.0, 0.0
        if contains_placeholder_token(candidate):
            return False, "placeholder_leakage", 0.0, 0.0
        if "```" in candidate or len(candidate) > 1200:
            return False, "format_or_length_leakage", 0.0, 0.0
        if normalize_text(old) == normalize_text(candidate):
            return False, "unchanged", 1.0, 1.0
        old_words = max(1, len(old.split()))
        ratio = len(candidate.split()) / old_words
        overlap = claim_overlap(old, candidate)
        if ratio < min_word_ratio or ratio > max_word_ratio:
            return False, f"word_ratio_out_of_range:{ratio:.3f}", overlap, ratio
        if len(content_tokens(old)) >= 4 and overlap < min_claim_overlap:
            return False, f"claim_overlap_too_low:{overlap:.3f}", overlap, ratio
        if not _preserves_numbers(old, candidate):
            return False, "number_or_measurement_changed", overlap, ratio
        visible_entities = prompts.protected_entities(config, visible_context)
        old_entities = prompts.protected_entities(config, old)
        new_entities = prompts.protected_entities(config, candidate)
        missing = old_entities - new_entities
        if missing:
            return False, "old_named_entities_missing:" + ",".join(sorted(missing)[:6]), overlap, ratio
        added = new_entities - visible_entities - old_entities
        if added:
            return False, "new_named_entities:" + ",".join(sorted(added)[:6]), overlap, ratio
        return True, "valid", overlap, ratio

    return validate


def _selfbert_candidate_validator(module: ModuleType, config: DomainConfig):
    original = module.validate_rewrite

    def validate(**kwargs: Any) -> tuple[bool, str]:
        ok, reason = original(**kwargs)
        if not ok:
            return ok, reason
        old = str(kwargs.get("old") or "")
        candidate = str(kwargs.get("candidate") or "")
        if not _preserves_numbers(old, candidate):
            return False, "number_or_measurement_changed"
        old_entities = prompts.protected_entities(config, old)
        new_entities = prompts.protected_entities(config, candidate)
        missing = old_entities - new_entities
        if missing:
            return False, "old_named_entities_missing:" + ",".join(sorted(missing)[:6])
        return True, "valid"

    return validate


def _retrying_chat_completion(original: Any):
    def call(**kwargs: Any) -> str:
        import os
        import time

        try:
            retries = max(1, int(os.environ.get("LLM_API_RETRIES", "6") or 6))
        except ValueError:
            retries = 6
        try:
            delay = max(0.0, float(os.environ.get("LLM_API_RETRY_DELAY", "10") or 10))
        except ValueError:
            delay = 10.0
        last_error: Exception | None = None
        for attempt in range(retries):
            try:
                text = str(original(**kwargs) or "").strip()
                if not text:
                    raise RuntimeError("empty completion")
                return text
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if attempt + 1 >= retries:
                    raise
                print(
                    f"[selfbert-llm-retry] attempt={attempt + 1}/{retries} "
                    f"error={type(exc).__name__}:{exc}",
                    flush=True,
                )
                time.sleep(delay * (attempt + 1))
        assert last_error is not None
        raise last_error

    return call


def _preserves_numbers(old: str, candidate: str) -> bool:
    return prompts.protected_numbers(old) <= prompts.protected_numbers(candidate)


def parse_candidate_response(raw: str) -> list[dict[str, str]]:
    """Parse strict JSON and salvage complete candidates from truncated output."""

    text = re.sub(r"^```(?:json)?\s*", "", str(raw or "").strip(), flags=re.I)
    text = re.sub(r"\s*```$", "", text)
    for candidate in (text, _repair_json(text)):
        try:
            payload = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        rows = payload.get("candidates") if isinstance(payload, dict) else payload
        normalized = _normalize_candidates(rows)
        if normalized:
            return normalized

    rows: list[dict[str, str]] = []
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{\s*\"(?:style|text|new_content)\"", text):
        try:
            row, _ = decoder.raw_decode(text[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    normalized = _normalize_candidates(rows)
    if normalized:
        print(
            f"[reviser-parse-warning] salvaged={len(normalized)} complete candidates "
            f"from truncated response chars={len(text)}",
            flush=True,
        )
    else:
        print(
            f"[reviser-parse-warning] no complete candidates in response chars={len(text)}",
            flush=True,
        )
    return normalized


def parse_selfbert_candidate_response(raw: str) -> list[dict[str, str]]:
    rows = _parse_candidate_rows(raw)
    output: list[dict[str, str]] = []
    seen: set[str] = set()
    for row in rows:
        text = str(row.get("text") or row.get("new_content") or "").strip()
        key = " ".join(text.lower().split())
        if not text or key in seen:
            continue
        seen.add(key)
        output.append(
            {
                "style": str(row.get("style") or row.get("discourse_job") or "domain_neutral_rewrite").strip(),
                "discourse_job": str(row.get("discourse_job") or row.get("style") or "minor_tangent").strip(),
                "preserved_tone": str(row.get("preserved_tone") or "yes").strip(),
                "preserved_story_mode": str(row.get("preserved_story_mode") or "yes").strip(),
                "preserved_stance": str(row.get("preserved_stance") or "yes").strip(),
                "preserved_reply_relation": str(row.get("preserved_reply_relation") or "yes").strip(),
                "text": text,
                "why_different": str(row.get("why_different") or "").strip(),
            }
        )
    return output


def _parse_candidate_rows(raw: str) -> list[dict[str, Any]]:
    text = re.sub(r"^```(?:json)?\s*", "", str(raw or "").strip(), flags=re.I)
    text = re.sub(r"\s*```$", "", text)
    for candidate in (text, _repair_json(text)):
        try:
            payload = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        rows = payload.get("candidates") if isinstance(payload, dict) else payload
        if isinstance(rows, list):
            normalized = [row for row in rows if isinstance(row, dict)]
            if normalized:
                return normalized

    rows: list[dict[str, Any]] = []
    decoder = json.JSONDecoder()
    for match in re.finditer(r'\{\s*"(?:style|text|new_content)"', text):
        try:
            row, _ = decoder.raw_decode(text[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    if rows:
        print(
            f"[reviser-parse-warning] salvaged={len(rows)} complete candidates "
            f"from truncated response chars={len(text)}",
            flush=True,
        )
    return rows


def _normalize_candidates(rows: Any) -> list[dict[str, str]]:
    if not isinstance(rows, list):
        return []
    output: list[dict[str, str]] = []
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        text = str(row.get("text") or row.get("new_content") or "").strip()
        style = str(row.get("style") or "domain_neutral_rewrite").strip()
        key = " ".join(text.lower().split())
        if not text or key in seen:
            continue
        seen.add(key)
        output.append({"style": style, "text": text})
    return output


def _repair_json(text: str) -> str:
    value = text.replace("“", '"').replace("”", '"').replace("’", "'")
    value = re.sub(r",\s*([}\]])", r"\1", value)
    return value
