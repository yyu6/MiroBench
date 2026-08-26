from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from .closing_move import build_closing_profile
from .comment_structure import build_structure_profile
from .data import load_real_thread_bank
from .domain import DomainConfig
from .entity_inventory import build_entity_inventory
from .entity_spread import build_entity_spread_profile
from .length_fidelity import build_length_fidelity_profile
from .lexical_quality import build_lexical_calibration
from .opener_profile import build_opener_profile
from .planning_quality import universal_viewpoints
from .generation_distribution import TONE_CLASSES
from .reference_metric_calibration import build_reference_metric_calibration
from .evaluative_register import build_evaluative_profile
from .opening_move import build_opening_profile
from .register_realization import build_register_profile
from .sentence_rhythm import build_rhythm_profile
from .surface_typography import (
    build_final_punctuation_profile,
    build_typography_profile,
)
from .tone_length_fit import build_tone_length_profile
from .reference_link import build_reference_link_inventory
from .viewpoint_bank import build_reference_viewpoints


# 18: the register profile becomes per-register. v99 measured `polite` only, and
# the v100 gate showed three of the four moves at exactly zero on every other
# register while real comments of those registers carry them -- real `impolite`
# comments carry an intensifier at 0.300 and a possessive at 0.182 against a
# generated 0.100 and 0.011. 75% of the corpus carried the deficit.
# 17: adds the measured per-band closing move. Real comments end on a concrete
# fact of the speaker's own (0.152) and almost never on an abstract verdict
# (0.014); generated output ends on a verdict 0.265 of the time, 19x, which is the
# root of the adjudication frame chased since v73 through phrase bans.
# 16: adds the measured per-band share of four positive-register surface moves
# among the evaluation classifier's own `polite` comments. The plan already puts
# polite on the right slots -- 0.275 against a real 0.288 -- and the Writer
# realizes it 19.3% of the time while realizing `impolite` 89.7%, so the register
# has to be drawn as a surface act rather than described in prose.
# 15: adds the measured per-band sentence rhythm and typing habits, drawn per
# slot, and the measured share of declarative endings left with no final
# punctuation. Two generated comments of the same size shared a function-word
# skeleton at cosine 0.502 against a real 0.368, which is where
# `self_bertscore_mean_f1` lives.
# 14: adds the measured P(tone class | comment size band) conditional, so the
# template's tone marginal is placed on the slot sizes that carry it in real
# threads instead of on the slots nearest each class's median length.
# 13: adds the measured per-size-band paragraph, list, and quoted-excerpt
# layout, so a long slot is asked for the shape a long comment actually has.
# 12: adds the measured typographic/keyboard punctuation share per class, drawn
# once per speaker so generated text carries a keyboard's punctuation instead of
# a language model's.
# 9: reference metric templates carry the classifier's measured
# somewhat_polite_rate, so a tone contract is a complete partition.
# 11: adds measured grammatical opener-type shares, scheduled per slot.
# 10: adds the evaluation-excluded equipment inventory. Restricting every
# comment's entities to the seed post made one generated thread reuse 23
# distinct models where the matched real thread used 117, which is a large
# share of the remaining self-BLEU gap.
# 21: adds the evaluation-excluded reference-link inventory. 4.47% of real
# comments in the excluded corpus carry a human reference URL and 0.00% of
# generated comments ever have. Removing URLs from real text and rescoring moves
# real's `self_bertscore` by +0.0094 -- 76% of the whole gap, and the only
# channel of the five tested that moves it at all.
# 22 adds two measured count distributions, both conditional on the habit being
# present so they compose with the existing share/routing decisions rather than
# replacing them: `rhythm_profile.bands[*].habit_counts` (v116, parenthetical
# asides per comment) and `reference_link_inventory.urls_per_carrier` (v117,
# reference URLs per carrying comment). An older profile simply lacks them and
# both arms fall back to a count of one, which is the pre-v116 behaviour.
PROFILE_SCHEMA_VERSION = 22
CARD_CONTEXT_DROPOUT_RATE = 0.42
CARD_CONTEXT_JITTER_RATE = 0.32
CARD_GENERATION_CONTROLS = {
    "advisor_max_share": 0.28,
    "question_max_share": 0.18,
    "micro_target_share": 0.07,
    "short_max_share": 0.18,
    "social_noise_min_share": 0.18,
    "gratitude_min_share": 0.12,
    "tone_harsh_max_share": 0.14,
    "tone_calm_min_share": 0.78,
    "tone_personal_min_share": 0.16,
    "story_personal_min_share": 0.16,
    "tone_polite_min_share": 0.24,
    "context_dropout_rate": CARD_CONTEXT_DROPOUT_RATE,
    "context_jitter_rate": CARD_CONTEXT_JITTER_RATE,
}
def build_domain_profile(
    config: DomainConfig,
    *,
    seed_pool_path: Path,
    output_path: Path,
    max_perspectives: int = 32,
) -> dict[str, Any]:
    """Build a frozen domain profile from threads outside the evaluation seed pool."""

    seed_payload = _load_json(seed_pool_path)
    excluded_ids = {
        str(row.get("source_raw_post_id") or "").strip()
        for row in seed_payload.get("seed_posts") or []
        if isinstance(row, dict)
    }
    unique: dict[str, dict[str, Any]] = {}
    for thread in load_real_thread_bank(config.raw_discussions_dir):
        post_id = str(thread.get("post_id") or "").strip()
        if not post_id or post_id in excluded_ids:
            continue
        old = unique.get(post_id)
        if old is None or int(thread.get("comment_count") or 0) > int(old.get("comment_count") or 0):
            unique[post_id] = thread
    reference_threads = sorted(unique.values(), key=lambda row: str(row.get("post_id") or ""))
    if len(reference_threads) < 20:
        raise ValueError(
            f"Domain profile requires at least 20 non-seed reference threads; found {len(reference_threads)}"
        )

    perspectives = universal_viewpoints(limit=max_perspectives)
    reference_viewpoints = build_reference_viewpoints(reference_threads)
    behavior_observations = _behavior_observations(reference_threads)
    lexical_quality = build_lexical_calibration(reference_threads)
    reference_metrics = build_reference_metric_calibration(
        config.raw_discussions_dir,
        reference_thread_ids=unique,
        excluded_seed_ids=excluded_ids,
    )
    behavior = dict(CARD_GENERATION_CONTROLS)
    metric_summary = reference_metrics.get("summary") or {}
    if reference_metrics.get("available"):
        behavior["story_personal_min_share"] = _clamp(
            float(metric_summary.get("mean_story_rate") or 0.0),
            0.0,
            0.30,
        )
        behavior["tone_polite_min_share"] = _clamp(
            float(metric_summary.get("mean_polite_rate") or 0.0),
            0.02,
            0.34,
        )
    payload: dict[str, Any] = {
        "schema_version": PROFILE_SCHEMA_VERSION,
        "domain_id": config.domain_id,
        "source": {
            "raw_discussions_dir": str(config.raw_discussions_dir),
            "seed_pool": str(seed_pool_path.resolve()),
            "excluded_seed_count": len(excluded_ids),
            "reference_thread_count": len(reference_threads),
            "reference_comment_count": sum(len(row.get("comments") or []) for row in reference_threads),
            "reference_viewpoint_count": len(reference_viewpoints),
            "seed_ids_sha256": _id_set_hash(excluded_ids),
            "reference_thread_ids_sha256": _id_set_hash(unique),
            "seed_reference_overlap_count": len(excluded_ids & set(unique)),
            "test_content_visible": False,
        },
        "configured_facets": list(config.topic_facets),
        "perspectives": perspectives,
        "reference_viewpoints": reference_viewpoints,
        "reference_link_inventory": build_reference_link_inventory(reference_threads),
        "behavior_targets": behavior,
        "behavior_observations": behavior_observations,
        "behavior_observation_method": (
            "surface-pattern diagnostics only; not used as semantic or metric ground truth"
        ),
        "lexical_quality": lexical_quality,
        "reference_metric_calibration": reference_metrics,
        "opener_profile": build_opener_profile(reference_threads),
        "typography_profile": build_typography_profile(reference_threads),
        "final_punctuation_profile": build_final_punctuation_profile(reference_threads),
        "structure_profile": build_structure_profile(reference_threads),
        "rhythm_profile": build_rhythm_profile(reference_threads),
        "closing_profile": build_closing_profile(reference_threads),
        "register_profile": build_register_profile(
            config.raw_discussions_dir,
            reference_thread_ids=unique,
        ),
        "opening_profile": build_opening_profile(
            config.raw_discussions_dir,
            reference_thread_ids=unique,
        ),
        "evaluative_profile": build_evaluative_profile(
            config.raw_discussions_dir,
            reference_thread_ids=unique,
        ),
        "tone_length_profile": build_tone_length_profile(
            config.raw_discussions_dir,
            reference_thread_ids=unique,
            tone_classes=TONE_CLASSES,
        ),
        "entity_inventory": build_entity_inventory(
            reference_threads,
            brand_terms=config.protected_entity_terms,
        ),
        "entity_spread_profile": build_entity_spread_profile(reference_threads),
        "length_fidelity_profile": build_length_fidelity_profile(reference_threads),
    }
    payload["profile_sha256"] = profile_hash(payload)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


def load_domain_profile(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {}
    payload = _load_json(Path(path).expanduser())
    if payload and payload.get("profile_sha256") != profile_hash(payload):
        raise ValueError(f"Domain profile hash mismatch: {path}")
    return payload


def profile_hash(payload: dict[str, Any]) -> str:
    copied = dict(payload)
    copied.pop("profile_sha256", None)
    serialized = json.dumps(copied, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def render_profile_for_planner(profile: dict[str, Any], *, limit: int = 24) -> str:
    rows = []
    for item in (profile.get("perspectives") or [])[:limit]:
        if not isinstance(item, dict):
            continue
        rows.append(
            f"- {item.get('perspective_id')}: {item.get('label')}; "
            f"axis={item.get('axis')}; decision question={item.get('decision_question')}"
        )
    return "\n".join(rows) or "- No frozen decision lenses are available."


def _behavior_observations(threads: list[dict[str, Any]]) -> dict[str, float]:
    comments = [
        str(row.get("body") or "").strip()
        for thread in threads
        for row in thread.get("comments") or []
        if str(row.get("body") or "").strip()
    ]
    total = max(1, len(comments))
    words = [len(text.split()) for text in comments]
    question = sum("?" in text for text in comments) / total
    gratitude = sum(bool(re.search(r"\b(thanks|thank you|appreciate|helpful|good to know)\b", text, re.I)) for text in comments) / total
    personal = sum(bool(re.search(r"\b(i|i'm|i've|my|mine|we|our)\b", text, re.I)) for text in comments) / total
    polite = sum(bool(re.search(r"\b(please|thanks|thank you|appreciate|would you|could you)\b", text, re.I)) for text in comments) / total
    harsh = sum(bool(re.search(r"\b(wrong|nonsense|stupid|ridiculous|bullshit|terrible|awful|hate)\b", text, re.I)) for text in comments) / total
    advisor = sum(bool(re.search(r"\b(you should|you could|try |consider |recommend|best bet|go with)\b", text, re.I)) for text in comments) / total
    social = sum(
        len(text.split()) <= 12
        and bool(re.search(r"\b(lol|lmao|haha|same|exactly|yep|nope|nice|agreed|this)\b", text, re.I))
        for text in comments
    ) / total
    micro = sum(value <= 5 for value in words) / total
    short = sum(value <= 18 for value in words) / total
    return {
        "advisor_max_share": _clamp(advisor * 1.25, 0.12, 0.38),
        "question_max_share": _clamp(question * 1.25, 0.08, 0.32),
        "micro_target_share": _clamp(micro, 0.02, 0.18),
        "short_max_share": _clamp(short * 1.15, 0.18, 0.48),
        "social_noise_min_share": _clamp(social * 0.80, 0.04, 0.24),
        "gratitude_min_share": _clamp(gratitude * 0.80, 0.01, 0.14),
        "tone_harsh_max_share": _clamp(harsh * 1.25, 0.03, 0.22),
        "tone_calm_min_share": _clamp((1.0 - harsh) * 0.30, 0.20, 0.55),
        "tone_personal_min_share": _clamp(personal * 0.70, 0.08, 0.38),
        "tone_polite_min_share": _clamp(polite * 0.80, 0.02, 0.24),
    }


def _clamp(value: float, low: float, high: float) -> float:
    return round(max(low, min(high, value)), 6)


def _id_set_hash(values: Any) -> str:
    normalized = "\n".join(sorted(str(value).strip() for value in values if str(value).strip()))
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}
