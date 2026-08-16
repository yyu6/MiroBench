from __future__ import annotations

import ast
import importlib.util
import inspect
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import replace
from pathlib import Path
from types import ModuleType
from typing import Any

from .actor_conditioning import (
    MODE_DOMAIN_DERIVED,
    MODE_NONE,
    actor_author,
    actor_state_from_plan,
    assignment_key,
    enrich_normalized_plans,
)
from .branch_routing import (
    parent_slot_schedule,
    required_branch_count,
    root_branch_schedule,
)
from .data import find_matched_real_thread, load_real_thread_bank
from .core_contract import GENERALIZED_V2_ENGINE_FILES, verify_core_contract
from .domain import DomainConfig, REPO_ROOT
from .domain_claim import enrich_domain_claim_fields, normalized_domain_claim
from .opener_profile import OPENER_TYPES
from .domain_profile import load_domain_profile
from .first_pass_policy import generation_coverage, retain_explicitly_planned_tasks
from .generation_distribution import (
    allocate_story_and_affect,
    apply_planner_distribution_fields,
    enrich_distribution_plan_fields,
    select_thread_template,
    set_social_contract_coherence,
)
from .generation_diversity import (
    build_thread_distribution_target,
    distribution_target_with_slot_progress,
    semantic_distribution_problem,
    semantic_thread_diagnostics,
)
from .lexical_quality import (
    lexical_overlap_problem as calibrated_lexical_overlap_problem,
)
from .length_policy import (
    is_soft_length_problem,
    soft_length_guidance,
    writer_provider_token_budget,
    writer_safety_token_cap,
)
from .long_form_planning import enrich_development_plan_fields
from .planner_distribution import (
    apply_slot_distribution_schedule,
    build_slot_distribution_schedule,
)
from .planning_quality import (
    PlanSemanticIndex,
    evaluate_plan_batch,
    ledger_entry,
    normalize_substantive_plan_shape,
)
from .persona_bridge import inject_persona_system
from .surface_contract import (
    infer_surface_shape,
    infer_surface_skeleton,
    reconcile_substantive_task,
    surface_only_label,
)
from .task_distribution import rebalance_card_surfaces, restore_planner_task_contract
from .writer_quality import (
    LOCAL_REPAIR_STRATEGIES,
    annotate_writer_attempts,
    deduplicate_problems,
    distribution_candidate_is_reachable,
    distribution_candidate_rank,
    hard_realization_problems,
    is_single_stage_diagnostic,
    last_writer_problems,
    only_distribution_writer_problems,
    only_repairable_writer_problems,
    only_style_problems,
    set_repetition_guard,
    writer_distribution_problems,
    writer_local_repair_task,
)
from .writer_grounding import (
    license_mode,
    system_prompt_fact_sentence,
)
from .speaker_roster import (
    EMPTY_ROSTER,
    SPEAKER_IDENTITY_MATCHED,
    SpeakerRoster,
    build_speaker_roster,
)
from . import prompts


CARD_SNAPSHOT_PROFILE = "card-snapshot"
GENERALIZED_V2_PROFILE = "generalized-v2"
DEFAULT_GENERATOR_PROFILE = GENERALIZED_V2_PROFILE
GENERATOR_PROFILES = (GENERALIZED_V2_PROFILE, CARD_SNAPSHOT_PROFILE)

CARD_SNAPSHOT_BACKEND = (
    REPO_ROOT
    / "artifacts"
    / "pipeline_snapshots"
    / "v37_gpt_writer_selfbleu_3rounds_20260704"
    / "source_snapshots"
    / "generator_v37_surface_tone_balanced.py"
)
GENERALIZED_V2_BACKEND = (
    REPO_ROOT
    / "scripts"
    / "sampling_generator"
    / "run_sampled_reddit_generator.py"
)
DEFAULT_BACKEND = GENERALIZED_V2_BACKEND


# Algorithms that must still be the shared implementation after the adapter has
# finished patching. Each name is read off the module before patching, so a name
# listed here has to exist. The CARD surface rebalancers used to be listed too;
# `task_distribution.rebalance_card_surfaces` discards the core rebalancer it is
# handed (`del core_rebalance`), so none of them could run and their definitions
# have been removed.
CORE_ALGORITHM_SYMBOLS = (
    "sample_thread_target",
    "plan_thread",
    "plan_comment_moves",
    "selected_matched_comments",
    "expand_plan_to_tasks",
    "choose_context_transform",
    "generate_post_from_tasks",
    "generate_writer_text_with_guards",
    "validate_writer_text",
    "writer_temperature",
    "writer_token_cap",
    "completed_seed_slots",
    "replace_or_append_post",
    "update_global_memory",
)


DOMAIN_ADAPTATION_BOUNDARIES = (
    "GENERATOR_NAME",
    "CLAIM_FAMILIES",
    "SYSTEM_PROMPTS.gpt54_reddit_writer",
    "load_real_thread_bank",
    "find_matched_real_thread",
    "build_planner_prompt",
    "build_comment_move_planner_prompt",
    "normalize_comment_move_plans",
    "plan_comment_move_batch",
    "build_writer_prompt",
    "render_parent_context_for_writer",
    "render_seed_context_for_writer",
    "seed_post_gist",
    "mask_high_salience_context_terms",
    "extract_product_anchors",
    "extract_term_anchors",
    "extract_concrete_anchors",
    "build_concrete_anchors_for_task",
    "infer_real_surface_shape",
    "infer_surface_skeleton",
    "infer_real_tone_slot",
    "infer_real_comment_social_overrides",
    "real_text_allows_first_person_frame",
    "real_text_allows_uncertainty_frame",
    "finalize_rebalanced_task",
    "expand_matched_real_sample_to_tasks",
    "sanitize_writer_text",
    "shape_writer_text_for_task",
    "contains_planner_skeleton_residue",
    "lexical_overlap_problem",
    "has_blocking_guard_failure",
    "degraded_task_for_guard_failure",
    "has_realistic_long_helpful_anchor",
    "writer_length_rule",
    "retry_note_for_problems",
    "guard_fallback_retry_note",
    "parse_json_object",
    "chat_completion_text",
    "preflight_openai_compatible_endpoint",
    "load_or_init_discussion",
    "run_self_test",
)

GENERALIZED_ALGORITHM_EXTENSIONS = (
    "rebalance_tasks_for_diversity",
    "normalize_branch_plan",
    "generate_writer_text_with_guards",
    "generate_post_from_tasks",
    "writer_token_cap",
    "backfill_repeated_claim_task",
)


def load_generator_backend(
    path: Path | None = None,
    *,
    profile: str = DEFAULT_GENERATOR_PROFILE,
) -> ModuleType:
    if profile not in GENERATOR_PROFILES:
        raise ValueError(f"Unknown generator profile: {profile}")
    if path is None:
        verify_core_contract(
            GENERALIZED_V2_ENGINE_FILES
            if profile == GENERALIZED_V2_PROFILE
            else ("generator",)
        )
    default_source = (
        GENERALIZED_V2_BACKEND
        if profile == GENERALIZED_V2_PROFILE
        else CARD_SNAPSHOT_BACKEND
    )
    source = (path or default_source).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"Generator backend not found: {source}")
    module_name = f"generalized_card_{profile.replace('-', '_')}_generator_backend"
    spec = importlib.util.spec_from_file_location(module_name, source)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load generator backend: {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def configure_generator_backend(
    module: ModuleType,
    config: DomainConfig,
    *,
    profile: str = DEFAULT_GENERATOR_PROFILE,
) -> ModuleType:
    """Inject domain-neutral boundaries while preserving the paper CARD core.

    The shared generator owns structural sampling, deterministic task
    balancing, context apertures, writer guards, output format, and resume
    semantics. The adapter changes data access and fixed credit-card language,
    then applies the explicitly declared domain-neutral distribution extensions.
    """

    if profile not in GENERATOR_PROFILES:
        raise ValueError(f"Unknown generator profile: {profile}")

    original_core_symbols = {
        name: getattr(module, name)
        for name in CORE_ALGORITHM_SYMBOLS
    }
    original_functions = {
        name: value
        for name, value in vars(module).items()
        if inspect.isfunction(value)
    }
    original_normalize_branch_plan = module.normalize_branch_plan
    original_normalize_comment_plans = module.normalize_comment_move_plans
    original_comment_planner_batch = module.plan_comment_move_batch
    original_writer_lifecycle = module.generate_writer_text_with_guards
    original_writer_token_cap = module.writer_token_cap
    original_generate_post = module.generate_post_from_tasks
    original_shape_writer_text = module.shape_writer_text_for_task
    original_seed_gist = module.seed_post_gist
    original_backfill_claim = getattr(module, "backfill_repeated_claim_task", None)
    original_system_prompt = module.SYSTEM_PROMPTS["gpt54_reddit_writer"]
    original_planner_residue_check = module.contains_planner_skeleton_residue
    original_blocking_guard_check = module.has_blocking_guard_failure
    comment_plan_ledgers: dict[str, list[dict[str, Any]]] = {}
    module.GENERALIZED_COMMENT_PLAN_HISTORY = []
    module.GENERALIZED_COMMENT_PLAN_FEEDBACK = ""
    module.GENERALIZED_COMMENT_PLAN_REPORTS = []
    module.GENERALIZED_STORY_AFFECT_REPORTS = []
    module.GENERALIZED_ACTIVE_DISTRIBUTION_TARGET = {}
    module.GENERALIZED_ACTIVE_REFERENCE_TEMPLATE = {}
    module.GENERALIZED_ACTIVE_SLOT_DISTRIBUTION_SCHEDULE = {}
    module.GENERALIZED_ACTIVE_SEED_KEY = ""
    module.GENERALIZED_ACTIVE_PLANNER_COVERAGE = {}
    module.GENERALIZED_ACTOR_MODE = os.environ.get(
        "GENERALIZED_CARD_ACTOR_CONDITIONING", MODE_NONE
    ).strip()
    module.GENERALIZED_STORY_PERSONAL_MIN_SHARE = _env_float(
        "GENERALIZED_CARD_STORY_PERSONAL_MIN_SHARE",
        0.16,
        minimum=0.0,
        maximum=1.0,
    )
    module.GENERALIZED_ACTOR_ASSIGNMENTS = {}
    module.GENERALIZED_DOMAIN_CLAIMS = {}
    # Ablation switch. A planned domain claim reached 508 of 522 comments in
    # v71 against 0 in v69, which is a large enough intervention that it has to
    # be separable from everything else in the same release. "off" keeps the
    # Planner's claim out of the Writer's prompt without changing planning.
    module.GENERALIZED_DOMAIN_CLAIM_MODE = (
        os.environ.get("GENERALIZED_CARD_DOMAIN_CLAIM", "planned").strip().lower()
        or "planned"
    )
    # "full" reproduces policy v73's Writer prompt exactly; "focused" keeps only
    # the controls a currently-passing metric depends on. See
    # `prompts._focused_writer_prompt` for the measurement behind the split.
    module.GENERALIZED_WRITER_PROMPT_MODE = (
        os.environ.get("GENERALIZED_CARD_WRITER_PROMPT", "focused").strip().lower()
        or "focused"
    )
    # Ablation switch for how the Planner's move reaches the Writer. "say_only"
    # reproduces v73/v74 on both sides -- the writer's "Say this, and only this"
    # and the reply planner's "a full sentence" schema. "own_words" states the
    # move as a specification. Measured plan echo (longest contiguous shared word
    # run >= 12 between `semantic_move` and its comment): v67 0.4%, v73 10.2%,
    # v74 25.8%. See `prompts._route_lock_mode`.
    module.GENERALIZED_WRITER_ROUTE_LOCK = (
        os.environ.get("GENERALIZED_CARD_WRITER_ROUTE_LOCK", "own_words").strip().lower()
        or "own_words"
    )
    module.GENERALIZED_SOCIAL_CONTRACT_COHERENCE = (
        os.environ.get("GENERALIZED_CARD_SOCIAL_CONTRACT_COHERENCE", "on")
        .strip()
        .lower()
        or "on"
    )
    set_social_contract_coherence(module.GENERALIZED_SOCIAL_CONTRACT_COHERENCE)
    module.GENERALIZED_REPLY_SIBLING_VISIBILITY = (
        os.environ.get("GENERALIZED_CARD_REPLY_SIBLING_VISIBILITY", "on")
        .strip()
        .lower()
        or "on"
    )
    # Ablation switch for the fact ban. "off" reproduces policy v75: one blanket
    # prohibition covering the seed product and the speaker's own history alike,
    # which put a permission ("Equipment you may claim as your own") and its
    # revocation ("do not invent ... or personal experiences") into the same
    # prompt for 170 of 522 slots. "own" splits them -- the product under
    # discussion stays grounded in what is visible, the speaker's own kit and
    # history become theirs to state concretely. Measured target: 0.08
    # specifications per comment against 0.55 real, 6.6 novel brand or model
    # tokens per thread against 47.3. See `writer_grounding`.
    module.GENERALIZED_OWN_FACT_LICENSE = (
        os.environ.get("GENERALIZED_CARD_OWN_FACT_LICENSE", "off").strip().lower()
        or "off"
    )
    # Ablation switch for participation structure. "off" reproduces every run up
    # to v76: the author name is a pure function of the slot index, so a
    # 186-comment thread is 186 people who each speak once. "matched" recovers
    # the matched real thread's own structure through `real_sample_id`, which for
    # the ten evaluation seeds is 265 named participants over 559 comments --
    # 2.11 each, with 68% of comment mass written by someone who speaks more than
    # once. See `speaker_roster`.
    module.GENERALIZED_SPEAKER_IDENTITY = (
        os.environ.get("GENERALIZED_CARD_SPEAKER_IDENTITY", "off").strip().lower()
        or "off"
    )
    # Ablation switch for the repetition guard. "off" keeps every prior release's
    # behaviour, in which `template_phrase_reused` fires and is discarded.
    # "blocking" lets it force another Writer attempt. See
    # `writer_quality.REPETITION_DIAGNOSTIC_PROBLEMS` for the measurement.
    module.GENERALIZED_REPETITION_GUARD = set_repetition_guard(
        os.environ.get("GENERALIZED_CARD_REPETITION_GUARD", "off")
    )
    module.GENERALIZED_ACTIVE_SPEAKER_ROSTER = EMPTY_ROSTER
    module.GENERALIZED_OPENER_TYPES = {}
    module.GENERALIZED_SELF_TEST_ACTIVE = False
    # This bounds retries for one Writer slot, not the number of threads or
    # comments that can be repaired.  A failed diversity candidate is never
    # silently retained after this budget is exhausted.
    module.GENERALIZED_WRITER_DIVERSITY_CONFIG = {
        "local_repair_rounds": _env_int(
            "GENERALIZED_CARD_WRITER_LOCAL_REPAIRS", 6, minimum=0
        ),
        # Candidate retries are bounded. A failed slot is escalated to the
        # post-level recovery loop rather than spending without limit.
        "slot_retry_limit": _env_int(
            "GENERALIZED_CARD_WRITER_SLOT_RETRY_LIMIT", 6, minimum=0
        ),
        # Hard completion is independent of metric-driven candidate retries.
        # It only replaces output that cannot be persisted at all.
        "hard_recovery_rounds": _env_int(
            "GENERALIZED_CARD_WRITER_HARD_RECOVERY_ROUNDS", 2, minimum=0
        ),
    }
    module.GENERALIZED_PLAN_QUALITY_CONFIG = {
        "repair_rounds": _env_int("GENERALIZED_CARD_PLAN_REPAIRS", 2, minimum=0),
        # A missing S# is incomplete JSON/schema output, not a quality choice.
        # Recover only that omitted slot before Writer work begins.
        "schema_recovery_rounds": _env_int(
            "GENERALIZED_CARD_PLAN_SCHEMA_RECOVERY_ROUNDS", 2, minimum=0
        ),
        "similarity_threshold": _env_float(
            "GENERALIZED_CARD_PLAN_SIMILARITY_THRESHOLD", 0.72, minimum=0.0, maximum=1.0
        ),
        "embedding_enabled": os.environ.get(
            "GENERALIZED_CARD_PLAN_EMBEDDING_ENABLED", "1"
        ) == "1",
        "embedding_model": os.environ.get(
            "GENERALIZED_CARD_PLAN_EMBEDDING_MODEL",
            "sentence-transformers/all-mpnet-base-v2",
        ).strip(),
        "embedding_device": os.environ.get(
            "GENERALIZED_CARD_PLAN_EMBEDDING_DEVICE", "cpu"
        ).strip(),
        "embedding_similarity_threshold": _env_float(
            "GENERALIZED_CARD_PLAN_EMBEDDING_THRESHOLD",
            0.70,
            minimum=0.0,
            maximum=1.0,
        ),
        "max_collision_rate": _env_float(
            "GENERALIZED_CARD_PLAN_MAX_COLLISION_RATE", 0.10, minimum=0.0, maximum=1.0
        ),
        "max_perspective_share": _env_float(
            "GENERALIZED_CARD_MAX_PERSPECTIVE_SHARE", 0.34, minimum=0.05, maximum=1.0
        ),
        "strict": os.environ.get("GENERALIZED_CARD_STRICT_PLAN_QUALITY", "1") == "1",
        "require_reply_novelty": True,
    }
    module.GENERALIZED_PLAN_SEMANTIC_INDEX = (
        PlanSemanticIndex(
            model_name=module.GENERALIZED_PLAN_QUALITY_CONFIG["embedding_model"],
            device=module.GENERALIZED_PLAN_QUALITY_CONFIG["embedding_device"],
        )
        if module.GENERALIZED_PLAN_QUALITY_CONFIG["embedding_enabled"]
        else None
    )

    module.GENERATOR_NAME = f"generalized_card_{config.domain_id}_planner_writer"
    module.GENERALIZED_DOMAIN_PROFILE = load_domain_profile(
        os.environ.get("GENERALIZED_CARD_DOMAIN_PROFILE", "").strip() or None
    )
    module.CLAIM_FAMILIES = prompts.GENERIC_CLAIM_FAMILIES
    # The core system prompt is pinned in `engine/vocabulary.py` and its own ban
    # ("... or product details unless they are visible in the prompt") is the
    # third layer of the fact prohibition. The license is appended here rather
    # than edited into the pinned string, so `off` leaves the core untouched.
    module.SYSTEM_PROMPTS["gpt54_reddit_writer"] = _generalize_instruction_text(
        original_system_prompt,
        config,
    ) + system_prompt_fact_sentence(mode=license_mode(module))
    module.run_self_test = lambda: _run_generalized_self_test(module, config)

    module.load_real_thread_bank = lambda raw_dir, max_threads=0, **_: load_real_thread_bank(
        Path(raw_dir), max_threads=max_threads
    )
    module.find_matched_real_thread = find_matched_real_thread
    reference_calibration = dict(
        module.GENERALIZED_DOMAIN_PROFILE.get("reference_metric_calibration") or {}
    )

    def activate_reference_template(seed_post: Any, target: Any) -> None:
        seed_key = str(
            getattr(seed_post, "source_raw_post_id", "")
            or getattr(seed_post, "index", "")
            or getattr(seed_post, "title", "")
        )
        comment_count = max(1, int(getattr(target, "target_comments", 0) or 0))
        if seed_key != module.GENERALIZED_ACTIVE_SEED_KEY:
            module.GENERALIZED_ACTIVE_SEED_KEY = seed_key
            module.GENERALIZED_ACTIVE_REFERENCE_TEMPLATE = (
                select_thread_template(
                    reference_calibration,
                    comment_count=comment_count,
                    seed_key=seed_key,
                )
                or {}
            )

    def generalized_planner_prompt(**kwargs: Any) -> str:
        activate_reference_template(kwargs["seed_post"], kwargs["target"])
        matched = dict(kwargs.get("matched_real_thread") or {})
        target = kwargs["target"]
        selected = (
            module.selected_matched_comments(
                matched_real_thread=matched,
                target=target,
                matched_real_comments=int(kwargs.get("matched_real_comments") or 0),
            )
            if matched
            else []
        )
        available_lenses = len(
            [
                item
                for item in (module.GENERALIZED_DOMAIN_PROFILE.get("perspectives") or [])
                if isinstance(item, dict) and item.get("perspective_id")
            ]
        )
        minimum = required_branch_count(
            selected,
            maximum=available_lenses or None,
        )
        module.GENERALIZED_ACTIVE_MIN_BRANCHES = minimum
        return prompts.planner_prompt(
            config,
            module,
            minimum_branch_count=minimum,
            **kwargs,
        )

    def generalized_comment_planner_prompt(**kwargs: Any) -> str:
        activate_reference_template(kwargs["seed_post"], kwargs["target"])
        all_comments = list(kwargs.get("all_comments") or [])
        if all_comments:
            module.GENERALIZED_ACTIVE_SLOT_DISTRIBUTION_SCHEDULE = (
                build_slot_distribution_schedule(
                    opener_profile=(module.GENERALIZED_DOMAIN_PROFILE or {}).get("opener_profile"),
                    template=dict(module.GENERALIZED_ACTIVE_REFERENCE_TEMPLATE or {}),
                    comments=all_comments,
                    total_comments=int(kwargs["target"].target_comments),
                )
            )
        return prompts.comment_planner_prompt(
            config,
            module,
            prior_plans=list(module.GENERALIZED_COMMENT_PLAN_HISTORY),
            validation_feedback=str(module.GENERALIZED_COMMENT_PLAN_FEEDBACK or ""),
            **kwargs,
        )

    if profile == GENERALIZED_V2_PROFILE:
        module.build_planner_prompt = generalized_planner_prompt

        def normalize_thread_branches(payload: dict[str, Any], **kwargs: Any):
            minimum = int(getattr(module, "GENERALIZED_ACTIVE_MIN_BRANCHES", 3) or 3)
            target = kwargs.get("target")
            normalization_kwargs = dict(kwargs)
            if target is not None:
                # The shared parser treats every root as a unique semantic
                # branch. Generalized planning instead uses a topology-derived
                # semantic-axis budget and routes repeated roots as distinct
                # discourse instances of that axis.
                normalization_kwargs["target"] = replace(
                    target,
                    top_level_comments=minimum,
                )
            branches = original_normalize_branch_plan(payload, **normalization_kwargs)
            if len(branches) < minimum:
                raise ValueError(
                    f"not enough independent branch plans: {len(branches)} < {minimum}"
                )
            available = [
                str(item.get("perspective_id") or "").strip().upper()
                for item in (module.GENERALIZED_DOMAIN_PROFILE.get("perspectives") or [])
                if isinstance(item, dict) and item.get("perspective_id")
            ]
            normalized = []
            used: set[str] = set()
            for index, branch in enumerate(branches):
                perspective = str(getattr(branch, "perspective_id", "") or "").strip().upper()
                if not perspective or perspective == "SEED_LOCAL" or perspective in used:
                    perspective = next(
                        (value for value in available if value not in used),
                        available[index % len(available)] if available else "SEED_LOCAL",
                    )
                used.add(perspective)
                normalized.append(replace(branch, perspective_id=perspective))
            return normalized

        module.normalize_branch_plan = normalize_thread_branches
        module.build_comment_move_planner_prompt = generalized_comment_planner_prompt

        def normalize_comment_plans(payload: dict[str, Any], **kwargs: Any):
            normalized = original_normalize_comment_plans(payload, **kwargs)
            if module.GENERALIZED_ACTOR_MODE == MODE_DOMAIN_DERIVED:
                enrich_normalized_plans(payload, normalized)
            normalized = enrich_distribution_plan_fields(payload, normalized)
            normalized = enrich_development_plan_fields(payload, normalized)
            normalized = enrich_domain_claim_fields(payload, normalized)
            return apply_slot_distribution_schedule(
                normalized,
                module.GENERALIZED_ACTIVE_SLOT_DISTRIBUTION_SCHEDULE,
            )

        module.normalize_comment_move_plans = normalize_comment_plans
        module.build_writer_prompt = lambda **kwargs: prompts.writer_prompt(
            config,
            module,
            **kwargs,
        )
        module.render_parent_context_for_writer = lambda **kwargs: prompts.render_parent_context(
            config,
            module,
            **kwargs,
        )
        module.render_seed_context_for_writer = lambda **kwargs: prompts.render_seed_context(
            config,
            module,
            **kwargs,
        )
    else:
        # The card-snapshot profile audits a frozen historical artifact that
        # still carries CARD's own prompt builders, so its originals are read
        # here rather than above: the generalized-v2 engine renders every prompt
        # from `prompts.py` and no longer defines them.
        original_planner_prompt = module.build_planner_prompt
        original_comment_planner_prompt = module.build_comment_move_planner_prompt
        original_writer_prompt = module.build_writer_prompt
        original_parent_context = module.render_parent_context_for_writer
        original_seed_context = module.render_seed_context_for_writer
        module.build_planner_prompt = lambda **kwargs: _generalize_instruction_text(
            original_planner_prompt(**kwargs),
            config,
        )
        module.build_comment_move_planner_prompt = lambda **kwargs: _generalize_instruction_text(
            original_comment_planner_prompt(**kwargs),
            config,
        )
        module.build_writer_prompt = lambda **kwargs: _generalize_instruction_text(
            original_writer_prompt(**kwargs),
            config,
        )
        module.render_parent_context_for_writer = lambda **kwargs: _generalize_instruction_text(
            original_parent_context(**kwargs),
            config,
        )
        module.render_seed_context_for_writer = lambda **kwargs: _generalize_instruction_text(
            original_seed_context(**kwargs),
            config,
        )
    module.seed_post_gist = lambda seed_post: _generalize_instruction_text(
        original_seed_gist(seed_post),
        config,
    )
    module.mask_high_salience_context_terms = prompts.mask_specifics
    module.extract_product_anchors = lambda text: prompts.extract_product_anchors(config, text)
    module.extract_term_anchors = lambda text: prompts.extract_term_anchors(config, text)
    module.extract_concrete_anchors = lambda text, source_label="", max_items=12: prompts.extract_concrete_anchors(
        config,
        text,
        source_label=source_label,
        max_items=max_items,
    )
    module.build_concrete_anchors_for_task = _anchor_builder(module)
    module.plan_comment_move_batch = _comment_planner_batch_with_history(
        module,
        original_comment_planner_batch,
        comment_plan_ledgers,
    )
    module.infer_real_surface_shape = _generic_real_surface_shape
    module.infer_surface_skeleton = infer_surface_skeleton
    module.infer_real_comment_social_overrides = _structural_real_comment_overrides
    module.real_text_allows_first_person_frame = _allows_first_person
    module.real_text_allows_uncertainty_frame = _allows_uncertainty
    module.infer_real_tone_slot = lambda row, **kwargs: _generic_real_tone_slot(
        config,
        row,
        **kwargs,
    )
    module.sanitize_writer_text = _sanitize_writer_text(module)
    # The shared CARD renderer injects literal "Thanks", "lol", and ellipses
    # for selected surface labels.  Those fixed strings are not a structural
    # contract and create artificial n-gram clusters, so generalized Writer
    # output is left semantically intact after sanitization.
    module.shape_writer_text_for_task = lambda text, task: original_shape_writer_text(text, replace(task, surface_texture=""))
    module.contains_planner_skeleton_residue = _planner_residue_check(
        module,
        original_planner_residue_check,
    )
    # `writer_quality` reads this off the module, so it is always installed. The
    # check is evaluator-aligned rather than a wrapper: it takes no core
    # implementation, and the engine no longer ships CARD's fixed-window one.
    module.lexical_overlap_problem = _evaluator_aligned_lexical_overlap_check(
        module,
        calibration=dict(module.GENERALIZED_DOMAIN_PROFILE.get("lexical_quality") or {}),
    )
    module.has_blocking_guard_failure = _blocking_guard_check(
        module,
        original_blocking_guard_check,
    )
    module.writer_length_rule = soft_length_guidance
    module.writer_token_cap = lambda bucket, **kwargs: writer_safety_token_cap(
        original_writer_token_cap,
        bucket,
        **kwargs,
    )
    module.degraded_task_for_guard_failure = _substantive_safe_degraded_task()
    if original_backfill_claim is not None:
        module.backfill_repeated_claim_task = _claim_backfill_preserving_semantics(
            original_backfill_claim
        )
    module.generate_writer_text_with_guards = _writer_lifecycle_with_candidate_recovery(
        module,
        original_writer_lifecycle,
        calibration=dict(module.GENERALIZED_DOMAIN_PROFILE.get("lexical_quality") or {}),
    )
    module.generate_post_from_tasks = _finalize_post_generation(
        module,
        original_generate_post,
    )
    module.has_realistic_long_helpful_anchor = _long_helpful_anchor(module)
    module.retry_note_for_problems = _retry_note_for_problems(module)
    module.guard_fallback_retry_note = _guard_fallback_retry_note
    module.parse_json_object = _parse_json_object
    module.chat_completion_text = _chat_completion_text
    module.preflight_openai_compatible_endpoint = _endpoint_preflight_with_retry(module)
    module.load_or_init_discussion = _discussion_loader(module, config)

    original_finalize = module.finalize_rebalanced_task

    def finalize_task(task: Any) -> Any:
        finalized = original_finalize(task)
        finalized = replace(
            finalized,
            allow_first_person_frame=bool(task.allow_first_person_frame),
            allow_uncertainty_frame=bool(task.allow_uncertainty_frame),
        )
        finalized = reconcile_substantive_task(finalized)
        return _generalize_task_instruction_language(finalized, config)

    module.finalize_rebalanced_task = finalize_task

    def rebalance_tasks(
        tasks: list[Any],
        *,
        rng: Any,
        advisor_max_share: float,
        question_max_share: float,
        micro_target_share: float,
        short_max_share: float,
        social_noise_min_share: float,
        gratitude_min_share: float,
        tone_harsh_max_share: float = 0.14,
        tone_calm_min_share: float = 0.78,
        tone_personal_min_share: float = 0.16,
        tone_polite_min_share: float = 0.16,
    ) -> list[Any]:
        card_balanced, card_report = rebalance_card_surfaces(
            list(tasks),
            rng=rng,
            advisor_max_share=advisor_max_share,
            question_max_share=question_max_share,
            micro_target_share=micro_target_share,
            short_max_share=short_max_share,
            social_noise_min_share=social_noise_min_share,
            gratitude_min_share=gratitude_min_share,
            tone_harsh_max_share=tone_harsh_max_share,
            tone_calm_min_share=tone_calm_min_share,
            tone_personal_min_share=tone_personal_min_share,
            tone_polite_min_share=tone_polite_min_share,
        )
        distributed, report = allocate_story_and_affect(
            card_balanced,
            personal_min_share=module.GENERALIZED_STORY_PERSONAL_MIN_SHARE,
            calibration=reference_calibration,
            rng=rng,
            template=dict(module.GENERALIZED_ACTIVE_REFERENCE_TEMPLATE or {}),
        )
        module.GENERALIZED_ACTIVE_DISTRIBUTION_TARGET = (
            build_thread_distribution_target(
                report.get("reference_template") or {},
                reference_calibration,
                generated_comment_count=len(card_balanced),
            )
        )
        report["writer_distribution_target"] = dict(
            module.GENERALIZED_ACTIVE_DISTRIBUTION_TARGET
        )
        report["planner_slot_distribution_schedule"] = dict(
            module.GENERALIZED_ACTIVE_SLOT_DISTRIBUTION_SCHEDULE or {}
        )
        report["card_surface_rebalance"] = card_report
        report["sequence_index"] = len(module.GENERALIZED_STORY_AFFECT_REPORTS)
        module.GENERALIZED_STORY_AFFECT_REPORTS.append(report)
        if not module.GENERALIZED_SELF_TEST_ACTIVE:
            _append_distribution_audit(report)
        return [module.finalize_rebalanced_task(task) for task in distributed]

    module.rebalance_tasks_for_diversity = rebalance_tasks
    original_expand = module.expand_matched_real_sample_to_tasks

    def expand_tasks(**kwargs: Any) -> list[Any]:
        tasks = original_expand(**kwargs)
        comment_plans = dict(kwargs.get("comment_plans") or {})
        roster = _build_thread_speaker_roster(module, config, kwargs)
        module.GENERALIZED_ACTIVE_SPEAKER_ROSTER = roster
        revised = []
        for task in tasks:
            must_not = str(task.must_not_do).replace(
                "Do not write a balanced card review.",
                "Do not write a complete or balanced product review.",
            )
            generalized = _generalize_task_instruction_language(
                replace(task, must_not_do=must_not),
                config,
            )
            speaker = roster.speaker_for(task.real_sample_id)
            if speaker is not None:
                generalized = replace(generalized, speaker_id=speaker.speaker_id)
            plan = comment_plans.get(int(task.real_sample_id or task.local_task_id)) or {}
            restored = restore_planner_task_contract(generalized, plan, core=module)
            revised.append(apply_planner_distribution_fields(restored, plan))
        retained, coverage = retain_explicitly_planned_tasks(revised, comment_plans)
        module.GENERALIZED_ACTIVE_PLANNER_COVERAGE = coverage
        if coverage["omitted_structural_slot_ids"]:
            print(
                "[planner-coverage] "
                f"structural={coverage['structural_slots']} "
                f"planned={coverage['planner_returned_slots']} "
                f"writer_tasks={coverage['writer_task_slots']} "
                "policy=omit_without_backfill",
                flush=True,
            )
        return retained

    module.expand_matched_real_sample_to_tasks = expand_tasks

    def _build_thread_speaker_roster(
        core: ModuleType, domain: Any, kwargs: dict[str, Any]
    ) -> SpeakerRoster:
        """Recover the matched thread's participation structure for this thread.

        `selected_matched_comments` is deterministic -- no rng, only
        `evenly_spaced_indices` and sorted sets -- so calling it again here
        returns the identical list the task expansion consumed, which is what
        makes `real_sample_id - 1` a valid index into it. Verified on seed 8:
        `real_word_count` agrees for 186 of 186 slots.
        """

        if str(
            getattr(core, "GENERALIZED_SPEAKER_IDENTITY", "off") or "off"
        ).strip().lower() != SPEAKER_IDENTITY_MATCHED:
            return EMPTY_ROSTER
        matched = kwargs.get("matched_real_thread")
        target = kwargs.get("target")
        if not matched or target is None:
            return EMPTY_ROSTER
        try:
            selected = core.selected_matched_comments(
                matched_real_thread=matched,
                target=target,
                matched_real_comments=int(kwargs.get("matched_real_comments") or 0),
            )
        except Exception:  # noqa: BLE001 - a roster is never worth failing a run
            return EMPTY_ROSTER
        profile = getattr(core, "GENERALIZED_DOMAIN_PROFILE", {}) or {}
        return build_speaker_roster(
            selected,
            inventory=profile.get("entity_inventory") or {},
            facets=tuple(getattr(domain, "topic_facets", ()) or ()),
        )

    changed_core = [
        name
        for name, original in original_core_symbols.items()
        if getattr(module, name) is not original
    ]
    unexpected_core = sorted(set(changed_core) - set(GENERALIZED_ALGORITHM_EXTENSIONS))
    if unexpected_core:
        raise RuntimeError(
            "Generalized adapter changed undeclared CARD core symbols: "
            + ", ".join(unexpected_core)
        )
    changed_functions = sorted(
        name
        for name, original in original_functions.items()
        if getattr(module, name) is not original
    )
    unexpected_functions = sorted(
        set(changed_functions)
        - set(DOMAIN_ADAPTATION_BOUNDARIES)
        - set(GENERALIZED_ALGORITHM_EXTENSIONS)
    )
    if unexpected_functions:
        raise RuntimeError(
            "Generalized adapter changed functions outside the declared domain "
            "boundary: " + ", ".join(unexpected_functions)
        )
    module.GENERALIZED_CARD_PARITY = {
        "generator_profile": profile,
        "backend_source": str(Path(module.__file__).resolve()),
        "core_algorithm_symbols": list(CORE_ALGORITHM_SYMBOLS),
        "changed_core_algorithm_symbols": changed_core,
        "generalized_algorithm_extensions": list(GENERALIZED_ALGORITHM_EXTENSIONS),
        "unexpected_core_algorithm_symbols": unexpected_core,
        "changed_backend_functions": changed_functions,
        "unexpected_backend_functions": unexpected_functions,
        "domain_adaptation_boundaries": list(DOMAIN_ADAPTATION_BOUNDARIES),
    }
    return module


def _run_generalized_self_test(module: ModuleType, config: DomainConfig) -> None:
    task = module.CommentTask(
        local_task_id=1,
        local_parent_task_id=None,
        depth=0,
        branch_id=1,
        branch_goal="acknowledge one local detail",
        visible_scope="seed",
        local_anchor="visible product detail",
        comment_function="reaction",
        content_angle="fit_use_case",
        evidence_mode="none_assertion",
        story_mode="no_story",
        voice="grateful",
        payload_type="low_info_reaction",
        length_bucket="short",
        speaker_role="gratitude_reply",
        utterance_mode="op_followup",
        surface_texture="gratitude_social",
        allow_first_person_frame=False,
        allow_uncertainty_frame=False,
        planner_intent="briefly acknowledge the visible detail",
        must_not_do="Do not add advice.",
        real_word_count=5,
        semantic_move="acknowledge one useful detail",
        local_topic="visible product detail",
        reply_relation="answers_parent",
        stance="agree",
        detail_focus="visible product detail",
        avoid_repeating="complete review",
        claim_key="detail_ack",
        claim_family="direct_answer",
        opening_style="bare acknowledgement",
        context_aperture="title_only",
        tone_shape="soft_ack",
    )
    seed = module.SeedPost(
        index=0,
        title="Question about one product detail",
        body="Does this detail affect normal use?",
        content="Question about one product detail\nDoes this detail affect normal use?",
        source_raw_post_id="self-test",
        real_num_comments=5,
        metadata={},
    )
    if module.GENERALIZED_ACTOR_MODE == MODE_DOMAIN_DERIVED:
        module.GENERALIZED_ACTOR_ASSIGNMENTS[assignment_key(seed, 1)] = (
            actor_state_from_plan(
                {
                    "semantic_move": task.semantic_move,
                    "detail_focus": task.detail_focus,
                    "evidence_mode": task.evidence_mode,
                    "reply_relation": task.reply_relation,
                    "opening_style": task.opening_style,
                    "actor_participant_key": "A1",
                    "actor_participation_goal": "acknowledge the visible local detail",
                    "actor_realization_route": "brief local acknowledgement without an explanation tail",
                },
                sample_id=1,
            )
        )
    task = module.finalize_rebalanced_task(task)
    distribution_tasks = [
        replace(
            task,
            local_task_id=index,
            local_parent_task_id=1 if index > 1 else None,
            depth=1 if index > 1 else 0,
            payload_type="soft_helpful",
            comment_function="explanation_analysis",
            length_bucket="medium",
            speaker_role="side_observer",
            utterance_mode="small_observation",
            voice="casual_neutral",
            story_mode="no_story",
            real_word_count=24,
        )
        for index in range(1, 11)
    ]
    previous_self_test_state = module.GENERALIZED_SELF_TEST_ACTIVE
    module.GENERALIZED_SELF_TEST_ACTIVE = True
    try:
        distribution_tasks = module.rebalance_tasks_for_diversity(
            distribution_tasks,
            rng=module.random.Random(42),
            advisor_max_share=0.28,
            question_max_share=0.18,
            micro_target_share=0.07,
            short_max_share=0.18,
            social_noise_min_share=0.18,
            gratitude_min_share=0.12,
            tone_harsh_max_share=0.14,
            tone_calm_min_share=0.78,
            tone_personal_min_share=0.16,
            tone_polite_min_share=0.24,
        )
    finally:
        module.GENERALIZED_SELF_TEST_ACTIVE = previous_self_test_state
    distribution_report = module.GENERALIZED_STORY_AFFECT_REPORTS[-1]
    assert distribution_report["policy"] == "audit_planner_template_contract_without_post_planner_reassignment"
    assert distribution_report["task_count"] == len(distribution_tasks)
    assert distribution_report["card_surface_rebalance"]["surface_rebalanced_count"] == 0
    assert distribution_report["card_surface_rebalance"]["policy"] == "planner_owned_generalized_surface_contract"
    assert all(item.length_bucket == "medium" for item in distribution_tasks)
    prompt = module.build_writer_prompt(
        profile="gpt54_reddit_writer",
        seed_post=seed,
        task=task,
        parent_comment=None,
        previous_comments=[
            {
                "content": "Thanks, that detail helps.",
                "depth": 1,
                "speaker_role": "gratitude_reply",
                "tone_shape": "soft_ack",
                "payload_type": "low_info_reaction",
                "utterance_mode": "op_followup",
                "voice": "grateful",
                "length_bucket": "short",
                "surface_texture": "gratitude_social",
                "story_mode": "no_story",
            }
        ],
    )
    lowered = prompt.lower()
    assert (
        "already generated in this thread" in lowered
        or "earlier generated comments" in lowered
    )
    assert "gratitude_reply" in lowered
    assert "soft_ack" in lowered
    assert "low_info_reaction" in lowered
    assert (
        "thread-level pressure for this next comment" in lowered
        or "thread-level distribution pressure" in lowered
    )
    assert "current sampled slot" in lowered
    assert "r/creditcards" not in lowered
    assert "bank/card" not in lowered
    assert "issuer" not in lowered
    assert "hard maximum:" not in lowered
    assert "target length:" not in lowered
    # The matched word count is a directional scale target, never a counted
    # acceptance gate.
    assert "not a counted requirement" in lowered
    assert not module.has_blocking_guard_failure(["real_slot_too_short"])
    assert not module.has_blocking_guard_failure(["length_too_long"])
    assert not module.has_blocking_guard_failure(["low_info_too_long"])
    assert not module.has_blocking_guard_failure(["substantive_length_floor:5<12"])
    for diagnostic in (
        "first_person_frame_unwanted",
        "missing_concrete_anchor",
        "meta_template_quote_heading",
        "question_mark_unwanted",
    ):
        assert not module.has_blocking_guard_failure([diagnostic])
    assert module.has_blocking_guard_failure(
        ["placeholder_literal", "real_slot_too_short"]
    )
    assert module.writer_token_cap(
        "short",
        payload_type="low_info_reaction",
        profile="gpt54_reddit_writer",
        max_writer_tokens=260,
    ) == 260
    substantive_marker_text = (
        "This is a substantive local explanation with several constraints and "
        "a concrete consequence for ordinary use, followed by a caveat that "
        "keeps the recommendation narrow instead of turning it into a generic "
        "answer. The final reaction is incidental rather than the whole point lol."
    )
    assert module.infer_real_surface_shape(
        {"body": substantive_marker_text, "author": "user"}
    ) == "full_answer"
    skeleton, _ = module.infer_surface_skeleton(substantive_marker_text)
    assert "joke" not in skeleton
    assert writer_provider_token_budget(
        replace(task, real_word_count=300), configured_max=260
    ) > 500
    story_task = next(
        (item for item in distribution_tasks if item.story_mode != "no_story"),
        None,
    )
    if story_task is not None:
        story_prompt = module.build_writer_prompt(
            profile="gpt54_reddit_writer",
            seed_post=seed,
            task=story_task,
            parent_comment=None,
            previous_comments=[],
        ).lower()
        assert "affect role" in story_prompt
        assert "synthetic story slot" in story_prompt
        assert story_task.affect_role in story_prompt
    calibration = dict(module.GENERALIZED_DOMAIN_PROFILE.get("lexical_quality") or {})
    if calibration.get("prefix_mean_upper") or calibration.get("thresholds"):
        overlap_problem = module.lexical_overlap_problem(
            text="same tiny reply",
            previous_comments=[{"content": "same tiny reply"}],
            task=replace(task, length_bucket="micro"),
        )
        if module.GENERALIZED_ACTOR_MODE == MODE_DOMAIN_DERIVED:
            assert overlap_problem == ""
        else:
            assert overlap_problem.startswith("lexical_overlap_high:")
    if module.GENERALIZED_ACTOR_MODE == MODE_DOMAIN_DERIVED:
        actor_prompt = module.build_writer_prompt(
            profile="gpt54_reddit_writer",
            seed_post=seed,
            task=task,
            parent_comment=None,
            previous_comments=[],
        )
        assert "Thread-local actor state composed by the Planner" in actor_prompt
        assert "acknowledge the visible local detail" in actor_prompt
    print(f"[generalized-self-test] PASS domain={config.domain_id}", flush=True)


def _structural_real_comment_overrides(
    row: dict[str, Any],
    *,
    payload_type: str,
    speaker_role: str,
) -> dict[str, str]:
    """Keep only identity/structure facts; do not infer semantics from text."""

    del payload_type
    author = str(row.get("author") or "").strip().lower()
    if author in {"automoderator", "moderator"}:
        return {"speaker_role": "mod_meta"}
    if bool(row.get("is_submitter")):
        return {"speaker_role": "op_followup"}
    return {"speaker_role": speaker_role} if speaker_role else {}


def _claim_backfill_preserving_semantics(original: Any):
    """Let CARD rotate claim metadata without erasing the Planner's move."""

    def backfill(task: Any, *, reason: str) -> Any:
        revised = original(task, reason=reason)
        return replace(
            revised,
            semantic_move=task.semantic_move,
            local_topic=task.local_topic,
            reply_relation=task.reply_relation,
            stance=task.stance,
            detail_focus=task.detail_focus,
            avoid_repeating=task.avoid_repeating,
            domain_intent=task.domain_intent,
            decision_boundary=task.decision_boundary,
            opening_style=task.opening_style,
            planner_intent=task.planner_intent,
        )

    return backfill


def _generic_real_tone_slot(
    config: DomainConfig,
    row: dict[str, Any],
    *,
    payload_type: str,
    speaker_role: str,
    voice: str,
    real_surface_shape: str,
    surface_texture: str,
) -> tuple[str, str]:
    del config, row
    # Do not infer tone, emotion, story, or discourse function from matched
    # evaluation text with keyword lists.  Those assignments come from the
    # evaluation-excluded template and Planner's coherent slot contract.
    if real_surface_shape in {"link_reference", "quote_link_reference"}:
        return (
            "reference_aside",
            "Use a source/reference-like social move. Keep it local and do not expand it into a broad explanation.",
        )
    if real_surface_shape == "template_notice" or speaker_role == "mod_meta":
        return (
            "subreddit_meta_notice",
            "Use a community/meta/template-style move, not personal product advice or a support reply.",
        )
    if real_surface_shape == "micro_reaction":
        return "tiny_reaction", "Use a tiny reaction or bare answer without explanation."
    if surface_texture == "gratitude_social" or speaker_role == "gratitude_reply":
        return (
            "pure_acknowledgement",
            "Make the social acknowledgement visible and local. Stop before advice, correction, or a new recommendation.",
        )
    return "", ""


def _generalize_task_instruction_language(task: Any, config: DomainConfig) -> Any:
    updates = {}
    for field in (
        "planner_intent",
        "must_not_do",
        "surface_instruction",
        "real_tone_instruction",
        "tone_overlay_instruction",
        "tone_target_instruction",
    ):
        if not hasattr(task, field):
            continue
        value = str(getattr(task, field, "") or "")
        generalized = _generalize_instruction_text(value, config)
        updates[field] = generalized
    return replace(task, **updates)


def _generalize_instruction_text(value: str, config: DomainConfig) -> str:
    text = str(value or "")
    replacements = (
        ("r/CreditCards", config.community_context),
        ("credit-card", config.display_name),
        ("credit card", config.display_name),
        ("bank/card/process", "product/service/process"),
        ("card/bank/process", "product/service/process"),
        ("claim, bank, card, policy, or process", "claim, product, rule, interface, service, or process"),
        ("bank, card, policy, or process", "product, rule, interface, service, or process"),
        ("bank, card, or process", "product, service, or process"),
        ("issuer/product/process", "product/service/process"),
        ("issuer, product, or process", "product, service, or process"),
        ("issuer/card", "product/service"),
        ("issuers, reward programs, fees, or benefits", "brands, product lines, prices, or features"),
        ("named products, issuers, reward programs, fees, or benefits", "named products, brands, prices, or features"),
        ("issuer rules, approval datapoints, credit-limit handling, utilization, rewards value, product comparisons, support/process details", "product rules, firsthand datapoints, setup or compatibility, price and feature tradeoffs, product comparisons, and support/process details"),
        ("hard pull vs soft pull, credit-line transfer, USBAR limit, utilization scoring, sockdrawer joke, branch banker, card-specific DP, hidden fee", "setup difference, compatibility constraint, product limit, performance tradeoff, unused-product joke, store or support interaction, product-specific datapoint, hidden cost"),
        ("fee_number", "price_or_measurement"),
        ("issuer_rule", "product_rule"),
        ("card-specific", "product-specific"),
        ("card names", "product names"),
        ("cards, banks", "products, brands"),
        ("cards, dates, fees, percentages, URLs, or reward numbers", "products, dates, prices, measurements, URLs, or feature values"),
        ("card/process-as-subject", "product/process-as-subject"),
        ("card-as-subject", "product-as-subject"),
        ("card advice", "product advice"),
        ("break-even explanation", "generic tradeoff calculation"),
        ("appreciate the DP", "appreciate the datapoint"),
        ("concrete DP", "concrete datapoint"),
        ("local DP/story", "local datapoint/story"),
        ("DP/story", "datapoint/story"),
    )
    for old, new in replacements:
        text = re.sub(re.escape(old), new, text, flags=re.I)
    return text


def _anchor_builder(module: ModuleType):
    def build_concrete_anchors_for_task(
        *,
        real_body: str,
        seed_post: Any | None,
        branch: Any,
        planned: dict[str, str],
        anchor: str,
        parent_task: Any | None,
        max_anchors: int = 10,
    ) -> tuple[str, ...]:
        payload_type = str(planned.get("payload_type") or "").strip().lower()
        comment_function = str(planned.get("comment_function") or "").strip().lower()
        if payload_type in {
            "low_info_reaction",
            "joke",
            "meta_or_template",
            "side_tangent",
        } or comment_function == "offtopic_noise":
            return ()

        parent_anchors = (
            _without_internal_control_anchors(parent_task.concrete_anchors)
            if parent_task is not None
            else []
        )
        # The matched evaluation comment supplies only an anonymous structural
        # slot. Its facts must never become Writer-visible anchors.
        planner_text = " ".join(
            str(planned.get(key) or "")
            for key in (
                "semantic_move",
                "local_topic",
                "detail_focus",
                "claim_key",
                "claim_family",
                "domain_intent",
                "reply_relation",
            )
        )
        planner_anchors = (
            _without_internal_control_anchors(
                module.extract_concrete_anchors(planner_text, source_label="planner")
            )
        )
        local_anchors = (
            _without_internal_control_anchors(
                module.extract_concrete_anchors(
                " ".join([anchor, branch.anchor_quote, branch.branch_goal]),
                source_label="local",
                )
            )
        )
        seed_anchors: list[str] = []
        if seed_post is not None:
            seed_anchors = list(
                module.extract_concrete_anchors(
                    seed_post.title + "\n" + seed_post.body,
                    source_label="seed",
                    max_items=12,
                )
            )
        anchors = _select_task_anchors(
            planner_anchors=planner_anchors,
            local_anchors=local_anchors,
            parent_anchors=parent_anchors,
            seed_anchors=seed_anchors,
            planner_text=planner_text,
            sample_id=_safe_int(planned.get("sample_id"), 0),
            max_anchors=min(max_anchors, 4),
        )
        return tuple(module.dedup_anchors(anchors, max_anchors=max_anchors))

    return build_concrete_anchors_for_task


def _select_task_anchors(
    *,
    planner_anchors: list[str],
    local_anchors: list[str],
    parent_anchors: list[str],
    seed_anchors: list[str],
    planner_text: str,
    sample_id: int,
    max_anchors: int,
) -> list[str]:
    """Expose only anchors needed by this slot instead of every seed entity."""

    selected: list[str] = []
    selected.extend(_rank_relevant_anchors(planner_anchors, planner_text)[:3])
    selected.extend(_rank_relevant_anchors(local_anchors, planner_text)[:1])
    if len(selected) < 2:
        selected.extend(_rank_relevant_anchors(parent_anchors, planner_text)[:1])
    if seed_anchors and len(selected) < max_anchors:
        seed_index = (
            max(0, sample_id - 1) % len(seed_anchors)
            if sample_id > 0
            else 0
        )
        seed_anchor = seed_anchors[seed_index]
        normalized_selected = {
            prompts.strip_anchor_source(item).lower() for item in selected
        }
        if prompts.strip_anchor_source(seed_anchor).lower() not in normalized_selected:
            selected.append(seed_anchor)
    return selected[:max_anchors]


def _rank_relevant_anchors(anchors: list[str], planner_text: str) -> list[str]:
    return sorted(
        anchors,
        key=lambda item: (-_anchor_relevance(item, planner_text), anchors.index(item)),
    )


def _anchor_relevance(anchor: str, planner_text: str) -> int:
    anchor_tokens = set(re.findall(r"[a-z0-9]+", prompts.strip_anchor_source(anchor).lower()))
    plan_tokens = set(re.findall(r"[a-z0-9]+", planner_text.lower()))
    return len(anchor_tokens & plan_tokens)


def _without_internal_control_anchors(values: Any) -> list[str]:
    result = []
    for value in values or ():
        base = prompts.strip_anchor_source(str(value))
        if prompts.INTERNAL_CONTROL_ID_RE.fullmatch(base):
            continue
        result.append(str(value))
    return result


def _comment_planner_batch_with_history(
    module: ModuleType,
    original: Any,
    ledgers: dict[str, list[dict[str, Any]]],
):
    def plan_comment_move_batch(**kwargs: Any) -> dict[int, dict[str, str]]:
        seed_post = kwargs["seed_post"]
        key = str(
            getattr(seed_post, "source_raw_post_id", "")
            or getattr(seed_post, "index", "")
            or getattr(seed_post, "title", "")
        )
        if int(kwargs.get("sample_offset") or 0) == 0:
            ledgers[key] = []
        ledger = ledgers.setdefault(key, [])
        config = dict(getattr(module, "GENERALIZED_PLAN_QUALITY_CONFIG", {}) or {})
        perspective_ids = {
            str(item.get("perspective_id") or "").strip().upper()
            for item in (getattr(module, "GENERALIZED_DOMAIN_PROFILE", {}).get("perspectives") or [])
            if isinstance(item, dict) and item.get("perspective_id")
        }
        module.GENERALIZED_COMMENT_PLAN_HISTORY = list(ledger)
        module.GENERALIZED_COMMENT_PLAN_FEEDBACK = ""
        repair_budget = int(config.get("repair_rounds", 0))
        control_normalizations: list[dict[str, Any]] = []
        raw_plans = original(**kwargs)
        sample_offset = int(kwargs.get("sample_offset") or 0)
        expected_ids = set(
            range(sample_offset + 1, sample_offset + len(kwargs.get("comments") or []) + 1)
        )
        unexpected_ids = sorted(set(raw_plans) - expected_ids)
        plans = {
            sample_id: plan
            for sample_id, plan in raw_plans.items()
            if sample_id in expected_ids
        }
        missing_ids_initial = sorted(expected_ids - set(plans))
        repair_counts: Counter[int] = Counter()
        schema_recovery_counts: Counter[int] = Counter()
        schema_recovery_budget = int(config.get("schema_recovery_rounds", 0))
        schema_recovery_events: list[dict[str, Any]] = []
        # Complete missing schema rows before semantic-quality evaluation. This
        # never replaces a valid plan and never ranks alternative plans.
        for sample_id in missing_ids_initial:
            for schema_attempt in range(1, schema_recovery_budget + 1):
                recovery_kwargs = _targeted_plan_repair_kwargs(kwargs, sample_id)
                module.GENERALIZED_COMMENT_PLAN_HISTORY = [
                    *ledger,
                    *(
                        ledger_entry(other_id, plan)
                        for other_id, plan in sorted(plans.items())
                    ),
                ]
                module.GENERALIZED_COMMENT_PLAN_FEEDBACK = (
                    "Schema completion only: return exactly the requested S"
                    f"{sample_id} plan. Do not modify other slots."
                )
                recovered = original(**recovery_kwargs)
                _annotate_plan_metadata(module, recovered, recovery_kwargs)
                control_normalizations.extend(
                    _canonicalize_plan_controls(
                        recovered,
                        perspective_ids=perspective_ids,
                        repair_attempt=schema_attempt,
                    )
                )
                replacement = recovered.get(sample_id)
                recovered_sample_id: int | None = None
                # A one-slot recovery prompt contains only the requested
                # anonymous structural slot. Some providers still echo the
                # illustrative schema ID (usually S1) instead of the global
                # requested ID. The semantic plan is nevertheless unambiguous
                # because no other slot was visible to that call, so normalize
                # the transport-level ID rather than dropping a valid plan or
                # restarting the whole thread.
                if replacement is None and len(recovered) == 1:
                    recovered_sample_id, replacement = next(iter(recovered.items()))
                    replacement = dict(replacement)
                    replacement["sample_id"] = str(sample_id)
                schema_recovery_counts[sample_id] += 1
                schema_recovery_events.append(
                    {
                        "sample_id": sample_id,
                        "attempt": schema_attempt,
                        "returned": replacement is not None,
                        "returned_sample_id": recovered_sample_id,
                        "canonicalized_single_slot_id": recovered_sample_id is not None,
                    }
                )
                if replacement is not None:
                    plans[sample_id] = replacement
                    break
        unresolved_missing_ids = sorted(expected_ids - set(plans))
        _annotate_plan_metadata(module, plans, kwargs)
        control_normalizations.extend(
            _canonicalize_plan_controls(plans, perspective_ids=perspective_ids)
        )
        semantic_index = getattr(module, "GENERALIZED_PLAN_SEMANTIC_INDEX", None)

        def evaluate(candidate_plans: dict[int, dict[str, Any]]):
            if semantic_index is not None:
                semantic_index.prepare([*ledger, *candidate_plans.values()])
            return evaluate_plan_batch(
                candidate_plans,
                prior_plans=ledger,
                perspective_ids=perspective_ids,
                similarity_threshold=float(config.get("similarity_threshold", 0.72)),
                embedding_similarity_threshold=float(
                    config.get("embedding_similarity_threshold", 0.82)
                ),
                semantic_similarity=(
                    semantic_index.similarity if semantic_index is not None else None
                ),
                max_perspective_share=float(config.get("max_perspective_share", 0.34)),
                required_branch_ids=root_branch_schedule(
                    list(kwargs.get("all_comments") or []),
                    branch_ids=[
                        int(getattr(branch, "branch_id", 0) or 0)
                        for branch in list(kwargs.get("branches") or [])
                    ],
                ),
                require_reply_novelty=bool(config.get("require_reply_novelty", False)),
                enforce_social_contract=(
                    str(
                        getattr(
                            module,
                            "GENERALIZED_SOCIAL_CONTRACT_COHERENCE",
                            "on",
                        )
                    )
                    != "off"
                ),
            )

        report = evaluate(plans)
        best_plans = plans
        best_report = report
        attempts = [report.to_dict()]
        while repair_budget > 0:
            if best_report.healthy:
                break
            failing_ids = list(
                dict.fromkeys(
                    issue.sample_id
                    for issue in best_report.repair_issues
                    if issue.sample_id in best_plans
                    and repair_counts[issue.sample_id] < repair_budget
                )
            )
            if not failing_ids:
                break
            accepted_in_pass = False
            for sample_id in failing_ids:
                if not any(
                    issue.sample_id == sample_id
                    for issue in best_report.repair_issues
                ):
                    continue
                repair_counts[sample_id] += 1
                repair_kwargs = _targeted_plan_repair_kwargs(kwargs, sample_id)
                module.GENERALIZED_COMMENT_PLAN_HISTORY = [
                    *ledger,
                    *(
                        ledger_entry(other_id, plan)
                        for other_id, plan in sorted(best_plans.items())
                        if other_id != sample_id
                    ),
                ]
                module.GENERALIZED_COMMENT_PLAN_FEEDBACK = best_report.feedback(
                    repair_attempt=repair_counts[sample_id],
                    sample_ids=(sample_id,),
                )
                repaired = original(**repair_kwargs)
                _annotate_plan_metadata(module, repaired, repair_kwargs)
                control_normalizations.extend(
                    _canonicalize_plan_controls(
                        repaired,
                        perspective_ids=perspective_ids,
                        repair_attempt=repair_counts[sample_id],
                    )
                )
                replacement = repaired.get(sample_id)
                if replacement is None:
                    attempt = best_report.to_dict()
                    attempt.update(
                        {
                            "repair_scope": [sample_id],
                            "repair_accepted": False,
                            "repair_error": "planner did not return the requested global sample_id",
                        }
                    )
                    attempts.append(attempt)
                    continue
                candidate_plans = dict(best_plans)
                candidate_plans[sample_id] = replacement
                candidate_report = evaluate(candidate_plans)
                accepted = candidate_report.issue_score < best_report.issue_score
                attempt = candidate_report.to_dict()
                attempt.update(
                    {
                        "repair_scope": [sample_id],
                        "repair_accepted": accepted,
                    }
                )
                attempts.append(attempt)
                if accepted:
                    best_plans = candidate_plans
                    best_report = candidate_report
                    accepted_in_pass = True
            if not accepted_in_pass and all(
                repair_counts[sample_id] >= repair_budget
                for sample_id in failing_ids
            ):
                break
        module.GENERALIZED_COMMENT_PLAN_FEEDBACK = ""
        module.GENERALIZED_COMMENT_PLAN_HISTORY = list(ledger)

        report_row = {
            "seed_key": key,
            "sample_offset": sample_offset,
            "batch_size": len(best_plans),
            "expected_sample_ids": sorted(expected_ids),
            "unexpected_sample_ids_discarded": unexpected_ids,
            "missing_sample_ids_initial": missing_ids_initial,
            "omitted_sample_ids": unresolved_missing_ids,
            "missing_slot_policy": "bounded_schema_recovery_then_hard_fail",
            "schema_recovery_attempts": sum(schema_recovery_counts.values()),
            "schema_recovery_attempts_by_sample": {
                str(sample_id): count
                for sample_id, count in sorted(schema_recovery_counts.items())
            },
            "schema_recovery_events": schema_recovery_events,
            "repair_attempts": sum(repair_counts.values()),
            "repair_strategy": "targeted_slot",
            "repair_attempts_by_sample": {
                str(sample_id): count
                for sample_id, count in sorted(repair_counts.items())
            },
            "control_normalizations": control_normalizations,
            "selected": best_report.to_dict(),
            "attempts": attempts,
        }
        print(
            "[plan-quality] "
            f"seed={key} offset={report_row['sample_offset']} plans={len(best_plans)} "
            f"repairs={report_row['repair_attempts']} "
            f"omitted={len(unresolved_missing_ids)} "
            f"collisions={len(best_report.colliding_samples)} "
            f"collision_rate={best_report.collision_rate:.3f} "
            f"embedding_threshold={float(config.get('embedding_similarity_threshold', 0.82)):.3f} "
            f"dominant={best_report.dominant_perspective or 'none'}:" 
            f"{best_report.dominant_perspective_share:.3f} issues={len(best_report.issues)}",
            flush=True,
        )

        invalid_perspective = any(
            issue.code == "invalid_perspective" for issue in best_report.issues
        )
        excessive_concentration = (
            best_report.substantive_count >= 8
            and best_report.dominant_perspective_share
            > float(config.get("max_perspective_share", 0.34)) + 0.12
        )
        if bool(config.get("strict", False)) and (
            invalid_perspective
            or best_report.collision_rate > float(config.get("max_collision_rate", 0.10))
            or excessive_concentration
        ):
            # Targeted planning repairs have already been attempted above. A
            # remaining collision is a quality warning, not a reason to throw
            # away healthy generated slots and restart the whole post. The
            # Writer-level ledger and evaluator-aligned local repair policy
            # still enforce the generated-text constraints per comment.
            report_row["unresolved_plan_quality_warning"] = {
                "collision_rate": best_report.collision_rate,
                "dominant_perspective_share": best_report.dominant_perspective_share,
                "issues": len(best_report.issues),
                "unresolved_samples": sorted(
                    {issue.sample_id for issue in best_report.issues}
                ),
            }
            print(
                "[plan-quality-warning] continuing after targeted repair: "
                f"collision_rate={best_report.collision_rate:.3f}, "
                f"dominant_perspective_share={best_report.dominant_perspective_share:.3f}, "
                f"issues={len(best_report.issues)}, "
                "unresolved_samples="
                f"{report_row['unresolved_plan_quality_warning']['unresolved_samples']}",
                flush=True,
            )

        reports = getattr(module, "GENERALIZED_COMMENT_PLAN_REPORTS", None)
        if not isinstance(reports, list):
            reports = []
            module.GENERALIZED_COMMENT_PLAN_REPORTS = reports
        reports.append(report_row)
        _append_plan_quality_report(report_row)

        for sample_id, plan in sorted(best_plans.items()):
            ledger.append(ledger_entry(sample_id, plan))
            if getattr(module, "GENERALIZED_ACTOR_MODE", "") == MODE_DOMAIN_DERIVED:
                module.GENERALIZED_ACTOR_ASSIGNMENTS[(key, int(sample_id))] = (
                    actor_state_from_plan(plan, sample_id=int(sample_id))
                )
            # `CommentTask` is a frozen dataclass in the pinned shared generator,
            # so a planned field that the generator does not declare is carried
            # in a keyed registry instead, the same way actor state is.
            claim = normalized_domain_claim(plan.get("domain_claim"))
            if claim and module.GENERALIZED_DOMAIN_CLAIM_MODE != "off":
                module.GENERALIZED_DOMAIN_CLAIMS[(key, int(sample_id))] = claim
            opener = str(plan.get("opener_type") or "").strip().lower()
            if opener in OPENER_TYPES:
                module.GENERALIZED_OPENER_TYPES[(key, int(sample_id))] = opener
        module.GENERALIZED_COMMENT_PLAN_HISTORY = list(ledger)
        return best_plans

    return plan_comment_move_batch


def _canonicalize_plan_controls(
    plans: dict[int, dict[str, Any]],
    *,
    perspective_ids: set[str],
    repair_attempt: int = 0,
) -> list[dict[str, Any]]:
    """Repair enum-only Planner mistakes without changing semantic content.

    A branch ID such as B3 has no semantic meaning as a decision lens. Mapping
    an unknown lens to ``seed_local`` is the schema-defined fallback and avoids
    spending LLM repair calls on a deterministic metadata correction.
    """

    allowed = {str(value).strip().upper() for value in perspective_ids}
    allowed.add("SEED_LOCAL")
    events: list[dict[str, Any]] = []
    for sample_id, plan in sorted(plans.items()):
        raw = str(plan.get("perspective_id") or "seed_local").strip()
        normalized = raw.upper()
        if normalized in allowed:
            plan["perspective_id"] = (
                "seed_local" if normalized == "SEED_LOCAL" else normalized
            )
        else:
            plan["perspective_id"] = "seed_local"
            events.append(
                {
                    "sample_id": int(sample_id),
                    "field": "perspective_id",
                    "raw_value": raw,
                    "normalized_value": "seed_local",
                    "reason": "invalid_frozen_decision_lens",
                    "repair_attempt": int(repair_attempt),
                }
            )
        affect = str(plan.get("affect_role") or "").strip().lower()
        role = str(plan.get("speaker_role") or "").strip().lower()
        function = str(plan.get("comment_function") or "").strip().lower()
        if affect in {"gratitude", "relief"} and not (
            role == "gratitude_reply" and function == "reaction"
        ):
            # Affect quotas are template controls, not permission to attach a
            # thank-you label to a second substantive recommendation.
            plan.update(
                {
                    "speaker_role": "gratitude_reply",
                    "comment_function": "reaction",
                    "evidence_mode": "none_assertion",
                    "story_mode": "no_story",
                    "voice": "grateful",
                    "utterance_mode": "op_followup",
                    "semantic_move": "briefly acknowledge the parent-local help without adding a factual claim",
                    "decision_boundary": "social acknowledgement only",
                    "reply_delta": "social close",
                    "reply_delta_type": "social_close",
                    "reply_novelty_anchor": "acknowledge the parent-local help without adding a factual claim",
                    "avoid_repeating": "do not add a second factual claim or recommendation",
                }
            )
            events.append(
                {
                    "sample_id": int(sample_id),
                    "field": "social_contract",
                    "reason": "affect_quota_requires_social_discourse_act",
                    "repair_attempt": int(repair_attempt),
                }
            )
        structural_event = normalize_substantive_plan_shape(plan)
        if structural_event is not None:
            events.append(
                {
                    "sample_id": int(sample_id),
                    "repair_attempt": int(repair_attempt),
                    "reason": "anonymous_substantive_slot_requires_contextual_payload",
                    **structural_event,
                }
            )
    return events


def _targeted_plan_repair_kwargs(
    kwargs: dict[str, Any],
    sample_id: int,
) -> dict[str, Any]:
    all_comments = list(kwargs.get("all_comments") or [])
    if sample_id <= 0 or sample_id > len(all_comments):
        raise RuntimeError(
            f"Cannot target plan repair for S{sample_id}: matched slot is unavailable"
        )
    return {
        **kwargs,
        "comments": [all_comments[sample_id - 1]],
        "sample_offset": sample_id - 1,
    }


def _annotate_plan_metadata(
    module: ModuleType,
    plans: dict[int, dict[str, Any]],
    kwargs: dict[str, Any],
) -> None:
    all_comments = list(kwargs.get("all_comments") or [])
    comment_index: dict[str, int] = {}
    branches = {
        int(getattr(branch, "branch_id", 0) or 0): branch
        for branch in list(kwargs.get("branches") or [])
    }
    required_branches = root_branch_schedule(
        all_comments,
        branch_ids=branches,
    )
    parent_slots = parent_slot_schedule(all_comments)
    for index, row in enumerate(all_comments, start=1):
        for key in module.real_comment_keys(row):
            comment_index[str(key)] = index
    for sample_id, plan in plans.items():
        plan["sample_id"] = str(sample_id)
        if 0 < int(sample_id) <= len(all_comments):
            slot = all_comments[int(sample_id) - 1]
            parent_id = str(slot.get("parent_id") or "")
            parent_sample = comment_index.get(parent_id, 0)
            plan["parent_sample_id"] = str(parent_sample or "")
            slot_body = str(slot.get("body") or "")
            plan["_slot_word_count"] = str(len(slot_body.split()))
            plan["_slot_surface_label"] = surface_only_label(slot_body)
        required_branch = required_branches.get(int(sample_id))
        if required_branch is not None:
            plan["_required_branch_id"] = str(required_branch)
            # Anonymous topology, rather than a free-form B# from the model,
            # determines root-chain ownership. This makes every reply inherit
            # its parent's branch before the Writer receives the task.
            plan["branch_id"] = str(required_branch)
            branch = branches.get(required_branch)
            plan["_required_branch_goal"] = str(
                getattr(branch, "branch_goal", "") or ""
            )
            plan["_required_branch_perspective"] = str(
                getattr(branch, "perspective_id", "") or "seed_local"
            )
            plan["_required_branch_exclusion"] = str(
                getattr(branch, "branch_exclusion", "") or ""
            )
            plan["_required_branch_subject"] = str(
                getattr(branch, "owned_decision_subject", "")
                or getattr(branch, "decision_boundary", "")
                or ""
            )
            # Branch ownership is structural Planner metadata, not a candidate
            # choice. Keep it attached even when the Comment Planner uses a
            # shorter local decision-boundary wording.
            plan["owned_decision_subject"] = plan["_required_branch_subject"]
            plan["perspective_id"] = plan["_required_branch_perspective"]
        parent_sample = parent_slots.get(int(sample_id))
        if parent_sample is not None:
            plan["_required_parent_sample_id"] = str(parent_sample)


def _append_plan_quality_report(report: dict[str, Any]) -> None:
    raw_path = os.environ.get("GENERALIZED_CARD_PLAN_AUDIT_JSONL", "").strip()
    if not raw_path:
        return
    path = Path(raw_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(report, ensure_ascii=False, sort_keys=True) + "\n")


def _env_int(name: str, default: int, *, minimum: int) -> int:
    try:
        value = int(os.environ.get(name, str(default)))
    except ValueError:
        value = default
    return max(minimum, value)


def _env_float(
    name: str,
    default: float,
    *,
    minimum: float,
    maximum: float,
) -> float:
    try:
        value = float(os.environ.get(name, str(default)))
    except ValueError:
        value = default
    return max(minimum, min(maximum, value))


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _planner_residue_check(module: ModuleType, original: Any):
    profile_ids = {
        str(item.get("perspective_id") or "").upper()
        for item in (module.GENERALIZED_DOMAIN_PROFILE.get("perspectives") or [])
        if isinstance(item, dict) and item.get("perspective_id")
    }

    def contains_planner_skeleton_residue(text: str, task: Any) -> bool:
        if original(text, task):
            return True
        allowed = {
            match.group(0).upper()
            for anchor in getattr(task, "concrete_anchors", ())
            if str(anchor).rstrip().lower().endswith("(seed)")
            for match in prompts.INTERNAL_CONTROL_ID_RE.finditer(prompts.strip_anchor_source(str(anchor)))
        }
        for match in prompts.INTERNAL_CONTROL_ID_RE.finditer(str(text or "")):
            label = match.group(0).upper()
            if (label in profile_ids or label.startswith(("S", "B"))) and label not in allowed:
                return True
        return False

    return contains_planner_skeleton_residue


def _blocking_guard_check(module: ModuleType, original: Any):
    def has_blocking_guard_failure(problems: list[str]) -> bool:
        checked = [
            problem for problem in problems if not is_soft_length_problem(problem)
        ]
        checked = [
            problem
            for problem in checked
            if not is_single_stage_diagnostic(problem)
        ]
        return original(checked)

    return has_blocking_guard_failure


def _evaluator_aligned_lexical_overlap_check(
    module: ModuleType,
    *,
    calibration: dict[str, Any],
):
    def lexical_overlap_problem(
        *,
        text: str,
        previous_comments: list[dict[str, Any]] | None,
        task: Any,
    ) -> str:
        if getattr(module, "GENERALIZED_ACTOR_MODE", "") == MODE_DOMAIN_DERIVED:
            # Actor-conditioned generation makes distribution metrics diagnostic.
            # They are evaluated after the single Writer realization rather than
            # optimized by repeatedly sampling the same slot.
            return ""
        previous = module.previous_comment_texts(previous_comments)
        target = distribution_target_with_slot_progress(
            getattr(module, "GENERALIZED_ACTIVE_DISTRIBUTION_TARGET", {}) or {},
            local_task_id=getattr(task, "local_task_id", 0),
        )
        lexical_problem = calibrated_lexical_overlap_problem(
            text=text,
            previous_texts=previous,
            calibration=calibration,
            thread_target=target,
        )
        semantic = semantic_thread_diagnostics(
            text=text,
            previous_texts=previous,
            thread_target=target,
            semantic_index=getattr(module, "GENERALIZED_PLAN_SEMANTIC_INDEX", None),
        )
        semantic_problem = semantic_distribution_problem(semantic)
        if lexical_problem and semantic_problem:
            return lexical_problem + ";" + semantic_problem
        return lexical_problem or semantic_problem

    return lexical_overlap_problem


def _writer_lifecycle_with_candidate_recovery(
    module: ModuleType,
    original: Any,
    *,
    calibration: dict[str, Any],
):
    """Repair one Writer slot without accepting a failed diversity candidate.

    The CARD core retries Writer failures at the comment level.  This adapter
    adds evaluator-aligned lexical/semantic checks after the native guards and
    sends only the failing slot through local repairs. It never substitutes a
    best failed candidate for an accepted one or silently drops a known,
    repairable slot.
    """

    def generate_writer_text_with_guards(**kwargs: Any) -> dict[str, Any]:
        kwargs = {
            **kwargs,
            "max_writer_tokens": writer_provider_token_budget(
                kwargs.get("task"),
                configured_max=kwargs.get("max_writer_tokens", 0),
            ),
        }
        result = original(**kwargs)
        previous_texts = module.previous_comment_texts(kwargs.get("previous_comments"))
        thread_target = distribution_target_with_slot_progress(
            getattr(module, "GENERALIZED_ACTIVE_DISTRIBUTION_TARGET", {}) or {},
            local_task_id=getattr(kwargs.get("task"), "local_task_id", 0),
        )
        attempts = annotate_writer_attempts(
            result.get("attempts") or [], start_at=0, repair_round=0
        )
        text = str(result.get("text") or "").strip()
        diagnostics, dynamic_problems = writer_distribution_problems(
            module,
            text=text,
            previous_comments=kwargs.get("previous_comments"),
            previous_texts=previous_texts,
            calibration=calibration,
            thread_target=thread_target,
            task=kwargs.get("task"),
        )
        native_problems = last_writer_problems(result)
        problems = deduplicate_problems([*native_problems, *dynamic_problems])
        if not result.get("skip") and not problems:
            result["distribution_diagnostics"] = diagnostics
            _append_writer_diversity_audit(
                diagnostics,
                task=kwargs.get("task"),
                selected_attempt=len(attempts),
                recovered=False,
                attempts=attempts,
                final_status="accepted",
            )
            return result

        recovery_config = dict(
            getattr(module, "GENERALIZED_WRITER_DIVERSITY_CONFIG", {}) or {}
        )
        hard_recovery_rounds = int(recovery_config.get("hard_recovery_rounds", 0))
        original_task = kwargs.get("task")
        hard_failures = hard_realization_problems(problems)
        hard_recovery_attempt = 0
        while hard_failures and hard_recovery_attempt < hard_recovery_rounds:
            hard_recovery_attempt += 1
            repaired_task = writer_local_repair_task(
                original_task,
                problems=problems,
                repair_round=hard_recovery_attempt,
                previous_candidate_text=text,
            )
            candidate = original(
                **{
                    **kwargs,
                    "task": repaired_task,
                    "writer_retries": 0,
                }
            )
            attempts.extend(
                annotate_writer_attempts(
                    candidate.get("attempts") or [],
                    start_at=len(attempts),
                    repair_round=hard_recovery_attempt,
                )
            )
            candidate_text = str(candidate.get("text") or "").strip()
            candidate_diagnostics, candidate_dynamic = writer_distribution_problems(
                module,
                text=candidate_text,
                previous_comments=kwargs.get("previous_comments"),
                previous_texts=previous_texts,
                calibration=calibration,
                thread_target=thread_target,
                task=repaired_task,
            )
            candidate_problems = deduplicate_problems(
                [*last_writer_problems(candidate), *candidate_dynamic]
            )
            result = candidate
            text = candidate_text
            diagnostics = candidate_diagnostics
            problems = candidate_problems
            hard_failures = hard_realization_problems(candidate_problems)
            if hard_failures:
                continue
            diagnostic_only = all(
                is_single_stage_diagnostic(problem)
                for problem in candidate_problems
            )
            if candidate_text and diagnostic_only:
                accepted = {
                    **candidate,
                    "skip": False,
                    "attempts": attempts,
                    "distribution_diagnostics": candidate_diagnostics,
                    "candidate_selection": {
                        "reason": "accepted_after_hard_slot_completion",
                        "hard_recovery_round": hard_recovery_attempt,
                        "diagnostic_only_problems": candidate_problems,
                    },
                }
                _append_writer_diversity_audit(
                    candidate_diagnostics,
                    task=repaired_task,
                    selected_attempt=len(attempts),
                    recovered=True,
                    attempts=attempts,
                    final_status="accepted_after_hard_slot_completion",
                )
                return accepted
        if (
            text
            and problems
            and all(
                is_single_stage_diagnostic(problem)
                for problem in problems
            )
        ):
            acceptance_reason = "accepted_first_pass_distribution_diagnostics"
            accepted = {
                **result,
                "skip": False,
                "distribution_diagnostics": diagnostics,
                "candidate_selection": {
                    "reason": acceptance_reason,
                    "diagnostic_only_problems": problems,
                },
            }
            _append_writer_diversity_audit(
                diagnostics,
                task=kwargs.get("task"),
                selected_attempt=len(attempts),
                recovered=False,
                attempts=attempts,
                final_status=acceptance_reason,
            )
            return accepted

        if not only_repairable_writer_problems(problems):
            rejected = {
                **result,
                "skip": True,
                "attempts": attempts,
                "skip_reason": ",".join(problems),
                "candidate_selection": {
                    "reason": "unknown_guard_failure",
                    "last_problems": problems,
                },
                "distribution_diagnostics": diagnostics,
            }
            _append_writer_diversity_audit(
                diagnostics,
                task=kwargs.get("task"),
                selected_attempt=0,
                recovered=False,
                attempts=attempts,
                final_status="rejected_unknown_guard_failure",
            )
            return rejected

        repair_rounds = int(recovery_config.get("local_repair_rounds", 0))
        configured_limit = int(recovery_config.get("slot_retry_limit", repair_rounds))
        # Zero used to mean infinite and caused a single impossible prefix to
        # consume thousands of calls. It now means one bounded pass over the
        # configured strategy set.
        slot_retry_limit = configured_limit if configured_limit > 0 else repair_rounds
        last_result = result
        last_diagnostics = diagnostics
        last_problems = problems
        best_distribution_candidate: tuple[
            tuple[float, ...],
            dict[str, Any],
            dict[str, Any],
            list[str],
            int,
        ] | None = None

        def consider_distribution_candidate(
            candidate_result: dict[str, Any],
            candidate_diagnostics: dict[str, Any],
            candidate_problems: list[str],
            candidate_round: int,
        ) -> None:
            nonlocal best_distribution_candidate
            if not str(candidate_result.get("text") or "").strip():
                return
            if not only_distribution_writer_problems(candidate_problems):
                return
            rank = distribution_candidate_rank(candidate_diagnostics)
            row = (
                rank,
                candidate_result,
                candidate_diagnostics,
                candidate_problems,
                candidate_round,
            )
            if best_distribution_candidate is None or rank < best_distribution_candidate[0]:
                best_distribution_candidate = row

        consider_distribution_candidate(result, diagnostics, problems, 0)
        repair_attempt = 0
        strategy_count = min(len(LOCAL_REPAIR_STRATEGIES), max(0, repair_rounds))
        while strategy_count > 0 and repair_attempt < slot_retry_limit:
            repair_attempt += 1
            strategy_round = (repair_attempt - 1) % strategy_count + 1
            repaired_task = writer_local_repair_task(
                original_task,
                problems=last_problems,
                repair_round=strategy_round,
                previous_candidate_text=str(last_result.get("text") or ""),
            )
            repaired_kwargs = {
                **kwargs,
                "task": repaired_task,
                # One native attempt per local repair keeps the repair scoped
                # to this slot rather than multiplying whole-thread retries.
                "writer_retries": 0,
            }
            candidate = original(**repaired_kwargs)
            candidate_attempts = annotate_writer_attempts(
                candidate.get("attempts") or [],
                start_at=len(attempts),
                repair_round=repair_attempt,
            )
            attempts.extend(candidate_attempts)
            candidate_text = str(candidate.get("text") or "").strip()
            candidate_diagnostics, candidate_dynamic = writer_distribution_problems(
                module,
                text=candidate_text,
                previous_comments=kwargs.get("previous_comments"),
                previous_texts=previous_texts,
                calibration=calibration,
                thread_target=thread_target,
                task=repaired_task,
            )
            candidate_problems = deduplicate_problems(
                [*last_writer_problems(candidate), *candidate_dynamic]
            )
            last_result = candidate
            last_diagnostics = candidate_diagnostics
            last_problems = candidate_problems
            consider_distribution_candidate(
                candidate,
                candidate_diagnostics,
                candidate_problems,
                repair_attempt,
            )
            if not candidate.get("skip") and not candidate_problems:
                accepted = {
                    **candidate,
                    "attempts": attempts,
                    "distribution_diagnostics": candidate_diagnostics,
                    "candidate_selection": {
                        "reason": "accepted_after_local_distribution_repair",
                        "repair_round": repair_attempt,
                        "strategy_round": strategy_round,
                        "selected_attempt": len(attempts),
                        "joint_target_distance": candidate_diagnostics.get(
                            "joint_target_distance"
                        ),
                        "self_bleu": candidate_diagnostics.get("self_bleu"),
                        "semantic_cosine": candidate_diagnostics.get(
                            "semantic_cosine"
                        ),
                    },
                }
                _append_writer_diversity_audit(
                    candidate_diagnostics,
                    task=repaired_task,
                    selected_attempt=len(attempts),
                    recovered=True,
                    attempts=attempts,
                    final_status="accepted_after_local_distribution_repair",
                )
                return accepted
            if not only_repairable_writer_problems(candidate_problems):
                break
            if repair_attempt == 1 or repair_attempt % 5 == 0:
                summary = ",".join(
                    item.split(":", 1)[0] for item in candidate_problems[:4]
                )
                print(
                    f"[writer-slot-retry] task={getattr(original_task, 'local_task_id', 0)} "
                    f"attempt={repair_attempt} problems={summary}",
                    flush=True,
                )

        if best_distribution_candidate is not None:
            (
                _,
                selected,
                selected_diagnostics,
                selected_problems,
                selected_round,
            ) = best_distribution_candidate
            if distribution_candidate_is_reachable(selected_diagnostics):
                accepted = {
                    **selected,
                    "skip": False,
                    "attempts": attempts,
                    "distribution_diagnostics": selected_diagnostics,
                    "candidate_selection": {
                        "reason": "accepted_best_bounded_distribution_candidate",
                        "repair_round": selected_round,
                        "candidate_budget": slot_retry_limit,
                        "residual_distribution_problems": selected_problems,
                        "joint_target_distance": selected_diagnostics.get(
                            "joint_target_distance"
                        ),
                    },
                }
                _append_writer_diversity_audit(
                    selected_diagnostics,
                    task=original_task,
                    selected_attempt=len(attempts),
                    recovered=True,
                    attempts=attempts,
                    final_status="accepted_best_bounded_distribution_candidate",
                )
                return accepted

        # Exhausting the retries on phrasing alone must not cost a comment. The
        # repetition guard made these codes non-distribution failures, so the
        # loop above never registered such a candidate as a fallback; v77 lost
        # 14 of 186 slots that way, which also shortens the matched thread the
        # structural metrics are scored against.
        if str(last_result.get("text") or "").strip() and only_style_problems(
            last_problems
        ):
            retained = {
                **last_result,
                "skip": False,
                "attempts": attempts,
                "distribution_diagnostics": last_diagnostics,
                "candidate_selection": {
                    "reason": "accepted_style_residual_after_repair",
                    "repair_rounds_attempted": repair_attempt,
                    "slot_retry_limit": slot_retry_limit,
                    "residual_problems": last_problems,
                },
            }
            _append_writer_diversity_audit(
                last_diagnostics,
                task=original_task,
                selected_attempt=len(attempts),
                recovered=True,
                attempts=attempts,
                final_status="accepted_style_residual_after_repair",
            )
            return retained

        rejected = {
            **last_result,
            "skip": True,
            "attempts": attempts,
            "skip_reason": ",".join(last_problems or problems),
            "candidate_selection": {
                "reason": "distribution_repair_exhausted",
                "repair_rounds_attempted": repair_attempt,
                "slot_retry_limit": slot_retry_limit,
                "last_problems": last_problems,
            },
            "distribution_diagnostics": last_diagnostics,
        }
        _append_writer_diversity_audit(
            last_diagnostics,
            task=original_task,
            selected_attempt=0,
            recovered=False,
            attempts=attempts,
            final_status="rejected_distribution_repair_exhausted",
        )
        return rejected

    return generate_writer_text_with_guards


def _substantive_safe_degraded_task():
    """Keep hard-failure recovery on the exact Planner slot.

    CARD degraded a failed slot into a safer, shorter one. That rewrote the
    Planner's role, payload, story, tone, and length, so this takes no core
    implementation to fall back on.
    """

    def degraded_task_for_guard_failure(task: Any, problems: list[str]) -> Any:
        del problems
        # Hard schema/safety recovery may ask the Writer to try the same slot
        # again, but it must not change role, payload, story, tone, or length.
        return task

    return degraded_task_for_guard_failure


def _finalize_post_generation(module: ModuleType, original: Any):
    """Annotate valid first-pass output while preserving audited omissions."""

    def generate_post_from_tasks(**kwargs: Any) -> dict[str, Any]:
        tasks = list(kwargs.get("tasks") or [])
        post = original(**kwargs)
        records = [
            row for row in (post.get("generation_records") or []) if isinstance(row, dict)
        ]
        skipped = [row for row in records if bool(row.get("skipped"))]
        coverage = generation_coverage(tasks, records)
        planner_coverage = dict(
            getattr(module, "GENERALIZED_ACTIVE_PLANNER_COVERAGE", {}) or {}
        )
        post.setdefault("thread_plan", {})["first_pass_coverage"] = {
            **planner_coverage,
            **coverage,
        }
        if skipped:
            print(
                "[writer-coverage] "
                f"tasks={coverage['writer_task_slots']} "
                f"generated={coverage['generated_comments']} "
                f"skipped={coverage['skipped_comments']} "
                "policy=persist_valid_comments",
                flush=True,
            )
        if getattr(module, "GENERALIZED_ACTOR_MODE", "") == MODE_DOMAIN_DERIVED:
            seed_post = kwargs.get("seed_post")
            run_index = _safe_int(kwargs.get("run_index"), 0)
            post_slot = _safe_int(kwargs.get("post_slot"), 0)
            actor_rows = []
            for row in records:
                task_row = row.get("task") or {}
                sample_id = task_row.get("real_sample_id") or task_row.get("local_task_id")
                state = module.GENERALIZED_ACTOR_ASSIGNMENTS.get(
                    assignment_key(seed_post, sample_id)
                )
                comment = row.get("comment")
                if state is None or not isinstance(comment, dict):
                    continue
                payload = state.to_dict()
                comment["actor_state"] = payload
                comment["actor_conditioning"] = MODE_DOMAIN_DERIVED
                comment["author"] = actor_author(
                    state,
                    run_index=run_index,
                    post_slot=post_slot,
                )
                actor_rows.append(payload)
            post.setdefault("thread_plan", {})["actor_conditioning"] = {
                "mode": MODE_DOMAIN_DERIVED,
                "source": "evaluation-excluded domain references plus visible thread",
                "participant_states": actor_rows,
            }
        return post

    return generate_post_from_tasks


def _append_distribution_audit(report: dict[str, Any]) -> None:
    raw_path = os.environ.get("GENERALIZED_CARD_DISTRIBUTION_AUDIT_JSONL", "").strip()
    if not raw_path:
        return
    path = Path(raw_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(report, ensure_ascii=False, sort_keys=True) + "\n")


def _append_writer_diversity_audit(
    diagnostics: dict[str, Any],
    *,
    task: Any,
    selected_attempt: int,
    recovered: bool,
    attempts: list[dict[str, Any]],
    final_status: str,
) -> None:
    raw_path = os.environ.get(
        "GENERALIZED_CARD_WRITER_DIVERSITY_AUDIT_JSONL", ""
    ).strip()
    if not raw_path:
        return
    payload = {
        "local_task_id": _safe_int(getattr(task, "local_task_id", 0), 0),
        "selected_attempt": int(selected_attempt),
        "recovered_after_exhaustion": bool(recovered),
        "final_status": final_status,
        "attempts": [
            {
                "attempt": _safe_int(row.get("attempt"), 0),
                "word_count": _safe_int(row.get("word_count"), 0),
                "problems": list(row.get("problems") or []),
                "degraded": bool(row.get("degraded")),
            }
            for row in attempts
            if isinstance(row, dict)
        ],
        **diagnostics,
    }
    path = Path(raw_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def _discussion_loader(module: ModuleType, config: DomainConfig):
    def load_or_init_discussion(*, run_dir: Path, run_index: int, args: Any) -> dict[str, Any]:
        path = run_dir / "discussion.json"
        if path.exists() and not args.force_post:
            return json.loads(path.read_text(encoding="utf-8"))
        return {
            "meta": {
                "run_id": f"sampled_reddit_run_{run_index:02d}",
                "generator": module.GENERATOR_NAME,
                "model": args.writer_model,
                "writer_profile": args.writer_profile,
                "planner_model": args.planner_model,
                "product_category": config.domain_id,
                "domain": config.to_public_dict(),
                "created_at": module.utc_now(),
                "seed": args.seed,
            },
            "posts": [],
        }

    return load_or_init_discussion


def _long_helpful_anchor(module: ModuleType):
    def has_anchor(text: str, task: Any | None = None) -> bool:
        if task is not None and module.output_uses_visible_concrete_anchor(text, task):
            return True
        if re.search(
            r"https?://|www\.|[$€£]\s*\d|\b\d+(?:\.\d+)?\s*(?:%|mm|cm|gb|tb|mah|hz|mp|fps|w|wh|kg|g|inch(?:es)?)\b",
            str(text),
            flags=re.I,
        ):
            return True
        if task is not None and module.has_task_anchor_overlap(text, task):
            return True
        return bool(
            re.search(
                r"\b(i|my|we)\b.{0,90}\b(bought|used|tried|tested|returned|replaced|updated|installed|"
                r"charged|measured|noticed|compared|fixed|failed|worked)\b",
                str(text),
                flags=re.I,
            )
        )

    return has_anchor


def _retry_note_for_problems(module: ModuleType):
    def note(problems: list[str], task: Any) -> str:
        notes: list[str] = []
        if "exact_duplicate" in problems:
            notes.append("Use a different opening, clause path, and local wording from every earlier comment.")
        if "parent_copy" in problems:
            notes.append("Reply to the parent without copying or continuing its wording.")
        if any(item in problems for item in ("placeholder_literal", "planner_skeleton_residue", "meta_template_quote_heading")):
            notes.append("Write an actual Reddit comment with no labels, fake links, bracket placeholders, or planner text.")
        if "long_helpful_too_generic" in problems:
            notes.append("Use one visible product, technical term, measurement, caveat, or parent-specific detail; otherwise shorten the reply.")
        if "missing_concrete_anchor" in problems:
            notes.append("Use one or two factual anchors explicitly listed in the prompt. Do not invent replacements.")
        if "length_too_long" in problems:
            low, high = module.LENGTH_BUCKET_BOUNDS.get(task.length_bucket, (8, 45))
            notes.append(f"Keep the rewrite near {low}-{high} words.")
        if "low_info_too_long" in problems:
            notes.append(f"Keep this low-information turn within {module.low_info_word_limit(task) or 10} words.")
        if "real_slot_too_short" in problems:
            notes.append("Keep the same narrow point but match the requested substantive length bucket.")
        if any(item in problems for item in ("opening_reused", "opener_family_reused", "template_phrase_reused")):
            notes.append("Change the entry shape and avoid repeated acknowledgements, first-person templates, and connective phrases.")
        overlap = next((item.split(":", 1)[1] for item in problems if item.startswith("lexical_overlap_high:")), "")
        if overlap:
            notes.append(
                "The measured lexical path is too close to this thread's held-out-real target. "
                f"Diagnostic and nearest prior excerpts: {overlap}. Preserve the local claim and factual anchors, "
                "but use a different opener, clause order, connective pattern, and wording. Do not copy the excerpts."
            )
        semantic = next(
            (
                item.split(":", 1)[1]
                for item in problems
                if item.startswith(("semantic_overlap_high:", "semantic_overlap_low:"))
            ),
            "",
        )
        embedded_semantic = next(
            (
                item.split(";semantic_overlap_", 1)[1]
                for item in problems
                if ";semantic_overlap_" in item
            ),
            "",
        )
        if semantic or embedded_semantic:
            detail = semantic or embedded_semantic
            if any("semantic_overlap_low:" in item for item in problems):
                notes.append(
                    "The candidate is too disconnected from the thread's held-out-real semantic range. "
                    f"Diagnostic: {detail}. Keep the sampled local move, parent relation, and visible anchor, but do not repeat prior wording."
                )
            else:
                notes.append(
                    "The candidate is semantically too close to earlier comments even if some words differ. "
                    f"Nearest prior excerpts and diagnostic: {detail}. Keep the assigned role and topic, but contribute a different implication, evidence role, stance, or decision lens. Do not paraphrase the excerpts."
                )
        if "first_person_frame_unwanted" in problems:
            notes.append("Remove the firsthand-experience frame and make the same direct local move.")
        if "uncertainty_frame_unwanted" in problems:
            notes.append("Remove the uncertainty preface while preserving the same claim.")
        if "question_mark_unwanted" in problems:
            notes.append("This slot is not a question; keep the same point as a statement or fragment.")
        if "empty" in problems:
            notes.append("Return a non-empty comment body only.")
        return " ".join(notes) or "Keep the same local task and rewrite only the failed surface form."

    return note


def _guard_fallback_retry_note(problems: list[str]) -> str:
    if any(item in problems for item in ("meta_template_quote_heading", "placeholder_literal", "planner_skeleton_residue")):
        return " Final fallback: write one plain Reddit sentence without labels, markdown headings, placeholders, or fake links."
    if any(item in problems for item in ("missing_concrete_anchor", "long_helpful_too_generic")):
        return " Final fallback: use one visible factual anchor and one narrow local point; do not add any new fact."
    return " Final fallback: write a raw local Reddit reply, not a complete assistant answer."


def _generic_real_surface_shape(row: dict[str, Any]) -> str:
    return infer_surface_shape(row)


def _allows_first_person(text: str) -> bool:
    return bool(re.search(r"\b(i|i'm|i’ve|i've|ive|my|mine|we|our)\b", str(text), flags=re.I))


def _allows_uncertainty(text: str) -> bool:
    return bool(
        re.search(
            r"\b(maybe|might|not sure|i think|i guess|probably|possibly|seems?|appears?|could)\b|\?",
            str(text),
            flags=re.I,
        )
    )


def _sanitize_writer_text(module: ModuleType):
    def sanitize(raw: str) -> str:
        text = str(raw or "").strip()
        if text.startswith("```"):
            text = module.strip_code_fence(text)
        text = module.remove_thinking_blocks(text)
        text = module.remove_space_token_leakage(text)
        for prefix in ("Comment:", "Answer:", "Assistant:", "assistant:", "Output:"):
            if text.startswith(prefix):
                text = text[len(prefix) :].strip()
        return text.strip().strip('"').strip()

    return sanitize


def _chat_completion_text(
    *,
    client: Any,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int,
    response_format_json: bool,
    extra_body: dict[str, Any] | None = None,
) -> str:
    messages = inject_persona_system(messages)
    retries = max(1, _int_env("LLM_API_RETRIES", 4))
    delay = max(0.0, _float_env("LLM_API_RETRY_DELAY", 10.0))
    last_error: Exception | None = None
    completion_boost = 0
    for attempt in range(retries):
        kwargs = _completion_kwargs(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format_json=response_format_json,
            extra_body=extra_body,
        )
        if completion_boost and "max_completion_tokens" in kwargs:
            kwargs["max_completion_tokens"] += completion_boost
        try:
            response = client.chat.completions.create(**kwargs)
            _record_usage(response, model=model)
            content = str(response.choices[0].message.content or "").strip()
            if content:
                sleep_seconds = _float_env("LLM_CALL_SLEEP_SECONDS", 0.0)
                if sleep_seconds > 0:
                    time.sleep(sleep_seconds)
                return content
            finish = getattr(response.choices[0], "finish_reason", None)
            last_error = RuntimeError(f"empty completion; finish_reason={finish}")
            completion_boost = _next_completion_boost(
                current=completion_boost,
                finish_reason=finish,
                reasoning_model="max_completion_tokens" in kwargs,
            )
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if _is_output_limit_error(exc) and "max_completion_tokens" in kwargs:
                completion_boost = _next_completion_boost(
                    current=completion_boost,
                    finish_reason="length",
                    reasoning_model=True,
                )
        if attempt + 1 < retries:
            print(
                f"[llm-retry] model={model} attempt={attempt + 1}/{retries} "
                f"error={type(last_error).__name__}:{last_error}",
                flush=True,
            )
            time.sleep(delay * (attempt + 1))
    assert last_error is not None
    raise last_error


def _endpoint_preflight_with_retry(module: ModuleType):
    checked_urls: set[str] = set()

    def preflight(
        *,
        role: str,
        base_url: str,
        api_key: str | None,
        allow_remote: bool,
    ) -> None:
        """Validate permanent endpoint errors without failing on transient timeouts."""

        normalized = base_url.strip().rstrip("/")
        if not normalized:
            raise SystemExit(f"{role} base URL is empty.")
        if normalized in checked_urls:
            return

        url = normalized + "/models"
        headers = {"Accept": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        request = urllib.request.Request(url, headers=headers, method="GET")
        retries = max(1, _int_env("ENDPOINT_PREFLIGHT_RETRIES", 3))
        timeout = max(1.0, _float_env("ENDPOINT_PREFLIGHT_TIMEOUT", 30.0))
        delay = max(0.0, _float_env("ENDPOINT_PREFLIGHT_RETRY_DELAY", 2.0))
        last_error: Exception | None = None

        for attempt in range(retries):
            try:
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    body = response.read().decode("utf-8", errors="replace")
                    content_type = str(response.headers.get("Content-Type") or "")
                lowered = body.lower()
                if "text/html" in content_type.lower() or "<html" in lowered or "page not found" in lowered:
                    raise SystemExit(
                        module.describe_bad_endpoint(
                            role=role,
                            base_url=base_url,
                            content_type=content_type,
                            body=body,
                        )
                    )
                if not allow_remote and "127.0.0.1" in normalized and "\"data\"" not in body and "\"object\"" not in body:
                    raise SystemExit(
                        f"{role} endpoint responded, but it does not look like an "
                        f"OpenAI-compatible `/v1/models` response:\n{base_url}"
                    )
                checked_urls.add(normalized)
                return
            except urllib.error.HTTPError as exc:
                if exc.code not in {408, 429} and exc.code < 500:
                    body = exc.read().decode("utf-8", errors="replace")
                    content_type = str(exc.headers.get("Content-Type") or "")
                    raise SystemExit(
                        module.describe_bad_endpoint(
                            role=role,
                            base_url=base_url,
                            content_type=content_type,
                            body=body,
                        )
                    ) from exc
                last_error = exc
            except (TimeoutError, urllib.error.URLError) as exc:
                last_error = exc

            if attempt + 1 < retries:
                print(
                    f"[endpoint-preflight-retry] role={role} attempt={attempt + 1}/{retries} "
                    f"error={type(last_error).__name__}:{last_error}",
                    flush=True,
                )
                if delay:
                    time.sleep(delay * (attempt + 1))

        # A GET /models timeout does not prove that chat completions are down.
        # Let the SDK's configured retries make the authoritative request.
        print(
            f"[endpoint-preflight-warning] role={role} url={url} "
            f"error={type(last_error).__name__}:{last_error}; continuing to SDK request",
            flush=True,
        )
        checked_urls.add(normalized)

    return preflight


def _completion_kwargs(
    *,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int,
    response_format_json: bool,
    extra_body: dict[str, Any] | None,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"model": model, "messages": messages}
    if response_format_json:
        kwargs["response_format"] = {"type": "json_object"}
    if extra_body:
        kwargs["extra_body"] = extra_body
    if _uses_max_completion_tokens(model):
        # GPT-5 and reasoning-model endpoints reject non-default temperature.
        # Short writer caps otherwise can be consumed entirely by hidden
        # reasoning. The visible text is still constrained by the prompt,
        # length-bucket shaping, and writer guards after generation.
        reserve = max(0, _int_env("GPT5_REASONING_TOKEN_RESERVE", 256))
        completion_limit = max_tokens + reserve if max_tokens <= 512 else max_tokens
        kwargs["max_completion_tokens"] = completion_limit
    else:
        kwargs["max_tokens"] = max_tokens
        kwargs["temperature"] = temperature
    reasoning_effort = os.environ.get("REASONING_EFFORT", "").strip()
    if reasoning_effort and model.lower().startswith("gpt-5"):
        kwargs["reasoning_effort"] = reasoning_effort
    return kwargs


def _next_completion_boost(*, current: int, finish_reason: Any, reasoning_model: bool) -> int:
    if finish_reason != "length" or not reasoning_model:
        return current
    return min(2048, max(128, current * 2))


def _is_output_limit_error(error: Exception) -> bool:
    message = str(error).lower()
    return "max_tokens or model output limit" in message or (
        "max_tokens" in message and "higher" in message
    )


def _record_usage(response: Any, *, model: str) -> None:
    scripts = REPO_ROOT / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    try:
        from token_usage_tracker import record_openai_usage

        record_openai_usage(
            response,
            model=model,
            component="generalized_card_generator",
        )
    except Exception:
        return


def _parse_json_object(raw: str) -> dict[str, Any]:
    text = str(raw or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    candidates = [text]
    start, end = text.find("{"), text.rfind("}")
    if 0 <= start < end:
        candidates.append(text[start : end + 1])
    for candidate in candidates:
        try:
            payload = json.loads(candidate)
            if isinstance(payload, dict):
                return payload
        except json.JSONDecodeError:
            pass
    original = candidates[-1]
    repaired = _repair_json(original)
    try:
        payload = json.loads(repaired)
    except json.JSONDecodeError:
        payload = ast.literal_eval(original)
    if not isinstance(payload, dict):
        raise ValueError("planner response was not a JSON object")
    return payload


def _repair_json(text: str) -> str:
    value = re.sub(r",\s*([}\]])", r"\1", text)
    value = value.replace("“", '"').replace("”", '"').replace("’", "'")
    value = re.sub(r"\bTrue\b", "true", value)
    value = re.sub(r"\bFalse\b", "false", value)
    value = re.sub(r"\bNone\b", "null", value)
    return value


def _uses_max_completion_tokens(model: str) -> bool:
    value = model.strip().lower()
    return value.startswith(("gpt-5", "o1", "o3"))


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)) or default)
    except ValueError:
        return default


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)) or default)
    except ValueError:
        return default
