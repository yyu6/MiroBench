#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from generalized_card.backend import (  # noqa: E402
    DEFAULT_GENERATOR_PROFILE,
    GENERATOR_PROFILES,
    CORE_ALGORITHM_SYMBOLS,
    DOMAIN_ADAPTATION_BOUNDARIES,
    GENERALIZED_ALGORITHM_EXTENSIONS,
)
from generalized_card.actor_conditioning import (  # noqa: E402
    ACTOR_MODES,
    MODE_DOMAIN_DERIVED,
)
from generalized_card.data import build_seed_pool  # noqa: E402
from generalized_card.domain_profile import (  # noqa: E402
    CARD_CONTEXT_DROPOUT_RATE,
    CARD_CONTEXT_JITTER_RATE,
    build_domain_profile,
    load_domain_profile,
)
from generalized_card.core_contract import (  # noqa: E402
    CORE_POLICY_VERSION,
    CURRENT_GENERATION_CORE_NAMES,
    GENERATION_ADAPTER_CORE_NAMES,
    GENERALIZED_V2_GENERATION_POLICY_VERSION,
    REVISION_CORE_POLICY_VERSION,
    verify_core_contract,
    verify_run_policy,
)
from generalized_card.domain import REPO_ROOT, load_domain_config  # noqa: E402
from generalized_card.persona_bridge import (  # noqa: E402
    MODE_NONE,
    PERSONA_MODES,
    annotate_generated_outputs,
    build_runtime,
)


DEFAULT_PRICES = {
    "gpt-5.4-mini": (0.75, 0.075, 4.50),
    "gpt-4o-mini": (0.15, 0.075, 0.60),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run generalized CARD planner + writer generation."
    )
    parser.add_argument("--domain", default="camera")
    parser.add_argument("--model", default="gpt-5.4-mini")
    parser.add_argument("--base-url", default="https://api.openai.com/v1")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--tag", required=True)
    parser.add_argument(
        "--generator-profile",
        choices=GENERATOR_PROFILES,
        default=DEFAULT_GENERATOR_PROFILE,
        help=(
            "generalized-v2 uses the proven domain-neutral Planner-Writer; "
            "card-snapshot is retained only for exact historical snapshot audits"
        ),
    )
    parser.add_argument("--pool-size", type=int, default=150)
    parser.add_argument("--max-posts", type=int, default=10)
    parser.add_argument("--posts-per-run", type=int, default=5)
    parser.add_argument(
        "--start-seed-index",
        type=int,
        default=0,
        help=(
            "Zero-based seed-pool offset for a focused, reproducible smoke run. "
            "The generated range is [start-seed-index, start-seed-index + max-posts)."
        ),
    )
    parser.add_argument("--sampling-seed", type=int, default=42)
    parser.add_argument("--max-comments-per-post", type=int, default=0)
    parser.add_argument("--comment-count-scale", type=float, default=1.0)
    parser.add_argument("--matched-real-comments", type=int, default=0)
    parser.add_argument(
        "--exact-matched-thread-size",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--context-dropout-rate", type=float)
    parser.add_argument("--context-jitter-rate", type=float)
    parser.add_argument(
        "--domain-profile",
        type=Path,
        help="Frozen profile built from non-seed real threads. Built inside the run directory by default.",
    )
    parser.add_argument(
        "--planner-max-tokens",
        type=int,
        default=10000,
        help=(
            "Output budget for the root-branch Planner. High-fanout real threads "
            "need enough space to return one compact contract per root."
        ),
    )
    parser.add_argument("--comment-planner-max-tokens", type=int, default=18000)
    parser.add_argument(
        "--comment-planner-batch-size",
        type=int,
        default=8,
        help=(
            "Number of comment slots planned with one shared semantic ledger. "
            "The shared ledger preserves complementary first-pass contributions "
            "while smaller batches prevent omitted JSON slots on busy threads."
        ),
    )
    parser.add_argument(
        "--plan-quality-repairs",
        type=int,
        default=3,
        help=(
            "Bounded slot-local repair rounds for semantic plan collisions. "
            "These run before Writer generation and preserve healthy plans."
        ),
    )
    parser.add_argument("--plan-similarity-threshold", type=float, default=0.72)
    parser.add_argument(
        "--plan-embedding-quality",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use local sentence embeddings to detect paraphrased plan collisions.",
    )
    parser.add_argument(
        "--plan-embedding-model",
        default="sentence-transformers/all-mpnet-base-v2",
    )
    parser.add_argument("--plan-embedding-threshold", type=float, default=0.70)
    parser.add_argument("--plan-embedding-device", default="cpu")
    parser.add_argument("--plan-max-collision-rate", type=float, default=0.10)
    parser.add_argument("--max-perspective-share", type=float, default=0.34)
    parser.add_argument(
        "--strict-plan-quality",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Record unresolved plan-quality warnings after the configured plan "
            "pass. Missing slots use bounded schema recovery and then fail before "
            "Writer generation; they are never omitted."
        ),
    )
    parser.add_argument("--writer-max-tokens", type=int, default=260)
    parser.add_argument("--api-retries", type=int, default=2)
    parser.add_argument("--writer-retries", type=int, default=0)
    parser.add_argument(
        "--writer-hard-recovery-rounds",
        type=int,
        default=2,
        help=(
            "Bounded slot-local completion for non-persistable Writer output "
            "such as empty text, exact duplicates, parent copies, or leaked "
            "planner placeholders. It never optimizes soft metrics."
        ),
    )
    parser.add_argument("--retry-delay", type=float, default=10.0)
    parser.add_argument("--call-sleep-seconds", type=float, default=0.0)
    parser.add_argument(
        "--post-retry-limit",
        type=int,
        default=1,
        help=(
            "Maximum total attempts for one unfinished post. The default 1 disables "
            "automatic whole-post regeneration; hard Writer failures use their "
            "bounded slot-local handling first."
        ),
    )
    parser.add_argument(
        "--post-retry-delay",
        type=float,
        default=15.0,
        help="Base delay in seconds between recoverable post attempts.",
    )
    parser.add_argument("--reasoning-effort", default="")
    parser.add_argument("--gpt5-reasoning-token-reserve", type=int, default=256)
    parser.add_argument(
        "--writer-prompt",
        choices=("focused", "full"),
        default="focused",
        help=(
            "Which Writer prompt to render. 'full' reproduces policy v73 exactly "
            "(mean 22,249 characters). 'focused' keeps the compact Planner "
            "discourse, distribution, and grounding contract without repeated "
            "control paraphrases; a rebuilt-thread A/B held within-thread "
            "diversity at 13%% of the old size."
        ),
    )
    parser.add_argument(
        "--writer-route-lock",
        choices=("own_words", "say_only"),
        default="own_words",
        help=(
            "How the Planner's semantic_move reaches the Writer. 'say_only' "
            "reproduces v73/v74 on both sides: the Writer is told 'Say this, and "
            "only this', and the reply planner is asked for 'a full sentence'. "
            "'own_words' states the move as a specification to realize. Plan echo "
            "(longest shared word run >= 12) measured 0.4%% in v67, 10.2%% in v73 "
            "and 25.8%% in v74."
        ),
    )
    parser.add_argument(
        "--social-contract-coherence",
        choices=("off", "on"),
        default="on",
        help=(
            "Whether v80 rejects contradictory story/tone plans and renders the "
            "matching Writer guidance. 'off' reproduces pre-v80 behavior."
        ),
    )
    parser.add_argument(
        "--reply-sibling-visibility",
        choices=("off", "on"),
        default="on",
        help=(
            "Whether direct-reply planning sees sibling delta coverage. 'off' "
            "reproduces the pre-v80 parent-only rows."
        ),
    )
    parser.add_argument(
        "--own-fact-license",
        choices=("off", "own", "named"),
        default="off",
        help=(
            "How much concrete detail a slot may state beyond what is visible. "
            "'off' reproduces v75: one blanket ban covering the seed product and "
            "the speaker's own past alike, which put a permission ('Equipment you "
            "may claim as your own') and its revocation ('do not invent ... or "
            "personal experiences') in the same prompt for 170 of 522 slots. "
            "'own' licenses the speaker's own kit and history on first-person "
            "slots; run v76b measured it and it moved concreteness the WRONG way "
            "(0.05 -> 0.02 per comment against a real 0.54), because 68%% of real "
            "concrete comments have no first-person frame -- kept only as a "
            "reproducible arm. 'named' is the correction: on any slot with room, "
            "license naming and quantifying, stated without domain vocabulary, "
            "since quantities (real 12.3x generated) and proper nouns (1.85x) are "
            "the two gaps that hold on all ten matched threads while "
            "specification-shaped tokens range from 0%% to 64%% of comments by "
            "thread."
        ),
    )
    parser.add_argument(
        "--speaker-identity",
        choices=("off", "matched"),
        default="matched",
        help=(
            "Whether a thread has matched recurring participants or one author "
            "per slot. 'matched' uses only author grouping and OP membership; "
            "real author strings and invented biographies never reach the Writer. "
            "'off' is the one-shot-author structural ablation."
        ),
    )
    parser.add_argument(
        "--domain-claim",
        choices=("planned", "off"),
        default="planned",
        help=(
            "Whether the Planner assigns and the Writer receives a separate "
            "domain claim. This is an "
            "ablation control: the claim went from 0 of 522 comments in v69 to "
            "508 of 522 in v71, so it has to be separable from the rest of a "
            "release when attributing a metric change."
        ),
    )
    parser.add_argument(
        "--actor-conditioning",
        choices=ACTOR_MODES,
        default=MODE_NONE,
        help=(
            "Optional thread-local actor state. The default preserves the V12 "
            "Planner-Writer path without an additional persona layer."
        ),
    )
    parser.add_argument(
        "--persona-conditioning",
        choices=PERSONA_MODES,
        default=MODE_NONE,
        help=(
            "MatrAIx persona mode. matraix-projected uses selected behavioral "
            "dimensions with the official MatrAIx system renderer; matraix-full "
            "renders the complete official profile and is intended for diagnostics."
        ),
    )
    parser.add_argument(
        "--matraix-root",
        type=Path,
        default=REPO_ROOT / "third_party" / "MatrAIx-Persona-8B",
    )
    parser.add_argument("--matraix-dataset", type=Path)
    parser.add_argument("--persona-seed", type=int, default=42)
    parser.add_argument("--price-input-per-1m", type=float)
    parser.add_argument("--price-cached-input-per-1m", type=float)
    parser.add_argument("--price-output-per-1m", type=float)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--extend-existing",
        action="store_true",
        help=(
            "Append a larger max-posts range to an existing complete prefix. "
            "All generation settings other than the size must match."
        ),
    )
    parser.add_argument(
        "--upgrade-generation-policy",
        action="store_true",
        help=(
            "Resume a contiguous historical prefix under the current audited "
            "generation policy and record the exact seed boundary."
        ),
    )
    parser.add_argument("--prepare-only", action="store_true")
    return parser


def _load_env_files() -> None:
    """Load the repo's .env files, matching `calibration/cli.py`.

    API keys in this repo live in `third_party/MiroFish/.env`, which the
    calibration CLI already loads. This entry point did not, so a run failed at
    the credential check after the whole preflight had passed. `load_dotenv` does
    not overwrite variables that are already set, so an exported key still wins.
    """

    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    for candidate in (
        REPO_ROOT / ".env",
        REPO_ROOT / "third_party" / "MiroFish" / ".env",
    ):
        if candidate.exists():
            load_dotenv(candidate)


def main() -> None:
    _load_env_files()
    args = build_parser().parse_args()
    if args.start_seed_index < 0:
        raise SystemExit("--start-seed-index must be non-negative")
    if args.pool_size < args.start_seed_index + args.max_posts:
        raise SystemExit(
            "--pool-size must cover the requested seed range: "
            "start-seed-index + max-posts"
        )
    if args.max_posts <= 0 or args.posts_per_run <= 0:
        raise SystemExit("--max-posts and --posts-per-run must be positive")
    if args.plan_quality_repairs < 0:
        raise SystemExit("--plan-quality-repairs must be non-negative")
    if args.comment_planner_batch_size <= 0:
        raise SystemExit("--comment-planner-batch-size must be positive")
    if args.writer_hard_recovery_rounds < 0:
        raise SystemExit("--writer-hard-recovery-rounds must be non-negative")
    if args.post_retry_limit <= 0 or args.post_retry_delay < 0:
        raise SystemExit(
            "--post-retry-limit must be positive and --post-retry-delay non-negative"
        )
    for name in (
        "plan_similarity_threshold",
        "plan_embedding_threshold",
        "plan_max_collision_rate",
        "max_perspective_share",
    ):
        if not 0.0 <= float(getattr(args, name)) <= 1.0:
            raise SystemExit(f"--{name.replace('_', '-')} must be between 0 and 1")
    if args.extend_existing and args.prepare_only:
        raise SystemExit("--extend-existing cannot be combined with --prepare-only")
    if args.extend_existing and args.upgrade_generation_policy:
        raise SystemExit(
            "--extend-existing and --upgrade-generation-policy are separate lineage operations"
        )
    if (
        args.actor_conditioning == MODE_DOMAIN_DERIVED
        and args.persona_conditioning != MODE_NONE
    ):
        raise SystemExit(
            "--actor-conditioning domain-derived cannot be combined with a fixed MatrAIx persona mode"
        )

    config = load_domain_config(args.domain)
    matraix_root = _resolve_repo_path(args.matraix_root)
    matraix_dataset = _resolve_repo_path(
        args.matraix_dataset
        or matraix_root / "persona" / "datasets" / "matraix-persona-dev-sample"
    )
    persona_runtime = build_runtime(
        mode=args.persona_conditioning,
        matraix_root=matraix_root,
        dataset_dir=matraix_dataset,
        assignment_seed=args.persona_seed,
        expertise_dimensions=config.persona_expertise_dimensions,
    )
    persona_config = persona_runtime.public_config()
    generator_core_name = (
        "generator_generalized_v2"
        if args.generator_profile == "generalized-v2"
        else "generator"
    )
    generation_core_names = (
        CURRENT_GENERATION_CORE_NAMES
        if args.generator_profile == "generalized-v2"
        else (generator_core_name, *GENERATION_ADAPTER_CORE_NAMES)
    )
    core_provenance = verify_core_contract(generation_core_names)
    generator_policy_version = (
        GENERALIZED_V2_GENERATION_POLICY_VERSION
        if args.generator_profile == "generalized-v2"
        else CORE_POLICY_VERSION
    )
    run_root = REPO_ROOT / "artifacts" / "generalized_card" / "runs" / args.tag
    generated_root = run_root / "generated"
    if run_root.exists() and not args.resume and not args.prepare_only:
        raise SystemExit(f"Run exists; pass --resume or choose a new --tag: {run_root}")
    seed_pool = (
        REPO_ROOT
        / "artifacts"
        / "generalized_card"
        / "seed_pools"
        / f"{config.domain_id}_{args.pool_size}_seed{args.sampling_seed}.json"
    )
    if not seed_pool.exists():
        build_seed_pool(
            config,
            seed_pool,
            count=args.pool_size,
            seed=args.sampling_seed,
        )

    domain_profile_path = (
        args.domain_profile.expanduser().resolve()
        if args.domain_profile
        else run_root / "domain_profile.json"
    )
    if domain_profile_path.exists():
        domain_profile = load_domain_profile(domain_profile_path)
    else:
        domain_profile = build_domain_profile(
            config,
            seed_pool_path=seed_pool,
            output_path=domain_profile_path,
        )
    if str(domain_profile.get("domain_id") or "") != config.domain_id:
        raise RuntimeError(
            f"Domain profile is for {domain_profile.get('domain_id')!r}, expected {config.domain_id!r}"
        )
    behavior_targets = dict(domain_profile.get("behavior_targets") or {})
    if args.context_dropout_rate is None:
        args.context_dropout_rate = float(
            behavior_targets.get("context_dropout_rate", CARD_CONTEXT_DROPOUT_RATE)
        )
    if args.context_jitter_rate is None:
        args.context_jitter_rate = float(
            behavior_targets.get("context_jitter_rate", CARD_CONTEXT_JITTER_RATE)
        )

    state_path = run_root / "run_state.json"
    existing_config = _load_json(run_root / "run_config.json")
    existing_max_posts = int(existing_config.get("max_posts") or 0)
    append_extension = bool(
        existing_config and args.extend_existing and args.max_posts > existing_max_posts
    )
    policy_upgrade = bool(
        existing_config
        and args.upgrade_generation_policy
        and str(existing_config.get("generator_policy_version") or "")
        != generator_policy_version
    )
    if args.extend_existing and not existing_config:
        raise RuntimeError("--extend-existing requires an existing run_config.json")
    if args.extend_existing and args.max_posts < existing_max_posts:
        raise RuntimeError(
            f"Append-only extension cannot shrink max_posts: {existing_max_posts}->{args.max_posts}"
        )
    if args.upgrade_generation_policy and not existing_config:
        raise RuntimeError(
            "--upgrade-generation-policy requires an existing run_config.json"
        )
    if existing_config and (args.resume or generated_root.exists()):
        verify_run_policy(
            existing_config,
            operation="extend generation" if append_extension else "resume generation",
            allow_historical=append_extension or policy_upgrade,
        )
    elif generated_root.exists():
        raise RuntimeError(
            "Cannot resume generation: generated output exists without a run policy. "
            "Use a new tag; old comments cannot be relabeled as parity-v3 output."
        )
    run_root.mkdir(parents=True, exist_ok=True)
    state = _load_json(state_path)
    prior_elapsed = float(state.get("elapsed_seconds") or 0.0)

    command = _generator_command(
        args=args,
        config_raw_dir=config.raw_discussions_dir,
        seed_pool=seed_pool,
        generated_root=generated_root,
        behavior_targets=behavior_targets,
    )
    requested_config = {
        "domain": config.to_public_dict(),
        "domain_config": args.domain,
        "tag": args.tag,
        "model": args.model,
        "base_url": args.base_url,
        "seed_pool": str(seed_pool),
        "domain_profile": str(domain_profile_path),
        "domain_profile_sha256": str(domain_profile.get("profile_sha256") or ""),
        "domain_profile_schema_version": int(domain_profile.get("schema_version") or 0),
        "reference_viewpoint_count": int(
            domain_profile.get("source", {}).get("reference_viewpoint_count") or 0
        ),
        "domain_behavior_targets": behavior_targets,
        "distribution_controls": {
            "story_personal_min_share": float(
                behavior_targets.get(
                    "story_personal_min_share",
                    behavior_targets.get("tone_personal_min_share", 0.16),
                )
            ),
            "affect_assignment": (
                "discourse-compatible sampling from evaluation-excluded real thread templates"
            ),
            "lexical_quality": dict(domain_profile.get("lexical_quality") or {}),
            "writer_distribution_controller": {
                "metrics": ["self_bleu_4", "semantic_mean_cosine"],
                "target": "same-size evaluation-excluded real metric template",
                "candidate_policy": "single Writer realization; distribution metrics are diagnostic",
            },
            "length_conditioning": {
                "mode": "anonymous_continuous_matched_scale",
                "word_count_acceptance_gate": False,
                "bucket_specific_token_cap": False,
                "provider_safety_max_tokens": args.writer_max_tokens,
            },
            "reference_metric_calibration": {
                key: value
                for key, value in dict(
                    domain_profile.get("reference_metric_calibration") or {}
                ).items()
                if key != "templates_by_size"
            },
        },
        "generated_root": str(generated_root),
        "pool_size": args.pool_size,
        "max_posts": args.max_posts,
        "posts_per_run": args.posts_per_run,
        "start_seed_index": args.start_seed_index,
        "sampling_seed": args.sampling_seed,
        "context_dropout_rate": args.context_dropout_rate,
        "context_jitter_rate": args.context_jitter_rate,
        "plan_quality": {
            "repair_rounds": args.plan_quality_repairs,
            "missing_slot_policy": "bounded_schema_recovery_then_hard_fail",
            "comment_planner_batch_size": args.comment_planner_batch_size,
            "similarity_threshold": args.plan_similarity_threshold,
            "embedding_enabled": args.plan_embedding_quality,
            "embedding_model": args.plan_embedding_model,
            "embedding_threshold": args.plan_embedding_threshold,
            "embedding_device": args.plan_embedding_device,
            "max_collision_rate": args.plan_max_collision_rate,
            "max_perspective_share": args.max_perspective_share,
            "strict": args.strict_plan_quality,
        },
        "domain_claim": args.domain_claim,
        "writer_prompt": args.writer_prompt,
        "writer_route_lock": args.writer_route_lock,
        "social_contract_coherence": args.social_contract_coherence,
        "reply_sibling_visibility": args.reply_sibling_visibility,
        "own_fact_license": args.own_fact_license,
        "speaker_identity": args.speaker_identity,
        "actor_conditioning": {
            "mode": args.actor_conditioning,
            "source": (
                "visible thread plus evaluation-excluded same-domain references"
                if args.actor_conditioning == MODE_DOMAIN_DERIVED
                else "disabled"
            ),
            "fixed_participant_catalog": False,
            "writer_distribution_resampling": False,
        },
        "post_recovery": {
            "retry_limit": args.post_retry_limit,
            "retry_delay_seconds": args.post_retry_delay,
            "recoverable_action": (
                "retry_same_post"
                if args.post_retry_limit > 1
                else "fail_incomplete_post_without_persistence"
            ),
            "writer_hard_recovery_rounds": args.writer_hard_recovery_rounds,
        },
        "reasoning_effort": args.reasoning_effort,
        "gpt5_reasoning_token_reserve": args.gpt5_reasoning_token_reserve,
        "persona_conditioning": persona_config,
        "generator_profile": args.generator_profile,
        "generator_policy_version": generator_policy_version,
        "revision_core_policy_version": REVISION_CORE_POLICY_VERSION,
        "generator_core_provenance": core_provenance,
        "card_core_algorithm_symbols": list(CORE_ALGORITHM_SYMBOLS),
        "generalized_algorithm_extensions": list(GENERALIZED_ALGORITHM_EXTENSIONS),
        "domain_adaptation_boundaries": list(DOMAIN_ADAPTATION_BOUNDARIES),
        "command": _redact(command),
    }
    if existing_config:
        _preserve_revision_lineage(existing_config, requested_config)
        if append_extension:
            _verify_append_extension(
                existing=existing_config,
                requested=requested_config,
                generated_root=generated_root,
                run_root=run_root,
            )
            requested_config["generation_lineage"] = _extended_generation_lineage(
                existing=existing_config,
                requested=requested_config,
                old_max_posts=existing_max_posts,
            )
        elif policy_upgrade:
            completed_prefix = _verify_policy_upgrade(
                existing=existing_config,
                requested=requested_config,
                generated_root=generated_root,
                run_root=run_root,
            )
            requested_config["generation_lineage"] = _upgraded_generation_lineage(
                existing=existing_config,
                requested=requested_config,
                completed_prefix=completed_prefix,
            )
        else:
            if "generation_lineage" in existing_config:
                requested_config["generation_lineage"] = existing_config[
                    "generation_lineage"
                ]
            _verify_resume_config(existing_config, requested_config)
    _write_json(run_root / "run_config.json", requested_config)
    if append_extension:
        _record_append_extension(
            run_root=run_root,
            generated_root=generated_root,
            existing=existing_config,
            requested=requested_config,
        )
    if policy_upgrade:
        _record_policy_upgrade(
            run_root=run_root,
            existing=existing_config,
            requested=requested_config,
            completed_prefix=completed_prefix,
        )
    print(
        f"[generalized-config] domain={config.domain_id} model={args.model}", flush=True
    )
    print(f"[generalized-config] seed_pool={seed_pool}", flush=True)
    print(
        f"[generalized-config] domain_profile={domain_profile_path} "
        f"reference_threads={domain_profile.get('source', {}).get('reference_thread_count', 0)} "
        f"reference_viewpoints={domain_profile.get('source', {}).get('reference_viewpoint_count', 0)}",
        flush=True,
    )
    print(f"[generalized-config] output={generated_root}", flush=True)
    print(
        f"[generalized-config] seed_range={args.start_seed_index}-"
        f"{args.start_seed_index + args.max_posts - 1}",
        flush=True,
    )
    print(
        f"[generalized-config] generator_profile={args.generator_profile} "
        f"generator_policy={generator_policy_version}",
        flush=True,
    )
    print(
        f"[generalized-config] reasoning_effort={args.reasoning_effort or 'default'} "
        f"gpt5_reasoning_token_reserve={args.gpt5_reasoning_token_reserve}",
        flush=True,
    )
    print(
        f"[generalized-config] context_dropout_rate={args.context_dropout_rate} "
        f"context_jitter_rate={args.context_jitter_rate}",
        flush=True,
    )
    print(
        f"[generalized-config] plan_quality_repairs={args.plan_quality_repairs} "
        f"comment_planner_batch_size={args.comment_planner_batch_size} "
        f"similarity_threshold={args.plan_similarity_threshold} "
        f"embedding={int(args.plan_embedding_quality)} "
        f"embedding_threshold={args.plan_embedding_threshold} "
        f"max_collision_rate={args.plan_max_collision_rate} "
        f"max_perspective_share={args.max_perspective_share} "
        f"strict={int(args.strict_plan_quality)}",
        flush=True,
    )
    print(
        f"[generalized-config] post_retry_limit={args.post_retry_limit} "
        f"post_retry_delay={args.post_retry_delay}",
        flush=True,
    )
    print(
        f"[generalized-config] writer_hard_recovery_rounds="
        f"{args.writer_hard_recovery_rounds}",
        flush=True,
    )
    print(
        f"[generalized-config] actor_conditioning={args.actor_conditioning} "
        "fixed_participant_catalog=0 writer_distribution_resampling=0",
        flush=True,
    )
    print(
        f"[generalized-config] persona_conditioning={persona_runtime.mode} "
        f"eligible_personas={persona_config.get('eligible_personas', 0)} "
        f"matraix_commit={persona_config.get('matraix_commit', 'disabled')}",
        flush=True,
    )
    print(f"[generalized-command] {' '.join(_redact(command))}", flush=True)
    if args.prepare_only:
        print("[prepare-only] no API calls were made", flush=True)
        return

    api_key = os.environ.get(args.api_key_env, "").strip()
    if not api_key:
        # Name the keys that are actually set. The credential check runs after the
        # whole preflight, so an unhelpful message here costs a full setup pass.
        available = sorted(
            name
            for name in os.environ
            if name.endswith("_API_KEY") and os.environ[name].strip()
        )
        hint = (
            f" Keys present in the environment: {', '.join(available)}."
            f" Pass --api-key-env with one of them."
            if available
            else " No *_API_KEY variable is set; check .env or export one."
        )
        raise SystemExit(
            f"API key is missing: environment variable {args.api_key_env}.{hint}"
        )
    env = os.environ.copy()
    env["GENERALIZED_CARD_DOMAIN"] = args.domain
    env["GENERALIZED_CARD_DOMAIN_PROFILE"] = str(domain_profile_path)
    env["GENERALIZED_CARD_GENERATOR_PROFILE"] = args.generator_profile
    env["GENERALIZED_CARD_ACTOR_CONDITIONING"] = args.actor_conditioning
    env["GENERALIZED_CARD_DOMAIN_CLAIM"] = args.domain_claim
    env["GENERALIZED_CARD_WRITER_PROMPT"] = args.writer_prompt
    env["GENERALIZED_CARD_WRITER_ROUTE_LOCK"] = args.writer_route_lock
    env["GENERALIZED_CARD_SOCIAL_CONTRACT_COHERENCE"] = args.social_contract_coherence
    env["GENERALIZED_CARD_REPLY_SIBLING_VISIBILITY"] = args.reply_sibling_visibility
    env["GENERALIZED_CARD_OWN_FACT_LICENSE"] = args.own_fact_license
    env["GENERALIZED_CARD_SPEAKER_IDENTITY"] = args.speaker_identity
    env["GENERALIZED_CARD_STORY_PERSONAL_MIN_SHARE"] = str(
        behavior_targets.get(
            "story_personal_min_share",
            behavior_targets.get("tone_personal_min_share", 0.16),
        )
    )
    env["GENERALIZED_CARD_PERSONA_MODE"] = persona_runtime.mode
    env["GENERALIZED_CARD_MATRAIX_ROOT"] = str(matraix_root)
    env["GENERALIZED_CARD_PERSONA_DATASET"] = str(matraix_dataset)
    env["GENERALIZED_CARD_PERSONA_SEED"] = str(args.persona_seed)
    env["GENERALIZED_CARD_PERSONA_EXPERTISE_DIMENSIONS"] = ",".join(
        config.persona_expertise_dimensions
    )
    env["GENERALIZED_CARD_PLAN_REPAIRS"] = str(args.plan_quality_repairs)
    env["GENERALIZED_CARD_PLAN_SIMILARITY_THRESHOLD"] = str(
        args.plan_similarity_threshold
    )
    env["GENERALIZED_CARD_PLAN_EMBEDDING_ENABLED"] = (
        "1" if args.plan_embedding_quality else "0"
    )
    env["GENERALIZED_CARD_PLAN_EMBEDDING_MODEL"] = args.plan_embedding_model
    env["GENERALIZED_CARD_PLAN_EMBEDDING_THRESHOLD"] = str(
        args.plan_embedding_threshold
    )
    env["GENERALIZED_CARD_PLAN_EMBEDDING_DEVICE"] = args.plan_embedding_device
    env["GENERALIZED_CARD_PLAN_MAX_COLLISION_RATE"] = str(args.plan_max_collision_rate)
    env["GENERALIZED_CARD_MAX_PERSPECTIVE_SHARE"] = str(args.max_perspective_share)
    env["GENERALIZED_CARD_STRICT_PLAN_QUALITY"] = (
        "1" if args.strict_plan_quality else "0"
    )
    env["GENERALIZED_CARD_WRITER_HARD_RECOVERY_ROUNDS"] = str(
        args.writer_hard_recovery_rounds
    )
    env["GENERALIZED_CARD_PLAN_AUDIT_JSONL"] = str(
        run_root / "logs" / "planning_quality.jsonl"
    )
    env["GENERALIZED_CARD_DISTRIBUTION_AUDIT_JSONL"] = str(
        run_root / "logs" / "story_affect_distribution.jsonl"
    )
    env["GENERALIZED_CARD_WRITER_DIVERSITY_AUDIT_JSONL"] = str(
        run_root / "logs" / "writer_distribution_control.jsonl"
    )
    env["OPENAI_API_KEY"] = api_key
    env["PLANNER_API_KEY"] = api_key
    env["WRITER_API_KEY"] = api_key
    env["LLM_API_KEY"] = api_key
    env["TOKEN_USAGE_LOG_JSONL"] = str(run_root / "logs" / "token_usage.jsonl")
    env["TOKEN_USAGE_RUN_TAG"] = args.tag
    env["LLM_API_RETRIES"] = str(args.api_retries)
    env["LLM_API_RETRY_DELAY"] = str(args.retry_delay)
    env["LLM_CALL_SLEEP_SECONDS"] = str(args.call_sleep_seconds)
    if args.reasoning_effort:
        env["REASONING_EFFORT"] = args.reasoning_effort
    env["GPT5_REASONING_TOKEN_RESERVE"] = str(max(0, args.gpt5_reasoning_token_reserve))
    _set_prices(env, args)

    self_test_command = [
        sys.executable,
        str(PACKAGE_ROOT / "scripts" / "run_generator_backend.py"),
        "--self-test",
    ]
    print(f"[generalized-preflight] {' '.join(self_test_command)}", flush=True)
    subprocess.run(self_test_command, cwd=REPO_ROOT, env=env, check=True)

    started = time.monotonic()
    status = "failed"
    return_code = 1
    annotation_error: Exception | None = None
    try:
        completed = subprocess.run(command, cwd=REPO_ROOT, env=env, check=False)
        return_code = int(completed.returncode)
        status = "complete" if return_code == 0 else "failed"
    except KeyboardInterrupt:
        status = "interrupted"
        return_code = 130
        print("[interrupted] completed post slots remain resumable", flush=True)
    finally:
        try:
            persona_manifest = annotate_generated_outputs(
                generated_root, persona_runtime
            )
            if persona_runtime.enabled:
                print(
                    f"[persona-manifest] comments={persona_manifest.get('comments', 0)} "
                    f"unique_personas={persona_manifest.get('unique_personas_used', 0)} "
                    f"path={run_root / 'persona_assignment_manifest.json'}",
                    flush=True,
                )
        except Exception as exc:  # noqa: BLE001
            annotation_error = exc
            status = "failed"
            return_code = 1
            print(f"[persona-manifest-error] {type(exc).__name__}: {exc}", flush=True)
        elapsed = prior_elapsed + (time.monotonic() - started)
        state = {
            "status": status,
            "return_code": return_code,
            "elapsed_seconds": elapsed,
            "updated_at_epoch": time.time(),
            "generated_root": str(generated_root),
            "token_log": str(run_root / "logs" / "token_usage.jsonl"),
        }
        _write_json(state_path, state)
        _summarize_usage(run_root, elapsed, env)
    if annotation_error is not None:
        raise annotation_error
    if return_code:
        raise SystemExit(return_code)


def _generator_command(
    *,
    args: argparse.Namespace,
    config_raw_dir: Path,
    seed_pool: Path,
    generated_root: Path,
    behavior_targets: dict[str, Any] | None = None,
) -> list[str]:
    runs = math.ceil(args.max_posts / args.posts_per_run)
    targets = behavior_targets or {}

    def value(key: str, default: float) -> str:
        return str(targets.get(key, default))

    context_dropout = (
        args.context_dropout_rate
        if args.context_dropout_rate is not None
        else targets.get("context_dropout_rate", CARD_CONTEXT_DROPOUT_RATE)
    )
    context_jitter = (
        args.context_jitter_rate
        if args.context_jitter_rate is not None
        else targets.get("context_jitter_rate", CARD_CONTEXT_JITTER_RATE)
    )
    return [
        sys.executable,
        str(PACKAGE_ROOT / "scripts" / "run_generator_backend.py"),
        "--seed-post-pool-json",
        str(seed_pool),
        "--real-comments-dir",
        str(config_raw_dir),
        "--output-dir",
        str(generated_root),
        "--runs",
        str(runs),
        "--posts-per-run",
        str(args.posts_per_run),
        "--max-total-posts",
        str(args.max_posts),
        "--start-seed-index",
        str(args.start_seed_index),
        "--seed",
        str(args.sampling_seed),
        "--max-comments-per-post",
        str(args.max_comments_per_post),
        "--comment-count-scale",
        str(args.comment_count_scale),
        "--exact-matched-thread-size"
        if args.exact_matched_thread_size
        else "--no-exact-matched-thread-size",
        "--planner-model",
        args.model,
        "--planner-base-url",
        args.base_url,
        "--planner-retries",
        str(args.api_retries),
        "--planner-max-tokens",
        str(args.planner_max_tokens),
        "--planner-timeout",
        "900",
        "--comment-planner-max-tokens",
        str(args.comment_planner_max_tokens),
        "--comment-planner-batch-size",
        str(args.comment_planner_batch_size),
        "--writer-model",
        args.model,
        "--writer-base-url",
        args.base_url,
        "--writer-timeout",
        "900",
        "--writer-profile",
        "gpt54_reddit_writer",
        "--writer-max-tokens",
        str(args.writer_max_tokens),
        "--writer-retries",
        str(args.writer_retries),
        "--post-retry-limit",
        str(args.post_retry_limit),
        "--post-retry-delay",
        str(args.post_retry_delay),
        "--matched-real-comments",
        str(args.matched_real_comments),
        "--claim-key-budget",
        "1",
        "--claim-family-max-share",
        "0.18",
        "--claim-family-min-budget",
        "3",
        "--opening-reuse-budget",
        "1",
        "--opener-family-reuse-budget",
        "5",
        "--template-phrase-reuse-budget",
        "4",
        "--advisor-max-share",
        value("advisor_max_share", 0.28),
        "--question-max-share",
        value("question_max_share", 0.18),
        "--micro-target-share",
        value("micro_target_share", 0.07),
        "--short-max-share",
        value("short_max_share", 0.18),
        "--social-noise-min-share",
        value("social_noise_min_share", 0.18),
        "--gratitude-min-share",
        value("gratitude_min_share", 0.12),
        "--tone-harsh-max-share",
        value("tone_harsh_max_share", 0.14),
        "--tone-calm-min-share",
        value("tone_calm_min_share", 0.30),
        "--tone-personal-min-share",
        value(
            "tone_personal_min_share",
            0.18,
        ),
        "--tone-polite-min-share",
        value("tone_polite_min_share", 0.10),
        "--context-dropout-rate",
        str(context_dropout),
        "--context-jitter-rate",
        str(context_jitter),
    ]


def _set_prices(env: dict[str, str], args: argparse.Namespace) -> None:
    defaults = DEFAULT_PRICES.get(args.model.lower())
    input_price = (
        args.price_input_per_1m
        if args.price_input_per_1m is not None
        else defaults[0]
        if defaults
        else None
    )
    cached_price = (
        args.price_cached_input_per_1m
        if args.price_cached_input_per_1m is not None
        else defaults[1]
        if defaults
        else None
    )
    output_price = (
        args.price_output_per_1m
        if args.price_output_per_1m is not None
        else defaults[2]
        if defaults
        else None
    )
    if input_price is not None:
        env["TOKEN_PRICE_INPUT_PER_1M"] = str(input_price)
    if cached_price is not None:
        env["TOKEN_PRICE_CACHED_INPUT_PER_1M"] = str(cached_price)
    if output_price is not None:
        env["TOKEN_PRICE_OUTPUT_PER_1M"] = str(output_price)


def _summarize_usage(run_root: Path, elapsed: float, env: dict[str, str]) -> None:
    token_log = run_root / "logs" / "token_usage.jsonl"
    summary = run_root / "logs" / "token_usage_summary.json"
    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "summarize_token_usage.py"),
            str(token_log),
            "--output",
            str(summary),
            "--elapsed-seconds",
            str(elapsed),
        ],
        cwd=REPO_ROOT,
        env=env,
        check=False,
    )


def _redact(command: list[str]) -> list[str]:
    output = list(command)
    for index, token in enumerate(output[:-1]):
        if token in {"--planner-api-key", "--writer-api-key", "--api-key"}:
            output[index + 1] = "[REDACTED]"
    return output


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


RUN_EXPERIMENT_FIELDS = (
    "domain",
    "domain_config",
    "tag",
    "model",
    "base_url",
    "seed_pool",
    "domain_profile",
    "domain_profile_sha256",
    "domain_profile_schema_version",
    "reference_viewpoint_count",
    "domain_behavior_targets",
    "distribution_controls",
    "generated_root",
    "pool_size",
    "posts_per_run",
    "start_seed_index",
    "sampling_seed",
    "context_dropout_rate",
    "context_jitter_rate",
    "plan_quality",
    "domain_claim",
    "writer_prompt",
    "writer_route_lock",
    "social_contract_coherence",
    "reply_sibling_visibility",
    "own_fact_license",
    "speaker_identity",
    "actor_conditioning",
    "reasoning_effort",
    "gpt5_reasoning_token_reserve",
    "persona_conditioning",
    "generator_profile",
    "revision_core_policy_version",
    "card_core_algorithm_symbols",
    "generalized_algorithm_extensions",
    "domain_adaptation_boundaries",
)


def _verify_resume_config(
    existing: dict[str, Any],
    requested: dict[str, Any],
) -> None:
    immutable = RUN_EXPERIMENT_FIELDS + (
        "max_posts",
        "post_recovery",
        "generator_policy_version",
        "generator_core_provenance",
        "generation_lineage",
        "command",
    )
    changed = [key for key in immutable if existing.get(key) != requested.get(key)]
    if changed:
        raise RuntimeError(
            "Cannot resume generation with changed configuration fields: "
            + ", ".join(changed)
            + ". Use the original command or choose a new tag."
        )


def _preserve_revision_lineage(
    existing: dict[str, Any],
    requested: dict[str, Any],
) -> None:
    """Keep reviser lineage stable while resuming an unchanged generator run."""

    if existing.get("revision_core_policy_version"):
        requested["revision_core_policy_version"] = existing[
            "revision_core_policy_version"
        ]
    if existing.get("revision_policy_history"):
        requested["revision_policy_history"] = existing["revision_policy_history"]


def _verify_append_extension(
    *,
    existing: dict[str, Any],
    requested: dict[str, Any],
    generated_root: Path,
    run_root: Path,
) -> None:
    old_max = int(existing.get("max_posts") or 0)
    new_max = int(requested.get("max_posts") or 0)
    if old_max <= 0 or new_max <= old_max:
        raise RuntimeError(f"Extension must increase max_posts: {old_max}->{new_max}")

    stable_fields = RUN_EXPERIMENT_FIELDS + ("post_recovery",)
    changed = [key for key in stable_fields if existing.get(key) != requested.get(key)]
    if changed:
        raise RuntimeError(
            "Cannot extend generation with changed configuration fields: "
            + ", ".join(changed)
        )
    if _size_neutral_command(existing.get("command")) != _size_neutral_command(
        requested.get("command")
    ):
        raise RuntimeError(
            "Cannot extend generation: backend command changed beyond --runs/--max-total-posts"
        )

    seed_indices = _generated_seed_indices(generated_root)
    expected = set(range(old_max))
    if seed_indices != expected:
        missing = sorted(expected - seed_indices)
        unexpected = sorted(seed_indices - expected)
        raise RuntimeError(
            "Existing generated prefix is not complete and contiguous: "
            f"expected=0..{old_max - 1} missing={missing[:10]} unexpected={unexpected[:10]}"
        )

    history_path = run_root / "full_revision_history.json"
    if history_path.exists():
        raise RuntimeError("Cannot extend a run after self-loop revision has started")
    artifact = _load_json(run_root / "current_artifact.json")
    if artifact and artifact.get("stage") != "initial_evaluation":
        raise RuntimeError(
            f"Cannot extend from revision artifact stage={artifact.get('stage')!r}"
        )


def _size_neutral_command(value: object) -> list[str]:
    command = [str(token) for token in value] if isinstance(value, list) else []
    normalized: list[str] = []
    skip_value = False
    for token in command:
        if skip_value:
            skip_value = False
            continue
        if token in {"--runs", "--max-total-posts"}:
            skip_value = True
            continue
        normalized.append(token)
    return normalized


def _policy_neutral_command(value: object) -> list[str]:
    command = [str(token) for token in value] if isinstance(value, list) else []
    normalized: list[str] = []
    skip_value = False
    for token in command:
        if skip_value:
            skip_value = False
            continue
        if token in {"--post-retry-limit", "--post-retry-delay"}:
            skip_value = True
            continue
        normalized.append(token)
    return normalized


def _verify_policy_upgrade(
    *,
    existing: dict[str, Any],
    requested: dict[str, Any],
    generated_root: Path,
    run_root: Path,
) -> int:
    """Validate an explicit, append-only code-policy transition."""

    stable_fields = RUN_EXPERIMENT_FIELDS + ("max_posts",)
    changed = [key for key in stable_fields if existing.get(key) != requested.get(key)]
    if changed:
        raise RuntimeError(
            "Cannot upgrade generation policy with changed experiment fields: "
            + ", ".join(changed)
        )
    if _policy_neutral_command(existing.get("command")) != _policy_neutral_command(
        requested.get("command")
    ):
        raise RuntimeError(
            "Cannot upgrade generation policy: backend command changed beyond "
            "the audited post-recovery controls"
        )
    indices = _generated_seed_indices(generated_root)
    completed_prefix = len(indices)
    if indices != set(range(completed_prefix)):
        raise RuntimeError(
            "Cannot upgrade generation policy: existing seeds are not a contiguous prefix"
        )
    if completed_prefix >= int(requested.get("max_posts") or 0):
        raise RuntimeError(
            "Generation is already complete; no policy upgrade is needed"
        )
    if (run_root / "full_revision_history.json").exists():
        raise RuntimeError(
            "Cannot upgrade generation policy after self-loop revision started"
        )
    return completed_prefix


def _upgraded_generation_lineage(
    *,
    existing: dict[str, Any],
    requested: dict[str, Any],
    completed_prefix: int,
) -> dict[str, Any]:
    prior = existing.get("generation_lineage")
    if isinstance(prior, dict) and isinstance(prior.get("segments"), list):
        source_segments = list(prior["segments"])
    else:
        source_segments = [
            {
                "seed_start": 0,
                "seed_end_exclusive": completed_prefix,
                "generator_policy_version": existing.get("generator_policy_version"),
                "generator_core_provenance": existing.get("generator_core_provenance"),
            }
        ]
    segments: list[dict[str, Any]] = []
    for raw in source_segments:
        if not isinstance(raw, dict):
            continue
        start = max(0, int(raw.get("seed_start") or 0))
        end = min(completed_prefix, int(raw.get("seed_end_exclusive") or 0))
        if start >= end:
            continue
        segment = dict(raw)
        segment["seed_start"] = start
        segment["seed_end_exclusive"] = end
        segments.append(segment)
    segments.append(
        {
            "seed_start": completed_prefix,
            "seed_end_exclusive": int(requested["max_posts"]),
            "generator_policy_version": requested.get("generator_policy_version"),
            "generator_core_provenance": requested.get("generator_core_provenance"),
        }
    )
    return {"mode": "append_only_policy_transition", "segments": segments}


def _record_policy_upgrade(
    *,
    run_root: Path,
    existing: dict[str, Any],
    requested: dict[str, Any],
    completed_prefix: int,
) -> None:
    event = {
        "seed_boundary": completed_prefix,
        "generator_policy_before": existing.get("generator_policy_version"),
        "generator_policy_after": requested.get("generator_policy_version"),
        "provenance_before": existing.get("generator_core_provenance"),
        "provenance_after": requested.get("generator_core_provenance"),
        "recorded_at_epoch": time.time(),
    }
    path = run_root / "generation_policy_upgrade_history.json"
    history = _load_json(path)
    events = list(history.get("upgrades") or [])
    if not any(
        int(item.get("seed_boundary") or -1) == completed_prefix
        and item.get("generator_policy_after")
        == requested.get("generator_policy_version")
        for item in events
        if isinstance(item, dict)
    ):
        events.append(event)
    _write_json(path, {"upgrades": events})
    _write_json(
        run_root / "evaluation_invalidated.json",
        {
            "reason": "generation_policy_upgraded_before_completion",
            "seed_boundary": completed_prefix,
            "invalidated_at_epoch": time.time(),
        },
    )
    print(
        f"[generation-policy-upgrade] preserved seeds=0-{completed_prefix - 1}; "
        f"new_policy_starts_at_seed={completed_prefix}",
        flush=True,
    )


def _resolve_repo_path(path: Path) -> Path:
    expanded = path.expanduser()
    return (
        expanded.resolve()
        if expanded.is_absolute()
        else (REPO_ROOT / expanded).resolve()
    )


def _generated_seed_indices(generated_root: Path) -> set[int]:
    indices: list[int] = []
    for path in sorted(generated_root.glob("run_*_sampled_reddit/discussion.json")):
        payload = _load_json(path)
        for post in payload.get("posts") or []:
            try:
                indices.append(int(post["seed_index"]))
            except (KeyError, TypeError, ValueError):
                raise RuntimeError(
                    f"Generated post lacks a valid seed_index: {path}"
                ) from None
    if len(indices) != len(set(indices)):
        raise RuntimeError(
            "Existing generated prefix contains duplicate seed_index values"
        )
    return set(indices)


def _extended_generation_lineage(
    *,
    existing: dict[str, Any],
    requested: dict[str, Any],
    old_max_posts: int,
) -> dict[str, Any]:
    prior = existing.get("generation_lineage")
    if isinstance(prior, dict) and isinstance(prior.get("segments"), list):
        segments = list(prior["segments"])
    else:
        segments = [
            {
                "seed_start": 0,
                "seed_end_exclusive": old_max_posts,
                "generator_policy_version": existing.get("generator_policy_version"),
                "generator_core_provenance": existing.get("generator_core_provenance"),
            }
        ]
    segments.append(
        {
            "seed_start": old_max_posts,
            "seed_end_exclusive": int(requested["max_posts"]),
            "generator_policy_version": requested.get("generator_policy_version"),
            "generator_core_provenance": requested.get("generator_core_provenance"),
        }
    )
    return {"mode": "append_only", "segments": segments}


def _record_append_extension(
    *,
    run_root: Path,
    generated_root: Path,
    existing: dict[str, Any],
    requested: dict[str, Any],
) -> None:
    old_max = int(existing["max_posts"])
    new_max = int(requested["max_posts"])
    event = {
        "old_max_posts": old_max,
        "new_max_posts": new_max,
        "generator_policy_before": existing.get("generator_policy_version"),
        "generator_policy_after": requested.get("generator_policy_version"),
        "recorded_at_epoch": time.time(),
    }
    history_path = run_root / "generation_extension_history.json"
    history = _load_json(history_path)
    events = list(history.get("extensions") or [])
    if not any(
        int(item.get("old_max_posts") or -1) == old_max
        and int(item.get("new_max_posts") or -1) == new_max
        for item in events
        if isinstance(item, dict)
    ):
        events.append(event)
    _write_json(history_path, {"extensions": events})
    extension_dir = generated_root / "_reproducibility_extensions"
    _write_json(extension_dir / f"seeds_{old_max:03d}_{new_max - 1:03d}.json", event)

    current_artifact = run_root / "current_artifact.json"
    current_artifact.unlink(missing_ok=True)
    _write_json(
        run_root / "evaluation_invalidated.json",
        {
            "reason": "generation_extended",
            "old_max_posts": old_max,
            "new_max_posts": new_max,
            "invalidated_at_epoch": time.time(),
        },
    )
    print(
        f"[generation-extension] verified complete seeds=0-{old_max - 1}; "
        f"appending seeds={old_max}-{new_max - 1}",
        flush=True,
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


if __name__ == "__main__":
    main()
