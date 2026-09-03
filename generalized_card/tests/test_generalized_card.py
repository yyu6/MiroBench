from __future__ import annotations

# The test imports a historical script after adding the repository script
# directory to sys.path; all package imports below must therefore follow it.
# ruff: noqa: E402

import importlib.util
import csv
import json
import os
import random
import re
import sys
import tempfile
import unittest
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from revision_memory import (  # noqa: E402
    build_memory,
    choose_history_aware_strategy,
    merge_strategy_history,
    render_prompt_feedback,
    restore_controller_state,
    summarize_reviser_output,
)


from generalized_card import legacy_reviser_prompts, prompts
from generalized_card.actor_conditioning import (
    MODE_DOMAIN_DERIVED,
    actor_state_from_plan,
    enrich_normalized_plans,
)
from generalized_card.audit import (
    _closest_real_overlap,
    _planner_rows_for_audit,
    audit_generated_root,
)
from generalized_card.backend import (
    CARD_SNAPSHOT_PROFILE,
    CORE_ALGORITHM_SYMBOLS,
    GENERALIZED_V2_PROFILE,
    _canonicalize_plan_controls,
    _annotate_plan_metadata,
    _comment_planner_batch_with_history,
    _completion_kwargs,
    _endpoint_preflight_with_retry,
    _finalize_post_generation,
    _is_output_limit_error,
    _next_completion_boost,
    _sample_repair_budget,
    _substantive_safe_degraded_task,
    _writer_lifecycle_with_candidate_recovery,
    configure_generator_backend,
    load_generator_backend,
)
from generalized_card.data import (
    build_seed_pool,
    find_matched_real_thread,
    load_real_thread_bank,
)
from generalized_card.core_contract import (
    CORE_POLICY_VERSION,
    GENERALIZED_V2_GENERATION_POLICY_VERSION,
    REVISION_CORE_POLICY_VERSION,
    upgrade_revision_policy_config,
    verify_core_contract,
    verify_revision_policy,
    verify_run_policy,
)
from generalized_card.domain import load_domain_config
from generalized_card.domain import DomainConfig
from generalized_card.domain_profile import (
    CARD_CONTEXT_DROPOUT_RATE,
    CARD_CONTEXT_JITTER_RATE,
    PROFILE_SCHEMA_VERSION,
    build_domain_profile,
    load_domain_profile,
)
from generalized_card.generation_diversity import (
    build_thread_distribution_target,
    distribution_target_with_slot_progress,
)
from generalized_card.generation_distribution import (
    apply_planner_distribution_fields,
    enrich_distribution_plan_fields,
)
from generalized_card.first_pass_policy import retain_explicitly_planned_tasks
from generalized_card.length_policy import (
    local_move_scope_guidance,
    soft_length_guidance,
    writer_provider_token_budget,
)
from generalized_card.length_calibration import (
    calibrated_word_ask,
    set_length_calibration,
)
from generalized_card.long_form_planning import enrich_development_plan_fields
from generalized_card.lexical_quality import (
    candidate_thread_bleu_diagnostics,
    lexical_overlap_problem as calibrated_lexical_overlap_problem,
)
from generalized_card.viewpoint_bank import (
    render_reference_viewpoints,
    retrieve_reference_viewpoints,
)
from generalized_card.persona_bridge import (
    AUDITED_MATRAIX_COMMIT,
    MODE_FULL,
    MODE_PROJECTED,
    annotate_generated_outputs,
    build_runtime,
    inject_persona_system,
    persona_marker_for_task,
    reset_runtime_cache,
)
from generalized_card.planner_schema import parse_sample_id
from generalized_card.planning_quality import (
    evaluate_plan_batch,
    social_contract_problem,
    universal_viewpoints,
)
from generalized_card.surface_contract import (
    infer_surface_shape,
    infer_surface_skeleton,
    infer_surface_texture,
    reconcile_substantive_task,
    surface_only_label,
)
from generalized_card.reviser_backend import (
    configure_reviser_backend,
    load_reviser_backend,
    parse_candidate_response,
    parse_selfbert_candidate_response,
)
from generalized_card.text_metric_reviser import (
    candidate_rank,
    validate_candidate as validate_text_metric_candidate,
)
from generalized_card.writer_quality import (
    parse_distribution_problems,
    planned_quote_has_distinct_reply,
    substantive_length_floor_problem,
    writer_hard_recovery_task,
)


def load_script_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load script module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class GeneralizedCardTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_domain_config("camera")

    def test_domain_alias_and_seed_pool_exact_matches(self) -> None:
        self.assertEqual(
            load_domain_config("camera_product").domain_id, "camera_product"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "seeds.json"
            payload = build_seed_pool(self.config, path, count=8, seed=7)
            rows = payload["seed_posts"]
            self.assertEqual(len(rows), 8)
            self.assertEqual(len({row["source_raw_post_id"] for row in rows}), 8)
            bank = load_real_thread_bank(self.config.raw_discussions_dir)

            class Seed:
                pass

            for row in rows:
                seed = Seed()
                seed.source_raw_post_id = row["source_raw_post_id"]
                seed.metadata = row
                matched = find_matched_real_thread(bank, seed)
                self.assertIsNotNone(matched)
                self.assertEqual(matched["post_id"], row["source_raw_post_id"])

    def test_matraix_official_renderer_and_deterministic_assignment(self) -> None:
        root = REPO_ROOT / "third_party" / "MatrAIx-Persona-8B"
        if not root.is_dir():
            self.skipTest("MatrAIx repository is not cloned")
        dataset = root / "persona" / "datasets" / "matraix-persona-dev-sample"
        task = SimpleNamespace(
            local_task_id=7,
            comment_id=10007,
            speaker_role="jokester",
            voice="sarcastic",
            tone_shape="light_joke",
        )
        projected = build_runtime(
            mode=MODE_PROJECTED,
            matraix_root=root,
            dataset_dir=dataset,
            assignment_seed=42,
            expertise_dimensions=self.config.persona_expertise_dimensions,
        )
        first = projected.assign(seed_index=3, task=task)
        second = projected.assign(seed_index=3, task=task)
        self.assertIsNotNone(first)
        self.assertEqual(first, second)
        self.assertEqual(projected.commit, AUDITED_MATRAIX_COMMIT)
        self.assertEqual(projected.public_config()["dataset_personas"], 200)
        # 147 records pass the English/adult filter. 14 render a system prompt
        # longer than the Writer's own task (dilution) and 10 carry no
        # behavioural dimension at all (an empty identity); both are dropped.
        self.assertEqual(projected.public_config()["eligible_personas"], 123)
        self.assertEqual(
            projected.public_config()["template_source"],
            "official-matraix-persona-system",
        )
        self.assertIn("## Who you are", first.system_prompt)
        self.assertLess(len(first.system_prompt), 5000)

        full = build_runtime(
            mode=MODE_FULL,
            matraix_root=root,
            dataset_dir=dataset,
            assignment_seed=42,
            expertise_dimensions=self.config.persona_expertise_dimensions,
        )
        full_assignment = full.assign(seed_index=3, task=task)
        self.assertEqual(first.persona_id, full_assignment.persona_id)
        self.assertGreater(len(full_assignment.system_prompt), len(first.system_prompt))

    def test_matraix_identity_is_a_system_message_and_marker_is_removed(self) -> None:
        root = REPO_ROOT / "third_party" / "MatrAIx-Persona-8B"
        if not root.is_dir():
            self.skipTest("MatrAIx repository is not cloned")
        dataset = root / "persona" / "datasets" / "matraix-persona-dev-sample"
        env = {
            "GENERALIZED_CARD_PERSONA_MODE": MODE_PROJECTED,
            "GENERALIZED_CARD_MATRAIX_ROOT": str(root),
            "GENERALIZED_CARD_PERSONA_DATASET": str(dataset),
            "GENERALIZED_CARD_PERSONA_SEED": "42",
            "GENERALIZED_CARD_PERSONA_EXPERTISE_DIMENSIONS": (
                "fam_photography,ind_consumer_electronics"
            ),
        }
        task = SimpleNamespace(
            local_task_id=5,
            comment_id=5,
            speaker_role="confused_asker",
            voice="uncertain",
            tone_shape="plain_question",
        )
        with patch.dict(os.environ, env, clear=False):
            reset_runtime_cache()
            marker = persona_marker_for_task(SimpleNamespace(index=2), task)
            messages = [
                {"role": "system", "content": "CARD writer contract"},
                {"role": "user", "content": f"{marker}\nWrite one comment."},
            ]
            revised = inject_persona_system(messages)
            self.assertIn("## Who you are", revised[0]["content"])
            self.assertIn("CARD writer contract", revised[0]["content"])
            self.assertNotIn("generalized-card-matraix", revised[1]["content"])
            self.assertEqual(revised[1]["content"], "Write one comment.")
        reset_runtime_cache()

    def test_matraix_annotation_writes_per_comment_provenance(self) -> None:
        root = REPO_ROOT / "third_party" / "MatrAIx-Persona-8B"
        if not root.is_dir():
            self.skipTest("MatrAIx repository is not cloned")
        runtime = build_runtime(
            mode=MODE_PROJECTED,
            matraix_root=root,
            dataset_dir=root / "persona" / "datasets" / "matraix-persona-dev-sample",
            assignment_seed=42,
            expertise_dimensions=self.config.persona_expertise_dimensions,
        )
        with tempfile.TemporaryDirectory() as directory:
            generated = Path(directory) / "generated"
            run = generated / "run_00_sampled_reddit"
            run.mkdir(parents=True)
            discussion = {
                "posts": [
                    {
                        "seed_index": 0,
                        "comments": [
                            {
                                "comment_id": 10001,
                                "speaker_role": "advisor",
                                "voice": "casual_neutral",
                                "tone_shape": "plain_claim",
                                "content": "Check whether the lens mount fits.",
                                "replies": [],
                            }
                        ],
                    }
                ]
            }
            path = run / "discussion.json"
            path.write_text(json.dumps(discussion), encoding="utf-8")
            manifest = annotate_generated_outputs(generated, runtime)
            revised = json.loads(path.read_text(encoding="utf-8"))
            meta = revised["posts"][0]["comments"][0]["persona_conditioning"]
            self.assertEqual(meta["provider"], "MatrAIx-Persona-8B")
            self.assertEqual(meta["matraix_commit"], AUDITED_MATRAIX_COMMIT)
            self.assertEqual(manifest["comments"], 1)
            self.assertEqual(manifest["unique_personas_used"], 1)
            self.assertTrue(
                (generated.parent / "persona_assignment_manifest.json").is_file()
            )

    def test_shared_card_core_hashes_are_pinned(self) -> None:
        self.assertEqual(
            CORE_POLICY_VERSION,
            "card-paper-v37-domain-neutral-profile-v4-20260807",
        )
        provenance = verify_core_contract(
            (
                "generator",
                "generator_generalized_v2",
                "selfbleu_controller",
                "generalized_selfbleu_controller",
                "selfbleu_reviser",
                "tone_controller",
                "tone_reviser",
            )
        )
        for row in provenance.values():
            self.assertEqual(row["actual_sha256"], row["expected_sha256"])

    def test_old_or_missing_run_policy_cannot_be_relabelled(self) -> None:
        self.assertEqual(
            verify_run_policy(
                {"card_core_policy_version": CORE_POLICY_VERSION},
                operation="test",
            ),
            CORE_POLICY_VERSION,
        )
        for config in ({}, {"card_core_policy_version": "history-v2"}):
            with self.assertRaisesRegex(RuntimeError, "cannot be relabeled"):
                verify_run_policy(config, operation="test")
        with patch.dict(
            os.environ,
            {"GENERALIZED_CARD_ALLOW_LINEAGE_MISMATCH": "1"},
        ):
            self.assertEqual(
                verify_run_policy(
                    {"card_core_policy_version": "history-v2"},
                    operation="test",
                ),
                "history-v2",
            )
        self.assertEqual(
            verify_run_policy(
                {
                    "generator_profile": GENERALIZED_V2_PROFILE,
                    "generator_policy_version": GENERALIZED_V2_GENERATION_POLICY_VERSION,
                },
                operation="test",
            ),
            GENERALIZED_V2_GENERATION_POLICY_VERSION,
        )
        historical = {
            "generator_profile": GENERALIZED_V2_PROFILE,
            "generator_policy_version": "generalized-card-v2-frozen-domain-profile-v2-20260807",
        }
        with self.assertRaisesRegex(RuntimeError, "cannot be relabeled"):
            verify_run_policy(historical, operation="resume generation")
        self.assertEqual(
            verify_run_policy(
                historical,
                operation="evaluate generation",
                allow_historical=True,
            ),
            historical["generator_policy_version"],
        )

    def test_revision_workspace_uses_current_reviser_without_relabeling_generator(
        self,
    ) -> None:
        config = {
            "source_generation_policy_version": "unversioned-legacy-generalized-generator",
            "revision_core_policy_version": REVISION_CORE_POLICY_VERSION,
        }
        self.assertEqual(
            verify_revision_policy(config, operation="test revision"),
            REVISION_CORE_POLICY_VERSION,
        )
        with self.assertRaisesRegex(
            RuntimeError, "Initialize an audited revision workspace"
        ):
            verify_revision_policy({}, operation="test revision")

    def test_revision_memory_records_effects_and_guides_next_strategy(self) -> None:
        before = {
            "status": "FAIL",
            "mwu": 0.01,
            "ks": 0.01,
            "cliff": 0.3,
            "wasserstein": 0.1,
            "real_mean": 0.2,
            "generated_mean": 0.4,
        }
        after = {**before, "mwu": 0.02, "generated_mean": 0.35}
        history = [
            {
                "round": 1,
                "target_metric": "self_bleu_4",
                "selected_profile": "high_tail",
                "profile_params": {"max-rewrite-budget": 6},
                "shape": {"failure_region": "high_tail", "q90_gap": 0.2},
                "input_root": "accepted-root",
                "output_root": "proposal",
                "before": before,
                "after": after,
                "improved": False,
                "protected_ok": True,
                "accepted_round": False,
                "decision": "rejected",
                "rejection_reasons": ["target_improvement_below_threshold"],
                "candidate_summary": {
                    "selected_comments": 3,
                    "accepted_comments": 2,
                    "generated_candidates": 18,
                    "candidate_rejection_reasons": {"claim_overlap_too_low": 7},
                    "accepted_styles": {"clause_reorder": 2},
                },
            }
        ]
        memory = build_memory(
            history,
            controller="metric",
            target_metrics=("self_bleu_4",),
            current_input_root=Path("accepted-root"),
        )
        self.assertEqual(memory["aggregate"]["attempted_rounds"], 1)
        self.assertEqual(
            memory["aggregate"]["candidate_totals"]["generated_candidates"],
            18,
        )
        self.assertEqual(memory["rounds"][0]["modification_direction"], "high_tail")
        self.assertEqual(
            memory["rounds"][0]["target_metrics_before"]["self_bleu_4"]["status"],
            "FAIL",
        )
        self.assertEqual(
            choose_history_aware_strategy(
                "high_tail",
                ("high_tail", "middle_mass", "shape_safe"),
                memory,
            ),
            "middle_mass",
        )
        feedback = render_prompt_feedback(memory)
        self.assertIn("claim_overlap_too_low", feedback)
        self.assertIn("high_tail", feedback)
        self.assertIn("accepted_comments=2/3", feedback)
        self.assertIn("generated_candidates=18", feedback)

    def test_generalized_selfbleu_controller_delegates_without_overrides(self) -> None:
        controller = load_script_module(
            "generalized_card_thin_selfbleu_controller",
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "run_selfbleu_revision_controller.py",
        )
        names = (
            "choose_profile",
            "choose_history_aware_strategy",
            "metric_improved",
            "protected_report",
        )
        before = {name: getattr(controller.card_controller, name) for name in names}
        with patch.object(controller.card_controller, "main") as delegated:
            with patch.object(sys, "argv", ["run_selfbleu_revision_controller.py"]):
                controller.main()
        delegated.assert_called_once_with()
        after = {name: getattr(controller.card_controller, name) for name in names}
        self.assertEqual(before, after)

    def test_selfbleu_adapter_preserves_card_prompt_and_adds_ngram_diagnostic(
        self,
    ) -> None:
        module = load_reviser_backend("selfbleu")
        target = module.CommentRef(
            comment_id=2,
            parent_comment_id=1,
            depth=1,
            content="The SD card feels slow, but the autofocus feels fine.",
            meta={"comment_function": "correction_caveat"},
            comment={},
        )
        comments = [
            module.CommentRef(
                1, None, 0, "The autofocus feels fine with this SD card.", {}, {}
            ),
            target,
            module.CommentRef(3, 1, 1, "The SD card feels slow in burst mode.", {}, {}),
        ]
        scored = module.ScoredComment(target, 0.2, 3, 0, 0.3, ["sd card feels"])
        configure_reviser_backend(module, kind="selfbleu", config=self.config)
        rendered = module.build_reviser_prompt(
            post={"title": "Camera buffer", "content": "Burst mode slows down."},
            comments=comments,
            target=target,
            scored=scored,
            candidates_per_comment=9,
            target_profile="high_tail",
            strategy_candidates=3,
            controller_feedback="Previous connector swaps did not help.",
        )
        for marker in (
            "Strategies:",
            "metadata=",
            "Nearby context:",
            "Previous connector swaps did not help.",
            "Cross-domain n-gram diagnostic:",
            "1-, 2-, 3-, and 4-gram overlap",
            "Repeated bigrams in this target:",
            "The SD card feels slow",
        ):
            self.assertIn(marker, rendered)
        self.assertNotIn("facts, cards, banks, dates, fees", rendered)

    def test_static_domain_adaptation_does_not_rewrite_dynamic_sd_card_text(
        self,
    ) -> None:
        original = (
            "You revise one generated Reddit comment.\n"
            "Preserve card names. Do not add new facts, cards, banks, dates, fees, "
            "percentages, URLs, or reward numbers.\nTarget: Replace the SD card."
        )
        adapted = legacy_reviser_prompts.adapt_card_reviser_prompt(
            self.config,
            original,
            kind="selfbleu",
        )
        self.assertIn("Preserve product/model names", adapted)
        self.assertIn("Replace the SD card", adapted)
        self.assertNotIn("facts, cards, banks", adapted)

    def test_selfbert_adapter_preserves_exact_card_prompt_fields(self) -> None:
        module = load_reviser_backend("selfbert")
        target_ref = module.CommentRef(
            2,
            1,
            1,
            "The autofocus hunts in low light on the Sony A7 IV.",
            {"comment_function": "correction_caveat", "story_mode": "no_story"},
            {},
        )
        parent = module.CommentRef(
            1, None, 0, "Low-light autofocus is my concern.", {}, {}
        )
        scored = module.ScoredComment(target_ref, 0.61, 3, 0.55, 0.1, 0.71, 0.64, 0.12)
        thread_target = module.ThreadTarget(
            {"self_bertscore_mean_f1": 0.54, "semantic_mean_cosine": 0.35},
            {"self_bertscore_mean_f1": 0.47, "semantic_mean_cosine": 0.32},
            0.07,
            2,
        )
        configure_reviser_backend(module, kind="selfbert", config=self.config)
        rendered = module.build_rewrite_prompt(
            post={"title": "A7 IV low light", "content": "How is autofocus?"},
            comments=[parent, target_ref],
            scored_comments=[scored],
            target=scored,
            thread_target=thread_target,
            accepted_so_far=[],
            candidates_per_comment=8,
            controller_feedback="Avoid the prior generic aside.",
        )
        for marker in (
            "exact_pair_mean=0.6400",
            "Most similar comments in this thread:",
            "Previously accepted rewrites in this same thread:",
            "Examples of possible discourse jobs",
            "Avoid the prior generic aside.",
        ):
            self.assertIn(marker, rendered)
        self.assertNotIn("Bank X, Card Y, issuer X", rendered)

    def test_tone_adapter_preserves_stage_specific_candidate_slate(self) -> None:
        module = load_reviser_backend("tone")
        parent = module.CommentRef(1, None, 0, "The focus misses in dim rooms.", {}, {})
        target = module.CommentRef(2, 1, 1, "No, that setup is just wrong.", {}, {})
        candidate = module.ToneCandidate(
            target,
            "impolite",
            "reduce_impolite_edge",
            ("neutral", "polite"),
            1.0,
            "matched-real tone gap",
            parent_ref=parent,
            parent_text=parent.content,
        )
        configure_reviser_backend(module, kind="tone", config=self.config)
        rendered = module.build_tone_prompt(
            post={"title": "Low-light setup", "content": "Which setting is wrong?"},
            comments=[parent, target],
            target=target,
            candidate=candidate,
            current_rates={"polite": 0.1, "neutral": 0.2, "impolite": 0.7},
            real_rates={"polite": 0.3, "neutral": 0.4, "impolite": 0.3},
            gaps={
                "polite": 0.2,
                "neutral": 0.2,
                "impolite": -0.4,
                "hard_disagree": -0.2,
            },
            candidates_per_comment=8,
            accepted=[],
            focus_stage="tone_gap_bestofn",
            controller_feedback="The previous polite prefix was rejected.",
        )
        for marker in (
            "Tone-gap best-of-N repair:",
            "same_claim_less_edge",
            "Thread wording to avoid repeating:",
            "Follow-up impolite-edge failure mode:",
            "The previous polite prefix was rejected.",
        ):
            self.assertIn(marker, rendered)
        self.assertNotIn("new card/bank/number", rendered)

    def test_story_adapter_preserves_card_strategy_and_distribution_target(
        self,
    ) -> None:
        module = load_reviser_backend("story")
        target = module.CommentRef(
            2,
            1,
            1,
            "I spent a week testing it, and the grip still felt cramped.",
            {},
            {},
        )
        parent = module.CommentRef(1, None, 0, "How is the grip?", {}, {})
        configure_reviser_backend(module, kind="story", config=self.config)
        rendered = module.build_prompt(
            post={"title": "Camera grip", "content": "Looking at the A7 IV."},
            target=target,
            comments=[parent, target],
            story_probability=0.81,
            real_thread_probability=0.32,
            candidate_count=6,
            strategy="concise_factual",
            controller_feedback="Direct-claim candidates preserved too much narration.",
        )
        for marker in (
            "current comment StorySeeker probability is 0.8100",
            "matched real thread mean is 0.3200",
            "Compress the anecdotal setup",
            "Direct-claim candidates preserved too much narration.",
        ):
            self.assertIn(marker, rendered)
        self.assertNotIn("r/CreditCards", rendered)
        self.assertNotIn("financial point", rendered)

    def test_revision_resume_rolls_back_rejected_round(self) -> None:
        history = [
            {
                "round": 1,
                "input_root": "initial",
                "input_scores_csv": "initial.csv",
                "input_matched_eval_dir": "initial-matched",
                "clean_root": "rejected-output",
                "eval_dir": "rejected-eval",
                "matched_eval_dir": "rejected-matched",
                "improved": False,
                "protected_ok": True,
                "decision": "rejected",
            }
        ]
        root, scores, matched, next_round = restore_controller_state(
            history,
            initial_root=Path("initial"),
            initial_scores=Path("initial.csv"),
            initial_matched=Path("initial-matched"),
        )
        self.assertEqual(root, Path("initial"))
        self.assertEqual(scores, Path("initial.csv"))
        self.assertEqual(matched, Path("initial-matched"))
        self.assertEqual(next_round, 2)

    def test_revision_resume_replays_legacy_history(self) -> None:
        history = [
            {
                "round": 1,
                "input_root": "initial",
                "clean_root": "accepted-root",
                "eval_dir": "accepted-eval",
                "matched_eval_dir": "accepted-matched",
                "improved": True,
                "protected_ok": True,
                "decision": "accepted",
            },
            {
                "round": 2,
                "clean_root": "rejected-root",
                "eval_dir": "rejected-eval",
                "matched_eval_dir": "rejected-matched",
                "improved": False,
                "protected_ok": True,
                "decision": "rejected",
            },
        ]
        root, scores, matched, next_round = restore_controller_state(
            history,
            initial_root=Path("initial"),
            initial_scores=Path("initial.csv"),
            initial_matched=Path("initial-matched"),
        )
        self.assertEqual(root, Path("accepted-root"))
        self.assertEqual(
            scores,
            Path("accepted-eval/revised_generated_thread_scores.csv"),
        )
        self.assertEqual(matched, Path("accepted-matched"))
        self.assertEqual(next_round, 3)

    def test_reviser_output_summary_counts_candidates_and_failures(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prompt_dir = root / "_prompts"
            prompt_dir.mkdir()
            (prompt_dir / "one.response.json").write_text(
                json.dumps(
                    {
                        "candidates": [
                            {"style": "a", "text": "one"},
                            {"style": "b", "text": "two"},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            (root / "selfbleu_lexical_reviser_report.json").write_text(
                json.dumps(
                    [
                        {
                            "status": "applied",
                            "candidate_comment_ids": [1],
                            "accepted_rewrites": [
                                {
                                    "style": "a",
                                    "candidate_evaluations": [
                                        {
                                            "accepted": False,
                                            "reason": "claim_overlap_too_low:0.4",
                                        },
                                        {"accepted": True, "reason": "accepted:gain"},
                                    ],
                                }
                            ],
                            "rejected_best_attempts": [],
                        }
                    ]
                ),
                encoding="utf-8",
            )
            summary = summarize_reviser_output(root)
        self.assertEqual(summary["generated_candidates"], 2)
        self.assertEqual(summary["accepted_comments"], 1)
        self.assertEqual(
            summary["candidate_rejection_reasons"]["claim_overlap_too_low"],
            1,
        )

    def test_backend_excludes_matched_real_text_from_writer_anchors(self) -> None:
        module = configure_generator_backend(load_generator_backend(), self.config)
        branch = module.BranchPlan(
            branch_id=1,
            anchor_quote="camera body",
            anchor_source="seed_detail",
            detour_type="comparison",
            branch_goal="compare handling",
            allowed_functions=("reaction",),
            evidence_modes=("none_assertion",),
            tone_palette=("casual_neutral",),
            story_modes=("no_story",),
            content_angles=("fit_use_case",),
        )
        seed = module.SeedPost(
            index=0,
            title="Sony A7 IV handling question",
            body="Is the grip comfortable?",
            content="Sony A7 IV handling question\nIs the grip comfortable?",
            source_raw_post_id="x",
            real_num_comments=5,
            metadata={},
        )
        anchors = module.build_concrete_anchors_for_task(
            real_body="CONFIDENTIAL_REAL_ONLY $9,999",
            seed_post=seed,
            branch=branch,
            planned={"semantic_move": "react to grip comfort"},
            anchor="grip comfort",
            parent_task=None,
        )
        rendered = " ".join(anchors)
        self.assertNotIn("$9,999", rendered)
        self.assertNotIn("CONFIDENTIAL_REAL_ONLY", rendered)
        self.assertIn("Sony", rendered)
        self.assertEqual(len(anchors), 1)
        self.assertEqual(module.sanitize_writer_text(""), "")

    def test_planned_domain_claim_becomes_a_writer_validation_anchor(self) -> None:
        module = configure_generator_backend(load_generator_backend(), self.config)
        branch = module.BranchPlan(
            branch_id=1,
            anchor_quote="camera body",
            anchor_source="seed_detail",
            detour_type="comparison",
            branch_goal="compare handling",
            allowed_functions=("reaction",),
            evidence_modes=("none_assertion",),
            tone_palette=("casual_neutral",),
            story_modes=("no_story",),
            content_angles=("fit_use_case",),
        )
        seed = module.SeedPost(
            index=0,
            title="Sony handling question",
            body="Is the grip comfortable?",
            content="Sony handling question\nIs the grip comfortable?",
            source_raw_post_id="x",
            real_num_comments=5,
            metadata={},
        )
        anchors = module.build_concrete_anchors_for_task(
            real_body="MATCHED_ONLY_SECRET",
            seed_post=seed,
            branch=branch,
            planned={
                "sample_id": "S2",
                "semantic_move": "state the assigned compatibility fact",
                "domain_claim": "Fujifilm X-T5 supports electronic aperture control",
            },
            anchor="handling",
            parent_task=None,
        )
        rendered = " ".join(anchors)
        self.assertIn("Fujifilm", rendered)
        self.assertNotIn("MATCHED_ONLY_SECRET", rendered)

    def test_domain_profile_excludes_evaluation_seed_threads(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "raw" / "sample"
            raw.mkdir(parents=True)
            post_rows = []
            comment_rows = []
            for index in range(25):
                post_id = f"p{index:02d}"
                post_rows.append(
                    json.dumps(
                        {
                            "id": post_id,
                            "title": f"Camera lens topic {index}",
                            "selftext": "Autofocus and low light comparison",
                        }
                    )
                )
                comment_rows.append(
                    json.dumps(
                        {
                            "post_id": post_id,
                            "comment_id": f"c{index:02d}",
                            "body": f"PRIVATE_REFERENCE_MARKER_{index} camera lens autofocus low light",
                        }
                    )
                )
            (raw / "sample.jsonl").write_text(
                "\n".join(post_rows) + "\n", encoding="utf-8"
            )
            (raw / "sample.comments.jsonl").write_text(
                "\n".join(comment_rows) + "\n",
                encoding="utf-8",
            )
            seed_pool = root / "seeds.json"
            seed_pool.write_text(
                json.dumps(
                    {
                        "seed_posts": [
                            {"source_raw_post_id": "p00"},
                            {"source_raw_post_id": "p01"},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            config = DomainConfig(
                domain_id="camera_test",
                display_name="Camera test",
                community_context="camera forum",
                raw_discussions_dir=root / "raw",
                real_scores_csv=root / "real.csv",
                topic_facets=("lens selection",),
                technical_terms=("autofocus", "low light"),
                protected_entity_terms=("camera",),
            )
            profile_path = root / "profile.json"
            profile = build_domain_profile(
                config,
                seed_pool_path=seed_pool,
                output_path=profile_path,
            )
            self.assertEqual(profile["source"]["reference_thread_count"], 23)
            self.assertEqual(profile["source"]["seed_reference_overlap_count"], 0)
            self.assertFalse(profile["source"]["test_content_visible"])
            self.assertGreater(profile["source"]["reference_viewpoint_count"], 0)
            self.assertIn("behavior_observations", profile)
            self.assertEqual(
                profile["behavior_targets"]["context_dropout_rate"],
                CARD_CONTEXT_DROPOUT_RATE,
            )
            self.assertEqual(
                profile["behavior_targets"]["context_jitter_rate"],
                CARD_CONTEXT_JITTER_RATE,
            )
            bank_text = " ".join(
                str(row.get("text") or "") for row in profile["reference_viewpoints"]
            )
            bank_ids = {
                str(row.get("source_post_id") or "")
                for row in profile["reference_viewpoints"]
            }
            self.assertNotIn("p00", bank_ids)
            self.assertNotIn("p01", bank_ids)
            bank_rows = {
                str(row.get("text") or "").split()[0]
                for row in profile["reference_viewpoints"]
            }
            self.assertNotIn("PRIVATE_REFERENCE_MARKER_0", bank_rows)
            self.assertNotIn("PRIVATE_REFERENCE_MARKER_1", bank_rows)
            self.assertIn("PRIVATE_REFERENCE_MARKER_2", bank_text)
            self.assertEqual(profile["schema_version"], PROFILE_SCHEMA_VERSION)
            self.assertFalse(profile["reference_metric_calibration"]["available"])
            self.assertIn("lexical_quality", profile)
            self.assertEqual(
                profile["lexical_quality"]["metric"],
                "mean symmetric pairwise BLEU-4 over all unordered thread-prefix pairs",
            )
            self.assertEqual(profile["perspectives"], universal_viewpoints())
            self.assertTrue(
                all(
                    row.get("axis") and row.get("decision_question")
                    for row in profile["perspectives"]
                )
            )
            self.assertNotIn(
                "autofocus",
                " ".join(row["label"] for row in profile["perspectives"]).lower(),
            )
            self.assertEqual(load_domain_profile(profile_path), profile)

    def test_reference_viewpoint_retrieval_is_relevant_diverse_and_deterministic(
        self,
    ) -> None:
        profile = {
            "profile_sha256": "test-profile",
            "reference_viewpoints": [
                {
                    "reference_id": "R00001",
                    "source_thread_hash": "a",
                    "thread_title": "Telephoto zoom for wildlife",
                    "thread_context": "Choosing reach for birds",
                    "text": "The extra reach matters more than a small low light gain.",
                    "surface_role": "full_answer",
                },
                {
                    "reference_id": "R00002",
                    "source_thread_hash": "b",
                    "thread_title": "Portrait lens",
                    "thread_context": "Studio portraits",
                    "text": "Would you use it indoors or outside?",
                    "surface_role": "narrow_question",
                },
                {
                    "reference_id": "R00003",
                    "source_thread_hash": "c",
                    "thread_title": "Compact camera zoom",
                    "thread_context": "Travel point and shoot",
                    "text": "I carried the smaller zoom on every trip.",
                    "surface_role": "personal_datapoint",
                },
            ],
        }
        first = retrieve_reference_viewpoints(
            profile,
            seed_title="Need a compact camera with telephoto zoom",
            seed_body="Mostly wildlife while traveling",
            limit=3,
        )
        second = retrieve_reference_viewpoints(
            profile,
            seed_title="Need a compact camera with telephoto zoom",
            seed_body="Mostly wildlife while traveling",
            limit=3,
        )
        self.assertEqual(first, second)
        self.assertIn("R00001", {row["reference_id"] for row in first})
        self.assertEqual(len({row["source_thread_hash"] for row in first}), 3)
        prefix = retrieve_reference_viewpoints(
            profile,
            seed_title="Need a compact camera with telephoto zoom",
            seed_body="Mostly wildlife while traveling",
            limit=1,
        )
        self.assertEqual(prefix, first[:1])

    def _render_comment_planner(self, backend, seed, target, slot_words):
        from generalized_card import prompts

        return prompts.comment_planner_prompt(
            self.config,
            backend,
            seed_post=seed,
            target=target,
            branches=[
                SimpleNamespace(
                    branch_id=1,
                    branch_goal="compare handling",
                    anchor_quote="grip",
                    allowed_functions=("reaction",),
                    content_angles=("fit_use_case",),
                )
            ],
            matched_real_thread={"comments": [{"body": "x " * slot_words}]},
            comments=[{"body": "x " * slot_words, "depth": 0}],
            all_comments=[{"body": "x " * slot_words, "depth": 0}],
        )

    def _planner_prompt_fixture(self):
        backend = SimpleNamespace(
            # A distinct sha: `viewpoint_bank._INDEX_CACHE` is keyed on
            # `profile_sha256` with no invalidation, so reusing another test's
            # sha with a different viewpoint list poisons that test.
            GENERALIZED_DOMAIN_PROFILE={
                "profile_sha256": "development-scope-fixture",
                "perspectives": [],
                "reference_viewpoints": [],
            },
            GENERALIZED_DOMAIN_CLAIM_MODE="planned",
            GENERALIZED_ACTIVE_REFERENCE_TEMPLATE={
                "comment_count": 1,
                "story_count": 0,
                "polite_rate": 1.0,
                "impolite_rate": 0.0,
                "neutral_rate": 0.0,
                "dominant_emotion_counts": {"curiosity": 1},
            },
            render_top_counts=lambda memory: "none",
            compact=lambda value, limit: str(value)[:limit],
        )
        seed = SimpleNamespace(
            title="Camera grip question",
            body="Is the grip comfortable?",
            content="Camera grip question\nIs the grip comfortable?",
        )
        target = SimpleNamespace(
            target_comments=1,
            max_depth_goal=1,
            top_level_comments=1,
            shape_label="quiet",
            length_mix_note="one short comment",
        )
        return backend, seed, target

    def test_development_plan_threshold_matches_the_beat_budget_exactly(self) -> None:
        """The prose rule and the three code gates must never disagree.

        `expected_development_beats` drives the slot schedule line, the capacity
        reconcile and the Writer cue. The Planner's prose rule is an f-string and
        has to be handed the number. If the two ever diverge the Planner receives
        `development_plan=N beats required` and `return the literal string none`
        for the same slot, in the same prompt.
        """

        from generalized_card.long_form_planning import (
            development_plan_word_threshold,
            expected_development_beats,
            set_development_scope,
        )

        try:
            for mode, expected_threshold in (("long_only", 100), ("measured", 34)):
                set_development_scope(mode)
                self.assertEqual(development_plan_word_threshold(), expected_threshold)
                for words in range(0, 901):
                    self.assertEqual(
                        expected_development_beats(words) > 0,
                        words > development_plan_word_threshold(),
                        f"{mode} disagrees at {words} words",
                    )
        finally:
            set_development_scope("long_only")

    def _rhythm_count_profile(self) -> dict:
        return {
            "available": True,
            "bands": {
                "very_long": {
                    "sample_count": 1104,
                    "median_words_per_sentence": 17.9,
                    "median_sentences": 8,
                    "multi_sentence_count": 1100,
                    "shares": {"parenthetical": 0.5489},
                    "habit_counts": {
                        "parenthetical": {
                            "1": 0.574, "2": 0.219, "3": 0.102, "4": 0.048, "5": 0.056,
                        }
                    },
                }
            },
        }

    # Named for this arm specifically: `_link_inventory` and `_link_task` already
    # exist in this class for the v113 tests, and a duplicate method name is taken
    # silently by whichever definition comes last.
    def _link_count_inventory(self) -> dict:
        return {
            "available": True,
            "urls": [f"https://example{i}.test/a/b?q={i}" for i in range(400)],
            "urls_per_carrier": {"1": 0.699, "2": 0.172, "3": 0.046, "4": 0.083},
            "mean_urls_per_carrier": 1.5133,
        }

    def _link_count_task(self, index: int):
        class _T:
            surface_texture = "link_reference"
            evidence_mode = ""
            real_sample_id = 0
            branch_id = ""
            claim_key = ""
        task = _T()
        task.local_task_id = index
        return task

    def test_reference_link_count_off_draws_exactly_one(self) -> None:
        """E1: the legacy value must reproduce v113 through v116."""

        from generalized_card.reference_link import (
            draw_reference_links, set_reference_link_count, set_reference_link_mode,
        )

        inv = self._link_count_inventory()
        try:
            set_reference_link_mode("measured")
            set_reference_link_count("off")
            counts = {len(draw_reference_links(self._link_count_task(i), inv)) for i in range(200)}
        finally:
            set_reference_link_count("off")
            set_reference_link_mode("off")
        self.assertEqual(counts, {1})

    def test_reference_link_count_measured_matches_the_inventory_distribution(self) -> None:
        from generalized_card.reference_link import (
            draw_reference_links, set_reference_link_count, set_reference_link_mode,
        )

        inv = self._link_count_inventory()
        try:
            set_reference_link_mode("measured")
            set_reference_link_count("measured")
            drawn = [len(draw_reference_links(self._link_count_task(i), inv)) for i in range(4000)]
        finally:
            set_reference_link_count("off")
            set_reference_link_mode("off")
        self.assertEqual(set(drawn), {1, 2, 3, 4})
        self.assertAlmostEqual(drawn.count(1) / len(drawn), 0.699, delta=0.03)
        self.assertAlmostEqual(sum(drawn) / len(drawn), 1.513, delta=0.06)

    def test_reference_link_count_never_repeats_a_url_in_one_slot(self) -> None:
        """A repeated link is a repeated n-gram and pushes self_bleu_4 the wrong way."""

        from generalized_card.reference_link import (
            draw_reference_links, set_reference_link_count, set_reference_link_mode,
        )

        inv = self._link_count_inventory()
        try:
            set_reference_link_mode("measured")
            set_reference_link_count("measured")
            multi = 0
            for i in range(600):
                urls = draw_reference_links(self._link_count_task(i), inv)
                self.assertEqual(len(urls), len(set(urls)), urls)
                if len(urls) > 1:
                    multi += 1
        finally:
            set_reference_link_count("off")
            set_reference_link_mode("off")
        self.assertGreater(multi, 50)

    # v118. Named apart from `_link_count_inventory` for the same reason that one
    # is named apart from `_link_inventory`: a duplicate method name is taken
    # silently by whichever definition comes last.
    def _link_host_inventory(self) -> dict:
        """Three hosts big enough for any k, plus 60 singletons."""

        urls = [f"https://big{g}.test/a/b/{i}" for g in range(3) for i in range(20)]
        urls += [f"https://solo{i}.test/x" for i in range(60)]
        return {
            "available": True,
            "urls": sorted(urls),
            "urls_per_carrier": {"1": 0.60, "2": 0.20, "3": 0.10, "4": 0.10},
            "mean_urls_per_carrier": 1.70,
            "same_host_rate": {"2": 0.771, "3": 0.640, "4": 0.417},
            "same_host_rate_pooled": 0.695,
            "same_host_sample_counts": {"2": 105, "3": 25, "4": 24},
        }

    def _donor_inventory(self) -> dict:
        return {
            "available": True,
            "sentence_count": 400,
            "sentences": [f"Thanks so much for tip number {i}." for i in range(400)],
        }

    def _donor_task(self, index: int, tone: str = "polite"):
        class _T:
            real_sample_id = 0
            branch_id = ""
            claim_key = ""
        task = _T()
        task.local_task_id = index
        task.tone_target = tone
        return task

    def test_tone_donor_reads_the_key_the_real_profile_writes(self) -> None:
        """The v120 run's whole failure, as a test.

        `_tone_donor_block` read `profile["domain"]`; a real domain profile writes
        **`domain_id`**. Every slot got "", `load_donor_inventory("")` found no
        file, and the arm rendered nothing across 186 prompts while
        `run_config.json` said `tone_donor: measured`. It passed verification
        because that check built its own `{"domain": ...}` dict instead of reading
        a profile off disk -- so this test reads one off disk.
        """

        import json
        from pathlib import Path as _Path

        from generalized_card.tone_donor import domain_of

        root = _Path(__file__).resolve().parents[2] / "artifacts/generalized_card/runs"
        profiles = sorted(root.glob("*/domain_profile.json"))
        if not profiles:
            self.skipTest("no domain profile in this tree")
        payload = json.loads(profiles[-1].read_text())
        self.assertNotIn(
            "domain", payload, "if a profile ever writes `domain`, revisit domain_of"
        )
        self.assertTrue(payload.get("domain_id"), "a profile must carry domain_id")
        self.assertEqual(domain_of(payload), payload["domain_id"])
        self.assertEqual(domain_of({"domain": "legacy"}), "legacy")
        self.assertEqual(domain_of({}), "")

    def test_tone_donor_on_with_no_inventory_raises(self) -> None:
        """E9: an arm that is on and cannot fire must stop the run, not go quiet."""

        from generalized_card.tone_donor import (
            require_donor_inventory,
            set_tone_donor_mode,
        )

        try:
            set_tone_donor_mode("measured")
            with self.assertRaises(RuntimeError):
                require_donor_inventory({"domain_id": "no_such_domain"})
            # and the arm off must stay silent
            set_tone_donor_mode("off")
            self.assertFalse(
                require_donor_inventory({"domain_id": "no_such_domain"})["available"]
            )
        finally:
            set_tone_donor_mode("off")

    def test_tone_matrix_follows_the_donor_arm(self) -> None:
        """G79: solving against the donor-free matrix is what made v120b overshoot.

        The Planner asked for 56% polite because the solver used a polite row of
        0.384; the donor then delivered 0.784 and `polite_rate` landed +34.7% above
        real -- a worse miss than the -55.5% it started from.
        """

        import generalized_card.tone_realization as tr
        from generalized_card.tone_donor import set_tone_donor_mode

        template = {
            "polite": 0.313,
            "somewhat_polite": 0.0951,
            "neutral": 0.1644,
            "impolite": 0.4275,
        }
        try:
            set_tone_donor_mode("off")
            tr._CACHE.clear()
            self.assertEqual(tr.active_matrix(), tr.REALIZATION_MATRIX)
            tr.set_tone_quota_mode("inverted")
            without = tr.invert_tone_rates(template)

            set_tone_donor_mode("measured")
            tr._CACHE.clear()
            armed = tr.active_matrix()
            self.assertAlmostEqual(sum(armed[0]), 1.0, places=3)
            self.assertGreater(armed[0][0], tr.REALIZATION_MATRIX[0][0])
            self.assertEqual(armed[1:], tr.REALIZATION_MATRIX[1:], "only the polite row moves")
            with_donor = tr.invert_tone_rates(template)
        finally:
            tr.set_tone_quota_mode("off")
            set_tone_donor_mode("off")
            tr._CACHE.clear()

        self.assertLess(
            with_donor["polite"],
            without["polite"] - 0.10,
            "with the donor on, the Planner must ask for materially less polite",
        )
        # and the projected realization must land on the template, not past it
        set_tone_donor_mode("measured")
        tr._CACHE.clear()
        try:
            realized = tr._realized(tuple(with_donor[t] for t in tr.TONE_ORDER))
        finally:
            set_tone_donor_mode("off")
            tr._CACHE.clear()
        for i, name in enumerate(tr.TONE_ORDER):
            if name == "somewhat_polite":
                continue
            self.assertAlmostEqual(realized[i], template[name], delta=0.02, msg=name)

    def test_tone_donor_off_draws_nothing(self) -> None:
        """E1: the legacy value must reproduce every release through v119."""

        from generalized_card.tone_donor import draw_donor_sentence, set_tone_donor_mode

        inv = self._donor_inventory()
        try:
            set_tone_donor_mode("off")
            drawn = [draw_donor_sentence(self._donor_task(i), inv) for i in range(200)]
        finally:
            set_tone_donor_mode("off")
        self.assertEqual(set(drawn), {""})

    def test_tone_donor_routes_on_the_polite_assignment_only(self) -> None:
        from generalized_card.tone_donor import draw_donor_sentence, set_tone_donor_mode

        inv = self._donor_inventory()
        try:
            set_tone_donor_mode("measured")
            polite = [draw_donor_sentence(self._donor_task(i), inv) for i in range(120)]
            others = [
                draw_donor_sentence(self._donor_task(i, tone), inv)
                for tone in ("impolite", "neutral", "somewhat_polite", "")
                for i in range(120)
            ]
        finally:
            set_tone_donor_mode("off")
        self.assertTrue(all(polite), "every polite slot must be offered a sentence")
        self.assertEqual(set(others), {""}, "no other assignment may be touched")

    def test_tone_donor_draw_is_deterministic_and_spread(self) -> None:
        """G37: a shared prescribed sentence converges the pairwise metrics."""

        from generalized_card.tone_donor import draw_donor_sentence, set_tone_donor_mode

        inv = self._donor_inventory()
        try:
            set_tone_donor_mode("measured")
            first = [draw_donor_sentence(self._donor_task(i), inv) for i in range(400)]
            again = [draw_donor_sentence(self._donor_task(i), inv) for i in range(400)]
        finally:
            set_tone_donor_mode("off")
        self.assertEqual(first, again, "the draw must be reproducible per slot")
        # 400 draws over a 400-sentence pool: a uniform hash gives ~253 distinct.
        self.assertGreater(len(set(first)), 200)

    def test_tone_donor_offer_carries_the_sentence_and_no_rationale(self) -> None:
        """G37 again: the sentence is per-slot distinct, an added rationale is not."""

        from generalized_card.tone_donor import donor_sentence_offer

        text = donor_sentence_offer("Thanks so much for sharing your experience!")
        self.assertIn("Thanks so much for sharing your experience!", text)
        self.assertEqual(donor_sentence_offer("   "), "")
        for banned in ("because", "so that", "in order to", "this makes"):
            self.assertNotIn(banned, text.lower())

    def test_tone_donor_reaches_all_three_writer_templates(self) -> None:
        """G23 and G41: a prompt fix that reaches one template silently caps the arm.

        `polite_rate` is a share over every comment, so the block has to render
        from `writer_prompt`'s two branches AND from `_low_info_writer_prompt`.
        """

        import inspect

        from generalized_card import prompts

        source = inspect.getsource(prompts)
        self.assertEqual(
            source.count("_tone_donor_block(backend, task)"),
            3,
            "the donor block must render from all three writer prompt templates",
        )
        self.assertIn("_tone_donor_block", inspect.getsource(prompts._low_info_writer_prompt))

    def test_tone_donor_inventory_is_topic_free(self) -> None:
        """A donor is prefixed to a comment about a DIFFERENT product."""

        import re

        from generalized_card.tone_donor import load_donor_inventory

        inv = load_donor_inventory("camera_product")
        if not inv.get("available"):
            self.skipTest("no harvested inventory in this tree")
        brand = re.compile(r"\b(?:canon|nikon|sony|fuji|olympus|ricoh|leica|sigma)\b", re.I)
        digit = re.compile(r"\d")
        for sentence in inv["sentences"]:
            self.assertIsNone(brand.search(sentence), sentence)
            self.assertIsNone(digit.search(sentence), sentence)
            self.assertLessEqual(len(sentence.split()), 12, sentence)
        self.assertGreater(inv["sentence_count"], 300)

    def test_tone_inversion_objective_ignores_the_unreported_class(self) -> None:
        """G66: `somewhat_polite` is never reported, so it is not in the loss.

        The guard is behavioural, not structural: moving the target's
        `somewhat_polite` mass while holding the three reported rates fixed must
        not change the solution, and moving a reported rate must.
        """

        from generalized_card.tone_realization import (
            REPORTED_TONES,
            TONE_ORDER,
            invert_tone_rates,
            set_tone_quota_mode,
        )

        self.assertNotIn("somewhat_polite", REPORTED_TONES)
        base = {"polite": 0.30, "somewhat_polite": 0.10, "neutral": 0.16, "impolite": 0.44}
        # Same three reported rates, different split of the unreported remainder
        # against a scaled total -- the normalised reported ratios are identical.
        try:
            set_tone_quota_mode("inverted")
            first = invert_tone_rates(base)
            moved = invert_tone_rates({**base, "polite": 0.20, "somewhat_polite": 0.20})
        finally:
            set_tone_quota_mode("off")
        self.assertEqual(set(first), set(TONE_ORDER))
        self.assertNotEqual(first, moved, "a reported rate must move the solution")

    def test_reference_link_host_off_leaves_the_v117_draw_untouched(self) -> None:
        """E1: the legacy value must reproduce v117 byte-for-byte."""

        from generalized_card.reference_link import (
            draw_reference_links, set_reference_link_count, set_reference_link_host,
            set_reference_link_mode,
        )

        inv = self._link_host_inventory()
        try:
            set_reference_link_mode("measured")
            set_reference_link_count("measured")
            set_reference_link_host("off")
            legacy = [draw_reference_links(self._link_count_task(i), inv) for i in range(300)]
            set_reference_link_host("measured")
            armed = [draw_reference_links(self._link_count_task(i), inv) for i in range(300)]
        finally:
            set_reference_link_host("off")
            set_reference_link_count("off")
            set_reference_link_mode("off")
        self.assertEqual([len(u) for u in legacy], [len(u) for u in armed],
                         "the arm must not change how many links a slot gets")
        singles = [i for i, u in enumerate(legacy) if len(u) == 1]
        self.assertGreater(len(singles), 100)
        for i in singles:
            self.assertEqual(legacy[i], armed[i], "single-link slots must not move")
        self.assertNotEqual(legacy, armed, "the arm must do something at k >= 2")

    def test_reference_link_host_draws_one_host_at_the_measured_rate(self) -> None:
        from generalized_card.reference_link import (
            draw_reference_links, folded_host, set_reference_link_count,
            set_reference_link_host, set_reference_link_mode,
        )

        inv = self._link_host_inventory()
        try:
            set_reference_link_mode("measured")
            set_reference_link_count("measured")
            set_reference_link_host("measured")
            by_k: dict[int, list[bool]] = {}
            for i in range(6000):
                urls = draw_reference_links(self._link_count_task(i), inv)
                self.assertEqual(len(urls), len(set(urls)), urls)
                if len(urls) >= 2:
                    by_k.setdefault(len(urls), []).append(
                        len({folded_host(u) for u in urls}) == 1
                    )
        finally:
            set_reference_link_host("off")
            set_reference_link_count("off")
            set_reference_link_mode("off")
        for k, target in ((2, 0.771), (3, 0.640), (4, 0.417)):
            rows = by_k.get(k) or []
            self.assertGreater(len(rows), 200, f"k={k} too rare to judge")
            self.assertAlmostEqual(sum(rows) / len(rows), target, delta=0.05,
                                   msg=f"k={k} one-host rate")

    def test_reference_link_host_falls_back_to_the_pooled_rate(self) -> None:
        """A k with too few carriers to estimate uses the pooled rate, not zero."""

        from generalized_card.reference_link import _same_host_rate

        inv = self._link_host_inventory()
        self.assertAlmostEqual(_same_host_rate(inv, 3), 0.640)
        thin = dict(inv, same_host_rate={"2": 0.771})
        self.assertAlmostEqual(_same_host_rate(thin, 4), 0.695)
        self.assertEqual(_same_host_rate({}, 2), 0.0)

    def test_folded_host_treats_one_place_as_one_host(self) -> None:
        from generalized_card.reference_link import folded_host

        self.assertEqual(folded_host("https://youtu.be/abc"), "youtube.com")
        self.assertEqual(folded_host("https://www.youtube.com/watch?v=abc"), "youtube.com")
        self.assertEqual(folded_host("https://np.reddit.com/r/x/"), "reddit.com")
        self.assertEqual(folded_host("https://en.wikipedia.org/wiki/X"), "wikipedia.org")
        self.assertEqual(folded_host("https://usa.canon.com/shop/p/x"), "canon.com")
        self.assertEqual(folded_host("www.dpreview.com/a"), "dpreview.com")

    def test_reference_link_host_inventory_measures_the_rate(self) -> None:
        from generalized_card.reference_link import build_reference_link_inventory

        threads = [{"comments": (
            # 3 same-host pairs, 1 mixed pair -> 0.75 at k=2
            [{"body": f"see https://a.test/{i} and https://a.test/{i}b"} for i in range(3)]
            + [{"body": "see https://a.test/z and https://b.test/z"}]
            # a 5-URL carrier is outside 2..MAX and must not be counted
            + [{"body": " ".join(f"https://c.test/{i}" for i in range(5))}]
        )}]
        inv = build_reference_link_inventory(threads)
        self.assertEqual(inv["same_host_sample_counts"]["2"], 4)
        self.assertEqual(inv["same_host_sample_counts"]["4"], 0)
        self.assertAlmostEqual(inv["same_host_rate_pooled"], 0.75)
        # 4 carriers is under MIN_HOST_RATE_SAMPLE, so no per-k rate is published
        self.assertEqual(inv["same_host_rate"], {})

    def test_reference_link_single_offer_is_byte_identical(self) -> None:
        from generalized_card.reference_link import (
            reference_link_offer, reference_links_offer,
        )

        url = "https://example7.test/a/b?q=7"
        self.assertEqual(reference_links_offer([url]), reference_link_offer(url))
        self.assertEqual(reference_links_offer([]), "")

    def test_reference_link_plural_offer_names_the_count_and_every_url(self) -> None:
        from generalized_card.reference_link import reference_links_offer

        urls = ["https://a.test/x", "https://b.test/y", "https://c.test/z"]
        text = reference_links_offer(urls)
        self.assertIn("3 of them", text)
        for url in urls:
            self.assertIn(url, text)
        self.assertIn("do not write any", text)

    def test_rhythm_count_off_always_asks_for_one(self) -> None:
        """E1: the legacy value must reproduce every release through v115."""

        from generalized_card.sentence_rhythm import set_rhythm_count, slot_habit_count

        profile = self._rhythm_count_profile()
        try:
            set_rhythm_count("off")
            drawn = {
                slot_habit_count(
                    profile, slot_key=f"s{i}", habit="parenthetical", word_count=160
                )
                for i in range(300)
            }
        finally:
            set_rhythm_count("off")
        self.assertEqual(drawn, {1})

    def test_rhythm_count_measured_reproduces_the_band_distribution(self) -> None:
        from generalized_card.sentence_rhythm import set_rhythm_count, slot_habit_count

        profile = self._rhythm_count_profile()
        try:
            set_rhythm_count("measured")
            drawn = [
                slot_habit_count(
                    profile, slot_key=f"s{i}", habit="parenthetical", word_count=160
                )
                for i in range(4000)
            ]
        finally:
            set_rhythm_count("off")
        self.assertEqual(set(drawn), {1, 2, 3, 4, 5})
        mean = sum(drawn) / len(drawn)
        # The band measures 1.88; the draw is capped at five and real's tail runs
        # to 22, so the drawn mean sits just under it.
        self.assertGreater(mean, 1.6)
        self.assertLess(mean, 1.9)
        self.assertAlmostEqual(drawn.count(1) / len(drawn), 0.574, delta=0.03)

    def test_rhythm_count_draw_is_stable_and_independent_of_the_habit_draw(self) -> None:
        from generalized_card.sentence_rhythm import (
            set_rhythm_count,
            slot_habit_count,
            slot_uses_habit,
        )

        profile = self._rhythm_count_profile()
        try:
            set_rhythm_count("measured")
            first = [
                slot_habit_count(
                    profile, slot_key=f"s{i}", habit="parenthetical", word_count=160
                )
                for i in range(200)
            ]
            second = [
                slot_habit_count(
                    profile, slot_key=f"s{i}", habit="parenthetical", word_count=160
                )
                for i in range(200)
            ]
            used = [
                slot_uses_habit(
                    profile, slot_key=f"s{i}", habit="parenthetical", word_count=160
                )
                for i in range(200)
            ]
        finally:
            set_rhythm_count("off")
        self.assertEqual(first, second)
        # Sharing one digest would couple "barely drew the habit" to "drew the
        # smallest count"; these must be independent.
        drew = [c for c, u in zip(first, used) if u]
        skipped = [c for c, u in zip(first, used) if not u]
        self.assertGreater(len(drew), 20)
        self.assertGreater(len(skipped), 20)
        self.assertAlmostEqual(
            sum(drew) / len(drew), sum(skipped) / len(skipped), delta=0.6
        )

    def test_rhythm_count_renders_words_not_figures(self) -> None:
        """The same rendered rule tells the Writer to write numbers as figures."""

        from generalized_card.sentence_rhythm import rhythm_guidance, set_rhythm_count

        profile = self._rhythm_count_profile()
        try:
            set_rhythm_count("off")
            legacy = rhythm_guidance(profile, slot_key="s3", word_count=160)
            set_rhythm_count("measured")
            rendered = [
                rhythm_guidance(profile, slot_key=f"s{i}", word_count=160)
                for i in range(400)
            ]
        finally:
            set_rhythm_count("off")
        asides = [r for r in rendered if "asides in parentheses" in r]
        self.assertGreater(len(asides), 20)
        for text in asides:
            fragment = text[text.index("Put ") : text.index("asides in parentheses")]
            self.assertFalse(
                any(ch.isdigit() for ch in fragment), fragment
            )
        # A drawn count of one must render the legacy string exactly.
        ones = [r for r in rendered if "Put one aside in parentheses." in r]
        self.assertGreater(len(ones), 20)
        self.assertIn("Put one aside in parentheses.", legacy)

    def test_tone_quota_calibrate_renders_a_flat_grid_covering_quota(self) -> None:
        """The measurement value: flat, template-independent, and clearly marked."""

        from generalized_card.generation_distribution import template_tone_rates
        from generalized_card.tone_realization import (
            CALIBRATION_RATES,
            realization_report,
            set_tone_quota_mode,
        )

        skewed = {
            "polite_rate": 0.05,
            "impolite_rate": 0.80,
            "neutral_rate": 0.10,
            "somewhat_polite_rate": 0.05,
        }
        try:
            set_tone_quota_mode("calibrate")
            rendered = template_tone_rates(skewed)
            report = realization_report(rendered)
        finally:
            set_tone_quota_mode("off")
        self.assertEqual(rendered, CALIBRATION_RATES)
        self.assertEqual(set(rendered), {"polite", "somewhat_polite", "neutral", "impolite"})
        self.assertTrue(all(abs(v - 0.25) < 1e-9 for v in rendered.values()))
        self.assertEqual(report["mode"], "calibrate")
        self.assertIn("measurement only", report["purpose"])

    def test_tone_quota_unknown_value_falls_back_to_off(self) -> None:
        from generalized_card.generation_distribution import (
            template_tone_rates,
            template_tone_rates_raw,
        )
        from generalized_card.tone_realization import set_tone_quota_mode

        template = {
            "polite_rate": 0.26,
            "impolite_rate": 0.46,
            "neutral_rate": 0.16,
            "somewhat_polite_rate": 0.12,
        }
        for value in ("", None, "INVERT", "yes", "measured"):
            self.assertFalse(set_tone_quota_mode(value), value)
            self.assertEqual(
                template_tone_rates(template), template_tone_rates_raw(template), value
            )
        set_tone_quota_mode("off")

    def test_tone_quota_off_is_byte_identical_to_the_template_rates(self) -> None:
        """E1: the legacy value must reproduce every release through v114."""

        from generalized_card.generation_distribution import (
            template_tone_rates,
            template_tone_rates_raw,
        )
        from generalized_card.tone_realization import set_tone_quota_mode

        template = {
            "polite_rate": 0.2595,
            "impolite_rate": 0.4640,
            "neutral_rate": 0.1591,
            "somewhat_polite_rate": 0.1174,
        }
        try:
            set_tone_quota_mode("off")
            self.assertEqual(
                template_tone_rates(template), template_tone_rates_raw(template)
            )
        finally:
            set_tone_quota_mode("off")

    def test_tone_quota_inverted_moves_the_assignment_not_the_target(self) -> None:
        """The rendered quota changes; the mix the metric reports against does not."""

        from generalized_card.generation_distribution import (
            template_tone_rates,
            template_tone_rates_raw,
        )
        from generalized_card.tone_realization import (
            POLITE_ASSIGNMENT_CAP,
            REALIZATION_MATRIX,
            TONE_ORDER,
            set_tone_quota_mode,
        )

        template = {
            "polite_rate": 0.2595,
            "impolite_rate": 0.4640,
            "neutral_rate": 0.1591,
            "somewhat_polite_rate": 0.1174,
        }
        raw = template_tone_rates_raw(template)
        try:
            set_tone_quota_mode("inverted")
            solved = template_tone_rates(template)
        finally:
            set_tone_quota_mode("off")

        self.assertEqual(template_tone_rates_raw(template), raw)
        self.assertNotEqual(solved, raw)
        self.assertAlmostEqual(sum(solved.values()), 1.0, places=6)
        # The cap is a shipped decision, not a free parameter: G66 sets it at the
        # value that maximises the WORST of the three reported metrics' closure
        # over both known realization matrices. It was 0.35 while the polite row
        # was only measured on `agree` slots.
        self.assertLessEqual(solved["polite"], POLITE_ASSIGNMENT_CAP + 1e-9)
        self.assertAlmostEqual(POLITE_ASSIGNMENT_CAP, 0.56)

        realized = {
            out: sum(
                solved[TONE_ORDER[i]] * REALIZATION_MATRIX[i][j]
                for i in range(len(TONE_ORDER))
            )
            for j, out in enumerate(TONE_ORDER)
        }
        # The point of the arm: the realized impolite share moves off the
        # generator's 0.607 and onto real's 0.464, without touching the Writer.
        today = sum(
            raw[TONE_ORDER[i]] * REALIZATION_MATRIX[i][3] for i in range(len(TONE_ORDER))
        )
        self.assertGreater(today, 0.58)
        self.assertLess(abs(realized["impolite"] - raw["impolite"]), abs(today - raw["impolite"]))
        self.assertGreater(realized["polite"], 
                           sum(raw[TONE_ORDER[i]] * REALIZATION_MATRIX[i][0]
                               for i in range(len(TONE_ORDER))))

    def test_tone_quota_inversion_is_deterministic_and_a_valid_distribution(self) -> None:
        from generalized_card.tone_realization import (
            invert_tone_rates,
            set_tone_quota_mode,
        )
        import generalized_card.tone_realization as tone_realization

        target = {
            "polite": 0.30,
            "somewhat_polite": 0.10,
            "neutral": 0.20,
            "impolite": 0.40,
        }
        try:
            set_tone_quota_mode("inverted")
            first = invert_tone_rates(target)
            tone_realization._CACHE.clear()
            second = invert_tone_rates(target)
        finally:
            set_tone_quota_mode("off")
        self.assertEqual(first, second)
        self.assertTrue(all(value >= 0.0 for value in first.values()))
        self.assertAlmostEqual(sum(first.values()), 1.0, places=6)

    def test_tone_quota_inverted_reaches_the_planner_schedule_too(self) -> None:
        """The rendered quota and the slot schedule must not disagree."""

        from generalized_card.planner_distribution import template_distribution_targets
        from generalized_card.tone_realization import set_tone_quota_mode

        template = {
            "polite_rate": 0.2595,
            "impolite_rate": 0.4640,
            "neutral_rate": 0.1591,
            "somewhat_polite_rate": 0.1174,
            "comment_count": 45,
        }
        try:
            set_tone_quota_mode("off")
            legacy = template_distribution_targets(template, total_comments=45)
            set_tone_quota_mode("inverted")
            inverted = template_distribution_targets(template, total_comments=45)
        finally:
            set_tone_quota_mode("off")
        self.assertNotEqual(legacy["tone_counts"], inverted["tone_counts"])
        self.assertLess(
            inverted["tone_counts"]["impolite"], legacy["tone_counts"]["impolite"]
        )
        self.assertEqual(sum(inverted["tone_counts"].values()), 45)

    def test_development_scope_long_only_renders_the_v110_planner_rule(self) -> None:
        from generalized_card.long_form_planning import set_development_scope

        backend, seed, target = self._planner_prompt_fixture()
        try:
            set_development_scope("long_only")
            rendered = self._render_comment_planner(backend, seed, target, 70)
        finally:
            set_development_scope("long_only")
        self.assertIn(
            "For every slot at or below 100 anonymous words, return the literal string",
            rendered,
        )
        self.assertNotIn("development_plan=", rendered)

    def test_development_scope_measured_moves_the_rule_and_the_schedule_together(
        self,
    ) -> None:
        """A 70-word slot must be asked for beats in BOTH places, or neither."""

        from generalized_card.long_form_planning import set_development_scope

        backend, seed, target = self._planner_prompt_fixture()
        try:
            set_development_scope("measured")
            rendered = self._render_comment_planner(backend, seed, target, 70)
        finally:
            set_development_scope("long_only")
        self.assertIn(
            "For every slot at or below 34 anonymous words, return the literal string",
            rendered,
        )
        self.assertNotIn("at or below 100 anonymous words", rendered)
        self.assertIn("development_plan=3 beats required", rendered)

    def test_development_scope_leaves_a_short_slot_alone_on_both_values(self) -> None:
        from generalized_card.long_form_planning import set_development_scope

        backend, seed, target = self._planner_prompt_fixture()
        rendered = {}
        try:
            for mode in ("long_only", "measured"):
                set_development_scope(mode)
                rendered[mode] = self._render_comment_planner(
                    backend, seed, target, 20
                )
        finally:
            set_development_scope("long_only")
        for mode in ("long_only", "measured"):
            self.assertNotIn("development_plan=", rendered[mode])

    def _link_backend(self, inventory):
        return SimpleNamespace(
            GENERALIZED_DOMAIN_PROFILE={
                "profile_sha256": "reference-link-fixture",
                "perspectives": [],
                "reference_viewpoints": [],
                "reference_link_inventory": inventory,
                "entity_inventory": {},
                "entity_spread_profile": {},
            },
            GENERALIZED_ACTIVE_SEED_KEY="seedkey",
            GENERALIZED_ACTIVE_THREAD_COMMENTS=20,
            GENERALIZED_OWN_FACT_LICENSE="off",
        )

    @staticmethod
    def _link_task(**over):
        base = dict(
            surface_texture="link_reference",
            evidence_mode="none_assertion",
            real_sample_id=7,
            local_task_id=7,
            branch_id=1,
            claim_key="handling_check",
            concrete_anchors=(),
            real_word_count=60,
            real_surface_shape="link_reference",
        )
        base.update(over)
        return SimpleNamespace(**base)

    def test_reference_link_off_offers_nothing_and_keeps_the_v112_prohibition(self) -> None:
        from generalized_card import prompts
        from generalized_card.reference_link import set_reference_link_mode

        inv = {"available": True, "urls": ["https://www.dpreview.com/reviews/x"]}
        try:
            set_reference_link_mode("off")
            block = prompts._equipment_and_referent_block(self._link_backend(inv), self._link_task())
            guidance = prompts._placeholder_guidance_block()
            texture = prompts._surface_texture_guidance("link_reference", task=self._link_task())
        finally:
            set_reference_link_mode("off")
        self.assertNotIn("dpreview", block)
        self.assertIn(
            "write a normal human reference sentence without inventing a URL", guidance
        )
        self.assertIn("without inventing a URL", texture)
        self.assertNotIn("supplied for this slot", texture)

    def test_reference_link_measured_hands_over_the_exact_url(self) -> None:
        from generalized_card import prompts
        from generalized_card.reference_link import set_reference_link_mode

        url = "https://www.dpreview.com/reviews/ricoh-gr-iii"
        inv = {"available": True, "urls": [url]}
        try:
            set_reference_link_mode("measured")
            block = prompts._equipment_and_referent_block(self._link_backend(inv), self._link_task())
            guidance = prompts._placeholder_guidance_block()
        finally:
            set_reference_link_mode("off")
        self.assertIn(url, block)
        self.assertIn("Include this exact URL once", block)
        self.assertIn("If this slot supplies an exact URL, use that one", guidance)

    def test_reference_link_offer_and_prohibition_never_contradict(self) -> None:
        """The v112 failure, asserted directly: an offer plus a blanket ban.

        Under `measured` the Writer is handed a URL, so a rule that forbids
        writing one at all would be a contradiction inside a single prompt.
        """

        from generalized_card import prompts
        from generalized_card.reference_link import set_reference_link_mode

        inv = {"available": True, "urls": ["https://www.dpreview.com/reviews/x"]}
        for mode in ("off", "measured"):
            try:
                set_reference_link_mode(mode)
                offered = "dpreview" in prompts._equipment_and_referent_block(
                    self._link_backend(inv), self._link_task()
                )
                banned_outright = (
                    "write a normal human reference sentence without inventing a URL"
                    in prompts._placeholder_guidance_block()
                )
            finally:
                set_reference_link_mode("off")
            self.assertFalse(
                offered and banned_outright,
                f"{mode}: the prompt both supplies a URL and forbids writing one",
            )

    def test_reference_link_only_routes_slots_whose_matched_comment_had_one(self) -> None:
        from generalized_card.reference_link import (
            draw_reference_link,
            reference_link_slot,
            set_reference_link_mode,
        )

        inv = {"available": True, "urls": ["https://a.example/1", "https://b.example/2"]}
        try:
            set_reference_link_mode("measured")
            routed = self._link_task()
            by_evidence = self._link_task(
                surface_texture="plain", evidence_mode="link_quote_reference"
            )
            plain = self._link_task(surface_texture="plain")
            self.assertTrue(reference_link_slot(routed))
            self.assertTrue(reference_link_slot(by_evidence))
            self.assertFalse(reference_link_slot(plain))
            self.assertTrue(draw_reference_link(routed, inv))
            self.assertEqual(draw_reference_link(plain, inv), "")
            self.assertEqual(
                draw_reference_link(routed, inv), draw_reference_link(routed, inv)
            )
        finally:
            set_reference_link_mode("off")

    def test_reference_link_extraction_survives_reddit_markdown(self) -> None:
        """The v113 gate's own failure, frozen as a test.

        Reddit renders a bare link as `[url](url)` and escapes `_` to `\\_`. The
        first extractor used `\\S+`, ran straight through `](` into the second
        copy, and put 166 of 690 malformed entries into the inventory; 6 of the
        23 links the gate wrote came out as `url](url`.
        """

        from generalized_card.reference_link import extract_urls

        cases = {
            r"https://youtu.be/1\_YaLe3rAbA?si=3fv0yQf\_OOAi4fYo](https://youtu.be/1_YaLe3rAbA?si=3fv0yQf_OOAi4fYo": [
                "https://youtu.be/1_YaLe3rAbA?si=3fv0yQf_OOAi4fYo",
                "https://youtu.be/1_YaLe3rAbA?si=3fv0yQf_OOAi4fYo",
            ],
            "http://optechusa.com/)**": ["http://optechusa.com/"],
            "www.eadr.co.uk](http://www.eadr.co.uk": [
                "www.eadr.co.uk",
                "http://www.eadr.co.uk",
            ],
            r"see https://youtu.be/fz2LSHQ8E\_w for the test": [
                "https://youtu.be/fz2LSHQ8E_w"
            ],
            "plain https://www.dpreview.com/reviews/x and text": [
                "https://www.dpreview.com/reviews/x"
            ],
        }
        for raw, expected in cases.items():
            self.assertEqual(extract_urls(raw), expected, raw)

    def test_reference_link_inventory_rejects_markdown_wrapped_links(self) -> None:
        from generalized_card.reference_link import build_reference_link_inventory

        threads = [
            {
                "comments": [
                    {
                        "body": "watch [https://youtu.be/abc\\_d](https://youtu.be/abc_d) first"
                    },
                    {"body": "and http://optechusa.com/)** too"},
                ]
            }
        ]
        inv = build_reference_link_inventory(threads)
        for url in inv["urls"]:
            self.assertNotIn("](", url)
            self.assertNotIn("\\", url)
            self.assertEqual(url.lower().count("http"), 1)

    def test_audit_flags_a_markdown_wrapped_link_in_output(self) -> None:
        from generalized_card.audit import _malformed_urls, _urls

        bad = "A link https://youtu.be/x](https://youtu.be/x here"
        self.assertTrue(_malformed_urls(bad))
        self.assertEqual(_urls(bad), ["https://youtu.be/x", "https://youtu.be/x"])
        self.assertFalse(_malformed_urls("see https://www.dpreview.com/reviews/x here"))

    def test_reference_link_inventory_excludes_media_and_overlong_urls(self) -> None:
        from generalized_card.reference_link import build_reference_link_inventory

        threads = [
            {
                "comments": [
                    {"body": "see https://www.dpreview.com/a for the samples"},
                    {"body": "photo https://preview.redd.it/abc.jpeg?s=1 here"},
                    {"body": "long https://x.example/" + "z" * 400},
                    {"body": "no link at all in this one"},
                ]
            }
        ]
        inv = build_reference_link_inventory(threads)
        self.assertEqual(inv["urls"], ["https://www.dpreview.com/a"])
        self.assertEqual(inv["reference_comment_count"], 4)
        self.assertAlmostEqual(inv["carrying_comment_share"], 0.25)

    def test_planners_receive_non_test_reference_text_but_writer_does_not(self) -> None:
        profile = {
            "profile_sha256": "prompt-profile",
            "perspectives": [],
            "reference_viewpoints": [
                {
                    "reference_id": "R00001",
                    "source_thread_hash": "non-test",
                    "thread_title": "Camera handling reference",
                    "thread_context": "A non-test discussion",
                    "text": "NON_TEST_VIEWPOINT_MARKER compare handling before buying",
                    "depth": 0,
                    "parent_scope": "op",
                    "word_count": 6,
                    "surface_role": "full_answer",
                }
            ],
        }
        backend = SimpleNamespace(
            GENERALIZED_DOMAIN_PROFILE=profile,
            GENERALIZED_DOMAIN_CLAIM_MODE="planned",
            GENERALIZED_ACTIVE_REFERENCE_TEMPLATE={
                "comment_count": 1,
                "story_count": 0,
                "polite_rate": 1.0,
                "impolite_rate": 0.0,
                "neutral_rate": 0.0,
                "dominant_emotion_counts": {"curiosity": 1},
            },
            render_top_counts=lambda memory: "none",
            compact=lambda value, limit: str(value)[:limit],
        )
        seed = SimpleNamespace(
            title="Camera grip question",
            body="Is the grip comfortable?",
            content="Camera grip question\nIs the grip comfortable?",
        )
        target = SimpleNamespace(
            target_comments=1,
            max_depth_goal=1,
            top_level_comments=1,
            shape_label="quiet",
            length_mix_note="one short comment",
        )
        planner = prompts.planner_prompt(
            self.config,
            backend,
            seed_post=seed,
            target=target,
            matched_real_thread={
                "comments": [{"body": "MATCHED_TEST_SECRET", "depth": 0}]
            },
            matched_real_comments=1,
            global_memory={},
        )
        comment_planner = prompts.comment_planner_prompt(
            self.config,
            backend,
            seed_post=seed,
            target=target,
            branches=[
                SimpleNamespace(
                    branch_id=1,
                    branch_goal="compare handling",
                    anchor_quote="grip",
                    allowed_functions=("reaction",),
                    content_angles=("fit_use_case",),
                )
            ],
            matched_real_thread={"comments": [{"body": "MATCHED_TEST_SECRET"}]},
            comments=[{"body": "MATCHED_TEST_SECRET", "depth": 0}],
            all_comments=[{"body": "MATCHED_TEST_SECRET", "depth": 0}],
        )
        self.assertIn("NON_TEST_VIEWPOINT_MARKER", planner)
        self.assertIn("NON_TEST_VIEWPOINT_MARKER", comment_planner)
        self.assertNotIn("MATCHED_TEST_SECRET", planner)
        self.assertNotIn("MATCHED_TEST_SECRET", comment_planner)
        self.assertIn("tone_class exact counts", planner)
        self.assertIn("polite=1", comment_planner)
        self.assertIn('"tone_class"', comment_planner)
        self.assertIn("same capacity function as validation", comment_planner)
        self.assertNotIn("per 35 words", comment_planner)
        self.assertNotIn("capped at 16 beats", comment_planner)
        self.assertIn('"affect_role"', comment_planner)
        self.assertEqual(
            comment_planner.count(
                "Synthesize an ordinary, non-verifiable first-person sequence"
            ),
            1,
        )
        self.assertNotIn(
            "Never include usernames, URLs, hidden anecdotes, or facts absent",
            comment_planner,
        )
        self.assertIn("These rows are also this domain's knowledge", comment_planner)
        self.assertIn("Give most substantive slots", comment_planner)
        self.assertIn('"reply_delta_type": "none"', comment_planner)
        self.assertIn("These are independent root slots", comment_planner)
        self.assertNotIn("Parent-local delta contracts for this batch", comment_planner)
        self.assertNotIn("operational_test: an observation or check", comment_planner)

        backend.GENERALIZED_DOMAIN_CLAIM_MODE = "off"
        claim_off = prompts.comment_planner_prompt(
            self.config,
            backend,
            seed_post=seed,
            target=target,
            branches=[
                SimpleNamespace(
                    branch_id=1,
                    branch_goal="compare handling",
                    anchor_quote="grip",
                    allowed_functions=("reaction",),
                    content_angles=("fit_use_case",),
                )
            ],
            matched_real_thread={"comments": [{"body": "MATCHED_TEST_SECRET"}]},
            comments=[{"body": "MATCHED_TEST_SECRET", "depth": 0}],
            all_comments=[{"body": "MATCHED_TEST_SECRET", "depth": 0}],
        )
        self.assertIn("NON_TEST_VIEWPOINT_MARKER", claim_off)
        self.assertIn('"domain_claim": "none"', claim_off)
        self.assertIn("information that the Writer will not receive", claim_off)
        self.assertNotIn("These rows are also this domain's knowledge", claim_off)
        self.assertNotIn("Give most substantive slots", claim_off)

        backend.GENERALIZED_DOMAIN_CLAIM_MODE = "selective"
        profile["reference_viewpoints"][0]["word_count"] = 30
        profile["reference_viewpoints"][0]["text"] = (
            "NON_TEST_VIEWPOINT_MARKER electronic aperture control remains "
            "available through the adapter while focus behavior depends on "
            "the body and lens combination used in practice"
        )
        profile["profile_sha256"] = "prompt-profile-selective"
        anonymous_long_slot = {
            "body": "MATCHED_TEST_SECRET " * 30,
            "depth": 0,
        }
        selective = prompts.comment_planner_prompt(
            self.config,
            backend,
            seed_post=seed,
            target=target,
            branches=[
                SimpleNamespace(
                    branch_id=1,
                    branch_goal="compare handling",
                    anchor_quote="grip",
                    allowed_functions=("reaction",),
                    content_angles=("fit_use_case",),
                )
            ],
            matched_real_thread={"comments": [anonymous_long_slot]},
            comments=[anonymous_long_slot],
            all_comments=[anonymous_long_slot],
        )
        self.assertIn("Selective factual slots in this request: S1", selective)
        self.assertIn("paired R# row", selective)
        self.assertIn("NON_TEST_VIEWPOINT_MARKER", selective)
        self.assertNotIn("MATCHED_TEST_SECRET", selective)
        self.assertNotIn("Give most substantive slots", selective)

        rendered_refs = render_reference_viewpoints(
            profile,
            seed_title=seed.title,
            seed_body=seed.body,
            limit=1,
        )
        self.assertIn("NON_TEST_VIEWPOINT_MARKER", rendered_refs)

    def test_planner_receives_matched_structure_without_real_comment_text(self) -> None:
        backend = SimpleNamespace(
            GENERALIZED_DOMAIN_PROFILE={},
            render_top_counts=lambda memory: "none",
            compact=lambda value, limit: str(value)[:limit],
        )
        rendered = prompts.planner_prompt(
            self.config,
            backend,
            seed_post=SimpleNamespace(
                title="Camera grip question",
                body="Is the grip comfortable?",
                content="Camera grip question\nIs the grip comfortable?",
            ),
            target=SimpleNamespace(
                target_comments=2,
                max_depth_goal=1,
                top_level_comments=1,
                shape_label="normal",
                length_mix_note="one short and one medium comment",
            ),
            matched_real_thread={
                "comments": [
                    {
                        "comment_id": "c1",
                        "parent_id": None,
                        "body": "CONFIDENTIAL_REAL_COMMENT $9,999",
                    }
                ]
            },
            matched_real_comments=1,
            global_memory={},
        )
        self.assertNotIn("CONFIDENTIAL_REAL_COMMENT", rendered)
        self.assertNotIn("$9,999", rendered)
        self.assertIn("matched real structural sample", rendered.lower())

    def test_generator_adapter_changes_only_declared_core_extensions(self) -> None:
        module = load_generator_backend()
        before = {name: getattr(module, name) for name in CORE_ALGORITHM_SYMBOLS}
        configured = configure_generator_backend(module, self.config)
        for name, original in before.items():
            if name in {
                "generate_post_from_tasks",
                "generate_writer_text_with_guards",
                "writer_token_cap",
            }:
                self.assertIsNot(getattr(configured, name), original, name)
            else:
                self.assertIs(getattr(configured, name), original, name)
        self.assertEqual(
            configured.GENERALIZED_CARD_PARITY["changed_core_algorithm_symbols"],
            [
                "generate_post_from_tasks",
                "generate_writer_text_with_guards",
                "writer_token_cap",
            ],
        )
        self.assertEqual(
            configured.GENERALIZED_CARD_PARITY["unexpected_core_algorithm_symbols"],
            [],
        )
        self.assertEqual(
            configured.GENERALIZED_CARD_PARITY["unexpected_backend_functions"],
            [],
        )

    def test_generalized_rebalance_runs_card_surface_core_and_preserves_invariants(
        self,
    ) -> None:
        module = load_generator_backend()
        calls = []

        def card_surface_rebalance(tasks, **kwargs):
            calls.append(kwargs)
            return [
                replace(
                    tasks[0],
                    speaker_role="datapoint_only",
                    payload_type="fragment_datapoint",
                    comment_function="personal_datapoint",
                    length_bucket="micro",
                    real_word_count=2,
                    semantic_move="incorrect replacement move",
                    claim_key="incorrect_claim",
                    perspective_id="incorrect_perspective",
                    surface_skeleton="datapoint-first local answer",
                )
            ]

        module.rebalance_tasks_for_diversity = card_surface_rebalance
        configured = configure_generator_backend(module, self.config)
        task = configured.CommentTask(
            local_task_id=1,
            local_parent_task_id=None,
            depth=0,
            branch_id=1,
            branch_goal="one local point",
            visible_scope="seed",
            local_anchor="visible detail",
            comment_function="explanation_analysis",
            content_angle="fit_use_case",
            evidence_mode="small_observation",
            story_mode="no_story",
            voice="casual_neutral",
            payload_type="soft_helpful",
            length_bucket="medium",
            speaker_role="side_observer",
            utterance_mode="small_observation",
            surface_texture="plain",
            allow_first_person_frame=True,
            allow_uncertainty_frame=True,
            planner_intent="preserve this exact local move",
            must_not_do="Do not add another claim.",
            semantic_move="distinguish the visible handling detail",
            local_topic="handling",
            reply_relation="adds_datapoint",
            stance="neutral",
            detail_focus="visible handling detail",
            avoid_repeating="the earlier size verdict",
            claim_key="handling_detail",
            claim_family="firsthand_datapoint",
            opening_style="concrete detail first",
            context_aperture="seed_gist_only",
        )
        revised = configured.rebalance_tasks_for_diversity(
            [task],
            rng=configured.random.Random(1),
            advisor_max_share=0.1,
            question_max_share=0.1,
            micro_target_share=0.1,
            short_max_share=0.1,
            social_noise_min_share=0.1,
            gratitude_min_share=0.1,
            tone_harsh_max_share=0.1,
            tone_calm_min_share=0.1,
            tone_personal_min_share=0.0,
            tone_polite_min_share=0.1,
        )
        self.assertEqual(len(calls), 0)
        self.assertEqual(revised[0].semantic_move, task.semantic_move)
        self.assertEqual(revised[0].claim_key, task.claim_key)
        self.assertEqual(revised[0].perspective_id, task.perspective_id)
        self.assertEqual(revised[0].length_bucket, task.length_bucket)
        self.assertEqual(revised[0].real_word_count, task.real_word_count)
        self.assertEqual(revised[0].payload_type, task.payload_type)
        self.assertEqual(revised[0].comment_function, task.comment_function)
        self.assertEqual(revised[0].surface_skeleton, task.surface_skeleton)
        self.assertTrue(revised[0].allow_first_person_frame)
        self.assertTrue(revised[0].allow_uncertainty_frame)

    def test_internal_planner_ids_never_become_writer_anchors(self) -> None:
        with patch(
            "generalized_card.backend.load_domain_profile",
            return_value={
                "perspectives": [{"perspective_id": "P03", "label": "autofocus"}]
            },
        ):
            module = configure_generator_backend(load_generator_backend(), self.config)
        seed = module.SeedPost(
            index=0,
            title="Canon EOS R5 autofocus question",
            body="How does subject tracking behave?",
            content="Canon EOS R5 autofocus question\nHow does subject tracking behave?",
            source_raw_post_id="seed-0",
            real_num_comments=4,
            metadata={},
        )
        anchors = module.build_concrete_anchors_for_task(
            real_body="",
            seed_post=seed,
            branch=SimpleNamespace(
                anchor_quote="compare P03 to S12",
                branch_goal="reply through B02 without repeating the AF point",
            ),
            planned={
                "semantic_move": "ask whether P03 changes the AF result",
                "perspective_id": "P03",
                "claim_key": "P03_af_comparison",
                "claim_family": "technical_explanation",
                "domain_intent": "compare the visible autofocus behavior",
                "reply_relation": "answers_parent",
            },
            anchor="P03 and S12",
            parent_task=None,
        )
        rendered = " | ".join(anchors)
        self.assertIn("Canon", rendered)
        self.assertNotRegex(rendered, r"\b(?:P03|S12|B02)\b")

        task = SimpleNamespace(surface_skeleton="", concrete_anchors=anchors)
        self.assertTrue(
            module.contains_planner_skeleton_residue("P03 was the same deal", task)
        )
        self.assertFalse(
            module.contains_planner_skeleton_residue("The R5 was the same deal", task)
        )

    def test_cross_batch_comment_plan_ledger_is_available_to_next_prompt(self) -> None:
        module = SimpleNamespace(
            GENERALIZED_COMMENT_PLAN_HISTORY=[],
            real_comment_keys=lambda row: (str(row.get("comment_id") or ""),),
        )
        ledgers = {}
        seen_history = []

        def original(**kwargs):
            seen_history.append(list(module.GENERALIZED_COMMENT_PLAN_HISTORY))
            sample_id = int(kwargs["sample_offset"]) + 1
            return {
                sample_id: {
                    "claim_key": f"claim_{sample_id}",
                    "perspective_id": "P01",
                    "reply_relation": "answers_parent",
                    "semantic_move": f"move {sample_id}",
                    "local_topic": "autofocus",
                }
            }

        wrapped = _comment_planner_batch_with_history(module, original, ledgers)
        seed = SimpleNamespace(source_raw_post_id="seed-1", index=1, title="seed")
        all_comments = [
            {"comment_id": f"c{index}", "parent_id": None} for index in range(1, 20)
        ]
        wrapped(
            seed_post=seed,
            sample_offset=0,
            comments=[all_comments[0]],
            all_comments=all_comments,
        )
        wrapped(
            seed_post=seed,
            sample_offset=18,
            comments=[all_comments[18]],
            all_comments=all_comments,
        )
        self.assertEqual(seen_history[0], [])
        self.assertEqual(seen_history[1][0]["claim_key"], "claim_1")

        prompt = prompts.comment_planner_prompt(
            self.config,
            SimpleNamespace(
                compact=lambda value, limit: str(value)[:limit],
                GENERALIZED_DOMAIN_PROFILE={},
            ),
            seed_post=SimpleNamespace(
                title="Camera question",
                body="Autofocus issue",
                content="Autofocus issue",
            ),
            target=SimpleNamespace(
                target_comments=2, max_depth_goal=1, shape_label="normal"
            ),
            branches=[
                SimpleNamespace(
                    branch_id=1,
                    branch_goal="autofocus behavior",
                    anchor_quote="autofocus",
                    allowed_functions=("reaction",),
                    content_angles=("setup_troubleshooting",),
                )
            ],
            matched_real_thread=None,
            comments=[
                {"comment_id": "c1", "parent_id": None, "depth": 0, "body": "hidden"}
            ],
            all_comments=[
                {"comment_id": "c1", "parent_id": None, "depth": 0, "body": "hidden"}
            ],
            prior_plans=module.GENERALIZED_COMMENT_PLAN_HISTORY,
        )
        self.assertIn("claim=claim_1", prompt)
        self.assertIn("Do not hide a repeated semantic move", prompt)

    def test_comment_plan_normalizer_preserves_reference_id(self) -> None:
        module = load_generator_backend()
        plans = module.normalize_comment_move_plans(
            {
                "comment_plans": [
                    {
                        "sample_id": 1,
                        "reference_id": "R00123",
                        "branch_id": 1,
                        "semantic_move": "compare the visible handling tradeoff",
                        "perspective_id": "P02",
                    }
                ]
            },
            branches=[SimpleNamespace(branch_id=1)],
        )
        self.assertEqual(plans[1]["reference_id"], "R00123")

    def test_comment_plan_normalizer_accepts_prompt_style_sample_ids(self) -> None:
        module = load_generator_backend()
        plans = module.normalize_comment_move_plans(
            {
                "comment_plans": [
                    {
                        "sample_id": "S1",
                        "branch_id": 1,
                        "semantic_move": "make one local observation",
                    },
                    {
                        "sample_id": "s38",
                        "branch_id": 1,
                        "semantic_move": "add one distinct consequence",
                    },
                ]
            },
            branches=[SimpleNamespace(branch_id=1)],
        )
        self.assertEqual(sorted(plans), [1, 38])

    def test_planner_enrichments_preserve_prompt_style_sample_ids(self) -> None:
        payload = {
            "comment_plans": [
                {
                    "sample_id": "S12",
                    "tone_class": "neutral",
                    "affect_role": "curiosity",
                    "decision_boundary": "separate the timing condition from the broader outcome",
                    "development_plan": [
                        "name the local condition",
                        "trace its consequence",
                    ],
                    "actor": {"attention_focus": "the local condition"},
                }
            ]
        }
        normalized = {12: {"sample_id": "12"}}
        enrich_distribution_plan_fields(payload, normalized)
        enrich_development_plan_fields(payload, normalized)
        enriched_actor = enrich_normalized_plans(payload, normalized)

        self.assertEqual(parse_sample_id("s12"), 12)
        self.assertEqual(
            normalized[12]["decision_boundary"],
            "separate the timing condition from the broader outcome",
        )
        self.assertEqual(
            normalized[12]["development_plan"],
            "name the local condition || trace its consequence",
        )
        self.assertEqual(
            enriched_actor[12]["actor_attention_focus"],
            "the local condition",
        )

    def test_structural_route_overrides_planner_branch_for_replies(self) -> None:
        module = SimpleNamespace(
            real_comment_keys=lambda row: (str(row.get("comment_id") or ""),)
        )
        branches = [
            SimpleNamespace(
                branch_id=1,
                branch_goal="root one",
                perspective_id="P01",
                branch_exclusion="leave root two alone",
                owned_decision_subject="first condition",
                decision_boundary="first condition",
            ),
            SimpleNamespace(
                branch_id=2,
                branch_goal="root two",
                perspective_id="P02",
                branch_exclusion="leave root one alone",
                owned_decision_subject="second condition",
                decision_boundary="second condition",
            ),
        ]
        comments = [
            {"comment_id": "c1", "parent_id": None},
            {"comment_id": "c2", "parent_id": None},
            {"comment_id": "c3", "parent_id": "t1_c2"},
        ]
        plans = {3: {"branch_id": "1", "semantic_move": "add a consequence"}}
        _annotate_plan_metadata(
            module,
            plans,
            {"all_comments": comments, "branches": branches},
        )
        self.assertEqual(plans[3]["branch_id"], "2")
        self.assertEqual(plans[3]["owned_decision_subject"], "second condition")

    def test_comment_planner_renders_explicit_parent_delta_contract(self) -> None:
        prompt = prompts.comment_planner_prompt(
            self.config,
            SimpleNamespace(
                compact=lambda value, limit: str(value)[:limit],
                GENERALIZED_DOMAIN_PROFILE={},
                GENERALIZED_ACTIVE_REFERENCE_TEMPLATE={},
                GENERALIZED_ACTIVE_SLOT_DISTRIBUTION_SCHEDULE={
                    "defaults": {"story_mode": "no_story"}
                },
            ),
            seed_post=SimpleNamespace(
                title="Camera question",
                body="Visible seed detail",
                content="Visible seed detail",
            ),
            target=SimpleNamespace(
                target_comments=2, max_depth_goal=1, shape_label="normal"
            ),
            branches=[
                SimpleNamespace(
                    branch_id=1,
                    branch_goal="root condition",
                    anchor_quote="visible detail",
                    perspective_id="P01",
                    branch_exclusion="other condition",
                    owned_decision_subject="root condition",
                    decision_boundary="root condition",
                )
            ],
            matched_real_thread=None,
            comments=[
                {"comment_id": "c2", "parent_id": "t1_c1", "depth": 1, "body": "hidden"}
            ],
            all_comments=[
                {"comment_id": "c1", "parent_id": None, "depth": 0, "body": "hidden"},
                {
                    "comment_id": "c2",
                    "parent_id": "t1_c1",
                    "depth": 1,
                    "body": "hidden",
                },
            ],
            sample_offset=1,
            prior_plans=[
                {
                    "sample_id": "1",
                    "semantic_move": "state the parent proposition",
                    "decision_boundary": "parent boundary",
                }
            ],
        )
        self.assertIn("Plan only the direct replies", prompt)
        self.assertIn(
            "Parent semantic move to exclude: state the parent proposition", prompt
        )
        self.assertIn("reply_novelty_anchor", prompt)
        self.assertIn("social_close", prompt)
        # social_close is offered only to a slot whose schedule allows it, so an
        # unallowed row simply omits it from its allowed increment list.
        self.assertIn("Allowed reply_delta_type:", prompt)
        self.assertIn("Story contract: no_story", prompt)
        allowed_line = next(
            line
            for line in prompt.splitlines()
            if line.strip().startswith("Allowed reply_delta_type:")
        )
        self.assertNotIn("social_close", allowed_line)
        self.assertNotIn("corroborating_datapoint", allowed_line)

    def test_direct_reply_planner_exposes_sibling_coverage(self) -> None:
        prompt = prompts.comment_planner_prompt(
            self.config,
            SimpleNamespace(
                compact=lambda value, limit: str(value)[:limit],
                GENERALIZED_DOMAIN_PROFILE={},
                GENERALIZED_ACTIVE_REFERENCE_TEMPLATE={},
                GENERALIZED_ACTIVE_SLOT_DISTRIBUTION_SCHEDULE={},
                GENERALIZED_WRITER_ROUTE_LOCK="own_words",
                CLAIM_FAMILIES=("direct_answer",),
            ),
            seed_post=SimpleNamespace(
                title="Camera question",
                body="Visible seed detail",
                content="Visible seed detail",
            ),
            target=SimpleNamespace(
                target_comments=3,
                max_depth_goal=1,
                shape_label="normal",
            ),
            branches=[],
            matched_real_thread=None,
            comments=[
                {
                    "comment_id": "c2",
                    "parent_id": "t1_c1",
                    "depth": 1,
                    "body": "hidden",
                },
                {
                    "comment_id": "c3",
                    "parent_id": "t1_c1",
                    "depth": 1,
                    "body": "hidden",
                },
            ],
            all_comments=[
                {"comment_id": "c1", "parent_id": None, "depth": 0, "body": "hidden"},
                {
                    "comment_id": "c2",
                    "parent_id": "t1_c1",
                    "depth": 1,
                    "body": "hidden",
                },
                {
                    "comment_id": "c3",
                    "parent_id": "t1_c1",
                    "depth": 1,
                    "body": "hidden",
                },
            ],
            sample_offset=1,
            prior_plans=[
                {
                    "sample_id": "1",
                    "semantic_move": "state the parent proposition",
                    "decision_boundary": "parent boundary",
                }
            ],
        )
        self.assertIn("Sibling coverage: S2,S3", prompt)
        self.assertIn("different delta type and novelty object", prompt)

    def test_writer_route_lock_keeps_reply_delta_ahead_of_parent_context(self) -> None:
        task = SimpleNamespace(
            semantic_move="add the exposure-consistency condition",
            decision_boundary="whether exposure remains consistent",
            owned_decision_subject="low-light performance",
            reply_delta="add only exposure consistency as the new condition",
            reply_delta_type="operational_test",
            reply_novelty_anchor="compare exposure across bright and dark parts of the scene",
            parent_semantic_move="the parent says image quality matters in low light",
            local_parent_task_id=2,
            planner_intent="reply with one condition",
        )
        rendered = prompts._semantic_route_lock(
            SimpleNamespace(compact=lambda value, limit: str(value)[:limit]),
            task,
            domain_profile={},
        )
        # The increment and the concrete object both reach the Writer, phrased
        # as plain statements rather than as roles in an argument.
        self.assertIn(task.reply_delta, rendered)
        self.assertIn(task.reply_novelty_anchor, rendered)
        # The block must not hand the Writer the construction it converges on.
        # An earlier wording said "the part that changes the parent" and the
        # Writer echoed "that's the part that..."; rewording that one line was
        # not enough, and the frame was still in 20% of v72 comments against 0
        # of 39,265 matched real tokens. Nothing here may name a "part", a
        # "proposition" or an "increment".
        for banned in (
            "the part that",
            "proposition",
            "increment",
            "decision boundary",
        ):
            self.assertNotIn(banned, rendered.lower(), banned)
        self.assertIn("concrete action or observation", rendered)
        # The parent's proposition is excluded once, by the structured
        # "parent contribution not to restate" field. The route lock must not
        # add a second copy: three separate prohibitions used to put the
        # parent's exact wording in front of the Writer three times per prompt,
        # and generated parent overlap fell to 0.129 against 0.197 in real
        # threads.
        self.assertNotIn(task.parent_semantic_move, rendered)
        self.assertNotIn("Do not state, summarize, or paraphrase", rendered)

    def test_reply_novelty_contract_flags_missing_increment(self) -> None:
        parent = {
            "sample_id": "1",
            "semantic_move": "compare the capability at the stated condition",
            "decision_boundary": "whether the condition is usable",
            "detail_focus": "the stated condition",
            "payload_type": "soft_helpful",
            "comment_function": "explanation_analysis",
        }
        child = {
            "sample_id": "2",
            "parent_sample_id": "1",
            "semantic_move": "say the condition still matters",
            "decision_boundary": "whether the condition is usable",
            "detail_focus": "the stated condition",
            "reply_delta": "say the condition still matters",
            "payload_type": "soft_helpful",
            "comment_function": "explanation_analysis",
            "reply_relation": "answers_parent",
            "perspective_id": "P01",
        }
        report = evaluate_plan_batch(
            {2: child},
            prior_plans=[parent],
            require_reply_novelty=True,
        )
        self.assertIn(
            "reply_increment_conflict",
            {issue.code for issue in report.issues},
        )

    def test_reply_novelty_default_scope_is_parent_only(self) -> None:
        parent = {
            "sample_id": "1",
            "semantic_move": "sets a boundary",
            "decision_boundary": "boundary one",
            "detail_focus": "detail one",
        }
        child = {
            "sample_id": "2",
            "parent_sample_id": "1",
            "semantic_move": "narrows the boundary",
            "decision_boundary": "boundary two",
            "detail_focus": "detail two",
            "reply_delta": "narrows the boundary",
            "reply_delta_type": "scope_limit",
            "reply_novelty_anchor": "a new concrete condition",
        }
        kwargs = dict(
            require_reply_novelty=True,
            semantic_similarity=lambda _left, _right: 0.90,
        )
        default_report = evaluate_plan_batch({2: child}, prior_plans=[parent], **kwargs)
        explicit_report = evaluate_plan_batch(
            {2: child}, prior_plans=[parent], reply_novelty_scope="parent_only", **kwargs
        )
        self.assertEqual(default_report.to_dict(), explicit_report.to_dict())

    def _marker_similarity(
        self, left: dict[str, Any], right: dict[str, Any]
    ) -> float:
        """A keyword-aware similarity stub: 0.90 if both probes share a
        `MARKER_X` token, 0.20 otherwise. A constant-return stub can't test
        which ancestor a reply's novelty anchor actually matched."""

        marker_re = re.compile(r"MARKER_[A-Z]")
        left_markers = set(marker_re.findall(" ".join(str(value) for value in left.values())))
        right_markers = set(marker_re.findall(" ".join(str(value) for value in right.values())))
        return 0.90 if left_markers & right_markers else 0.20

    def test_reply_novelty_parent_only_scope_misses_grandparent_restatement(self) -> None:
        grandparent = {
            "sample_id": "1",
            "semantic_move": "argues MARKER_A applies to this case",
            "decision_boundary": "boundary one MARKER_A",
            "detail_focus": "detail one",
        }
        parent = {
            "sample_id": "2",
            "parent_sample_id": "1",
            "semantic_move": "argues MARKER_B applies instead",
            "decision_boundary": "boundary two MARKER_B",
            "detail_focus": "detail two",
        }
        child = {
            "sample_id": "3",
            "parent_sample_id": "2",
            "semantic_move": "restates MARKER_A again for this case",
            "decision_boundary": "boundary three MARKER_A",
            "detail_focus": "detail three",
            "reply_delta_type": "scope_limit",
            "reply_novelty_anchor": "a plain new condition with no marker",
        }
        report = evaluate_plan_batch(
            {3: child},
            prior_plans=[grandparent, parent],
            require_reply_novelty=True,
            reply_novelty_scope="parent_only",
            semantic_similarity=self._marker_similarity,
        )
        self.assertNotIn(
            "reply_increment_conflict",
            {issue.code for issue in report.issues},
        )

    def test_reply_novelty_chain_scope_catches_grandparent_restatement(self) -> None:
        grandparent = {
            "sample_id": "1",
            "semantic_move": "argues MARKER_A applies to this case",
            "decision_boundary": "boundary one MARKER_A",
            "detail_focus": "detail one",
        }
        parent = {
            "sample_id": "2",
            "parent_sample_id": "1",
            "semantic_move": "argues MARKER_B applies instead",
            "decision_boundary": "boundary two MARKER_B",
            "detail_focus": "detail two",
        }
        child = {
            "sample_id": "3",
            "parent_sample_id": "2",
            "semantic_move": "restates MARKER_A again for this case",
            "decision_boundary": "boundary three MARKER_A",
            "detail_focus": "detail three",
            "reply_delta_type": "scope_limit",
            "reply_novelty_anchor": "a plain new condition with no marker",
        }
        report = evaluate_plan_batch(
            {3: child},
            prior_plans=[grandparent, parent],
            require_reply_novelty=True,
            reply_novelty_scope="chain",
            semantic_similarity=self._marker_similarity,
        )
        issue = next(
            (issue for issue in report.issues if issue.code == "reply_increment_conflict"),
            None,
        )
        self.assertIsNotNone(issue)
        self.assertIn("S1", issue.message)

    def test_reply_novelty_chain_scope_walks_past_two_hops(self) -> None:
        great_grandparent = {
            "sample_id": "1",
            "semantic_move": "argues MARKER_A applies to this case",
            "decision_boundary": "boundary one MARKER_A",
            "detail_focus": "detail one",
        }
        grandparent = {
            "sample_id": "2",
            "parent_sample_id": "1",
            "semantic_move": "argues MARKER_B applies instead",
            "decision_boundary": "boundary two MARKER_B",
            "detail_focus": "detail two",
        }
        parent = {
            "sample_id": "3",
            "parent_sample_id": "2",
            "semantic_move": "argues MARKER_C applies now",
            "decision_boundary": "boundary three MARKER_C",
            "detail_focus": "detail three",
        }
        child = {
            "sample_id": "4",
            "parent_sample_id": "3",
            "semantic_move": "restates MARKER_A again, three hops later",
            "decision_boundary": "boundary four MARKER_A",
            "detail_focus": "detail four",
            "reply_delta_type": "scope_limit",
            "reply_novelty_anchor": "a plain new condition with no marker",
        }
        report = evaluate_plan_batch(
            {4: child},
            prior_plans=[great_grandparent, grandparent, parent],
            require_reply_novelty=True,
            reply_novelty_scope="chain",
            semantic_similarity=self._marker_similarity,
        )
        issue = next(
            (issue for issue in report.issues if issue.code == "reply_increment_conflict"),
            None,
        )
        self.assertIsNotNone(issue)
        self.assertIn("S1", issue.message)

    def test_reply_novelty_conflict_message_has_no_domain_vocabulary(self) -> None:
        parent = {
            "sample_id": "1",
            "semantic_move": "argues MARKER_A applies to this case",
            "decision_boundary": "boundary one MARKER_A",
            "detail_focus": "detail one",
        }
        child = {
            "sample_id": "2",
            "parent_sample_id": "1",
            "semantic_move": "restates MARKER_A again",
            "decision_boundary": "boundary two MARKER_A",
            "detail_focus": "detail two",
            "reply_delta_type": "scope_limit",
            "reply_novelty_anchor": "MARKER_A restated as a new condition",
        }
        banned = (
            "gear",
            "camera",
            "lens",
            "shot",
            "shoot",
            "photo",
            "iso",
            "aperture",
            "specification",
            "megapixel",
            "sensor",
            "product",
        )
        for scope in ("parent_only", "chain"):
            report = evaluate_plan_batch(
                {2: child},
                prior_plans=[parent],
                require_reply_novelty=True,
                reply_novelty_scope=scope,
                semantic_similarity=self._marker_similarity,
            )
            issue = next(
                (issue for issue in report.issues if issue.code == "reply_increment_conflict"),
                None,
            )
            self.assertIsNotNone(issue)
            for word in banned:
                self.assertNotIn(word, issue.message.lower(), f"{word!r} in {issue.message!r}")

    def test_generalized_plan_normalizer_carries_long_form_plan_into_task(self) -> None:
        module = configure_generator_backend(load_generator_backend(), self.config)
        branches = [
            module.BranchPlan(
                branch_id=1,
                anchor_quote="visible workflow detail",
                anchor_source="seed",
                detour_type="none",
                branch_goal="explain one workflow consequence",
                allowed_functions=("explanation_analysis",),
                evidence_modes=("small_observation",),
                tone_palette=("casual_neutral",),
                story_modes=("no_story",),
                content_angles=("fit_use_case",),
            )
        ]
        normalized = module.normalize_comment_move_plans(
            {
                "comment_plans": [
                    {
                        "sample_id": 1,
                        "branch_id": 1,
                        "payload_type": "soft_helpful",
                        "comment_function": "explanation_analysis",
                        "semantic_move": "explain one visible workflow consequence",
                        "development_plan": [
                            "start from the visible constraint",
                            "trace the workflow consequence",
                            "add one boundary",
                        ],
                    }
                ]
            },
            branches=branches,
        )
        self.assertEqual(
            normalized[1]["development_plan"],
            "start from the visible constraint || trace the workflow consequence || add one boundary",
        )

        tasks = module.expand_matched_real_sample_to_tasks(
            branches=branches,
            target=module.ThreadTarget(1, 1, 0, "quiet", "matched"),
            seed_post=module.SeedPost(
                index=0,
                title="visible workflow question",
                body="Which setup fits this workflow?",
                content="visible workflow question Which setup fits this workflow?",
                source_raw_post_id="seed",
                real_num_comments=1,
                metadata={},
            ),
            matched_real_thread={
                "comments": [
                    {
                        "body": " ".join(f"word{index}" for index in range(260)),
                        "comment_id": "real_1",
                        "comment_fullname": "t1_real_1",
                        "parent_id": "t3_seed",
                    }
                ]
            },
            matched_real_comments=100,
            comment_plans=normalized,
            rng=random.Random(42),
        )
        self.assertEqual(tasks[0].development_plan, normalized[1]["development_plan"])
        rendered = module.build_writer_prompt(
            profile="gpt54_reddit_writer",
            seed_post=module.SeedPost(
                index=0,
                title="visible workflow question",
                body="Which setup fits this workflow?",
                content="visible workflow question Which setup fits this workflow?",
                source_raw_post_id="seed",
                real_num_comments=1,
                metadata={},
            ),
            task=module.finalize_rebalanced_task(tasks[0]),
            parent_comment=None,
            previous_comments=[],
        )
        self.assertIn("development sequence", rendered)
        self.assertIn("trace the workflow consequence", rendered)

    def test_plan_quality_detects_semantic_collision_behind_new_claim_keys(
        self,
    ) -> None:
        plans = {
            1: {
                "reference_id": "R00001",
                "claim_key": "first_name",
                "perspective_id": "P02",
                "semantic_move": "compare the weight tradeoff for daily carrying",
                "local_topic": "daily carrying weight",
                "detail_focus": "weight versus convenience",
                "domain_intent": "weigh carrying comfort against performance",
                "payload_type": "advice",
                "comment_function": "recommendation_advice",
                "content_angle": "fit_use_case",
                "claim_family": "recommendation",
                "reply_relation": "adds_datapoint",
                "stance": "mixed",
                "evidence_mode": "technical_or_policy_reasoning",
            },
            2: {
                "reference_id": "R00002",
                "claim_key": "renamed_claim",
                "perspective_id": "P02",
                "semantic_move": "compare daily carry weight as a convenience tradeoff",
                "local_topic": "weight for everyday carrying",
                "detail_focus": "convenience versus weight",
                "domain_intent": "weigh carrying comfort against performance",
                "payload_type": "advice",
                "comment_function": "recommendation_advice",
                "content_angle": "fit_use_case",
                "claim_family": "recommendation",
                "reply_relation": "adds_datapoint",
                "stance": "mixed",
                "evidence_mode": "technical_or_policy_reasoning",
            },
        }
        report = evaluate_plan_batch(
            plans,
            similarity_threshold=0.55,
        )
        self.assertEqual(report.colliding_samples, (2,))
        self.assertIn("semantic_collision", {issue.code for issue in report.issues})

    def test_plan_quality_embedding_catches_lexically_distinct_paraphrase(self) -> None:
        plans = {
            1: {
                "reference_id": "R1",
                "claim_key": "routine_keeper",
                "perspective_id": "P06",
                "semantic_move": "choose what fits an ordinary routine",
                "local_topic": "daily workflow",
                "detail_focus": "habit",
                "domain_intent": "evaluate practical fit",
                "payload_type": "advice",
                "comment_function": "recommendation_advice",
            },
            2: {
                "reference_id": "R2",
                "claim_key": "novelty_wears_off",
                "perspective_id": "P10",
                "semantic_move": "prefer the option a person keeps reaching for",
                "local_topic": "long term ownership",
                "detail_focus": "continued use",
                "domain_intent": "judge durable satisfaction",
                "payload_type": "personal_story",
                "comment_function": "personal_datapoint",
            },
        }
        report = evaluate_plan_batch(
            plans,
            similarity_threshold=0.95,
            embedding_similarity_threshold=0.82,
            semantic_similarity=lambda _left, _right: 0.88,
        )
        self.assertEqual(report.colliding_samples, (2,))
        self.assertIn("semantic_collision", {issue.code for issue in report.issues})

    def test_duplicate_reference_is_feedback_not_semantic_collision(self) -> None:
        plans = {
            1: {
                "reference_id": "R1",
                "claim_key": "price",
                "perspective_id": "P03",
                "semantic_move": "compare the price",
                "local_topic": "cost",
                "detail_focus": "budget",
                "domain_intent": "judge value",
                "payload_type": "advice",
                "comment_function": "recommendation_advice",
            },
            2: {
                "reference_id": "R1",
                "claim_key": "compatibility",
                "perspective_id": "P07",
                "semantic_move": "check whether the existing setup works",
                "local_topic": "compatibility",
                "detail_focus": "current equipment",
                "domain_intent": "test setup compatibility",
                "payload_type": "narrow_question",
                "comment_function": "question_followup",
            },
        }
        report = evaluate_plan_batch(
            plans,
            similarity_threshold=0.95,
        )
        self.assertEqual(report.colliding_samples, ())
        self.assertTrue(report.healthy)
        self.assertIn("duplicate_reference", {issue.code for issue in report.issues})

    def test_plan_quality_feedback_repair_is_selected_before_writer(self) -> None:
        module = SimpleNamespace(
            GENERALIZED_COMMENT_PLAN_HISTORY=[],
            GENERALIZED_COMMENT_PLAN_FEEDBACK="",
            GENERALIZED_COMMENT_PLAN_REPORTS=[],
            GENERALIZED_DOMAIN_PROFILE={
                "perspectives": [
                    {"perspective_id": "P01"},
                    {"perspective_id": "P02"},
                ]
            },
            GENERALIZED_PLAN_QUALITY_CONFIG={
                "repair_rounds": 1,
                "similarity_threshold": 0.55,
                "max_collision_rate": 0.10,
                "max_perspective_share": 0.60,
                "strict": True,
            },
            real_comment_keys=lambda row: (str(row.get("comment_id") or ""),),
        )
        calls = []

        def original(**call_kwargs):
            calls.append(module.GENERALIZED_COMMENT_PLAN_FEEDBACK)
            if len(calls) == 1:
                return {
                    1: {
                        "reference_id": "R00001",
                        "claim_key": "first",
                        "perspective_id": "P01",
                        "semantic_move": "compare daily carrying weight",
                        "local_topic": "daily carrying weight",
                        "detail_focus": "carrying weight",
                        "domain_intent": "compare carrying weight",
                        "payload_type": "advice",
                        "comment_function": "recommendation_advice",
                    },
                    2: {
                        "reference_id": "R00001",
                        "claim_key": "second",
                        "perspective_id": "P01",
                        "semantic_move": "compare weight for daily carrying",
                        "local_topic": "daily carrying weight",
                        "detail_focus": "carrying weight",
                        "domain_intent": "compare carrying weight",
                        "payload_type": "advice",
                        "comment_function": "recommendation_advice",
                    },
                    99: {
                        "reference_id": "R99999",
                        "claim_key": "out_of_batch",
                        "perspective_id": "P02",
                        "semantic_move": "an out of batch move",
                        "local_topic": "out of batch",
                        "detail_focus": "out of batch",
                        "domain_intent": "out of batch",
                        "payload_type": "advice",
                        "comment_function": "recommendation_advice",
                    },
                }
            self.assertEqual(call_kwargs["sample_offset"], 1)
            self.assertEqual(
                [row["comment_id"] for row in call_kwargs["comments"]],
                ["c2"],
            )
            return {
                2: {
                    "reference_id": "R00002",
                    "claim_key": "existing_setup",
                    "perspective_id": "P02",
                    "semantic_move": "ask what existing setup must remain compatible",
                    "local_topic": "existing setup",
                    "detail_focus": "compatibility constraint",
                    "domain_intent": "clarify the setup dependency",
                    "payload_type": "narrow_question",
                    "comment_function": "question_followup",
                },
            }

        wrapped = _comment_planner_batch_with_history(module, original, {})
        selected = wrapped(
            seed_post=SimpleNamespace(
                source_raw_post_id="seed-repair", index=0, title="seed"
            ),
            sample_offset=0,
            comments=[
                {"comment_id": "c1", "parent_id": None},
                {"comment_id": "c2", "parent_id": None},
            ],
            all_comments=[
                {"comment_id": "c1", "parent_id": None},
                {"comment_id": "c2", "parent_id": None},
            ],
        )
        self.assertEqual(len(calls), 2)
        self.assertIn("Regenerate only the displayed S#", calls[1])
        self.assertEqual(selected[1]["claim_key"], "first")
        self.assertEqual(selected[2]["reference_id"], "R00002")
        self.assertNotIn(99, selected)
        self.assertEqual(
            module.GENERALIZED_COMMENT_PLAN_REPORTS[-1]["repair_attempts"], 1
        )
        self.assertEqual(
            module.GENERALIZED_COMMENT_PLAN_REPORTS[-1][
                "unexpected_sample_ids_discarded"
            ],
            [99],
        )
        self.assertEqual(
            module.GENERALIZED_COMMENT_PLAN_REPORTS[-1]["repair_strategy"],
            "targeted_slot_with_blocking_field_merge",
        )
        self.assertEqual(
            module.GENERALIZED_COMMENT_PLAN_REPORTS[-1][
                "initial_slot_contract_overrides"
            ],
            [],
        )

    def test_missing_planner_slots_use_bounded_structural_completion(self) -> None:
        module = SimpleNamespace(
            GENERALIZED_COMMENT_PLAN_HISTORY=[],
            GENERALIZED_COMMENT_PLAN_FEEDBACK="",
            GENERALIZED_COMMENT_PLAN_REPORTS=[],
            GENERALIZED_DOMAIN_PROFILE={"perspectives": [{"perspective_id": "P01"}]},
            GENERALIZED_PLAN_QUALITY_CONFIG={
                "repair_rounds": 0,
                "schema_recovery_rounds": 1,
                "similarity_threshold": 0.95,
                "max_collision_rate": 1.0,
                "max_perspective_share": 1.0,
                "strict": False,
            },
            real_comment_keys=lambda row: (str(row.get("comment_id") or ""),),
        )
        calls: list[tuple[int, tuple[str, ...], str]] = []

        def original(**call_kwargs):
            offset = int(call_kwargs.get("sample_offset") or 0)
            comment_ids = tuple(row["comment_id"] for row in call_kwargs["comments"])
            calls.append(
                (offset, comment_ids, module.GENERALIZED_COMMENT_PLAN_FEEDBACK)
            )
            sample_ids = range(offset + 1, offset + len(comment_ids) + 1)
            if len(calls) == 1:
                # Reproduce the observed provider response: a 24-slot request
                # returned a valid JSON object containing only S1-S6.
                sample_ids = range(1, 7)
            return {
                sample_id: {
                    "reference_id": f"R{sample_id}",
                    "claim_key": f"claim_{sample_id}",
                    "perspective_id": "P01",
                    "semantic_move": f"perform distinct local move {sample_id}",
                    "local_topic": f"local topic {sample_id}",
                    "detail_focus": f"detail {sample_id}",
                    "domain_intent": f"intent {sample_id}",
                    "payload_type": "advice",
                    "comment_function": "recommendation_advice",
                }
                for sample_id in sample_ids
            }

        comments = [
            {"comment_id": f"c{sample_id}", "parent_id": None}
            for sample_id in range(1, 25)
        ]
        selected = _comment_planner_batch_with_history(module, original, {})(
            seed_post=SimpleNamespace(
                source_raw_post_id="seed-structural", index=0, title="seed"
            ),
            sample_offset=0,
            comments=comments,
            all_comments=comments,
        )

        self.assertEqual(set(selected), set(range(1, 25)))
        self.assertEqual(len(calls), 19)
        self.assertEqual(calls[1][0], 6)
        self.assertEqual(calls[1][1], ("c7",))
        self.assertIn("Schema completion only", calls[1][2])
        report = module.GENERALIZED_COMMENT_PLAN_REPORTS[-1]
        self.assertEqual(report["repair_attempts"], 0)
        self.assertEqual(report["missing_sample_ids_initial"], list(range(7, 25)))
        self.assertEqual(report["omitted_sample_ids"], [])
        self.assertEqual(report["schema_recovery_attempts"], 18)
        self.assertEqual(
            report["missing_slot_policy"], "bounded_schema_recovery_then_hard_fail"
        )

    def test_single_slot_schema_completion_canonicalizes_echoed_example_id(
        self,
    ) -> None:
        module = SimpleNamespace(
            GENERALIZED_COMMENT_PLAN_HISTORY=[],
            GENERALIZED_COMMENT_PLAN_FEEDBACK="",
            GENERALIZED_COMMENT_PLAN_REPORTS=[],
            GENERALIZED_DOMAIN_PROFILE={"perspectives": [{"perspective_id": "P01"}]},
            GENERALIZED_PLAN_QUALITY_CONFIG={
                "repair_rounds": 0,
                "schema_recovery_rounds": 1,
                "similarity_threshold": 0.95,
                "max_collision_rate": 1.0,
                "max_perspective_share": 1.0,
                "strict": False,
            },
            real_comment_keys=lambda row: (str(row.get("comment_id") or ""),),
        )
        calls = 0

        def original(**call_kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                return {}
            self.assertEqual(call_kwargs["sample_offset"], 37)
            self.assertEqual(
                [row["comment_id"] for row in call_kwargs["comments"]], ["c38"]
            )
            # Provider copied an illustrative schema ID although only S38
            # appeared in the recovery prompt.
            return {
                1: {
                    "reference_id": "R38",
                    "claim_key": "slot_38",
                    "perspective_id": "P01",
                    "semantic_move": "add one distinct local observation",
                    "local_topic": "local topic",
                    "detail_focus": "local detail",
                    "domain_intent": "local intent",
                    "payload_type": "advice",
                    "comment_function": "recommendation_advice",
                }
            }

        comments = [
            {"comment_id": f"c{sample_id}", "parent_id": None}
            for sample_id in range(1, 39)
        ]
        selected = _comment_planner_batch_with_history(module, original, {})(
            seed_post=SimpleNamespace(
                source_raw_post_id="seed-schema-id", index=0, title="seed"
            ),
            sample_offset=37,
            comments=[comments[-1]],
            all_comments=comments,
        )

        self.assertEqual(set(selected), {38})
        self.assertEqual(selected[38]["claim_key"], "slot_38")
        event = module.GENERALIZED_COMMENT_PLAN_REPORTS[-1]["schema_recovery_events"][0]
        self.assertEqual(event["returned_sample_id"], 1)
        self.assertTrue(event["canonicalized_single_slot_id"])

    def test_unresolved_plan_collision_is_logged_without_restarting_post(self) -> None:
        module = SimpleNamespace(
            GENERALIZED_COMMENT_PLAN_HISTORY=[],
            GENERALIZED_COMMENT_PLAN_FEEDBACK="",
            GENERALIZED_COMMENT_PLAN_REPORTS=[],
            GENERALIZED_DOMAIN_PROFILE={"perspectives": [{"perspective_id": "P01"}]},
            GENERALIZED_PLAN_QUALITY_CONFIG={
                "repair_rounds": 1,
                "similarity_threshold": 0.55,
                "max_collision_rate": 0.10,
                "max_perspective_share": 0.60,
                "strict": True,
            },
            real_comment_keys=lambda row: (str(row.get("comment_id") or ""),),
        )

        def repeated_plan(**kwargs):
            sample_offset = int(kwargs.get("sample_offset") or 0)
            sample_ids = range(
                sample_offset + 1,
                sample_offset + len(kwargs["comments"]) + 1,
            )
            return {
                sample_id: {
                    "reference_id": f"R{sample_id}",
                    "claim_key": f"claim_{sample_id}",
                    "perspective_id": "P01",
                    "semantic_move": "compare daily carrying weight",
                    "local_topic": "daily carrying weight",
                    "detail_focus": "carrying weight",
                    "domain_intent": "compare carrying weight",
                    "payload_type": "advice",
                    "comment_function": "recommendation_advice",
                }
                for sample_id in sample_ids
            }

        selected = _comment_planner_batch_with_history(module, repeated_plan, {})(
            seed_post=SimpleNamespace(
                source_raw_post_id="seed-warning", index=0, title="seed"
            ),
            sample_offset=0,
            comments=[
                {"comment_id": "c1", "parent_id": None},
                {"comment_id": "c2", "parent_id": None},
            ],
            all_comments=[
                {"comment_id": "c1", "parent_id": None},
                {"comment_id": "c2", "parent_id": None},
            ],
        )
        self.assertEqual(set(selected), {1, 2})
        warning = module.GENERALIZED_COMMENT_PLAN_REPORTS[-1][
            "unresolved_plan_quality_warning"
        ]
        self.assertGreater(warning["collision_rate"], 0.10)

    def test_contract_compiler_establishes_realizability_before_repair(self) -> None:
        module = SimpleNamespace(
            GENERALIZED_COMMENT_PLAN_HISTORY=[],
            GENERALIZED_COMMENT_PLAN_FEEDBACK="",
            GENERALIZED_COMMENT_PLAN_REPORTS=[],
            GENERALIZED_DOMAIN_PROFILE={"perspectives": [{"perspective_id": "P01"}]},
            GENERALIZED_PLAN_QUALITY_CONFIG={
                "repair_rounds": 1,
                "schema_recovery_rounds": 0,
                "similarity_threshold": 0.95,
                "embedding_similarity_threshold": 0.95,
                "max_collision_rate": 1.0,
                "max_perspective_share": 1.0,
                "strict": False,
                "require_reply_novelty": False,
            },
            real_comment_keys=lambda row: (str(row.get("comment_id") or ""),),
        )
        calls = 0

        def plans_with_tradeoff(**call_kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                return {
                    1: {
                        "claim_key": "initial_story",
                        "perspective_id": "P01",
                        "semantic_move": "describe a distinct first local condition",
                        "payload_type": "advice",
                        "comment_function": "recommendation_advice",
                        "story_mode": "specific_personal_story",
                        "evidence_mode": "technical_or_policy_reasoning",
                    },
                    2: {
                        "claim_key": "shared_increment",
                        "perspective_id": "P01",
                        "semantic_move": "report the shared handling outcome",
                        "payload_type": "soft_helpful",
                        "comment_function": "explanation_analysis",
                        "story_mode": "no_story",
                        "evidence_mode": "small_observation",
                    },
                }
            sample_id = int(call_kwargs["sample_offset"]) + 1
            if sample_id == 1:
                return {
                    1: {
                        "claim_key": "shared_increment",
                        "perspective_id": "P01",
                        "semantic_move": "report the shared handling outcome",
                        "payload_type": "personal_story",
                        "comment_function": "personal_datapoint",
                        "story_mode": "specific_personal_story",
                        "evidence_mode": "firsthand_experience",
                    }
                }
            return {
                2: {
                    "claim_key": "shared_increment",
                    "perspective_id": "P01",
                    "semantic_move": "report the shared handling outcome",
                    "payload_type": "soft_helpful",
                    "comment_function": "explanation_analysis",
                    "story_mode": "no_story",
                    "evidence_mode": "small_observation",
                }
            }

        comments = [
            {
                "comment_id": f"c{sample_id}",
                "parent_id": None,
                "body": "ordinary structural capacity " * 14,
            }
            for sample_id in (1, 2)
        ]
        selected = _comment_planner_batch_with_history(
            module,
            plans_with_tradeoff,
            {},
        )(
            seed_post=SimpleNamespace(
                source_raw_post_id="seed-priority", index=0, title="seed"
            ),
            sample_offset=0,
            comments=comments,
            all_comments=comments,
        )

        self.assertEqual(selected[1]["evidence_mode"], "firsthand_experience")
        self.assertEqual(selected[1]["payload_type"], "fragment_datapoint")
        self.assertEqual(selected[1]["comment_function"], "personal_datapoint")
        report = module.GENERALIZED_COMMENT_PLAN_REPORTS[-1]
        self.assertEqual(report["selected"]["blocking_issue_count"], 0)
        self.assertEqual(report["repair_attempts"], 0)
        self.assertIn(
            "scheduled_story_joint_contract",
            {event["reason"] for event in report["control_normalizations"]},
        )

    def test_contract_compilation_leaves_only_one_long_form_repair(
        self,
    ) -> None:
        """The v93 S9 story conflict is compiled before its beat repair."""

        module = SimpleNamespace(
            GENERALIZED_COMMENT_PLAN_HISTORY=[],
            GENERALIZED_COMMENT_PLAN_FEEDBACK="",
            GENERALIZED_COMMENT_PLAN_REPORTS=[],
            GENERALIZED_SOCIAL_CONTRACT_COHERENCE="on",
            GENERALIZED_DOMAIN_PROFILE={"perspectives": [{"perspective_id": "P10"}]},
            GENERALIZED_PLAN_QUALITY_CONFIG={
                "repair_rounds": 3,
                "schema_recovery_rounds": 0,
                "similarity_threshold": 0.95,
                "embedding_similarity_threshold": 0.95,
                "max_collision_rate": 1.0,
                "max_perspective_share": 1.0,
                "strict": False,
                "require_reply_novelty": False,
            },
            real_comment_keys=lambda row: (str(row.get("comment_id") or ""),),
        )
        development = "one || two || three || four || five"
        base = {
            "reference_id": "R00563",
            "claim_key": "long_term_ownership_feel",
            "perspective_id": "P10",
            "semantic_move": "ground the choice in one long-term ownership outcome",
            "local_topic": "long-term ownership outcome",
            "detail_focus": "practical ownership history",
            "domain_intent": "ground the choice in lived long-term use",
            "payload_type": "fragment_datapoint",
            "comment_function": "personal_datapoint",
            "speaker_role": "datapoint_only",
            "story_mode": "no_story",
        }
        responses = [
            {**base, "evidence_mode": "firsthand_experience", "development_plan": ""},
            {
                **base,
                "evidence_mode": "firsthand_experience",
                "development_plan": development,
            },
        ]
        feedback: list[str] = []

        def alternating_repairs(**_kwargs):
            feedback.append(module.GENERALIZED_COMMENT_PLAN_FEEDBACK)
            return {9: responses[len(feedback) - 1]}

        comments = [
            {"comment_id": f"c{index}", "parent_id": None, "body": "short"}
            for index in range(1, 9)
        ]
        comments.append({"comment_id": "c9", "parent_id": None, "body": "word " * 108})
        selected = _comment_planner_batch_with_history(
            module,
            alternating_repairs,
            {},
        )(
            seed_post=SimpleNamespace(
                source_raw_post_id="1lt0yq3", index=2, title="seed"
            ),
            sample_offset=8,
            comments=[comments[-1]],
            all_comments=comments,
        )

        self.assertEqual(len(feedback), 2)
        self.assertIn("Only development_plan is still failing", feedback[-1])
        self.assertEqual(selected[9]["development_plan"], development)
        self.assertEqual(selected[9]["evidence_mode"], "small_observation")
        report = module.GENERALIZED_COMMENT_PLAN_REPORTS[-1]
        self.assertEqual(report["selected"]["blocking_issue_count"], 0)
        final_attempt = report["attempts"][-1]
        self.assertTrue(final_attempt["repair_accepted"])
        self.assertEqual(final_attempt["repair_merge_fields"], ["development_plan"])
        self.assertEqual(
            final_attempt["candidate_plan"]["evidence_mode"],
            "small_observation",
        )
        self.assertEqual(
            final_attempt["applied_candidate_plan"]["evidence_mode"],
            "small_observation",
        )

    def test_nonblocking_quality_uses_one_repair(self) -> None:
        tone_only = SimpleNamespace(
            repair_issues=(SimpleNamespace(sample_id=7, code="tone_role_mismatch"),)
        )
        self.assertEqual(_sample_repair_budget(tone_only, 7, 3), 1)

        mixed = SimpleNamespace(
            repair_issues=(
                SimpleNamespace(sample_id=7, code="tone_role_mismatch"),
                SimpleNamespace(sample_id=7, code="semantic_collision"),
            )
        )
        self.assertEqual(_sample_repair_budget(mixed, 7, 3), 1)

    def test_dependent_story_evidence_is_reconciled_before_writer(self) -> None:
        module = SimpleNamespace(
            GENERALIZED_COMMENT_PLAN_HISTORY=[],
            GENERALIZED_COMMENT_PLAN_FEEDBACK="",
            GENERALIZED_COMMENT_PLAN_REPORTS=[],
            GENERALIZED_SOCIAL_CONTRACT_COHERENCE="on",
            GENERALIZED_DOMAIN_PROFILE={"perspectives": [{"perspective_id": "P01"}]},
            GENERALIZED_PLAN_QUALITY_CONFIG={
                "repair_rounds": 0,
                "schema_recovery_rounds": 0,
                "similarity_threshold": 0.95,
                "embedding_similarity_threshold": 0.95,
                "max_collision_rate": 1.0,
                "max_perspective_share": 1.0,
                "strict": False,
                "require_reply_novelty": False,
            },
            real_comment_keys=lambda row: (str(row.get("comment_id") or ""),),
        )

        def incompatible_plan(**_kwargs):
            return {
                1: {
                    "perspective_id": "P01",
                    "semantic_move": "report a past shooting event",
                    "local_topic": "camera handling",
                    "detail_focus": "one handling detail",
                    "domain_intent": "report handling",
                    "payload_type": "fragment_datapoint",
                    "comment_function": "personal_datapoint",
                    "story_mode": "no_story",
                    "evidence_mode": "firsthand_experience",
                }
            }

        planner = _comment_planner_batch_with_history(module, incompatible_plan, {})
        selected = planner(
            seed_post=SimpleNamespace(
                source_raw_post_id="seed-social", index=0, title="seed"
            ),
            sample_offset=0,
            comments=[
                {
                    "comment_id": "c1",
                    "parent_id": None,
                    "body": "ordinary local words " * 8,
                }
            ],
            all_comments=[
                {
                    "comment_id": "c1",
                    "parent_id": None,
                    "body": "ordinary local words " * 8,
                }
            ],
        )

        self.assertEqual(selected[1]["story_mode"], "no_story")
        self.assertEqual(selected[1]["evidence_mode"], "small_observation")
        report = module.GENERALIZED_COMMENT_PLAN_REPORTS[-1]
        self.assertEqual(report["selected"]["blocking_issue_count"], 0)
        self.assertIn(
            "no_story_uses_non_narrative_evidence",
            {event["reason"] for event in report["control_normalizations"]},
        )

    def test_social_close_contract_is_bidirectional(self) -> None:
        coherent = {
            "speaker_role": "gratitude_reply",
            "reply_delta_type": "social_close",
            "affect_role": "gratitude",
            "comment_function": "reaction",
            "payload_type": "low_info_reaction",
            "story_mode": "no_story",
        }
        self.assertEqual(social_contract_problem(coherent), "")

        for mismatch in (
            {
                **coherent,
                "affect_role": "neutral",
                "payload_type": "meta_or_template",
            },
            {
                **coherent,
                "speaker_role": "side_observer",
                "affect_role": "neutral",
            },
        ):
            with self.subTest(mismatch=mismatch):
                problem = social_contract_problem(mismatch)
                self.assertIn("reply_delta_type=social_close", problem)
                self.assertIn("affect_role=gratitude or relief", problem)

    def test_residual_content_contract_is_audited_without_aborting_post(self) -> None:
        module = SimpleNamespace(
            GENERALIZED_COMMENT_PLAN_HISTORY=[],
            GENERALIZED_COMMENT_PLAN_FEEDBACK="",
            GENERALIZED_COMMENT_PLAN_REPORTS=[],
            GENERALIZED_SOCIAL_CONTRACT_COHERENCE="on",
            GENERALIZED_DOMAIN_PROFILE={"perspectives": [{"perspective_id": "P01"}]},
            GENERALIZED_PLAN_QUALITY_CONFIG={
                "repair_rounds": 0,
                "schema_recovery_rounds": 0,
                "similarity_threshold": 0.95,
                "embedding_similarity_threshold": 0.95,
                "max_collision_rate": 1.0,
                "max_perspective_share": 1.0,
                "strict": False,
                "require_reply_novelty": False,
            },
            real_comment_keys=lambda row: (str(row.get("comment_id") or ""),),
        )

        def short_story(**_kwargs):
            return {
                1: {
                    "perspective_id": "P01",
                    "semantic_move": "give one compact personal sequence",
                    "payload_type": "personal_story",
                    "comment_function": "personal_datapoint",
                    "story_mode": "specific_personal_story",
                    "evidence_mode": "firsthand_experience",
                }
            }

        selected = _comment_planner_batch_with_history(module, short_story, {})(
            seed_post=SimpleNamespace(
                source_raw_post_id="seed-warning", index=0, title="seed"
            ),
            sample_offset=0,
            comments=[
                {
                    "comment_id": "c1",
                    "parent_id": None,
                    "body": "one two three four five six seven eight nine ten",
                }
            ],
            all_comments=[
                {
                    "comment_id": "c1",
                    "parent_id": None,
                    "body": "one two three four five six seven eight nine ten",
                }
            ],
        )

        self.assertEqual(set(selected), {1})
        report = module.GENERALIZED_COMMENT_PLAN_REPORTS[-1]
        self.assertEqual(report["selected"]["blocking_issue_count"], 1)
        self.assertEqual(
            report["unresolved_plan_contract_warning"][0]["code"],
            "surface_capacity_conflict",
        )

    def test_invalid_branch_id_perspective_is_deterministically_normalized(
        self,
    ) -> None:
        plans = {
            1: {"perspective_id": "B3", "semantic_move": "compare handling"},
            2: {"perspective_id": "p04", "semantic_move": "compare autofocus"},
        }
        events = _canonicalize_plan_controls(
            plans,
            perspective_ids={"P03", "P04"},
        )
        self.assertEqual(plans[1]["perspective_id"], "seed_local")
        self.assertEqual(plans[2]["perspective_id"], "P04")
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["raw_value"], "B3")
        self.assertEqual(events[0]["reason"], "invalid_frozen_decision_lens")

    def test_root_slot_clears_model_supplied_reply_only_contract(self) -> None:
        plans = {
            9: {
                "parent_sample_id": "",
                "perspective_id": "P09",
                "reply_delta": "none",
                "reply_delta_type": "social_close",
                "reply_novelty_anchor": "an invalid root-only reply object",
                "affect_role": "approval",
                "speaker_role": "side_observer",
                "comment_function": "verdict_evaluation",
                "payload_type": "soft_helpful",
                "story_mode": "no_story",
                "_slot_word_count": "108",
                "development_plan": "one || two || three || four || five",
            },
            10: {
                "parent_sample_id": "2",
                "perspective_id": "P09",
                "reply_delta": "acknowledge the parent",
                "reply_delta_type": "social_close",
                "reply_novelty_anchor": "none",
            },
        }

        events = _canonicalize_plan_controls(
            plans,
            perspective_ids={"P09"},
            repair_attempt=2,
        )

        self.assertEqual(plans[9]["reply_delta"], "")
        self.assertEqual(plans[9]["reply_delta_type"], "")
        self.assertEqual(plans[9]["reply_novelty_anchor"], "")
        self.assertEqual(plans[10]["reply_delta_type"], "social_close")
        root_events = [
            event
            for event in events
            if event.get("reason") == "root_slot_has_no_reply_contract"
        ]
        self.assertEqual(
            {event["field"] for event in root_events},
            {"reply_delta_type", "reply_novelty_anchor"},
        )
        self.assertTrue(all(event["repair_attempt"] == 2 for event in root_events))
        report = evaluate_plan_batch({9: plans[9]}, enforce_social_contract=True)
        self.assertFalse(
            any(issue.code == "social_contract_conflict" for issue in report.issues)
        )

    def test_plan_quality_perspective_pressure_spans_prior_batches(self) -> None:
        prior = [
            {
                "sample_id": index,
                "claim_key": f"prior_{index}",
                "perspective_id": "P01",
                "semantic_move": f"distinct prior action {index}",
                "local_topic": f"topic {index}",
                "detail_focus": f"detail {index}",
                "domain_intent": f"intent {index}",
                "payload_type": "advice",
                "comment_function": "recommendation_advice",
            }
            for index in range(1, 9)
        ]
        current = {
            9: {
                "claim_key": "new_nine",
                "perspective_id": "P01",
                "semantic_move": "ask about a separate visible constraint",
                "local_topic": "separate constraint",
                "detail_focus": "new constraint",
                "domain_intent": "clarify another requirement",
                "payload_type": "narrow_question",
                "comment_function": "question_followup",
            }
        }
        report = evaluate_plan_batch(
            current,
            prior_plans=prior,
            max_perspective_share=0.50,
        )
        self.assertEqual(report.dominant_perspective, "P01")
        self.assertEqual(report.dominant_perspective_share, 1.0)
        self.assertIn(
            "perspective_concentration", {issue.code for issue in report.issues}
        )
        self.assertNotIn(
            "perspective_concentration",
            {issue.code for issue in report.repair_issues},
        )
        self.assertTrue(report.healthy)

    def test_plan_quality_tail_collision_uses_thread_denominator(self) -> None:
        prior = [
            {
                "sample_id": index,
                "claim_key": f"prior_{index}",
                "perspective_id": "P01",
                "semantic_move": f"distinct prior move {index}",
                "local_topic": f"prior topic {index}",
                "detail_focus": f"prior detail {index}",
                "domain_intent": f"prior intent {index}",
                "payload_type": "advice",
                "comment_function": "recommendation_advice",
            }
            for index in range(1, 25)
        ]
        current = {
            25: {
                "claim_key": "prior_1",
                "perspective_id": "P02",
                "semantic_move": "repeat prior move one",
                "local_topic": "prior topic one",
                "detail_focus": "prior detail one",
                "domain_intent": "prior intent one",
                "payload_type": "advice",
                "comment_function": "recommendation_advice",
            }
        }
        report = evaluate_plan_batch(
            current,
            prior_plans=prior,
        )
        self.assertEqual(report.colliding_samples, (25,))
        self.assertEqual(report.thread_substantive_count, 25)
        self.assertAlmostEqual(report.collision_rate, 1 / 25)

    def test_first_pass_distribution_findings_are_diagnostic(self) -> None:
        with patch.dict(os.environ, {"GENERALIZED_CARD_ACTOR_CONDITIONING": "none"}):
            module = configure_generator_backend(load_generator_backend(), self.config)
        self.assertFalse(module.has_blocking_guard_failure(["opening_reused"]))
        self.assertFalse(
            module.has_blocking_guard_failure(["lexical_overlap_high:good to know"])
        )
        self.assertFalse(module.has_blocking_guard_failure(["length_too_long"]))
        self.assertFalse(
            module.has_blocking_guard_failure(["substantive_length_floor:5<12"])
        )
        self.assertTrue(
            module.has_blocking_guard_failure(
                ["placeholder_literal", "real_slot_too_short"]
            )
        )

    def test_substantive_floor_blocks_fragment_for_medium_real_slot(self) -> None:
        task = SimpleNamespace(
            real_word_count=24,
            real_surface_shape="compact_datapoint",
            payload_type="soft_helpful",
            comment_function="explanation_analysis",
            utterance_mode="local_advice",
        )
        self.assertEqual(
            substantive_length_floor_problem("Yep, separate tiers there.", task),
            "substantive_length_floor:4<12",
        )
        self.assertEqual(
            substantive_length_floor_problem(
                "The tiers are separate, so I would check the bundle details before deciding.",
                task,
            ),
            "",
        )

    def test_short_bleu_prefix_uses_full_thread_reachability(self) -> None:
        calibration = {
            "prefix_mean_upper": {"tiny": 0.06},
            "prefix_mean_lower": {"tiny": 0.01},
            "prefix_mean_median": {"tiny": 0.03},
        }
        target = {
            "comment_count": 10,
            "self_bleu_4": 0.03,
            "metric_bands": {"self_bleu_4": {"q10": 0.02, "q90": 0.05}},
        }
        diagnostics = candidate_thread_bleu_diagnostics(
            text="zoom preview artifact",
            previous_texts=["weird lines"],
            calibration=calibration,
            thread_target=target,
        )
        self.assertGreater(diagnostics["proposed_mean"], 0.30)
        self.assertTrue(diagnostics["completion_feasible"])
        self.assertEqual(
            calibrated_lexical_overlap_problem(
                text="zoom preview artifact",
                previous_texts=["weird lines"],
                calibration=calibration,
                thread_target=target,
            ),
            "",
        )

        completed_target = {**target, "comment_count": 2}
        completed = candidate_thread_bleu_diagnostics(
            text="zoom preview artifact",
            previous_texts=["weird lines"],
            calibration=calibration,
            thread_target=completed_target,
        )
        self.assertFalse(completed["completion_feasible"])
        self.assertTrue(
            calibrated_lexical_overlap_problem(
                text="zoom preview artifact",
                previous_texts=["weird lines"],
                calibration=calibration,
                thread_target=completed_target,
            ).startswith("lexical_overlap_high:")
        )

        repeated = calibrated_lexical_overlap_problem(
            text="same tiny reply",
            previous_texts=["same tiny reply"],
            calibration=calibration,
            thread_target=target,
        )
        self.assertTrue(repeated.startswith("lexical_overlap_high:"))

    def test_distribution_target_uses_actual_generated_comment_budget(self) -> None:
        target = build_thread_distribution_target(
            {
                "comment_count": 40,
                "self_bleu_4": 0.03,
                "semantic_mean_cosine": 0.30,
            },
            {},
            generated_comment_count=19,
        )
        self.assertEqual(target["comment_count"], 19)
        self.assertEqual(target["template_comment_count"], 40)

    def test_short_bleu_reachability_accounts_for_already_skipped_slots(self) -> None:
        target = distribution_target_with_slot_progress(
            {
                "comment_count": 10,
                "self_bleu_4": 0.03,
                "metric_bands": {"self_bleu_4": {"q10": 0.02, "q90": 0.05}},
            },
            local_task_id=10,
        )
        diagnostics = candidate_thread_bleu_diagnostics(
            text="zoom preview artifact",
            previous_texts=["weird lines"],
            calibration={
                "prefix_mean_upper": {"tiny": 0.06},
                "prefix_mean_lower": {"tiny": 0.01},
                "prefix_mean_median": {"tiny": 0.03},
            },
            thread_target=target,
        )
        self.assertEqual(diagnostics["configured_comment_count"], 10)
        self.assertEqual(diagnostics["planned_comment_count"], 2)
        self.assertFalse(diagnostics["completion_feasible"])

    def test_hard_recovery_preserves_planned_move_and_failed_output(self) -> None:
        @dataclass(frozen=True)
        class Slot:
            planner_intent: str = "Keep the planned caveat."
            must_not_do: str = "Do not add facts."

        repaired = writer_hard_recovery_task(
            Slot(),
            problems=["empty"],
            previous_candidate_text="Could this maybe be the same issue?",
        )
        self.assertIn("same assigned discussion move", repaired.planner_intent)
        self.assertIn("non-empty ordinary comment", repaired.planner_intent)
        self.assertIn("Could this maybe be the same issue?", repaired.planner_intent)
        self.assertIn("Do not add facts.", repaired.must_not_do)

    def test_thread_memory_keeps_early_short_utterances_in_long_threads(self) -> None:
        comments = [
            {
                "content": "heat or battery?" if index == 0 else f"short route {index}",
                "depth": 0,
            }
            for index in range(100)
        ]
        task = SimpleNamespace(
            speaker_role="side_observer",
            tone_shape="plain_neutral",
            payload_type="low_info_reaction",
            utterance_mode="fragment_only",
            voice="casual_neutral",
            story_mode="no_story",
            affect_role="neutral",
            length_bucket="micro",
            surface_skeleton="",
            surface_texture="plain_text",
            perspective_id="seed_local",
            claim_key="local_claim",
            domain_intent="one local reaction",
            opening_style="bare fragment",
        )
        memory = prompts._thread_memory(
            SimpleNamespace(compact=lambda value, _limit: str(value)),
            comments,
            current_task=task,
            domain_profile={},
        )
        self.assertIn("heat or battery?", memory)
        self.assertIn("short route 99", memory)

    def test_thread_memory_stays_bounded_as_a_thread_grows(self) -> None:
        # The ledgers previously scaled with thread length, so by comment 140 the
        # blackboard was 81% of the Writer prompt and the slot's own assignment
        # was 19%. The blackboard must plateau instead of growing.
        def memory_for(count: int) -> str:
            comments = [
                {
                    "content": (
                        f"Comment {index} makes a distinct local observation "
                        f"about detail {index} at some length."
                    ),
                    "depth": index % 3,
                    "semantic_move": f"establish local condition {index}",
                    "decision_boundary": f"boundary {index}",
                    "speaker_role": "advisor",
                    "payload_type": "advice",
                }
                for index in range(count)
            ]
            return prompts._thread_memory(
                SimpleNamespace(compact=lambda value, limit: str(value)[:limit]),
                comments,
                current_task=SimpleNamespace(
                    speaker_role="advisor",
                    tone_shape="neutral_fact",
                    payload_type="advice",
                    utterance_mode="local_answer_with_context",
                    voice="casual_neutral",
                    story_mode="no_story",
                    affect_role="neutral",
                    length_bucket="long",
                    real_word_count=140,
                    surface_skeleton="",
                    surface_texture="plain",
                    perspective_id="seed_local",
                    claim_key="local_claim",
                    domain_intent="one local point",
                    opening_style="verdict first",
                    semantic_move="establish local condition 40",
                    decision_boundary="boundary 40",
                    detail_focus="detail 40",
                    local_topic="local topic",
                ),
                domain_profile={},
            )

        at_40 = len(memory_for(40))
        at_200 = len(memory_for(200))
        self.assertLess(at_200, 12000, at_200)
        # Growth from 40 to 200 comments must be marginal, not linear.
        self.assertLess(at_200, at_40 * 1.6, (at_40, at_200))

    def _coverage_task(self) -> SimpleNamespace:
        return SimpleNamespace(
            speaker_role="advisor",
            tone_shape="neutral_fact",
            payload_type="advice",
            utterance_mode="local_answer_with_context",
            voice="casual_neutral",
            story_mode="no_story",
            affect_role="neutral",
            length_bucket="long",
            real_word_count=140,
            surface_skeleton="",
            surface_texture="plain",
            perspective_id="seed_local",
            claim_key="local_claim",
            domain_intent="one local point",
            opening_style="verdict first",
            semantic_move="a new move",
            decision_boundary="a new boundary",
            detail_focus="a new detail",
            local_topic="local topic",
        )

    def _coverage_comments(self) -> list[dict[str, Any]]:
        return [
            {
                "content": f"Earlier comment {index} makes its own point at length.",
                "depth": 0,
                "semantic_move": f"earlier move {index}",
                "decision_boundary": f"earlier boundary {index}",
                "speaker_role": "advisor",
                "payload_type": "advice",
            }
            for index in range(3)
        ]

    def test_semantic_coverage_nonrepeat_default_is_the_legacy_wording(self) -> None:
        memory = prompts._thread_memory(
            SimpleNamespace(compact=lambda value, limit: str(value)[:limit]),
            self._coverage_comments(),
            current_task=self._coverage_task(),
            domain_profile={},
        )
        self.assertIn("Semantic contributions already covered in this thread:\n", memory)
        self.assertNotIn(prompts.SEMANTIC_COVERAGE_NONREPEAT_INSTRUCTION, memory)
        # Byte-for-byte: the coverage block is followed by exactly one blank
        # line before the next header, the same as every version through v107.
        self.assertIn(
            "\nSentence- or clause-entry routes already used in this thread:\n",
            memory,
        )
        coverage_header = "Semantic contributions already covered in this thread:\n"
        routes_header = "Sentence- or clause-entry routes already used in this thread:\n"
        between = memory[
            memory.index(coverage_header) + len(coverage_header) : memory.index(routes_header)
        ]
        self.assertTrue(between.endswith("\n\n"))
        self.assertNotIn("Do not restate", between)

    def test_semantic_coverage_nonrepeat_on_adds_the_instruction(self) -> None:
        backend = SimpleNamespace(
            compact=lambda value, limit: str(value)[:limit],
            GENERALIZED_SEMANTIC_COVERAGE_NONREPEAT="on",
        )
        memory = prompts._thread_memory(
            backend,
            self._coverage_comments(),
            current_task=self._coverage_task(),
            domain_profile={},
        )
        self.assertIn(prompts.SEMANTIC_COVERAGE_NONREPEAT_INSTRUCTION, memory)
        coverage_header = "Semantic contributions already covered in this thread:\n"
        routes_header = "Sentence- or clause-entry routes already used in this thread:\n"
        self.assertLess(
            memory.index(coverage_header), memory.index(prompts.SEMANTIC_COVERAGE_NONREPEAT_INSTRUCTION)
        )
        self.assertLess(
            memory.index(prompts.SEMANTIC_COVERAGE_NONREPEAT_INSTRUCTION), memory.index(routes_header)
        )

    def test_semantic_coverage_nonrepeat_does_not_touch_the_sibling_blocks(self) -> None:
        off = prompts._thread_memory(
            SimpleNamespace(compact=lambda value, limit: str(value)[:limit]),
            self._coverage_comments(),
            current_task=self._coverage_task(),
            domain_profile={},
        )
        on = prompts._thread_memory(
            SimpleNamespace(
                compact=lambda value, limit: str(value)[:limit],
                GENERALIZED_SEMANTIC_COVERAGE_NONREPEAT="on",
            ),
            self._coverage_comments(),
            current_task=self._coverage_task(),
            domain_profile={},
        )
        for header in (
            "Short utterances already used anywhere in this thread:",
            "Do not output one of these lines again or a trivial polarity-swapped paraphrase.",
            "Sentence- or clause-entry routes already used in this thread:",
            "Do not reuse one of these clause paths",
        ):
            self.assertIn(header, off)
            self.assertIn(header, on)

    def test_semantic_coverage_nonrepeat_instruction_carries_no_domain_vocabulary(
        self,
    ) -> None:
        banned = ("camera", "lens", "sensor", "megapixel", "shutter")
        lowered = prompts.SEMANTIC_COVERAGE_NONREPEAT_INSTRUCTION.lower()
        for term in banned:
            self.assertNotIn(term, lowered)

    # The v108 gate shipped touching only `_thread_memory`, but `focused` is
    # `_writer_prompt_mode`'s default and every real run in this project's
    # history has used it -- confirmed by grepping the gate's own saved
    # `generation_records.json` for the instruction string: 0 of 186. These
    # tests pin the actually-live path (`_focused_thread_ledger`), and the one
    # after them pins that `focused` really is the silent default, so a future
    # change to that default fails a test instead of a paid run.

    def test_focused_thread_ledger_default_is_the_legacy_wording(self) -> None:
        ledger = prompts._focused_thread_ledger(
            SimpleNamespace(compact=lambda value, limit: str(value)[:limit]),
            self._coverage_comments(),
            current_task=self._coverage_task(),
            recent_openings=[],
        )
        self.assertIn("Semantic contributions already covered in this thread:\n", ledger)
        self.assertNotIn(prompts.SEMANTIC_COVERAGE_NONREPEAT_INSTRUCTION, ledger)

    def test_focused_thread_ledger_on_adds_the_instruction(self) -> None:
        backend = SimpleNamespace(
            compact=lambda value, limit: str(value)[:limit],
            GENERALIZED_SEMANTIC_COVERAGE_NONREPEAT="on",
        )
        ledger = prompts._focused_thread_ledger(
            backend,
            self._coverage_comments(),
            current_task=self._coverage_task(),
            recent_openings=[],
        )
        self.assertIn(prompts.SEMANTIC_COVERAGE_NONREPEAT_INSTRUCTION, ledger)
        coverage_header = "Semantic contributions already covered in this thread:\n"
        self.assertLess(
            ledger.index(coverage_header),
            ledger.index(prompts.SEMANTIC_COVERAGE_NONREPEAT_INSTRUCTION),
        )

    def test_focused_writer_prompt_mode_is_the_silent_default(self) -> None:
        # This is the exact fact the v108 bug got wrong: `writer_prompt`
        # dispatches to `_focused_thread_ledger`, not `_thread_memory`, unless
        # something explicitly sets GENERALIZED_WRITER_PROMPT_MODE. A fix that
        # only touches `_thread_memory` never reaches a real generation run.
        self.assertEqual(prompts._writer_prompt_mode(SimpleNamespace()), "focused")

    def test_semantic_coverage_nonrepeat_reaches_the_default_writer_prompt_mode(
        self,
    ) -> None:
        mode = prompts._writer_prompt_mode(SimpleNamespace())
        self.assertEqual(mode, "focused")
        ledger = prompts._focused_thread_ledger(
            SimpleNamespace(
                compact=lambda value, limit: str(value)[:limit],
                GENERALIZED_SEMANTIC_COVERAGE_NONREPEAT="on",
            ),
            self._coverage_comments(),
            current_task=self._coverage_task(),
            recent_openings=[],
        )
        self.assertIn(prompts.SEMANTIC_COVERAGE_NONREPEAT_INSTRUCTION, ledger)

    def test_the_cli_records_the_semantic_coverage_nonrepeat_arm(self) -> None:
        source = (
            REPO_ROOT / "generalized_card" / "scripts" / "run_generate.py"
        ).read_text()
        self.assertIn('"--semantic-coverage-nonrepeat"', source)
        self.assertIn(
            '"semantic_coverage_nonrepeat": args.semantic_coverage_nonrepeat', source
        )
        self.assertIn(
            'env["GENERALIZED_CARD_SEMANTIC_COVERAGE_NONREPEAT"]', source
        )
        fields = source[source.index("RUN_EXPERIMENT_FIELDS = ("):]
        self.assertIn('"semantic_coverage_nonrepeat"', fields[: fields.index(")")])

    def test_distribution_problem_parser_preserves_diagnostic_fields(self) -> None:
        combined = (
            "lexical_overlap_high:thread_mean_bleu4=0.20;target=0.03;"
            "shared=same route;nearest=Earlier lexical route;"
            "semantic_overlap_high:thread_mean_cosine=0.70;target=0.30;"
            "nearest=Earlier semantic route"
        )
        problems = parse_distribution_problems(combined)
        self.assertEqual(len(problems), 2)
        self.assertTrue(problems[0].startswith("lexical_overlap_high:"))
        self.assertIn("target=0.03", problems[0])
        self.assertTrue(problems[1].startswith("semantic_overlap_high:"))

    def test_actor_conditioned_writer_does_not_resample_soft_distribution_failure(
        self,
    ) -> None:
        module = SimpleNamespace(
            previous_comment_texts=lambda comments: [
                str(row.get("content") or "") for row in comments or []
            ],
            lexical_overlap_problem=lambda **_: "",
            GENERALIZED_PLAN_SEMANTIC_INDEX=None,
            GENERALIZED_ACTIVE_DISTRIBUTION_TARGET={},
            GENERALIZED_WRITER_DIVERSITY_CONFIG={
                "local_repair_rounds": 6,
                "slot_retry_limit": 6,
            },
            GENERALIZED_ACTOR_MODE=MODE_DOMAIN_DERIVED,
        )
        calls = 0

        def original(**_):
            nonlocal calls
            calls += 1
            return {
                "skip": False,
                "text": "one direct realization",
                "raw": "one direct realization",
                "prompt": "prompt",
                "attempts": [
                    {
                        "attempt": 1,
                        "text": "one direct realization",
                        "problems": ["opening_reused"],
                    }
                ],
            }

        wrapped = _writer_lifecycle_with_candidate_recovery(
            module,
            original,
            calibration={},
        )
        result = wrapped(
            task=SimpleNamespace(
                local_task_id=2,
                real_word_count=4,
                real_surface_shape="short_turn",
                payload_type="bare_answer",
                comment_function="reaction",
                utterance_mode="fragment_only",
            ),
            previous_comments=[{"content": "earlier comment"}],
        )
        self.assertEqual(calls, 1)
        self.assertFalse(result["skip"])
        self.assertEqual(
            result["candidate_selection"]["reason"],
            "accepted_first_pass_distribution_diagnostics",
        )

    def test_exact_duplicate_gets_bounded_hard_slot_completion(self) -> None:
        @dataclass(frozen=True)
        class Slot:
            local_task_id: int = 72
            planner_intent: str = "Ask which practical limit matters."
            must_not_do: str = "Do not repeat an earlier short question."
            real_word_count: int = 3
            real_surface_shape: str = "short_question"
            payload_type: str = "narrow_question"
            comment_function: str = "question_followup"
            utterance_mode: str = "question_only"

        module = SimpleNamespace(
            previous_comment_texts=lambda comments: [
                str(row.get("content") or "") for row in comments or []
            ],
            lexical_overlap_problem=lambda **_: "",
            GENERALIZED_PLAN_SEMANTIC_INDEX=None,
            GENERALIZED_ACTIVE_DISTRIBUTION_TARGET={},
            GENERALIZED_WRITER_DIVERSITY_CONFIG={
                "hard_recovery_rounds": 2,
                "local_repair_rounds": 0,
                "slot_retry_limit": 0,
            },
            GENERALIZED_ACTOR_MODE=MODE_DOMAIN_DERIVED,
        )
        calls = 0

        def original(**call_kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                text = "heat or battery?"
                problems = ["exact_duplicate", "opening_reused"]
                skip = True
            else:
                self.assertIn("heat or battery?", call_kwargs["task"].planner_intent)
                text = "battery the actual limit here?"
                problems = []
                skip = False
            return {
                "skip": skip,
                "text": text,
                "raw": text,
                "prompt": "prompt",
                "attempts": [
                    {
                        "attempt": 1,
                        "text": text,
                        "problems": problems,
                    }
                ],
            }

        result = _writer_lifecycle_with_candidate_recovery(
            module,
            original,
            calibration={},
        )(
            task=Slot(),
            previous_comments=[{"content": "heat or battery?"}],
        )
        self.assertEqual(calls, 2)
        self.assertFalse(result["skip"])
        self.assertEqual(result["text"], "battery the actual limit here?")
        self.assertEqual(
            result["candidate_selection"]["reason"],
            "accepted_after_hard_slot_completion",
        )

    def test_actor_conditioned_writer_does_not_resample_soft_length_failure(
        self,
    ) -> None:
        module = SimpleNamespace(
            previous_comment_texts=lambda comments: [
                str(row.get("content") or "") for row in comments or []
            ],
            lexical_overlap_problem=lambda **_: "",
            GENERALIZED_PLAN_SEMANTIC_INDEX=None,
            GENERALIZED_ACTIVE_DISTRIBUTION_TARGET={},
            GENERALIZED_WRITER_DIVERSITY_CONFIG={
                "local_repair_rounds": 6,
                "slot_retry_limit": 6,
            },
            GENERALIZED_ACTOR_MODE=MODE_DOMAIN_DERIVED,
        )
        calls = 0

        def original(**_):
            nonlocal calls
            calls += 1
            return {
                "skip": True,
                "text": "This stays as one natural local reply despite the old bucket.",
                "raw": "This stays as one natural local reply despite the old bucket.",
                "prompt": "prompt",
                "attempts": [
                    {
                        "attempt": 1,
                        "text": "This stays as one natural local reply despite the old bucket.",
                        "problems": ["real_slot_too_short"],
                    }
                ],
            }

        wrapped = _writer_lifecycle_with_candidate_recovery(
            module,
            original,
            calibration={},
        )
        result = wrapped(
            task=SimpleNamespace(
                local_task_id=2,
                real_word_count=34,
                real_surface_shape="full_answer",
                payload_type="soft_helpful",
                comment_function="explanation_analysis",
                utterance_mode="local_advice",
            ),
            previous_comments=[],
        )
        self.assertEqual(calls, 1)
        self.assertFalse(result["skip"])
        self.assertEqual(
            result["candidate_selection"]["diagnostic_only_problems"],
            ["real_slot_too_short", "substantive_length_floor:11<17"],
        )

    def test_actor_conditioned_writer_keeps_nonempty_style_diagnostics(self) -> None:
        module = SimpleNamespace(
            previous_comment_texts=lambda _: [],
            lexical_overlap_problem=lambda **_: "",
            GENERALIZED_PLAN_SEMANTIC_INDEX=None,
            GENERALIZED_ACTIVE_DISTRIBUTION_TARGET={},
            GENERALIZED_WRITER_DIVERSITY_CONFIG={
                "local_repair_rounds": 0,
                "slot_retry_limit": 0,
            },
            GENERALIZED_ACTOR_MODE=MODE_DOMAIN_DERIVED,
        )
        calls = 0
        style_problems = [
            "first_person_frame_unwanted",
            "missing_concrete_anchor",
            "meta_template_quote_heading",
            "question_mark_unwanted",
        ]

        def original(**_):
            nonlocal calls
            calls += 1
            return {
                "skip": True,
                "text": "A non-empty local comment?",
                "raw": "A non-empty local comment?",
                "prompt": "prompt",
                "attempts": [
                    {
                        "attempt": 1,
                        "text": "A non-empty local comment?",
                        "problems": style_problems,
                    }
                ],
            }

        result = _writer_lifecycle_with_candidate_recovery(
            module,
            original,
            calibration={},
        )(
            previous_comments=[],
            task=SimpleNamespace(local_task_id=1, real_word_count=0),
        )
        self.assertEqual(calls, 1)
        self.assertFalse(result["skip"])
        self.assertEqual(
            result["candidate_selection"]["diagnostic_only_problems"],
            style_problems,
        )

    def test_matched_real_length_overrides_low_info_fragment_shape(self) -> None:
        # Slot expansion calls the domain-neutral real-comment classifiers the
        # adapter installs, so an unconfigured module tests nothing that runs.
        module = configure_generator_backend(load_generator_backend(), self.config)
        medium_mode = module.infer_utterance_mode(
            payload_type="low_info_reaction",
            speaker_role="advisor",
            comment_function="reaction",
            voice="casual_neutral",
            real_word_count=34,
        )
        self.assertEqual(medium_mode, "local_advice")
        medium_task = SimpleNamespace(
            real_word_count=34,
            real_surface_shape="compact_datapoint",
            utterance_mode=medium_mode,
            payload_type="low_info_reaction",
            length_bucket="medium",
        )
        self.assertIsNone(module.low_info_word_limit(medium_task))

        short_mode = module.infer_utterance_mode(
            payload_type="low_info_reaction",
            speaker_role="advisor",
            comment_function="reaction",
            voice="casual_neutral",
            real_word_count=15,
        )
        self.assertEqual(short_mode, "direct_answer")

        tasks = module.expand_matched_real_sample_to_tasks(
            branches=[
                module.BranchPlan(
                    branch_id=1,
                    anchor_quote="one visible detail",
                    anchor_source="seed",
                    detour_type="none",
                    branch_goal="one local reply",
                    allowed_functions=("reaction",),
                    evidence_modes=("none_assertion",),
                    tone_palette=("casual_neutral",),
                    story_modes=("no_story",),
                    content_angles=("unclear_mixed",),
                )
            ],
            target=module.ThreadTarget(1, 1, 0, "quiet", "matched"),
            seed_post=module.SeedPost(
                index=0,
                title="visible post",
                body="visible body",
                content="visible post visible body",
                source_raw_post_id="seed",
                real_num_comments=1,
                metadata={},
            ),
            matched_real_thread={
                "comments": [
                    {
                        "body": " ".join(f"word{index}" for index in range(34)),
                        "comment_id": "real_1",
                        "comment_fullname": "t1_real_1",
                        "parent_id": "t3_seed",
                    }
                ]
            },
            matched_real_comments=100,
            comment_plans={
                1: {
                    "branch_id": "1",
                    "comment_function": "reaction",
                    "payload_type": "low_info_reaction",
                    "semantic_move": "make one narrow direct point",
                }
            },
            rng=random.Random(42),
        )
        self.assertEqual(tasks[0].real_word_count, 34)
        self.assertEqual(tasks[0].length_bucket, "medium")
        self.assertEqual(tasks[0].utterance_mode, "local_advice")
        self.assertIsNone(module.low_info_word_limit(tasks[0]))

    def test_direct_reply_receives_parent_semantic_exclusion_without_replanning(
        self,
    ) -> None:
        module = configure_generator_backend(load_generator_backend(), self.config)
        tasks = module.expand_matched_real_sample_to_tasks(
            branches=[
                module.BranchPlan(
                    branch_id=1,
                    anchor_quote="one visible constraint",
                    anchor_source="seed",
                    detour_type="none",
                    branch_goal="separate one local tradeoff",
                    allowed_functions=("reaction",),
                    evidence_modes=("none_assertion",),
                    tone_palette=("casual_neutral",),
                    story_modes=("no_story",),
                    content_angles=("fit_use_case",),
                    owned_decision_subject="whether the seating distance makes reach decisive",
                )
            ],
            target=module.ThreadTarget(2, 1, 1, "quiet", "matched"),
            seed_post=module.SeedPost(
                index=0,
                title="visible post",
                body="visible body",
                content="visible post visible body",
                source_raw_post_id="seed",
                real_num_comments=2,
                metadata={},
            ),
            matched_real_thread={
                "comments": [
                    {
                        "body": "parent slot has enough ordinary words to remain substantive",
                        "comment_id": "parent",
                        "comment_fullname": "t1_parent",
                        "parent_id": "t3_seed",
                    },
                    {
                        "body": "child slot has enough ordinary words to remain substantive too",
                        "comment_id": "child",
                        "comment_fullname": "t1_child",
                        "parent_id": "t1_parent",
                    },
                ]
            },
            matched_real_comments=100,
            comment_plans={
                1: {
                    "branch_id": "1",
                    "semantic_move": "ask whether reach matters from distant seating",
                    "decision_boundary": "whether distance changes the choice",
                },
                2: {
                    "branch_id": "1",
                    "payload_type": "correction",
                    "comment_function": "correction_caveat",
                    "content_angle": "risk_reliability_support",
                    "evidence_mode": "small_observation",
                    "voice": "blunt",
                    "speaker_role": "contrarian",
                    "semantic_move": "add one new caveat about the same local tradeoff",
                    "reply_relation": "limits_parent",
                    "stance": "hard_disagree",
                    "detail_focus": "loose framing at the subject edge",
                    "avoid_repeating": "the parent's general reach verdict",
                    "domain_intent": "name the narrower framing failure",
                    "decision_boundary": "whether another constraint changes the choice",
                    "reply_delta": "state the separate consequence of choosing too little reach",
                    "reply_delta_type": "downstream_consequence",
                    "reply_novelty_anchor": "the subject cannot be recovered once the framing is too loose",
                },
            },
            rng=random.Random(42),
        )
        child = tasks[1]
        self.assertEqual(child.parent_semantic_move, tasks[0].semantic_move)
        self.assertEqual(child.parent_decision_boundary, tasks[0].decision_boundary)
        # The reply keeps the Planner's own semantic move. The novelty anchor is
        # the concrete object it introduces and reaches the Writer as its own
        # field; it must not replace the move, which collapsed a reply's
        # semantic contract to a bare noun phrase.
        self.assertEqual(
            child.semantic_move, "add one new caveat about the same local tradeoff"
        )
        self.assertEqual(
            child.reply_novelty_anchor,
            "the subject cannot be recovered once the framing is too loose",
        )
        self.assertNotEqual(child.semantic_move, child.reply_novelty_anchor)
        # The configured path keeps the Planner's own decision boundary and only
        # the semantic move is replaced by the novelty anchor. CARD's raw
        # expansion overwrote both; the planner-owned field list restores one.
        self.assertEqual(
            child.decision_boundary, "whether another constraint changes the choice"
        )
        # The parent exclusion travels as its own structured field, not as a
        # second sentence appended to the Planner's avoid list.
        self.assertEqual(child.parent_semantic_move, tasks[0].semantic_move)
        self.assertNotIn(
            "Do not restate the parent contribution", child.avoid_repeating
        )
        serialized = module.task_to_dict(child)
        self.assertEqual(serialized["reply_delta"], child.reply_delta)
        self.assertEqual(serialized["reply_novelty_anchor"], child.reply_novelty_anchor)
        self.assertEqual(serialized["parent_semantic_move"], child.parent_semantic_move)
        self.assertEqual(
            serialized["parent_decision_boundary"], child.parent_decision_boundary
        )
        self.assertEqual(
            child.reply_delta,
            "state the separate consequence of choosing too little reach",
        )
        rendered = module.build_writer_prompt(
            profile="gpt54_reddit_writer",
            seed_post=module.SeedPost(
                index=0,
                title="visible post",
                body="visible body",
                content="visible post visible body",
                source_raw_post_id="seed",
                real_num_comments=2,
                metadata={},
            ),
            task=module.finalize_rebalanced_task(child),
            parent_comment={"content": "The generated parent gives a reach verdict."},
            previous_comments=[],
            recent_openings=[],
        )
        for expected in (
            "- payload form: correction",
            "- function: correction caveat",
            "- speaker role: contrarian",
            "- voice: blunt",
            "- stance: hard_disagree",
            "- reply relation: limits_parent",
            "- decision intent: name the narrower framing failure",
            "- content to avoid: the parent's general reach verdict",
            "- You are replying. What you add: state the separate consequence",
        ):
            self.assertEqual(rendered.count(expected), 1, expected)

    def test_tree_order_plans_all_parents_before_direct_replies(self) -> None:
        module = load_generator_backend()
        ordered = module.order_comments_by_thread_tree(
            [
                {"comment_id": "root_a", "parent_id": "t3_seed", "depth": 0},
                {"comment_id": "child_a", "parent_id": "t1_root_a", "depth": 1},
                {"comment_id": "root_b", "parent_id": "t3_seed", "depth": 0},
                {"comment_id": "child_b", "parent_id": "t1_root_b", "depth": 1},
                {"comment_id": "grandchild_a", "parent_id": "t1_child_a", "depth": 2},
            ]
        )
        self.assertEqual(
            [row["comment_id"] for row in ordered],
            ["root_a", "root_b", "child_a", "child_b", "grandchild_a"],
        )
        self.assertEqual(
            module.planner_batch_ranges_by_depth(ordered, batch_size=8),
            [(0, 2), (2, 4), (4, 5)],
        )

    def test_tree_order_normalizes_missing_parent_to_effective_root(self) -> None:
        module = load_generator_backend()
        ordered = module.order_comments_by_thread_tree(
            [
                {
                    "comment_id": "root",
                    "parent_id": "t3_seed",
                    "depth": 0,
                    "post_id": "seed",
                },
                {
                    "comment_id": "orphan",
                    "parent_id": "t1_deleted",
                    "depth": 1,
                    "post_id": "seed",
                },
                {
                    "comment_id": "child",
                    "parent_id": "t1_root",
                    "depth": 1,
                    "post_id": "seed",
                },
            ]
        )
        self.assertEqual(
            [row["comment_id"] for row in ordered],
            ["root", "orphan", "child"],
        )
        orphan = ordered[1]
        self.assertEqual(orphan["parent_id"], "t3_seed")
        self.assertEqual(orphan["depth"], 0)
        self.assertEqual(
            module.planner_batch_ranges_by_depth(ordered, batch_size=8),
            [(0, 2), (2, 3)],
        )

    def test_direct_reply_replaces_planner_none_delta_with_structural_increment(
        self,
    ) -> None:
        module = load_generator_backend()
        self.assertIn(
            "different narrow follow-up",
            module.default_reply_delta("asks_narrow_followup"),
        )

    def test_distribution_diagnostic_is_audited_without_resampling(self) -> None:
        module = SimpleNamespace(
            previous_comment_texts=lambda _: [],
            GENERALIZED_WRITER_DIVERSITY_CONFIG={"hard_recovery_rounds": 2},
        )
        wrapped = _writer_lifecycle_with_candidate_recovery(
            module,
            lambda **_: {
                "skip": True,
                "text": "Same narrow point in the same words.",
                "attempts": [
                    {
                        "attempt": 1,
                        "text": "Same narrow point in the same words.",
                        "problems": ["lexical_overlap_high:test"],
                    }
                ],
            },
            calibration={},
        )
        result = wrapped(previous_comments=[])
        self.assertFalse(result["skip"])
        self.assertEqual(
            result["candidate_selection"]["reason"],
            "accepted_first_pass_distribution_diagnostics",
        )

    def test_candidate_recovery_never_bypasses_safety_failure(self) -> None:
        module = SimpleNamespace(previous_comment_texts=lambda _: [])
        wrapped = _writer_lifecycle_with_candidate_recovery(
            module,
            lambda **_: {
                "skip": True,
                "attempts": [
                    {
                        "attempt": 1,
                        "text": "unsafe",
                        "problems": ["placeholder_literal"],
                    }
                ],
            },
            calibration={},
        )
        self.assertTrue(wrapped(previous_comments=[])["skip"])

    def test_candidate_recovery_rewrites_known_content_failure(self) -> None:
        module = SimpleNamespace(
            previous_comment_texts=lambda _: [],
            GENERALIZED_WRITER_DIVERSITY_CONFIG={
                "hard_recovery_rounds": 2,
            },
        )
        calls = 0

        def original(**_):
            nonlocal calls
            calls += 1
            if calls == 1:
                return {
                    "skip": True,
                    "text": "[insert answer]",
                    "attempts": [
                        {
                            "attempt": 1,
                            "text": "[insert answer]",
                            "problems": ["placeholder_literal"],
                        }
                    ],
                }
            return {
                "skip": False,
                "text": "That crop looks like a preview artifact.",
                "raw": "That crop looks like a preview artifact.",
                "prompt": "repair",
                "attempts": [
                    {
                        "attempt": 1,
                        "text": "That crop looks like a preview artifact.",
                        "problems": [],
                    }
                ],
            }

        wrapped = _writer_lifecycle_with_candidate_recovery(
            module,
            original,
            calibration={},
        )
        result = wrapped(previous_comments=[], task=None)
        self.assertFalse(result["skip"])
        self.assertEqual(calls, 2)
        self.assertEqual(
            result["candidate_selection"]["reason"],
            "accepted_after_hard_slot_completion",
        )

    def test_soft_distribution_findings_do_not_resample_writer(self) -> None:
        @dataclass(frozen=True)
        class Slot:
            local_task_id: int = 2
            planner_intent: str = "Keep the planned caveat."
            must_not_do: str = "Do not add facts."

        module = SimpleNamespace(
            previous_comment_texts=lambda comments: [
                str(row.get("content") or "") for row in comments or []
            ],
            GENERALIZED_WRITER_DIVERSITY_CONFIG={
                "local_repair_rounds": 6,
                "slot_retry_limit": 2,
            },
        )
        calls = 0

        def original(**_):
            nonlocal calls
            calls += 1
            text = f"candidate route {calls}"
            return {
                "skip": True,
                "text": text,
                "raw": text,
                "prompt": "prompt",
                "attempts": [
                    {
                        "attempt": 1,
                        "text": text,
                        "problems": ["lexical_overlap_high:test"],
                    }
                ],
            }

        wrapped = _writer_lifecycle_with_candidate_recovery(
            module,
            original,
            calibration={},
        )
        result = wrapped(
            previous_comments=[{"content": "prior route"}],
            task=Slot(),
        )
        self.assertEqual(calls, 1)
        self.assertFalse(result["skip"])
        self.assertEqual(
            result["candidate_selection"]["reason"],
            "accepted_first_pass_distribution_diagnostics",
        )

    def test_degraded_retry_preserves_normalized_planner_slot(self) -> None:
        @dataclass(frozen=True)
        class Slot:
            real_word_count: int = 120
            real_surface_shape: str = "full_answer"
            length_bucket: str = "micro"
            payload_type: str = "low_info_reaction"
            comment_function: str = "reaction"
            utterance_mode: str = "fragment_only"
            story_mode: str = "no_story"
            must_not_do: str = "Keep one local point."

        # There is no native task mutation left to fall back on: hard-failure
        # recovery retries the same slot rather than degrading it.
        fallback = _substantive_safe_degraded_task()
        repaired = fallback(Slot(), ["question_mark_unwanted"])
        self.assertEqual(repaired.length_bucket, "micro")
        self.assertEqual(repaired.payload_type, "low_info_reaction")
        self.assertEqual(repaired.comment_function, "reaction")

    def test_incomplete_post_is_rejected_before_persistence(self) -> None:
        wrapped = _finalize_post_generation(
            SimpleNamespace(GENERALIZED_ACTOR_MODE="none"),
            lambda **_: {
                "generation_records": [
                    {"task": {"local_task_id": 1}, "comment": {"content": "ok"}},
                    {"task": {"local_task_id": 2}, "comment": None, "skipped": True},
                ],
                "thread_plan": {},
            },
        )
        with self.assertRaisesRegex(
            RuntimeError,
            r"incomplete post was not persisted.*failed_task_ids=\[2\]",
        ):
            wrapped(tasks=[object(), object()])

    def test_planned_quote_excerpt_can_survive_parent_copy_guard(self) -> None:
        parent = "Buffer lag after switching modes is the ugly part of this workflow."
        text = (
            "> switching modes\n\n"
            "That is where autofocus consistency mattered more for my use, "
            "because the subject kept moving while the files cleared."
        )
        task = SimpleNamespace(
            local_task_id=2,
            real_sample_id=2,
            real_word_count=25,
            real_surface_shape="ordinary_turn",
            payload_type="soft_helpful",
            comment_function="explanation_analysis",
            utterance_mode="local_answer_with_context",
        )
        seed = SimpleNamespace(source_raw_post_id="seed-quote")
        module = SimpleNamespace(
            previous_comment_texts=lambda _: [],
            lexical_overlap_problem=lambda **_: "",
            GENERALIZED_PLAN_SEMANTIC_INDEX=None,
            GENERALIZED_ACTIVE_DISTRIBUTION_TARGET={},
            GENERALIZED_WRITER_DIVERSITY_CONFIG={"hard_recovery_rounds": 0},
            GENERALIZED_OPENER_TYPES={("seed-quote", 2): "quote"},
            GENERALIZED_ACTOR_MODE="none",
        )

        def original(**_):
            return {
                "skip": True,
                "text": text,
                "raw": text,
                "prompt": "prompt",
                "attempts": [{"attempt": 1, "text": text, "problems": ["parent_copy"]}],
            }

        with tempfile.TemporaryDirectory() as directory, patch.dict(
            os.environ,
            {
                "GENERALIZED_CARD_WRITER_DIVERSITY_AUDIT_JSONL": str(
                    Path(directory) / "writer.jsonl"
                )
            },
        ):
            result = _writer_lifecycle_with_candidate_recovery(
                module,
                original,
                calibration={},
            )(
                seed_post=seed,
                task=task,
                parent_comment={"content": parent},
                previous_comments=[],
            )
            audit_row = json.loads(
                (Path(directory) / "writer.jsonl").read_text(encoding="utf-8")
            )
        self.assertFalse(result["skip"])
        self.assertTrue(
            result["candidate_selection"]["planned_quote_parent_copy_waived"]
        )
        self.assertTrue(audit_row["planned_quote_parent_copy_waived"])
        self.assertTrue(planned_quote_has_distinct_reply(text, parent))
        self.assertFalse(
            planned_quote_has_distinct_reply(
                f"> {parent}\n\nThat is still a separate reply with enough words.",
                parent,
            )
        )

    def test_complete_post_persists_domain_actor_provenance(self) -> None:
        seed = SimpleNamespace(source_raw_post_id="seed-actor", index=0, title="seed")
        state = actor_state_from_plan(
            {
                "actor_participant_key": "A1",
                "actor_participation_goal": "add one local observation",
                "actor_realization_route": "detail first, short consequence second",
            },
            sample_id=1,
        )
        module = SimpleNamespace(
            GENERALIZED_ACTOR_MODE=MODE_DOMAIN_DERIVED,
            GENERALIZED_ACTOR_ASSIGNMENTS={("seed-actor", 1): state},
            GENERALIZED_ACTIVE_REFERENCE_TEMPLATE={
                "self_bleu_4": 0.03,
                "raw_text_included": False,
            },
        )
        comment = {"content": "local observation"}
        wrapped = _finalize_post_generation(
            module,
            lambda **_: {
                "generation_records": [
                    {
                        "task": {"local_task_id": 1, "real_sample_id": 1},
                        "comment": comment,
                    }
                ],
                "thread_plan": {},
            },
        )
        post = wrapped(
            tasks=[object()],
            seed_post=seed,
            run_index=3,
            post_slot=2,
        )
        self.assertEqual(comment["actor_conditioning"], MODE_DOMAIN_DERIVED)
        self.assertEqual(comment["actor_state"]["participant_key"], "A1")
        self.assertTrue(comment["author"].startswith("sampled_actor_3_2_"))
        self.assertEqual(
            post["thread_plan"]["actor_conditioning"]["source"],
            "evaluation-excluded domain references plus visible thread",
        )
        self.assertEqual(
            post["thread_plan"]["reference_metric_template"]["self_bleu_4"],
            0.03,
        )
        self.assertFalse(
            post["thread_plan"]["reference_metric_template_provenance"][
                "raw_text_included"
            ]
        )

    def test_generalized_ngram_guard_blocks_repeated_entity_sequence(self) -> None:
        with patch(
            "generalized_card.backend.load_domain_profile",
            return_value={
                "lexical_quality": {
                    "thresholds": {
                        "micro": 0.05,
                        "short": 0.05,
                        "medium": 0.05,
                        "long": 0.05,
                        "very_long": 0.05,
                    }
                }
            },
        ):
            with patch.dict(
                os.environ, {"GENERALIZED_CARD_ACTOR_CONDITIONING": "none"}
            ):
                module = configure_generator_backend(
                    load_generator_backend(), self.config
                )
        task = SimpleNamespace(length_bucket="medium", payload_type="advice")
        problem = module.lexical_overlap_problem(
            text=(
                "Canon Sony Fujifilm Panasonic are all options, but choose based on "
                "the handling that fits your own use."
            ),
            previous_comments=[
                {
                    "content": (
                        "Canon Sony Fujifilm Panasonic all came up, although the venue "
                        "rules matter more than another feature list."
                    )
                }
            ],
            task=task,
        )
        self.assertTrue(problem.startswith("lexical_overlap_high:"))

    def test_active_writer_guard_uses_semantic_metric_and_targeted_retry_context(
        self,
    ) -> None:
        class FakeSemanticIndex:
            vectors = {
                "Earlier autofocus claim.": [1.0, 0.0],
                "Another autofocus paraphrase.": [0.98, 0.2],
                "Same autofocus point again.": [0.99, 0.1],
            }

            def encode_texts(self, texts):
                import numpy as np

                rows = []
                for text in texts:
                    vector = np.asarray(self.vectors[text], dtype=float)
                    rows.append(vector / np.linalg.norm(vector))
                return rows

        with patch(
            "generalized_card.backend.load_domain_profile",
            return_value={
                "lexical_quality": {},
                "reference_metric_calibration": {"available": True},
            },
        ):
            with patch.dict(
                os.environ, {"GENERALIZED_CARD_ACTOR_CONDITIONING": "none"}
            ):
                module = configure_generator_backend(
                    load_generator_backend(), self.config
                )
        module.GENERALIZED_PLAN_SEMANTIC_INDEX = FakeSemanticIndex()
        module.GENERALIZED_ACTIVE_DISTRIBUTION_TARGET = {
            "comment_count": 3,
            "semantic_mean_cosine": 0.30,
            "metric_bands": {"semantic_mean_cosine": {"q10": 0.25, "q90": 0.35}},
        }
        task = SimpleNamespace(length_bucket="medium", payload_type="advice")
        problem = module.lexical_overlap_problem(
            text="Same autofocus point again.",
            previous_comments=[
                {"content": "Earlier autofocus claim."},
                {"content": "Another autofocus paraphrase."},
            ],
            task=task,
        )
        self.assertTrue(problem.startswith("semantic_overlap_high:"))
        retry = module.retry_note_for_problems([problem], task)
        self.assertIn("Earlier autofocus claim", retry)
        self.assertIn("different implication", retry)
        self.assertFalse(module.has_blocking_guard_failure([problem]))

    def test_gpt5_completion_omits_temperature(self) -> None:
        with patch.dict(os.environ, {"GPT5_REASONING_TOKEN_RESERVE": "256"}):
            kwargs = _completion_kwargs(
                model="gpt-5.4-mini",
                messages=[{"role": "user", "content": "test"}],
                temperature=0.55,
                max_tokens=6500,
                response_format_json=True,
                extra_body=None,
            )
            short_kwargs = _completion_kwargs(
                model="gpt-5.4-mini",
                messages=[{"role": "user", "content": "test"}],
                temperature=0.88,
                max_tokens=64,
                response_format_json=False,
                extra_body=None,
            )
        self.assertNotIn("temperature", kwargs)
        self.assertNotIn("max_tokens", kwargs)
        self.assertEqual(kwargs["max_completion_tokens"], 6500)
        self.assertEqual(short_kwargs["max_completion_tokens"], 320)

        legacy = _completion_kwargs(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "test"}],
            temperature=0.55,
            max_tokens=6500,
            response_format_json=True,
            extra_body=None,
        )
        self.assertEqual(legacy["temperature"], 0.55)
        self.assertEqual(legacy["max_tokens"], 6500)

        self.assertEqual(
            _next_completion_boost(
                current=0, finish_reason="length", reasoning_model=True
            ),
            128,
        )
        self.assertEqual(
            _next_completion_boost(
                current=128, finish_reason="length", reasoning_model=True
            ),
            256,
        )
        self.assertEqual(
            _next_completion_boost(
                current=128, finish_reason="stop", reasoning_model=True
            ),
            128,
        )
        self.assertTrue(
            _is_output_limit_error(
                RuntimeError(
                    "Could not finish because max_tokens or model output limit was reached. "
                    "Please try again with higher max_tokens."
                )
            )
        )
        self.assertFalse(_is_output_limit_error(RuntimeError("rate limit")))

    def test_generation_defaults_match_card_run_controls(self) -> None:
        script = load_script_module(
            "generalized_card_run_generate_defaults",
            Path(__file__).resolve().parents[1] / "scripts" / "run_generate.py",
        )
        parser = script.build_parser()
        help_text = parser.format_help()
        self.assertIn("13% of the old size", help_text)
        self.assertIn("invented biographies never reach the Writer", help_text)
        args = parser.parse_args(["--tag", "test"])
        self.assertEqual(args.generator_profile, GENERALIZED_V2_PROFILE)
        self.assertEqual(args.comment_planner_batch_size, 8)
        self.assertEqual(args.max_comments_per_post, 0)
        self.assertEqual(args.comment_count_scale, 1.0)
        self.assertEqual(args.matched_real_comments, 0)
        self.assertTrue(args.exact_matched_thread_size)
        self.assertEqual(args.api_retries, 2)
        self.assertEqual(args.writer_retries, 0)
        self.assertEqual(args.writer_hard_recovery_rounds, 2)
        self.assertEqual(args.social_contract_coherence, "on")
        self.assertEqual(args.reply_sibling_visibility, "on")
        self.assertEqual(args.speaker_identity, "matched")
        self.assertEqual(args.actor_conditioning, "none")
        self.assertEqual(args.retry_delay, 10.0)
        self.assertEqual(args.posts_per_run, 5)
        self.assertEqual(args.planner_max_tokens, 10000)
        self.assertEqual(args.comment_planner_max_tokens, 18000)
        self.assertEqual(args.writer_max_tokens, 260)
        self.assertEqual(args.plan_quality_repairs, 3)
        self.assertEqual(args.plan_similarity_threshold, 0.72)
        self.assertTrue(args.plan_embedding_quality)
        self.assertEqual(args.plan_embedding_threshold, 0.70)
        self.assertEqual(args.plan_max_collision_rate, 0.10)
        self.assertEqual(args.max_perspective_share, 0.34)
        self.assertTrue(args.strict_plan_quality)
        self.assertEqual(args.post_retry_limit, 1)
        self.assertEqual(args.post_retry_delay, 15.0)
        command = script._generator_command(
            args=args,
            config_raw_dir=Path("real"),
            seed_pool=Path("seed.json"),
            generated_root=Path("generated"),
        )
        writer_retries = command.index("--writer-retries")
        planner_retries = command.index("--planner-retries")
        batch_size = command.index("--comment-planner-batch-size")
        post_retry_limit = command.index("--post-retry-limit")
        self.assertEqual(command[writer_retries + 1], "0")
        self.assertEqual(command[planner_retries + 1], "2")
        self.assertEqual(command[batch_size + 1], "8")
        self.assertEqual(command[post_retry_limit + 1], "1")

    def test_first_pass_policy_rejects_unplanned_structural_slots(self) -> None:
        tasks = [
            SimpleNamespace(real_sample_id=1),
            SimpleNamespace(real_sample_id=2),
            SimpleNamespace(real_sample_id=3),
        ]
        with self.assertRaisesRegex(RuntimeError, "omitted required structural slots"):
            retain_explicitly_planned_tasks(
                tasks,
                {1: {"semantic_move": "first"}, 3: {"semantic_move": "third"}},
            )

    def test_soft_length_guidance_exposes_continuous_anonymous_scale(self) -> None:
        guidance = soft_length_guidance(SimpleNamespace(real_word_count=83))
        self.assertIn(f"roughly {calibrated_word_ask(83)} words", guidance)
        # The scale is a directional target, never a counted acceptance gate.
        self.assertIn("not a counted requirement", guidance)
        self.assertNotIn("hard maximum", guidance)

    def test_the_rendered_scale_is_the_ask_not_the_matched_count(self) -> None:
        # The cue carries the number that realizes the slot, not the slot's own
        # count: through v97 the two were identical and realized/target ran
        # 1.42x at the shortest slots and 0.71x at 251-400 words.
        # See `length_calibration`.
        self.assertIn("roughly 12 words", soft_length_guidance(SimpleNamespace(real_word_count=13)))
        self.assertIn("roughly 274 words", soft_length_guidance(SimpleNamespace(real_word_count=220)))
        set_length_calibration("off")
        try:
            self.assertIn(
                "roughly 220 words",
                soft_length_guidance(SimpleNamespace(real_word_count=220)),
            )
        finally:
            set_length_calibration("measured")

    def test_soft_length_guidance_names_undershoot_for_long_slots(self) -> None:
        long_guidance = soft_length_guidance(SimpleNamespace(real_word_count=240))
        self.assertIn(f"roughly {calibrated_word_ask(240)} words", long_guidance)
        self.assertIn("do not trim toward a medium-length answer", long_guidance)
        short_guidance = soft_length_guidance(SimpleNamespace(real_word_count=20))
        self.assertIn("Do not pad past it", short_guidance)

    def test_long_slot_scope_asks_for_connected_beats_not_one_repeated_thesis(
        self,
    ) -> None:
        guidance = local_move_scope_guidance(SimpleNamespace(real_word_count=340))
        self.assertIn("connected beats", guidance)
        self.assertIn("long-tail slot", guidance)
        # "one local thesis" was the wording that produced a single-paragraph
        # comment at every size. A slot this long is laid out as several
        # paragraphs by `comment_structure`, each with its own point.
        self.assertNotIn("one local thesis", guidance)

    def test_incidental_humor_does_not_collapse_substantive_surface(self) -> None:
        text = (
            "This is a substantive local explanation with several constraints and "
            "a concrete consequence for ordinary use, followed by a caveat that "
            "keeps the recommendation narrow instead of turning it into a generic "
            "answer. The final reaction is incidental rather than the whole point lol."
        )
        self.assertEqual(
            infer_surface_shape({"body": text, "author": "user"}),
            "full_answer",
        )
        self.assertEqual(surface_only_label(text), "ordinary_turn")
        skeleton, instruction = infer_surface_skeleton(text)
        self.assertNotIn("joke", skeleton)
        self.assertIn("assigned local move", instruction)

    def test_incidental_link_does_not_collapse_long_surface(self) -> None:
        text = (
            "The reference is useful for one narrow reason, but the rest of this "
            "comment explains how the constraint changes ordinary use and why the "
            "tradeoff matters in practice. It also leaves room for a different "
            "choice when that condition does not apply. https://example.invalid/ref "
            "That link is supporting context rather than the whole reply."
        )
        self.assertEqual(
            infer_surface_shape({"body": text, "author": "user"}),
            "full_answer",
        )
        self.assertEqual(surface_only_label(text), "ordinary_turn")

    def test_matched_lexical_roles_do_not_become_surface_shapes(self) -> None:
        self.assertEqual(
            infer_surface_shape(
                {
                    "body": "Side note: I think maybe this is unrelated to the main point, but it remains an ordinary local comment.",
                    "author": "user",
                }
            ),
            "full_answer",
        )
        self.assertEqual(
            infer_surface_shape(
                {
                    "body": "!template this is ordinary user text with enough words to avoid a short structural label",
                    "author": "user",
                }
            ),
            "full_answer",
        )
        self.assertEqual(
            infer_surface_shape(
                {
                    "body": " ".join("word" for _ in range(80)),
                    "author": "user",
                }
            ),
            "long_turn",
        )
        self.assertEqual(
            infer_surface_shape(
                {"body": "ordinary moderator notice", "author": "moderator"}
            ),
            "template_notice",
        )

    def test_matched_gratitude_words_do_not_assign_gratitude_tone(self) -> None:
        self.assertEqual(
            infer_surface_texture(
                "Thanks, I really appreciate it. Good to know.",
                payload_type="rant",
                speaker_role="ranter",
                utterance_mode="complaint_only",
            ),
            "plain",
        )
        self.assertEqual(
            infer_surface_texture(
                "That answered the narrow question.",
                payload_type="bare_answer",
                speaker_role="gratitude_reply",
                utterance_mode="op_followup",
            ),
            "gratitude_social",
        )

    def test_writer_provider_budget_tracks_long_tail_without_length_gate(self) -> None:
        self.assertEqual(
            writer_provider_token_budget(
                SimpleNamespace(real_word_count=80), configured_max=260
            ),
            260,
        )
        self.assertGreater(
            writer_provider_token_budget(
                SimpleNamespace(real_word_count=300), configured_max=260
            ),
            500,
        )

    def test_substantive_slot_removes_whole_comment_short_mode(self) -> None:
        @dataclass(frozen=True)
        class Task:
            real_word_count: int
            real_surface_shape: str
            utterance_mode: str

        reconciled = reconcile_substantive_task(Task(55, "full_answer", "joke_only"))
        self.assertEqual(reconciled.utterance_mode, "humorous_local_turn")

    def test_plan_quality_rejects_short_payload_for_substantive_slot(self) -> None:
        report = evaluate_plan_batch(
            {
                1: {
                    "sample_id": 1,
                    "perspective_id": "seed_local",
                    "payload_type": "joke",
                    "comment_function": "reaction",
                    "semantic_move": "compare the two visible constraints",
                    "local_topic": "the visible local choice",
                    "detail_focus": "one visible constraint",
                    "_slot_word_count": 45,
                    "_slot_surface_label": "ordinary_turn",
                }
            }
        )
        self.assertTrue(
            any(
                issue.code == "surface_density_conflict"
                for issue in report.repair_issues
            )
        )

    def test_tone_personal_target_does_not_use_story_share(self) -> None:
        script = load_script_module(
            "generalized_card_run_generate_tone_mapping",
            Path(__file__).resolve().parents[1] / "scripts" / "run_generate.py",
        )
        args = script.build_parser().parse_args(["--tag", "test-tone"])
        command = script._generator_command(
            args=args,
            config_raw_dir=Path("real"),
            seed_pool=Path("seed.json"),
            generated_root=Path("generated"),
            behavior_targets={
                "tone_personal_min_share": 0.21,
                "story_personal_min_share": 0.07,
            },
        )
        index = command.index("--tone-personal-min-share")
        self.assertEqual(command[index + 1], "0.21")

    def test_domain_actor_is_composed_from_planner_row_without_fixed_persona(
        self,
    ) -> None:
        normalized = {
            1: {
                "semantic_move": "separate a viewing artifact from a saved-file artifact",
                "detail_focus": "whether the line remains in the exported file",
                "evidence_mode": "small_observation",
                "reply_relation": "asks_narrow_followup",
                "context_aperture": "parent_only",
                "opening_style": "observation before implication",
            }
        }
        payload = {
            "comment_plans": [
                {
                    "sample_id": 1,
                    "actor": {
                        "participant_key": "A7",
                        "knowledge_boundary": "has only the visible symptom and parent reply",
                        "participation_goal": "isolate where the artifact appears",
                        "evidence_access": "can ask for one observable comparison",
                        "attention_focus": "viewer versus exported file",
                        "interaction_tendency": "asks one diagnostic follow-up",
                        "context_visibility": "parent-local symptom only",
                        "realization_route": "comparison first, then one short question",
                    },
                }
            ]
        }
        enriched = enrich_normalized_plans(payload, normalized)
        state = actor_state_from_plan(enriched[1], sample_id=1)
        self.assertEqual(state.participant_key, "A7")
        self.assertIn("artifact", state.participation_goal)
        self.assertIn("comparison first", state.realization_route)
        self.assertNotIn("photographer", " ".join(state.to_dict().values()).lower())

    def test_comment_planner_defines_actor_in_same_json_response(self) -> None:
        prompt = prompts.comment_planner_prompt(
            self.config,
            SimpleNamespace(
                GENERALIZED_DOMAIN_PROFILE={"perspectives": []},
                GENERALIZED_ACTOR_MODE=MODE_DOMAIN_DERIVED,
                compact=lambda value, limit: str(value)[:limit],
            ),
            seed_post=SimpleNamespace(
                title="Visible issue", body="One visible symptom", content=""
            ),
            target=SimpleNamespace(
                target_comments=1, max_depth_goal=1, shape_label="quiet"
            ),
            branches=[
                SimpleNamespace(
                    branch_id=1,
                    branch_goal="inspect one symptom",
                    anchor_quote="visible symptom",
                    allowed_functions=("question_followup",),
                    content_angles=("setup_troubleshooting",),
                )
            ],
            matched_real_thread=None,
            comments=[{"body": "hidden", "depth": 0, "parent_id": "t3_post"}],
            all_comments=[{"body": "hidden", "depth": 0, "parent_id": "t3_post"}],
        )
        self.assertIn('"actor":', prompt)
        self.assertIn("There is no fixed", prompt)
        self.assertIn("evaluation-excluded R# pattern", prompt)
        self.assertIn("realize it once", prompt)

        no_actor_prompt = prompts.comment_planner_prompt(
            self.config,
            SimpleNamespace(
                GENERALIZED_DOMAIN_PROFILE={"perspectives": []},
                GENERALIZED_ACTOR_MODE="none",
                compact=lambda value, limit: str(value)[:limit],
            ),
            seed_post=SimpleNamespace(
                title="Visible issue", body="One visible symptom", content=""
            ),
            target=SimpleNamespace(
                target_comments=1, max_depth_goal=1, shape_label="quiet"
            ),
            branches=[
                SimpleNamespace(
                    branch_id=1,
                    branch_goal="inspect one symptom",
                    anchor_quote="visible symptom",
                    allowed_functions=("question_followup",),
                    content_angles=("setup_troubleshooting",),
                )
            ],
            matched_real_thread=None,
            comments=[{"body": "hidden", "depth": 0, "parent_id": "t3_post"}],
            all_comments=[{"body": "hidden", "depth": 0, "parent_id": "t3_post"}],
        )
        self.assertNotIn('"actor":', no_actor_prompt)
        self.assertNotIn("thread-local A#", no_actor_prompt)

    def test_domain_actor_mode_makes_distribution_guard_diagnostic(self) -> None:
        with patch.dict(
            os.environ,
            {"GENERALIZED_CARD_ACTOR_CONDITIONING": MODE_DOMAIN_DERIVED},
        ):
            module = configure_generator_backend(load_generator_backend(), self.config)
        problem = module.lexical_overlap_problem(
            text="same repeated phrase",
            previous_comments=[{"content": "same repeated phrase"}],
            task=SimpleNamespace(local_task_id=2, length_bucket="short"),
        )
        self.assertEqual(problem, "")
        self.assertFalse(module.has_blocking_guard_failure(["opening_reused"]))

    def test_endpoint_preflight_retries_transient_timeout_then_defers_to_sdk(
        self,
    ) -> None:
        module = SimpleNamespace(describe_bad_endpoint=lambda **_: "bad endpoint")
        preflight = _endpoint_preflight_with_retry(module)
        with (
            patch.dict(
                os.environ,
                {
                    "ENDPOINT_PREFLIGHT_RETRIES": "3",
                    "ENDPOINT_PREFLIGHT_TIMEOUT": "1",
                    "ENDPOINT_PREFLIGHT_RETRY_DELAY": "0",
                },
            ),
            patch(
                "generalized_card.backend.urllib.request.urlopen",
                side_effect=TimeoutError("temporary timeout"),
            ) as urlopen,
        ):
            preflight(
                role="planner",
                base_url="https://api.openai.com/v1",
                api_key="test",
                allow_remote=True,
            )
            preflight(
                role="writer",
                base_url="https://api.openai.com/v1",
                api_key="test",
                allow_remote=False,
            )
        self.assertEqual(urlopen.call_count, 3)

    def test_generation_resume_rejects_changed_configuration(self) -> None:
        script = load_script_module(
            "generalized_card_run_generate_resume_config",
            Path(__file__).resolve().parents[1] / "scripts" / "run_generate.py",
        )
        existing = {
            "tag": "same",
            "model": "gpt-5.4-mini",
            "posts_per_run": 5,
            "command": ["python", "generator", "--posts-per-run", "5"],
        }
        script._verify_resume_config(existing, dict(existing))
        changed = dict(existing, posts_per_run=1)
        with self.assertRaisesRegex(RuntimeError, "posts_per_run"):
            script._verify_resume_config(existing, changed)

        behavior_changed = dict(existing, writer_prompt="full")
        with self.assertRaisesRegex(RuntimeError, "writer_prompt"):
            script._verify_resume_config(existing, behavior_changed)
        self.assertTrue(
            {
                "domain_claim",
                "writer_prompt",
                "writer_route_lock",
                "social_contract_coherence",
                "reply_sibling_visibility",
                "own_fact_license",
                "speaker_identity",
                "actor_conditioning",
            }.issubset(script.RUN_EXPERIMENT_FIELDS)
        )

    def test_generation_append_extension_requires_complete_prefix(self) -> None:
        script = load_script_module(
            "generalized_card_run_generate_extension",
            Path(__file__).resolve().parents[1] / "scripts" / "run_generate.py",
        )
        with tempfile.TemporaryDirectory() as directory:
            run_root = Path(directory)
            generated = run_root / "generated"
            run_dir = generated / "run_00_sampled_reddit"
            run_dir.mkdir(parents=True)
            (run_dir / "discussion.json").write_text(
                json.dumps({"posts": [{"seed_index": 0}, {"seed_index": 1}]}),
                encoding="utf-8",
            )
            stable = {
                "domain": {"domain_id": "camera"},
                "domain_config": "camera",
                "tag": "same",
                "model": "gpt-5.4-mini",
                "base_url": "https://api.openai.com/v1",
                "seed_pool": "seed.json",
                "domain_profile": "profile.json",
                "domain_profile_sha256": "abc",
                "domain_behavior_targets": {},
                "generated_root": str(generated),
                "pool_size": 150,
                "posts_per_run": 2,
                "sampling_seed": 42,
                "context_dropout_rate": 0.2,
                "context_jitter_rate": 0.15,
                "reasoning_effort": "low",
                "gpt5_reasoning_token_reserve": 256,
                "generator_profile": "generalized-v2",
                "card_core_algorithm_symbols": ["plan_thread"],
                "domain_adaptation_boundaries": ["build_planner_prompt"],
            }
            existing = dict(
                stable,
                max_posts=2,
                generator_policy_version="old",
                generator_core_provenance={"generator": "old"},
                command=[
                    "python",
                    "generator",
                    "--runs",
                    "1",
                    "--max-total-posts",
                    "2",
                ],
            )
            requested = dict(
                stable,
                max_posts=3,
                generator_policy_version="new",
                generator_core_provenance={"generator": "new"},
                command=[
                    "python",
                    "generator",
                    "--runs",
                    "2",
                    "--max-total-posts",
                    "3",
                ],
            )
            script._verify_append_extension(
                existing=existing,
                requested=requested,
                generated_root=generated,
                run_root=run_root,
            )
            lineage = script._extended_generation_lineage(
                existing=existing,
                requested=requested,
                old_max_posts=2,
            )
            self.assertEqual(
                [
                    (row["seed_start"], row["seed_end_exclusive"])
                    for row in lineage["segments"]
                ],
                [(0, 2), (2, 3)],
            )

            (run_dir / "discussion.json").write_text(
                json.dumps({"posts": [{"seed_index": 0}]}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "not complete and contiguous"):
                script._verify_append_extension(
                    existing=existing,
                    requested=requested,
                    generated_root=generated,
                    run_root=run_root,
                )

    def test_generation_policy_upgrade_records_actual_seed_boundary(self) -> None:
        script = load_script_module(
            "generalized_card_run_generate_policy_upgrade",
            Path(__file__).resolve().parents[1] / "scripts" / "run_generate.py",
        )
        with tempfile.TemporaryDirectory() as directory:
            run_root = Path(directory)
            generated = run_root / "generated"
            run_dir = generated / "run_00_sampled_reddit"
            run_dir.mkdir(parents=True)
            (run_dir / "discussion.json").write_text(
                json.dumps({"posts": [{"seed_index": 0}, {"seed_index": 1}]}),
                encoding="utf-8",
            )
            stable = {
                "domain": {"domain_id": "camera"},
                "domain_config": "camera",
                "tag": "same",
                "model": "gpt-5.4-mini",
                "base_url": "https://api.openai.com/v1",
                "seed_pool": "seed.json",
                "domain_profile": "profile.json",
                "domain_profile_sha256": "abc",
                "domain_behavior_targets": {},
                "generated_root": str(generated),
                "pool_size": 150,
                "max_posts": 3,
                "posts_per_run": 1,
                "sampling_seed": 42,
                "context_dropout_rate": 0.42,
                "context_jitter_rate": 0.32,
                "plan_quality": {"strict": True},
                "reasoning_effort": "",
                "gpt5_reasoning_token_reserve": 256,
                "persona_conditioning": {"mode": "none"},
                "generator_profile": "generalized-v2",
                "revision_core_policy_version": "revision",
                "card_core_algorithm_symbols": ["plan_thread"],
                "domain_adaptation_boundaries": ["build_planner_prompt"],
            }
            existing = dict(
                stable,
                generator_policy_version="v12",
                generator_core_provenance={"generator": "old"},
                command=["python", "generator", "--max-total-posts", "3"],
                generation_lineage={
                    "mode": "append_only",
                    "segments": [
                        {
                            "seed_start": 0,
                            "seed_end_exclusive": 3,
                            "generator_policy_version": "v12",
                        }
                    ],
                },
            )
            requested = dict(
                stable,
                post_recovery={"retry_limit": 0},
                generator_policy_version="v13",
                generator_core_provenance={"generator": "new"},
                command=[
                    "python",
                    "generator",
                    "--post-retry-limit",
                    "0",
                    "--post-retry-delay",
                    "15.0",
                    "--max-total-posts",
                    "3",
                ],
            )
            boundary = script._verify_policy_upgrade(
                existing=existing,
                requested=requested,
                generated_root=generated,
                run_root=run_root,
            )
            self.assertEqual(boundary, 2)
            lineage = script._upgraded_generation_lineage(
                existing=existing,
                requested=requested,
                completed_prefix=boundary,
            )
            self.assertEqual(
                [
                    (row["seed_start"], row["seed_end_exclusive"])
                    for row in lineage["segments"]
                ],
                [(0, 2), (2, 3)],
            )

    def test_post_recovery_retries_only_recoverable_failures(self) -> None:
        script = load_script_module(
            "sampled_generator_post_recovery",
            Path(__file__).resolve().parents[2]
            / "scripts"
            / "sampling_generator"
            / "run_sampled_reddit_generator.py",
        )
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            calls = 0

            def transient_operation() -> bool:
                nonlocal calls
                calls += 1
                if calls < 3:
                    raise RuntimeError("temporary strict plan-quality failure")
                return True

            result = script.run_post_with_recovery(
                operation=transient_operation,
                output_dir=output_dir,
                run_index=14,
                post_slot=2,
                seed_index=72,
                retry_limit=0,
                retry_delay=0.0,
            )
            self.assertTrue(result)
            self.assertEqual(calls, 3)
            retry_log = output_dir / "_generation_failures" / "post_retries.jsonl"
            rows = [json.loads(line) for line in retry_log.read_text().splitlines()]
            self.assertEqual(len(rows), 2)
            self.assertTrue(all(row["seed_index"] == 72 for row in rows))
            self.assertTrue(all(row["action"] == "retry_same_post" for row in rows))

            permanent_calls = 0

            def permanent_operation() -> bool:
                nonlocal permanent_calls
                permanent_calls += 1
                raise RuntimeError("Insufficient Balance")

            with self.assertRaisesRegex(RuntimeError, "Insufficient Balance"):
                script.run_post_with_recovery(
                    operation=permanent_operation,
                    output_dir=output_dir,
                    run_index=14,
                    post_slot=2,
                    seed_index=72,
                    retry_limit=0,
                    retry_delay=0.0,
                )
            self.assertEqual(permanent_calls, 1)
            self.assertEqual(len(retry_log.read_text().splitlines()), 2)

    def test_truncated_candidate_json_is_salvaged(self) -> None:
        raw = (
            '{"candidates":['
            '{"style":"a","text":"Keep Sony A7 IV and 24mm."},'
            '{"style":"b","text":"Second complete candidate."},'
            '{"style":"c","text":"truncated'
        )
        rows = parse_candidate_response(raw)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["style"], "a")

    def test_output_audit_rejects_placeholder(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run_00_sampled_reddit"
            run_dir.mkdir(parents=True)
            (run_dir / "discussion.json").write_text(
                json.dumps(
                    {
                        "posts": [
                            {
                                "id": "p1",
                                "comments": [
                                    {
                                        "id": "c1",
                                        "content": "[placeholder response]",
                                        "replies": [],
                                    }
                                ],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            report = audit_generated_root(Path(directory))
            self.assertFalse(report["healthy"])
            self.assertFalse(report["evaluable"])
            self.assertEqual(report["placeholder_comments"], 1)

    def test_output_audit_keeps_perspective_concentration_evaluable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run_00_sampled_reddit"
            run_dir.mkdir(parents=True)
            comments = [
                {
                    "comment_id": index,
                    "perspective_id": "P01",
                    "payload_type": "substantive_advice",
                    "comment_function": "independent_contribution",
                    "semantic_move": f"distinct observation route {index}",
                    "content": f"Distinct camera observation {index} with a separate practical detail.",
                    "replies": [],
                }
                for index in range(1, 9)
            ]
            (run_dir / "discussion.json").write_text(
                json.dumps(
                    {
                        "posts": [
                            {
                                "post_id": "p1",
                                "thread_plan": {"target_comments": len(comments)},
                                "comments": comments,
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            report = audit_generated_root(Path(directory))
            self.assertTrue(report["evaluable"])
            self.assertFalse(report["healthy"])
            self.assertEqual(report["overconcentrated_perspective_posts"], 1)

    def test_output_audit_rejects_one_skipped_slot_even_at_high_acceptance(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run_00_sampled_reddit"
            run_dir.mkdir(parents=True)
            comments = [
                {
                    "comment_id": index,
                    "content": (
                        f"Distinct camera observation number {index} stays locally grounded."
                    ),
                    "replies": [],
                }
                for index in range(1, 101)
            ]
            records = [
                {"comment": comment, "skipped": False} for comment in comments
            ] + [{"comment": None, "skipped": True}]
            (run_dir / "discussion.json").write_text(
                json.dumps(
                    {
                        "posts": [
                            {
                                "post_id": "p1",
                                "thread_plan": {"target_comments": 101},
                                "comments": comments,
                                "generation_records": records,
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            report = audit_generated_root(Path(directory))
            self.assertGreater(report["accepted_share"], 0.98)
            self.assertFalse(report["evaluable"])
            self.assertFalse(report["complete_structural_coverage"])
            self.assertEqual(report["incomplete_recorded_posts"], 1)
            self.assertEqual(report["incomplete_structural_posts"], 1)
            self.assertEqual(report["skipped_generation_slots"], 1)

    def test_output_audit_rejects_legacy_target_count_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run_00_sampled_reddit"
            run_dir.mkdir(parents=True)
            (run_dir / "discussion.json").write_text(
                json.dumps(
                    {
                        "posts": [
                            {
                                "post_id": "legacy",
                                "thread_plan": {"target_comments": 2},
                                "comments": [
                                    {
                                        "comment_id": 1,
                                        "content": "One complete legacy camera comment remains here.",
                                        "replies": [],
                                    }
                                ],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            report = audit_generated_root(Path(directory))
            self.assertFalse(report["evaluable"])
            self.assertFalse(report["complete_structural_coverage"])
            self.assertEqual(report["incomplete_recorded_posts"], 0)
            self.assertEqual(report["incomplete_structural_posts"], 1)

    def test_output_audit_rejects_missing_direct_reply_contract(self) -> None:
        rows = [
            {
                "comment_id": 1,
                "content": "A root contribution with a distinct local point.",
                "replies": [
                    {
                        "comment_id": 2,
                        "parent_comment_id": 1,
                        "content": "A reply that just repeats the local point.",
                        "replies": [],
                    }
                ],
            }
        ]
        records = [
            {
                "task": {
                    "local_task_id": 1,
                    "local_parent_task_id": None,
                    "payload_type": "soft_helpful",
                    "comment_function": "explanation_analysis",
                    "semantic_move": "state the root local point",
                    "decision_boundary": "the root condition",
                    "perspective_id": "P01",
                }
            },
            {
                "task": {
                    "local_task_id": 2,
                    "local_parent_task_id": 1,
                    "payload_type": "soft_helpful",
                    "comment_function": "explanation_analysis",
                    "semantic_move": "repeat the root local point",
                    "decision_boundary": "the root condition",
                    "perspective_id": "P01",
                    "reply_delta_type": "none",
                    "reply_novelty_anchor": "",
                }
            },
        ]
        flattened = list(row for row in rows)
        flattened.extend(rows[0]["replies"])
        recovered = _planner_rows_for_audit(rows=flattened, records=records)
        self.assertEqual(recovered[2]["parent_sample_id"], 1)
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run_00_sampled_reddit"
            run_dir.mkdir(parents=True)
            (run_dir / "discussion.json").write_text(
                json.dumps(
                    {
                        "posts": [
                            {
                                "post_id": "p1",
                                "thread_plan": {"target_comments": 2},
                                "comments": rows,
                                "generation_records": records,
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            report = audit_generated_root(Path(directory))
            self.assertFalse(report["healthy"])
            self.assertEqual(report["reply_contract_violations"], 1)

    def test_output_audit_rejects_internal_perspective_label(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run_00_sampled_reddit"
            run_dir.mkdir(parents=True)
            (run_dir / "discussion.json").write_text(
                json.dumps(
                    {
                        "posts": [
                            {
                                "post_id": "p1",
                                "title": "Camera autofocus question",
                                "thread_plan": {"target_comments": 1},
                                "comments": [
                                    {
                                        "comment_id": "c1",
                                        "perspective_id": "P03",
                                        "content": "P03 had the same autofocus result for me.",
                                        "replies": [],
                                    }
                                ],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            report = audit_generated_root(Path(directory))
            self.assertFalse(report["healthy"])
            self.assertEqual(report["internal_control_label_comments"], 1)
            self.assertEqual(report["prompt_leak_comments"], 1)

    def test_output_audit_allows_natural_placeholder_word(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run_00_sampled_reddit"
            run_dir.mkdir(parents=True)
            (run_dir / "discussion.json").write_text(
                json.dumps(
                    {
                        "posts": [
                            {
                                "post_id": "p1",
                                "thread_plan": {"target_comments": 1},
                                "comments": [
                                    {
                                        "comment_id": "c1",
                                        "content": (
                                            "Would the kit lens just be a placeholder for now, "
                                            "or would you use it for a while?"
                                        ),
                                        "replies": [],
                                    }
                                ],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            report = audit_generated_root(Path(directory))
            self.assertTrue(report["healthy"])
            self.assertTrue(report["evaluable"])
            self.assertEqual(report["placeholder_comments"], 0)

    def test_output_audit_ignores_short_common_exact_match(self) -> None:
        overlap = _closest_real_overlap(
            "one of us one of us",
            ["ONE OF US ONE OF US"],
        )
        self.assertIsNone(overlap)

    def test_output_audit_flags_long_exact_match(self) -> None:
        copied = (
            "this exact camera recommendation was copied from the reference discussion"
        )
        overlap = _closest_real_overlap(copied, [copied])
        self.assertIsNotNone(overlap)
        self.assertTrue(overlap["exact_match"])

    def test_revision_artifact_uses_latest_accepted_round(self) -> None:
        script = load_script_module(
            "generalized_card_run_revise",
            Path(__file__).resolve().parents[1] / "scripts" / "run_revise.py",
        )
        with tempfile.TemporaryDirectory() as directory:
            prefix = Path(directory) / "diversity_lineage_a00"
            history = [
                {
                    "round": 1,
                    "improved": True,
                    "protected_ok": True,
                    "clean_root": "accepted-round-1",
                    "eval_dir": "accepted-round-1-eval",
                    "matched_eval_dir": "accepted-round-1-matched",
                },
                {
                    "round": 2,
                    "improved": False,
                    "protected_ok": True,
                    "clean_root": "rejected-round-2",
                    "eval_dir": "rejected-round-2-eval",
                    "matched_eval_dir": "rejected-round-2-matched",
                },
            ]
            Path(f"{prefix}_controller_history.json").write_text(
                json.dumps(history),
                encoding="utf-8",
            )
            result = script._resolve_final_artifact(
                stage="diversity",
                prefix=prefix,
                fallback={
                    "root": "initial",
                    "scores": "initial.csv",
                    "matched": "initial-matched",
                },
            )
            self.assertEqual(result["root"], "accepted-round-1")
            self.assertEqual(
                result["scores"],
                "accepted-round-1-eval/revised_generated_thread_scores.csv",
            )
            self.assertEqual(result["matched"], "accepted-round-1-matched")

    def test_revision_lineage_changes_with_accepted_input(self) -> None:
        script = load_script_module(
            "generalized_card_run_revise_lineage",
            Path(__file__).resolve().parents[1] / "scripts" / "run_revise.py",
        )
        first = script._artifact_lineage(
            Path("root-a"), Path("scores-a"), Path("matched-a")
        )
        repeated = script._artifact_lineage(
            Path("root-a"), Path("scores-a"), Path("matched-a")
        )
        second = script._artifact_lineage(
            Path("root-b"), Path("scores-b"), Path("matched-b")
        )
        self.assertEqual(first, repeated)
        self.assertNotEqual(first, second)

    def test_cross_prefix_strategy_history_merges_without_duplicate_rounds(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            history_path = Path(directory) / "strategy_history.json"
            prior = {
                "round": 1,
                "input_root": "root-a",
                "output_root": "proposal-a",
                "selected_profile": "high_tail",
                "decision": "rejected",
            }
            history_path.write_text(json.dumps([prior]), encoding="utf-8")
            merged = merge_strategy_history(
                history_path,
                [
                    prior,
                    {
                        "round": 1,
                        "input_root": "root-a",
                        "output_root": "proposal-b",
                        "selected_profile": "middle_mass",
                        "decision": "accepted",
                    },
                ],
            )
            self.assertEqual(len(merged), 2)
            memory = build_memory(
                merged,
                controller="selfbleu",
                target_metrics=("self_bleu_4",),
                current_input_root=Path("root-a"),
            )
            self.assertEqual(memory["aggregate"]["attempted_rounds"], 2)

    def test_selfbert_candidate_parser_preserves_controller_fields(self) -> None:
        rows = parse_selfbert_candidate_response(
            json.dumps(
                {
                    "candidates": [
                        {
                            "style": "local_aside",
                            "discourse_job": "minor_tangent",
                            "preserved_tone": "yes",
                            "preserved_story_mode": "yes",
                            "preserved_stance": "yes",
                            "preserved_reply_relation": "yes",
                            "text": "The lens price is still the awkward part.",
                        }
                    ]
                }
            )
        )
        self.assertEqual(rows[0]["discourse_job"], "minor_tangent")
        self.assertEqual(rows[0]["preserved_stance"], "yes")

    def test_full_revision_order_targets_only_nonpassing_groups(self) -> None:
        script = load_script_module(
            "generalized_card_run_full_revise",
            Path(__file__).resolve().parents[1] / "scripts" / "run_full_revise.py",
        )
        parser = script.build_parser()
        args = parser.parse_args(["--tag", "test"])

        def row(
            *, mwu: float, ks: float, generated: float, real: float
        ) -> dict[str, float]:
            return {
                "mwu_p_value": mwu,
                "ks_p_value": ks,
                "generated_mean": generated,
                "real_mean": real,
            }

        evaluation = {
            metric: row(mwu=0.2, ks=0.2, generated=0.1, real=0.1)
            for metric in script.ALL_METRICS
        }
        evaluation["self_bleu_4"] = row(mwu=0.001, ks=0.001, generated=0.05, real=0.03)
        evaluation["self_bertscore_mean_f1"] = row(
            mwu=0.001,
            ks=0.001,
            generated=0.51,
            real=0.49,
        )
        evaluation["polite_rate"] = row(mwu=0.001, ks=0.001, generated=0.1, real=0.3)
        evaluation["structural_virality"] = row(
            mwu=0.03,
            ks=0.2,
            generated=1.8,
            real=2.2,
        )
        decisions = [
            script.stage_decision(stage, evaluation, args)
            for stage in ("diversity", "selfbert", "tone", "story", "structure")
        ]
        self.assertEqual(
            [item.stage for item in decisions if item.required],
            [
                "diversity",
                "selfbert",
                "tone",
                "structure",
            ],
        )
        self.assertEqual(decisions[-1].story_rounds, 0)
        self.assertEqual(decisions[-1].structure_rounds, 7)

    def test_distribution_diagnosis_uses_middle_mass_for_broad_shift(self) -> None:
        diagnostics = load_script_module(
            "generalized_card_distribution_diagnostics",
            REPO_ROOT / "scripts" / "distribution_diagnostics.py",
        )
        real = diagnostics.pd.Series([0.01, 0.02, 0.03, 0.04, 0.05] * 20)
        generated = diagnostics.pd.Series([0.03, 0.04, 0.05, 0.06, 0.07] * 20)
        result = diagnostics.diagnose_distribution(generated, real)
        self.assertEqual(result.region, "broad_high")
        self.assertEqual(result.direction, "decrease")
        self.assertEqual(result.recommended_profile, "middle_mass")

    def test_generated_parent_repair_recomputes_descendant_depth(self) -> None:
        module = load_generator_backend(profile=GENERALIZED_V2_PROFILE)

        @dataclass(frozen=True)
        class Task:
            local_task_id: int
            local_parent_task_id: int | None
            depth: int
            context_transform: str

        root = Task(1, None, 0, "seed_visible")
        skipped_parent = Task(2, 1, 1, "parent_full")
        child = Task(3, 2, 2, "parent_full")
        grandchild = Task(4, 3, 3, "parent_full")
        task_by_id = {
            task.local_task_id: task
            for task in (root, skipped_parent, child, grandchild)
        }
        actual = {1: {"comment_id": 1, "depth": 0}}

        aligned_child, parent = module.align_task_to_generated_parent(
            child,
            task_by_id=task_by_id,
            actual_by_task=actual,
        )
        self.assertEqual(aligned_child.local_parent_task_id, 1)
        self.assertEqual(aligned_child.depth, 1)
        self.assertEqual(parent["comment_id"], 1)
        actual[3] = {"comment_id": 3, "depth": aligned_child.depth}

        aligned_grandchild, parent = module.align_task_to_generated_parent(
            grandchild,
            task_by_id=task_by_id,
            actual_by_task=actual,
        )
        self.assertEqual(aligned_grandchild.local_parent_task_id, 3)
        self.assertEqual(aligned_grandchild.depth, 2)
        self.assertEqual(parent["comment_id"], 3)

    def test_full_revision_defaults_to_historical_card_core_chain(self) -> None:
        script = load_script_module(
            "generalized_card_run_full_revise_profile",
            Path(__file__).resolve().parents[1] / "scripts" / "run_full_revise.py",
        )
        args = script.build_parser().parse_args(["--tag", "test"])
        self.assertEqual(args.revision_profile, "card-core")
        self.assertEqual(
            script.revision_stages(args.revision_profile),
            ("diversity", "tone"),
        )
        self.assertEqual(
            script.revision_stages("extended"),
            (
                "diversity",
                "selfbert",
                "semantic",
                "tone",
                "emotion",
                "length",
                "story",
                "structure",
            ),
        )

    def test_extended_stage_order_covers_unattempted_metrics_before_repeats(
        self,
    ) -> None:
        script = load_script_module(
            "generalized_card_run_full_revise_coverage_order",
            Path(__file__).resolve().parents[1] / "scripts" / "run_full_revise.py",
        )
        stages = script.revision_stages("extended")
        ordered = script.coverage_order(
            stages,
            {"diversity": 2, "selfbert": 1, "tone": 1},
        )
        self.assertEqual(ordered[:4], ("semantic", "emotion", "length", "story"))
        self.assertLess(ordered.index("structure"), ordered.index("selfbert"))
        self.assertEqual(ordered[-1], "diversity")

    def test_full_revision_includes_exact_remaining_metric_stages(self) -> None:
        script = load_script_module(
            "generalized_card_run_full_revise_exact_metrics",
            Path(__file__).resolve().parents[1] / "scripts" / "run_full_revise.py",
        )
        args = script.build_parser().parse_args(["--tag", "test"])
        evaluation = {
            metric: {
                "mwu_p_value": 0.2,
                "ks_p_value": 0.2,
                "generated_mean": 0.2,
                "real_mean": 0.2,
            }
            for metric in script.ALL_METRICS
        }
        for metric in ("semantic_mean_cosine", "emotion_entropy", "length_cv"):
            evaluation[metric]["mwu_p_value"] = 0.001
            evaluation[metric]["ks_p_value"] = 0.001
        required = [
            stage
            for stage in ("semantic", "emotion", "length")
            if script.stage_decision(stage, evaluation, args).required
        ]
        self.assertEqual(required, ["semantic", "emotion", "length"])

    def test_text_metric_candidate_guard_is_domain_aware(self) -> None:
        ok, reason, overlap, ratio = validate_text_metric_candidate(
            old="The Sony A7 IV costs $2,000 and the grip feels cramped.",
            candidate="The Sony A7 IV is $2,000, and that cramped grip is still the issue.",
            visible_context="Sony A7 IV camera discussion",
            config=self.config,
            metric="semantic_mean_cosine",
        )
        self.assertTrue(ok, reason)
        self.assertGreater(overlap, 0.5)
        self.assertGreater(ratio, 0.5)
        bad, bad_reason, _, _ = validate_text_metric_candidate(
            old="The Sony A7 IV costs $2,000 and the grip feels cramped.",
            candidate="The Canon R5 costs $2,500 and feels better.",
            visible_context="Sony A7 IV camera discussion",
            config=self.config,
            metric="semantic_mean_cosine",
        )
        self.assertFalse(bad)
        self.assertIn("claim_overlap", bad_reason)

    def test_text_metric_candidate_rank_prefers_exact_gap_reduction(self) -> None:
        from generalized_card.text_metric_reviser import CandidateDecision

        weaker = CandidateDecision(
            1, True, "a", "b", "x", "accepted", 0.5, 0.48, 0.02, 0.8, 1.0
        )
        stronger = CandidateDecision(
            1, True, "a", "c", "x", "accepted", 0.5, 0.44, 0.06, 0.7, 1.0
        )
        self.assertGreater(candidate_rank(stronger), candidate_rank(weaker))

    def test_reviser_adapter_preserves_core_comment_budget_functions(self) -> None:
        module = SimpleNamespace(
            build_reviser_prompt=lambda **_: "original",
            filtered_named_entities=lambda _: set(),
            preserves_numbers=lambda *_: True,
            parse_reviser_response=lambda _: [],
            select_candidate_comments=lambda *, comments, limit: list(comments)[:limit],
            rewrite_budget_for_thread=lambda **_: 1,
        )
        original_select = module.select_candidate_comments
        original_budget = module.rewrite_budget_for_thread
        configure_reviser_backend(module, kind="selfbleu", config=self.config)
        comments = ["a", "b", "c", "d"]
        self.assertIs(module.select_candidate_comments, original_select)
        self.assertIs(module.rewrite_budget_for_thread, original_budget)
        self.assertEqual(
            module.select_candidate_comments(comments=comments, limit=1), ["a"]
        )
        self.assertEqual(module.rewrite_budget_for_thread(comment_count=4), 1)

    def test_real_reviser_adapters_only_replace_domain_boundaries(self) -> None:
        algorithm_functions = {
            "selfbleu": (
                "select_thread_targets",
                "rewrite_budget_for_thread",
                "select_candidate_comments",
                "choose_rewrite",
                "decision_rank",
            ),
            "selfbert": ("select_thread_targets", "rewrite_budget", "process_thread"),
            "tone": (
                "select_tone_targets",
                "rewrite_budget_for_thread",
                "adaptive_polite_rewrite_budget",
                "select_candidate_comments",
                "choose_tone_rewrite",
                "decision_rank",
            ),
            "story": ("select_story_targets", "revise_target"),
            "structure": (
                "select_structure_targets",
                "enumerate_move_candidates",
                "revise_target",
            ),
        }
        for kind, names in algorithm_functions.items():
            module = load_reviser_backend(kind)
            before = {name: getattr(module, name) for name in names}
            configure_reviser_backend(module, kind=kind, config=self.config)
            for name in names:
                self.assertIs(getattr(module, name), before[name], f"{kind}.{name}")
            self.assertEqual(
                module.GENERALIZED_CARD_REVISER_PARITY["unexpected_backend_functions"],
                [],
                kind,
            )

    def test_generalized_controller_exposes_no_thread_cap(self) -> None:
        full = load_script_module(
            "generalized_card_run_full_revise_no_cap",
            Path(__file__).resolve().parents[1] / "scripts" / "run_full_revise.py",
        )
        stage = load_script_module(
            "generalized_card_run_revise_no_cap",
            Path(__file__).resolve().parents[1] / "scripts" / "run_revise.py",
        )
        self.assertNotIn(
            "max_threads_per_round",
            {action.dest for action in full.build_parser()._actions},
        )
        self.assertNotIn(
            "max_threads_per_round",
            {action.dest for action in stage.build_parser()._actions},
        )

    def test_diversity_command_uses_core_local_budgets_without_thread_cap(self) -> None:
        stage = load_script_module(
            "generalized_card_run_revise_core_budget",
            Path(__file__).resolve().parents[1] / "scripts" / "run_revise.py",
        )
        args = stage.build_parser().parse_args(
            ["--tag", "test", "--stage", "diversity"]
        )
        command = stage._controller_command(
            args=args,
            config={
                "seed_pool": "seed.json",
                "domain": {"real_scores_csv": "real.csv"},
                "max_posts": 150,
                "posts_per_run": 5,
            },
            generated_root=Path("generated"),
            scores=Path("scores.csv"),
            matched=Path("matched"),
            model="gpt-5.4-mini",
            base_url="https://api.openai.com/v1",
            prefix=Path("revision"),
            strategy_history=Path("strategy-history.json"),
        )
        self.assertIn("--unbounded-coverage", command)
        self.assertNotIn("--max-threads-per-round", command)
        self.assertIn("--protected-metrics", command)
        protected = command.index("--protected-metrics")
        self.assertIn("avg_depth", command[protected + 1])
        self.assertIn("structural_virality", command[protected + 1])
        self.assertIn("emotion_entropy", command[protected + 1])
        improvement = command.index("--min-round-improvement")
        self.assertEqual(command[improvement + 1], "0.001")
        self.assertTrue(command[1].endswith("run_selfbleu_revision_controller.py"))
        self.assertIn("--continue-after-reject", command)
        self.assertIn("--verbose-candidates", command)
        self.assertIn("--playbook", command)
        strategy_history = command.index("--strategy-history-json")
        self.assertEqual(command[strategy_history + 1], "strategy-history.json")

        card_controller = load_script_module(
            "generalized_card_core_metric_controller",
            Path(__file__).resolve().parents[2]
            / "scripts"
            / "run_metric_revision_controller.py",
        )
        high_tail = card_controller.profile_params("high_tail")
        self.assertEqual(high_tail["max-rewrite-share"], 0.14)
        self.assertEqual(high_tail["max-rewrite-budget"], 6)
        self.assertEqual(high_tail["candidate-comments-per-thread"], 14)

    def test_tone_command_uses_paper_card_acceptance_thresholds(self) -> None:
        stage = load_script_module(
            "generalized_card_run_revise_tone_thresholds",
            Path(__file__).resolve().parents[1] / "scripts" / "run_revise.py",
        )
        args = stage.build_parser().parse_args(["--tag", "test", "--stage", "tone"])
        command = stage._controller_command(
            args=args,
            config={
                "seed_pool": "seed.json",
                "domain": {"real_scores_csv": "real.csv"},
                "max_posts": 150,
                "posts_per_run": 5,
            },
            generated_root=Path("generated"),
            scores=Path("scores.csv"),
            matched=Path("matched"),
            model="gpt-5.4-mini",
            base_url="https://api.openai.com/v1",
            prefix=Path("revision"),
            strategy_history=Path("strategy-history.json"),
        )
        tolerance = command.index("--protected-quality-drop-tolerance")
        improvement = command.index("--min-round-improvement")
        self.assertEqual(command[tolerance + 1], "0.01")
        self.assertEqual(command[improvement + 1], "0.005")
        self.assertIn("--continue-after-reject", command)
        self.assertIn("--unbounded-coverage", command)

    def test_extension_metric_profiles_do_not_set_global_thread_cap(self) -> None:
        selfbert = load_script_module(
            "generalized_card_selfbert_controller_no_cap",
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "run_selfbert_revision_controller.py",
        )
        text_metric = load_script_module(
            "generalized_card_text_controller_no_cap",
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "run_text_metric_revision_controller.py",
        )
        forbidden = {"max-threads"}
        for profile in ("high_tail", "middle_mass", "shape_safe"):
            self.assertTrue(forbidden.isdisjoint(selfbert.profile_params(profile)))
            self.assertTrue(
                forbidden.isdisjoint(
                    text_metric.profile_params("semantic_mean_cosine", profile)
                )
            )

    def test_all_local_reviser_stages_use_deviation_driven_coverage(self) -> None:
        stage = load_script_module(
            "generalized_card_run_revise_dynamic_coverage",
            Path(__file__).resolve().parents[1] / "scripts" / "run_revise.py",
        )
        config = {
            "seed_pool": "seed.json",
            "domain": {"real_scores_csv": "real.csv"},
            "max_posts": 150,
            "posts_per_run": 5,
        }
        for stage_name, expected_flag in (
            ("diversity", "--unbounded-coverage"),
            ("tone", "--unbounded-coverage"),
            ("story-structure", "--deviation-driven-coverage"),
        ):
            args = stage.build_parser().parse_args(
                ["--tag", "test", "--stage", stage_name]
            )
            command = stage._controller_command(
                args=args,
                config=config,
                generated_root=Path("generated"),
                scores=Path("scores.csv"),
                matched=Path("matched"),
                model="gpt-5.4-mini",
                base_url="https://api.openai.com/v1",
                prefix=Path("revision"),
                strategy_history=Path("strategy-history.json"),
            )
            self.assertIn(expected_flag, command, stage_name)

        selfbert = load_script_module(
            "generalized_card_selfbert_dynamic_coverage",
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "run_selfbert_revision_controller.py",
        )
        args = selfbert.build_parser().parse_args(
            [
                "generated",
                "--scores-csv",
                "scores.csv",
                "--matched-eval-dir",
                "matched",
                "--seed-post-pool-json",
                "seed.json",
                "--real-scores-csv",
                "real.csv",
                "--output-prefix",
                "revision",
            ]
        )
        command = selfbert.build_reviser_command(
            args=args,
            profile="middle_mass",
            input_root=Path("generated"),
            scores_csv=Path("scores.csv"),
            matched_eval_dir=Path("matched"),
            output_root=Path("output"),
            direction="increase",
        )
        self.assertIn("--deviation-driven-coverage", command)
        self.assertEqual(command[command.index("--direction") + 1], "increase")

    def test_full_revision_repairs_lower_than_real_selfbert(self) -> None:
        script = load_script_module(
            "generalized_card_run_full_revise_lower_selfbert",
            Path(__file__).resolve().parents[1] / "scripts" / "run_full_revise.py",
        )
        args = script.build_parser().parse_args(["--tag", "test"])
        evaluation = {
            "self_bertscore_mean_f1": {
                "mwu_p_value": 0.001,
                "ks_p_value": 0.001,
                "generated_mean": 0.45,
                "real_mean": 0.49,
            }
        }
        decision = script.stage_decision("selfbert", evaluation, args)
        self.assertTrue(decision.required)
        self.assertNotIn("monitor_only", decision.reason)

    def test_selfbert_exact_gate_supports_increase_direction(self) -> None:
        module = load_reviser_backend("selfbert")
        target = module.CommentRef(1, None, 0, "old", {}, {})
        comments = [
            target,
            module.CommentRef(2, None, 0, "peer one", {}, {}),
            module.CommentRef(3, None, 0, "peer two", {}, {}),
        ]
        gate = module.ExactSelfBertGate.__new__(module.ExactSelfBertGate)
        gate.score_target_pairs = lambda replacement_text, *_: (
            [0.40, 0.40] if replacement_text == "old" else [0.50, 0.50]
        )
        result = gate.evaluate(
            old_text="old",
            candidate="new",
            comments=comments,
            target=target,
            current_thread_selfbert=0.40,
            real_thread_selfbert=0.50,
            min_thread_gain=0.0001,
            max_pair_drop=0.20,
            max_real_undershoot=0.02,
            direction="increase",
        )
        self.assertTrue(result["accepted"], result["reason"])
        self.assertGreater(result["projected_thread_selfbert"], 0.40)
        self.assertFalse(module.target_reached(0.45, 0.48, "increase"))
        self.assertTrue(module.target_reached(0.49, 0.48, "increase"))

    def test_story_reviser_supports_claim_safe_increase_direction(self) -> None:
        module = load_reviser_backend("story")
        intended = module.CommentRef(
            1,
            None,
            0,
            "I tried the lens in rain.",
            {"story_mode": "tiny_personal_context"},
            {},
        )
        ordinary = module.CommentRef(
            2,
            None,
            0,
            "The lens is sealed.",
            {"story_mode": "no_story"},
            {},
        )
        self.assertTrue(module.story_intended(intended))
        self.assertFalse(module.story_intended(ordinary))
        args = SimpleNamespace(
            min_claim_overlap=0.5,
            min_word_ratio=0.5,
            max_word_ratio=1.5,
            min_comment_probability_drop=0.03,
            min_gap_reduction=0.001,
            max_real_undershoot=0.025,
        )
        accepted = module.candidate_rejection_reason(
            old_text="I tried the lens in rain.",
            new_text="I tried the lens in rain, then checked the seal.",
            old_probability=0.20,
            new_probability=0.50,
            old_gap=0.30,
            new_gap=0.10,
            new_thread_probability=0.40,
            real_thread_probability=0.50,
            claim_overlap_value=0.8,
            word_ratio_value=1.2,
            args=args,
            direction="increase",
        )
        self.assertEqual(accepted, "accepted")
        wrong_direction = module.candidate_rejection_reason(
            old_text="I tried the lens in rain.",
            new_text="I tried the lens in rain, then checked the seal.",
            old_probability=0.20,
            new_probability=0.10,
            old_gap=0.30,
            new_gap=0.40,
            new_thread_probability=0.10,
            real_thread_probability=0.50,
            claim_overlap_value=0.8,
            word_ratio_value=1.2,
            args=args,
            direction="increase",
        )
        self.assertIn("increase_too_small", wrong_direction)

    def test_extended_no_fail_requires_all_five_target_metrics(self) -> None:
        script = load_script_module(
            "generalized_card_run_full_revise_required_metrics",
            Path(__file__).resolve().parents[1] / "scripts" / "run_full_revise.py",
        )
        summary = {
            metric: {"status": "PASS"} for metric in script.REQUIRED_EXTENDED_METRICS
        }
        self.assertEqual(script.required_nonpassing_metrics(summary), [])
        summary["mean_story_probability"]["status"] = "PARTIAL"
        summary["emotion_entropy"]["status"] = "FAIL"
        self.assertEqual(
            script.required_nonpassing_metrics(summary),
            ["mean_story_probability", "emotion_entropy"],
        )

    def test_revision_policy_upgrade_does_not_relabel_generator(self) -> None:
        source = {
            "generator_policy_version": "generator-v15",
            "revision_core_policy_version": (
                "generalized-card-revision-v6-dynamic-coverage-history-20260807"
            ),
        }
        upgraded = upgrade_revision_policy_config(source)
        self.assertEqual(upgraded["generator_policy_version"], "generator-v15")
        self.assertEqual(
            upgraded["revision_core_policy_version"],
            REVISION_CORE_POLICY_VERSION,
        )
        self.assertEqual(
            upgraded["revision_policy_history"],
            ["generalized-card-revision-v6-dynamic-coverage-history-20260807"],
        )
        self.assertEqual(
            source["revision_core_policy_version"],
            "generalized-card-revision-v6-dynamic-coverage-history-20260807",
        )

    def test_generation_resume_preserves_revision_lineage(self) -> None:
        script = load_script_module(
            "generalized_card_run_generate_revision_lineage",
            Path(__file__).resolve().parents[1] / "scripts" / "run_generate.py",
        )
        existing = {
            "generator_policy_version": "generator-v17",
            "revision_core_policy_version": "revision-v6",
            "revision_policy_history": ["revision-v5"],
        }
        requested = {
            "generator_policy_version": "generator-v17",
            "revision_core_policy_version": "revision-v7",
        }
        script._preserve_revision_lineage(existing, requested)
        self.assertEqual(requested["generator_policy_version"], "generator-v17")
        self.assertEqual(requested["revision_core_policy_version"], "revision-v6")
        self.assertEqual(requested["revision_policy_history"], ["revision-v5"])

    def test_full_revision_reports_pass_regressions_for_audit(self) -> None:
        script = load_script_module(
            "generalized_card_run_full_revise_outer_gate",
            Path(__file__).resolve().parents[1] / "scripts" / "run_full_revise.py",
        )
        before = {metric: {"status": "PASS"} for metric in script.ALL_METRICS}
        after = {metric: dict(row) for metric, row in before.items()}
        after["neutral_rate"]["status"] = "PARTIAL"
        after["avg_depth"]["status"] = "FAIL"
        self.assertEqual(
            script.passing_metric_regressions(before, after),
            ["neutral_rate", "avg_depth"],
        )

    def test_protected_pass_to_partial_rejects_round(self) -> None:
        controller = load_script_module(
            "generalized_card_metric_controller_protection",
            REPO_ROOT / "scripts" / "run_metric_revision_controller.py",
        )
        before = {
            "neutral_rate": {
                "mwu_p_value": 0.20,
                "ks_p_value": 0.20,
                "cliffs_delta": 0.01,
                "wasserstein_distance": 0.01,
                "real_mean": 0.2,
                "generated_mean": 0.2,
            }
        }
        after = {
            "neutral_rate": {
                **before["neutral_rate"],
                "mwu_p_value": 0.04,
            }
        }
        report = controller.protected_report(before, after, "neutral_rate")
        self.assertTrue(report["hard_failure"])
        self.assertEqual(report["metrics"][0]["before"]["status"], "PASS")
        self.assertEqual(report["metrics"][0]["after"]["status"], "PARTIAL")

    def test_full_revision_only_skips_truly_unsupported_reverse_directions(
        self,
    ) -> None:
        script = load_script_module(
            "generalized_card_run_full_revise_directions",
            Path(__file__).resolve().parents[1] / "scripts" / "run_full_revise.py",
        )
        args = script.build_parser().parse_args(
            ["--tag", "test", "--revision-profile", "extended"]
        )
        evaluation = {
            metric: {
                "mwu_p_value": 0.001,
                "ks_p_value": 0.001,
                "generated_mean": 0.1,
                "real_mean": 0.2,
            }
            for metric in script.ALL_METRICS
        }
        diversity = script.stage_decision("diversity", evaluation, args)
        story = script.stage_decision("story", evaluation, args)
        structure = script.stage_decision("structure", evaluation, args)
        self.assertFalse(diversity.required)
        self.assertIn("monitor_only", diversity.reason)
        self.assertTrue(story.required)
        self.assertTrue(structure.required)
        self.assertEqual(story.story_rounds, 7)
        self.assertEqual(structure.structure_rounds, 7)

    def test_writer_blackboard_preserves_core_tags_and_pressure(self) -> None:
        backend = configure_generator_backend(load_generator_backend(), self.config)
        task = SimpleNamespace(
            speaker_role="contrarian",
            tone_shape="mild_caveat",
            voice="blunt",
            payload_type="correction",
            comment_function="correction_caveat",
            evidence_mode="none_assertion",
            story_mode="no_story",
            utterance_mode="correction_only",
            length_bucket="short",
            surface_texture="plain",
            surface_skeleton="short local correction",
            perspective_id="P03",
            domain_intent="compare autofocus behavior",
            claim_key="autofocus_behavior",
        )
        comments = [
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
            },
            {
                "content": "I would check the mount first.",
                "depth": 2,
                "speaker_role": "advisor",
                "tone_shape": "neutral_fact",
                "payload_type": "advice",
                "utterance_mode": "local_advice",
                "voice": "casual_neutral",
                "length_bucket": "medium",
                "surface_texture": "plain",
                "story_mode": "no_story",
            },
        ]
        rendered = prompts._thread_memory(backend, comments, current_task=task)
        self.assertIn("gratitude_reply", rendered)
        self.assertIn("soft_ack", rendered)
        self.assertIn("advice", rendered)
        self.assertIn("Thread-level distribution pressure:", rendered)
        self.assertIn("Recent discourse shapes:", rendered)
        self.assertIn("Current sampled slot:", rendered)
        self.assertNotIn("P03", rendered)
        self.assertIn("domain perspective", rendered)
        self.assertIn("autofocus_behavior", rendered)
        self.assertIn("Thanks, that detail helps.", rendered)
        self.assertIn("I would check the mount first.", rendered)
        self.assertIn("generated text and private controls only", rendered)

    def _tone_writer_prompt(self, module: Any, tone: str, **overrides: Any) -> str:
        """Render a substantive writer prompt carrying one tone contract.

        `overrides` are applied to the finalized task, so a test can supply a
        field a real run populates from the matched real comment but this
        fixture has no source for -- `surface_skeleton` above all.
        """

        task = module.CommentTask(
            local_task_id=1,
            local_parent_task_id=None,
            depth=0,
            branch_id=1,
            branch_goal="weigh the grip against handling for long shoots",
            visible_scope="seed",
            local_anchor="Sony A7 IV grip",
            comment_function="verdict_evaluation",
            content_angle="fit_use_case",
            evidence_mode="firsthand_experience",
            story_mode="no_story",
            voice="casual_neutral",
            payload_type="soft_helpful",
            length_bucket="long",
            speaker_role="datapoint_only",
            utterance_mode="local_answer_with_context",
            surface_texture="plain",
            allow_first_person_frame=False,
            allow_uncertainty_frame=False,
            planner_intent="give a verdict on grip comfort for long shoots",
            must_not_do="Do not add a full review.",
            real_word_count=140,
            semantic_move="commit to a verdict on grip comfort over a long shoot",
            local_topic="grip comfort",
            reply_relation="answers_parent",
            stance="agree",
            detail_focus="grip comfort",
            avoid_repeating="complete review",
            claim_key="grip_verdict",
            claim_family="direct_answer",
            opening_style="verdict then the condition that produced it",
            context_aperture="full_seed",
            tone_shape="neutral_fact",
        )
        task = apply_planner_distribution_fields(task, {"tone_class": tone})
        self.assertEqual(task.tone_target, tone)
        seed = module.SeedPost(
            index=0,
            title="Sony A7 IV grip question",
            body="Is the grip comfortable over a long shoot?",
            content="Sony A7 IV grip question\nIs the grip comfortable over a long shoot?",
            source_raw_post_id="x",
            real_num_comments=8,
            metadata={},
        )
        finalized = module.finalize_rebalanced_task(task)
        if overrides:
            finalized = replace(finalized, **overrides)
        return module.build_writer_prompt(
            profile="gpt54_reddit_writer",
            seed_post=seed,
            task=finalized,
            parent_comment=None,
            previous_comments=[],
            recent_openings=[],
        )

    def test_polite_writer_contract_is_not_cancelled_by_the_substitution_rule(
        self,
    ) -> None:
        # The tone control asks for an appreciative acknowledgement while the
        # hard rules used to forbid acknowledgement and first-person framing
        # outright, so the register could not be realized at all.
        module = configure_generator_backend(load_generator_backend(), self.config)
        rendered = self._tone_writer_prompt(module, "polite")
        self.assertNotIn(
            "Do not replace it with a generic agreement, acknowledgement",
            rendered,
        )
        self.assertIn("first-person positive frame is required here", rendered)
        self.assertIn("does not bar the interpersonal", rendered)
        self.assertIn("Ordinary hedges and brief thanks are allowed", rendered)
        self.assertIn("Do not narrate a sequence of events", rendered)

    def _actor_prompt(self, module: Any, *, low_info: bool) -> str:
        """Render a writer prompt with actor conditioning on, through the real path."""

        from generalized_card.actor_conditioning import (
            MODE_DOMAIN_DERIVED,
            actor_state_from_plan,
            assignment_key,
        )

        seed = module.SeedPost(
            index=0,
            title="Sony A7 IV grip question",
            body="Is the grip comfortable over a long shoot?",
            content="Sony A7 IV grip question\nIs the grip comfortable over a long shoot?",
            source_raw_post_id="actor-seed",
            real_num_comments=8,
            metadata={},
        )
        task = module.CommentTask(
            local_task_id=1,
            local_parent_task_id=None,
            depth=0,
            branch_id=1,
            branch_goal="weigh the grip against handling for long shoots",
            visible_scope="seed",
            local_anchor="Sony A7 IV grip",
            comment_function="reaction" if low_info else "verdict_evaluation",
            content_angle="fit_use_case",
            evidence_mode="none_assertion" if low_info else "firsthand_experience",
            story_mode="no_story",
            voice="casual_neutral",
            payload_type="low_info_reaction" if low_info else "soft_helpful",
            length_bucket="micro" if low_info else "long",
            speaker_role="side_observer" if low_info else "datapoint_only",
            utterance_mode="local_answer_with_context",
            surface_texture="plain",
            allow_first_person_frame=False,
            allow_uncertainty_frame=False,
            planner_intent="give a verdict on grip comfort for long shoots",
            must_not_do="Do not add a full review.",
            real_word_count=4 if low_info else 140,
            semantic_move="commit to a verdict on grip comfort over a long shoot",
            local_topic="grip comfort",
            reply_relation="answers_parent",
            stance="agree",
            detail_focus="grip comfort",
            avoid_repeating="complete review",
            claim_key="grip_verdict",
            claim_family="direct_answer",
            opening_style="verdict then the condition that produced it",
            context_aperture="full_seed",
            tone_shape="neutral_fact",
        )
        task = module.finalize_rebalanced_task(task)
        module.GENERALIZED_ACTOR_MODE = MODE_DOMAIN_DERIVED
        module.GENERALIZED_ACTOR_ASSIGNMENTS = {
            assignment_key(seed, 1): actor_state_from_plan(
                {
                    "actor_participant_key": "A7",
                    "actor_knowledge_boundary": "only what the seed shows about grip",
                    "actor_participation_goal": "add one narrow handling datapoint",
                    "actor_evidence_access": "own long-shoot handling",
                    "actor_attention_focus": "grip depth",
                    "actor_interaction_tendency": "brief and concrete",
                    "actor_context_visibility": "the seed's handling question",
                    "actor_realization_route": "TESTROUTE clause then a short qualifier",
                },
                sample_id=1,
            )
        }
        try:
            return module.build_writer_prompt(
                profile="gpt54_reddit_writer",
                seed_post=seed,
                task=task,
                parent_comment=None,
                previous_comments=[],
                recent_openings=[],
            )
        finally:
            module.GENERALIZED_ACTOR_MODE = "none"
            module.GENERALIZED_ACTOR_ASSIGNMENTS = {}

    def _profile_with_conversations(self) -> dict[str, Any]:
        """A minimal excluded-reference bank carrying two real-shaped exchanges.

        One thread is topically close to the seed and one is far from it, so a
        test can assert that the far one is preferred: the fragments are chosen
        for the shape of the exchange, not for content.
        """

        rows: list[dict[str, Any]] = []

        def add(post_id: str, depth: int, text: str) -> None:
            rows.append(
                {
                    "reference_id": f"R{len(rows) + 1:05d}",
                    "source_post_id": post_id,
                    "source_thread_hash": post_id,
                    "thread_title": f"thread {post_id}",
                    "thread_context": "",
                    "text": text,
                    "depth": str(depth),
                    "parent_scope": "op" if depth == 0 else "reply",
                    "word_count": len(text.split()),
                    "surface_role": "local_turn",
                }
            )

        # far from the seed: printers, and it is the richer exchange
        add("far", 0, "The duplex unit on this printer jams every third job and support blames the paper.")
        add("far", 0, "https://example.org/teardown shows the roller assembly is the same part as the older model.")
        add("far", 1, "> support blames the paper\nThat is what they told me too, then it jammed with their own brand.")
        add("far", 2, "Mine did the same until I reseated the tray. Took two minutes.")
        add("far", 1, "wait, third job? mine goes further than that")
        # close to the seed: compact cameras for travel, but flat, no replies
        add("near", 0, "The Ricoh GR III is a great compact for travel and street photography work.")
        add("near", 0, "An RX100 compact travel camera covers more range for street use than a GR III.")
        add("near", 0, "For travel and street, a compact camera like the GR III or RX100 is the usual pick.")
        add("near", 0, "Street and travel shooters usually land on the GR III or the RX100 compact.")
        return {"profile_sha256": "test-conversations", "reference_viewpoints": rows}

    def _planner_prompt(self, module: Any) -> str:
        """Render a real Planner prompt through the shipped builder."""

        seed = module.SeedPost(
            index=0,
            title="Which compact should I get for travel?",
            body="Looking at the Ricoh GR III and the RX100. Mostly street and travel.",
            content="Which compact should I get for travel?\nRicoh GR III or RX100.",
            source_raw_post_id="conv-seed",
            real_num_comments=18,
            metadata={},
        )
        target = module.sample_thread_target(
            seed_post=seed,
            rng=random.Random(20260828),
            max_comments_per_post=0,
            count_scale=1.0,
        )
        return module.build_planner_prompt(
            seed_post=seed,
            target=target,
            matched_real_thread=None,
            matched_real_comments=0,
            global_memory={},
        )

    def test_interaction_scope_off_leaves_the_planner_prompt_untouched(self) -> None:
        from generalized_card import conversation_reference as cr

        module = configure_generator_backend(load_generator_backend(), self.config)
        cr.set_interaction_scope("off")
        rendered = self._planner_prompt(module)
        self.assertNotIn("How people actually interact here", rendered)

    def test_interaction_scope_reaches_the_planner_prompt_with_real_exchanges(
        self,
    ) -> None:
        # E15 and the 2026-08-27 lesson: an arm's compliance gate is an OFFLINE
        # test that has to pass before the run is priced. v126 cost $0.81 to
        # learn this from an artifact instead.
        from generalized_card import conversation_reference as cr

        module = configure_generator_backend(load_generator_backend(), self.config)
        module.GENERALIZED_DOMAIN_PROFILE = self._profile_with_conversations()
        cr.set_interaction_scope("conversation")
        try:
            rendered = self._planner_prompt(module)
        finally:
            cr.set_interaction_scope("off")
        self.assertIn("How people actually interact here", rendered)
        self.assertIn("Conversation 1", rendered)
        # every fragment must carry at least one reply, which is the whole point
        self.assertIn("[reply at depth", rendered)

    def test_interaction_fragments_are_whole_threads_with_replies(self) -> None:
        from generalized_card import conversation_reference as cr

        module = configure_generator_backend(load_generator_backend(), self.config)
        profile = self._profile_with_conversations()
        cr.set_interaction_scope("conversation")
        try:
            frags = cr.select_conversation_fragments(
                profile,
                seed_title="Which compact should I get for travel?",
                seed_body="Ricoh GR III or RX100, street and travel.",
                exclude_post_ids={"conv-seed"},
            )
        finally:
            cr.set_interaction_scope("off")
        self.assertGreaterEqual(len(frags), 1)
        for group in frags:
            self.assertGreaterEqual(len(group), cr.MIN_FRAGMENT_COMMENTS)
            depths = [int(str(row.get("depth") or 0) or 0) for row in group]
            self.assertTrue(any(d >= 1 for d in depths), "a fragment carries no reply")
            posts = {str(row.get("source_post_id")) for row in group}
            self.assertEqual(len(posts), 1, "a fragment mixes source threads")

    def test_interaction_fragments_are_topically_distant_from_the_seed(self) -> None:
        # The fragments are here for their shape. Distance is what makes them
        # safer than the ranked rows beside them, not riskier.
        from generalized_card import conversation_reference as cr
        from generalized_card.viewpoint_bank import reference_viewpoint_window

        module = configure_generator_backend(load_generator_backend(), self.config)
        profile = self._profile_with_conversations()
        title = "Which compact should I get for travel?"
        body = "Ricoh GR III or RX100, street and travel."
        cr.set_interaction_scope("conversation")
        try:
            frags = cr.select_conversation_fragments(
                profile, seed_title=title, seed_body=body, exclude_post_ids=set()
            )
        finally:
            cr.set_interaction_scope("off")
        ranked = reference_viewpoint_window(
            profile, seed_title=title, seed_body=body, limit=36
        )
        seed_tokens = cr._tokens(f"{title} {body}")

        def overlap(rows: Any) -> float:
            return sum(
                len(cr._tokens(row.get("text")) & seed_tokens) for row in rows
            ) / max(1, len(rows))

        flat = [row for group in frags for row in group]
        self.assertLess(overlap(flat), overlap(ranked))

    def test_reply_material_only_under_full_scope(self) -> None:
        from generalized_card import conversation_reference as cr

        cr.set_interaction_scope("conversation")
        self.assertFalse(cr.reply_material_enabled())
        self.assertTrue(cr.planner_fragments_enabled())
        cr.set_interaction_scope("full")
        self.assertTrue(cr.reply_material_enabled())
        cr.set_interaction_scope("off")
        self.assertFalse(cr.planner_fragments_enabled())

    def test_sentence_pacing_states_the_slots_own_ratio_in_a_real_prompt(self) -> None:
        # E15: an arm's compliance is measured on the RENDERED prompt, through
        # the dispatcher, not on the plan. G113: the cue has to carry a concrete
        # number because `pacing` is a category and a category buys 0.23.
        from generalized_card import length_policy

        module = configure_generator_backend(load_generator_backend(), self.config)
        self.assertEqual(prompts._writer_prompt_mode(module), "focused")
        length_policy.set_sentence_pacing("measured")
        try:
            rendered = self._tone_writer_prompt(
                module,
                "polite",
                surface_skeleton="long uneven Reddit paragraph, about 14 sentences",
            )
        finally:
            length_policy.set_sentence_pacing("off")
        self.assertIn("averaging about", rendered)
        self.assertIn("sentences", rendered)
        self.assertIn("do not drift toward a comfortable middle length", rendered)

    def test_sentence_pacing_off_leaves_the_cue_untouched(self) -> None:
        from generalized_card import length_policy

        module = configure_generator_backend(load_generator_backend(), self.config)
        length_policy.set_sentence_pacing("off")
        rendered = self._tone_writer_prompt(
            module,
            "polite",
            surface_skeleton="long uneven Reddit paragraph, about 14 sentences",
        )
        self.assertNotIn("averaging about", rendered)

    def test_sentence_pacing_preserves_the_matched_ratio_not_the_count(self) -> None:
        # The sentence count is rescaled onto the calibrated word ask so the two
        # numbers cannot contradict each other; what is preserved is the matched
        # comment's own words-per-sentence.
        from generalized_card import length_policy

        length_policy.set_sentence_pacing("measured")
        try:
            task = SimpleNamespace(
                real_word_count=120,
                surface_skeleton="long uneven Reddit paragraph, about 12 sentences",
            )
            self.assertEqual(length_policy.skeleton_sentence_count(task.surface_skeleton), 12)
            cue = length_policy.sentence_pacing_cue(task, asked_words=60)
            # 120 words over 12 sentences is 10 words each; a 60-word ask at the
            # same ratio is 6 sentences, not 12.
            self.assertIn("about 6 sentences", cue)
            self.assertIn("about 10 words", cue)
            terse = SimpleNamespace(
                real_word_count=120,
                surface_skeleton="3-sentence local comment",
            )
            self.assertIn("about 40 words", length_policy.sentence_pacing_cue(terse, asked_words=120))
        finally:
            length_policy.set_sentence_pacing("off")

    def test_sentence_pacing_is_silent_without_a_countable_skeleton(self) -> None:
        from generalized_card import length_policy

        length_policy.set_sentence_pacing("measured")
        try:
            task = SimpleNamespace(real_word_count=40, surface_skeleton="tiny fragment reaction")
            self.assertEqual(length_policy.sentence_pacing_cue(task, asked_words=40), "")
        finally:
            length_policy.set_sentence_pacing("off")

    def test_actor_state_reaches_the_focused_writer_prompt(self) -> None:
        # The bug this pins: `writer_prompt` dispatches to three builders and
        # `_focused_writer_prompt` -- the shipped default, and the path most
        # substantive slots take -- was the one that never rendered the actor
        # state. A live v126 run carried the actor fields on 100% of plans and
        # reached only 26.5% of Writer prompts, which makes the arm
        # unattributable. See ORIENTATION section 7, "apply the change to every
        # path".
        module = configure_generator_backend(load_generator_backend(), self.config)
        self.assertEqual(prompts._writer_prompt_mode(module), "focused")
        rendered = self._actor_prompt(module, low_info=False)
        self.assertIn("Thread-local actor state composed by the Planner", rendered)
        self.assertIn("TESTROUTE clause then a short qualifier", rendered)
        self.assertIn("Realize the thread-local actor state", rendered)

    def test_actor_state_reaches_the_low_information_writer_prompt(self) -> None:
        module = configure_generator_backend(load_generator_backend(), self.config)
        rendered = self._actor_prompt(module, low_info=True)
        self.assertIn("Thread-local actor state composed by the Planner", rendered)
        self.assertIn("TESTROUTE clause then a short qualifier", rendered)

    def test_actor_state_is_absent_when_the_arm_is_off(self) -> None:
        module = configure_generator_backend(load_generator_backend(), self.config)
        module.GENERALIZED_ACTOR_MODE = "none"
        module.GENERALIZED_ACTOR_ASSIGNMENTS = {}
        rendered = self._tone_writer_prompt(module, "polite")
        self.assertNotIn("Thread-local actor state", rendered)

    def test_non_polite_writer_contract_keeps_the_substitution_ban(self) -> None:
        module = configure_generator_backend(load_generator_backend(), self.config)
        rendered = self._tone_writer_prompt(module, "impolite")
        self.assertIn(
            "Do not replace it with a generic agreement, acknowledgement",
            rendered,
        )
        self.assertIn("Tone target selector: impolite", rendered)
        self.assertIn("Ordinary non-targeted profanity is allowed", rendered)
        self.assertNotIn(">> impolite:", rendered)
        self.assertNotIn("Lead with the", rendered)

    def test_writer_sees_one_copy_of_its_assigned_register(self) -> None:
        # v80 repeated the assigned definition in both the selector and a
        # contrast block. The second copy did not improve realization and
        # competed with story and affect controls in the focused prompt.
        module = configure_generator_backend(load_generator_backend(), self.config)
        rendered = self._tone_writer_prompt(module, "polite")
        self.assertIn("Tone target selector: polite", rendered)
        self.assertNotIn(">> polite:", rendered)
        self.assertNotIn("not somewhat_polite:", rendered)
        self.assertNotIn(">> impolite:", rendered)
        self.assertNotIn("not neutral:", rendered)

    def test_polite_slot_keeps_the_first_person_frame_without_a_story(self) -> None:
        # The warm register is realized through personal appraisal, so a
        # no-story polite slot must not have that surface switched off.
        @dataclass(frozen=True)
        class ToneTask:
            local_task_id: int = 1
            tone_target: str = ""
            tone_target_instruction: str = ""
            affect_role: str = ""
            affect_instruction: str = ""
            story_mode: str = "no_story"
            story_instruction: str = ""
            allow_first_person_frame: bool = False

        task = ToneTask()
        polite = apply_planner_distribution_fields(
            task, {"tone_class": "polite", "story_mode": "no_story"}
        )
        self.assertTrue(polite.allow_first_person_frame)
        self.assertEqual(polite.story_mode, "no_story")
        impolite = apply_planner_distribution_fields(
            task, {"tone_class": "impolite", "story_mode": "no_story"}
        )
        self.assertFalse(impolite.allow_first_person_frame)

    def test_writer_restores_low_info_core_path_without_card_language(self) -> None:
        module = configure_generator_backend(load_generator_backend(), self.config)
        task = module.CommentTask(
            local_task_id=1,
            local_parent_task_id=None,
            depth=0,
            branch_id=1,
            branch_goal="acknowledge one camera detail",
            visible_scope="seed",
            local_anchor="Sony A7 IV grip",
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
            planner_intent="briefly acknowledge the grip detail",
            must_not_do="Do not add advice.",
            real_word_count=5,
            semantic_move="acknowledge the useful grip detail",
            local_topic="grip comfort",
            reply_relation="answers_parent",
            stance="agree",
            detail_focus="grip comfort",
            avoid_repeating="complete review",
            claim_key="grip_ack",
            claim_family="direct_answer",
            opening_style="bare acknowledgement",
            context_aperture="title_only",
            tone_shape="soft_ack",
            tone_overlay_slot="legacy_soften",
            tone_overlay_instruction="Legacy overlay text must not reach the Writer.",
        )
        seed = module.SeedPost(
            index=0,
            title="Sony A7 IV grip question",
            body="Is the grip comfortable?",
            content="Sony A7 IV grip question\nIs the grip comfortable?",
            source_raw_post_id="x",
            real_num_comments=5,
            metadata={},
        )
        task = module.finalize_rebalanced_task(task)
        rendered = module.build_writer_prompt(
            profile="gpt54_reddit_writer",
            seed_post=seed,
            task=task,
            parent_comment=None,
            previous_comments=[
                {
                    "content": "The grip felt fine to me.",
                    "depth": 0,
                    "speaker_role": "datapoint_only",
                    "tone_shape": "personal_dp",
                    "payload_type": "fragment_datapoint",
                    "utterance_mode": "one_datapoint",
                    "voice": "casual_neutral",
                    "length_bucket": "short",
                    "story_mode": "tiny_personal_context",
                }
            ],
        )
        self.assertIn("Write exactly one low-information Reddit comment", rendered)
        self.assertIn("What kind of turn this is:", rendered)
        self.assertIn("- function: reaction", rendered)
        self.assertIn("- payload form: low info reaction", rendered)
        self.assertIn("- speaker role: gratitude reply", rendered)
        self.assertIn("- relation to post: answers_post", rendered)
        self.assertNotIn("Earlier generated comments", rendered)
        self.assertNotIn("Thread-level distribution pressure", rendered)
        self.assertNotIn("One-shot semantic difference contract", rendered)
        self.assertNotIn("Required local move:", rendered)
        self.assertIn("Short utterances already used anywhere in this thread", rendered)
        self.assertIn("The grip felt fine to me.", rendered)
        # The low-info path renders the same route lock, so it needs the same
        # counterweight. 106 of 522 v74 slots took this branch; giving 80% of
        # slots half a fix is what made the v74 result impossible to attribute.
        self.assertIn("specification of what to say, never wording to", rendered)
        self.assertNotIn("Say this, and only this", rendered)
        self.assertNotIn("Core metric guidance", rendered)
        self.assertNotIn("hard maximum:", rendered)
        self.assertNotIn("target length:", rendered.lower())
        self.assertNotIn("tone overlay", rendered.lower())
        self.assertNotIn("legacy_soften", rendered)
        self.assertNotIn("Legacy overlay text", rendered)
        self.assertNotIn("tone_overlay_slot", module.controls_for_task(task))
        self.assertNotIn("tone_overlay_slot", module.render_sampled_plan_block(task))
        self.assertIn("not a counted requirement", rendered.lower())
        self.assertLess(len(rendered), 6000)
        self.assertEqual(
            module.writer_token_cap(
                "short",
                payload_type="low_info_reaction",
                profile="gpt54_reddit_writer",
                max_writer_tokens=260,
            ),
            260,
        )
        lowered = rendered.lower()
        self.assertNotIn("r/creditcards", lowered)
        self.assertNotIn("bank/card", lowered)
        self.assertNotIn("issuer", lowered)
        self.assertNotIn("reward program", lowered)
        self.assertNotIn("5/24", lowered)

        bare_task = replace(
            task,
            payload_type="bare_answer",
            utterance_mode="direct_answer",
            semantic_move="reject a price drop across the entire product line",
            avoid_repeating="the earlier answer that only denies a whole-line drop",
            planner_intent="answer the narrow product-line scope question",
        )
        prior_comments = [
            {
                "content": "Not the whole A7 line, no",
                "depth": 0,
                "semantic_move": "deny a whole-line price drop",
                "detail_focus": "whole product line",
                "reply_relation": "answers_seed",
                "stance": "disagree",
            }
        ]
        prior_comments.extend(
            {
                "content": (
                    f"This deliberately longer prior comment number {index} carries enough "
                    "words to stay outside the short-utterance exclusion list."
                ),
                "depth": 0,
                "semantic_move": f"cover unrelated route {index}",
            }
            for index in range(1, 14)
        )
        bare_rendered = module.build_writer_prompt(
            profile="gpt54_reddit_writer",
            seed_post=seed,
            task=bare_task,
            parent_comment=None,
            previous_comments=prior_comments,
        )
        self.assertIn(
            "The point this comment makes, in your own words: reject a price drop across the entire product line",
            bare_rendered,
        )
        self.assertIn(
            "content to avoid: the earlier answer that only denies a whole-line drop",
            bare_rendered,
        )
        self.assertNotIn("Required local move:", bare_rendered)
        self.assertIn("Not the whole A7 line, no", bare_rendered)

    def test_short_utterance_shape_cannot_downgrade_substantive_payload(self) -> None:
        module = configure_generator_backend(load_generator_backend(), self.config)
        base = module.CommentTask(
            local_task_id=1,
            local_parent_task_id=None,
            depth=0,
            branch_id=1,
            branch_goal="check one autofocus condition",
            visible_scope="seed",
            local_anchor="autofocus tracking",
            comment_function="question_followup",
            content_angle="fit_use_case",
            evidence_mode="none_assertion",
            story_mode="no_story",
            voice="casual_neutral",
            payload_type="narrow_question",
            length_bucket="short",
            speaker_role="op_followup",
            utterance_mode="op_followup",
            surface_texture="plain",
            allow_first_person_frame=False,
            allow_uncertainty_frame=False,
            planner_intent="ask one short tracking question",
            must_not_do="Do not broaden the question.",
            real_word_count=8,
            semantic_move="ask whether tracking holds on a sideways cut",
            local_topic="tracking on a sideways cut",
            reply_relation="asks_narrow_followup",
            stance="neutral",
            detail_focus="a sideways cut",
            avoid_repeating="the parent verdict",
            claim_key="sideways_cut_question",
            claim_family="clarification_question",
        )
        self.assertTrue(module.should_use_low_info_writer(base))

        for payload, function in (
            ("soft_helpful", "explanation_analysis"),
            ("correction", "correction_caveat"),
        ):
            task = replace(base, payload_type=payload, comment_function=function)
            self.assertFalse(module.should_use_low_info_writer(task), payload)
            rendered = module.build_writer_prompt(
                profile="gpt54_reddit_writer",
                seed_post=module.SeedPost(
                    index=0,
                    title="Autofocus tracking question",
                    body="Does tracking hold when a runner cuts sideways?",
                    content="Autofocus tracking question",
                    source_raw_post_id="routing-test",
                    real_num_comments=1,
                    metadata={},
                ),
                task=task,
                parent_comment=None,
                previous_comments=[],
                recent_openings=[],
            )
            self.assertNotIn("Write exactly one low-information", rendered)
            self.assertIn("Write exactly one Reddit comment", rendered)
            self.assertIn(f"- payload form: {payload.replace('_', ' ')}", rendered)

    def test_final_task_refreshes_stale_acknowledgement_controls(self) -> None:
        module = configure_generator_backend(load_generator_backend(), self.config)
        task = module.CommentTask(
            local_task_id=1,
            local_parent_task_id=None,
            depth=0,
            branch_id=1,
            branch_goal="report one autofocus datapoint",
            visible_scope="seed",
            local_anchor="autofocus tracking",
            comment_function="personal_datapoint",
            content_angle="fit_use_case",
            evidence_mode="firsthand_experience",
            story_mode="no_story",
            voice="casual_neutral",
            payload_type="fragment_datapoint",
            length_bucket="short",
            speaker_role="datapoint_only",
            utterance_mode="one_datapoint",
            surface_texture="gratitude_social",
            allow_first_person_frame=True,
            allow_uncertainty_frame=False,
            planner_intent="report one short tracking datapoint",
            must_not_do="Do not add advice.",
            real_word_count=9,
            real_surface_shape="short_direct_answer",
            real_tone_slot="pure_acknowledgement",
            real_tone_instruction="Make the social acknowledgement visible.",
        )
        refreshed = module.finalize_rebalanced_task(task)
        self.assertEqual(refreshed.surface_texture, "plain")
        self.assertEqual(refreshed.real_tone_slot, "")
        self.assertEqual(refreshed.real_tone_instruction, "")

        gratitude = module.finalize_rebalanced_task(
            replace(
                task,
                speaker_role="gratitude_reply",
                voice="grateful",
                payload_type="bare_answer",
                comment_function="reaction",
                evidence_mode="none_assertion",
                affect_role="gratitude",
            )
        )
        self.assertEqual(gratitude.surface_texture, "gratitude_social")
        self.assertEqual(gratitude.real_tone_slot, "pure_acknowledgement")

    def test_focused_ledger_is_bounded_and_deduped_from_openings(self) -> None:
        module = configure_generator_backend(load_generator_backend(), self.config)
        comments = [
            {
                "content": f"short prior line {index}",
                "semantic_move": (
                    "briefly acknowledge the parent-local help without adding a factual claim"
                    if index == 1
                    else f"cover distinct local point {index}"
                ),
                "decision_boundary": f"boundary {index}",
            }
            for index in range(60)
        ]
        task = SimpleNamespace(
            length_bucket="short",
            real_word_count=8,
            reply_delta_type="social_close",
            semantic_move="briefly acknowledge the parent-local help without adding a factual claim",
        )
        ledger = prompts._focused_thread_ledger(
            module,
            comments,
            current_task=task,
            recent_openings=["short prior line 59"],
        )
        short_block, coverage_block = ledger.split(
            "Semantic contributions already covered in this thread:\n",
            1,
        )
        short_rows = [
            line for line in short_block.splitlines() if line.startswith("- ")
        ]
        coverage_rows = [
            line for line in coverage_block.splitlines() if line.startswith("- ")
        ]
        self.assertLessEqual(len(short_rows), 32)
        self.assertLessEqual(len(coverage_rows), 8)
        self.assertNotIn("short prior line 59", short_block)
        self.assertEqual(
            ledger.count(
                "briefly acknowledge the parent-local help without adding a factual claim"
            ),
            0,
        )

    def test_domain_neutral_real_slot_classifiers_are_bound(self) -> None:
        module = configure_generator_backend(load_generator_backend(), self.config)
        self.assertEqual(
            module.infer_real_surface_shape(
                {
                    "body": "My autofocus misses focus occasionally indoors when the subject moves near the edge.",
                    "author": "user",
                }
            ),
            "full_answer",
        )
        slot, instruction = module.infer_real_tone_slot(
            {"body": "I tested the autofocus twice and noticed the same lag."},
            payload_type="fragment_datapoint",
            speaker_role="datapoint_only",
            voice="casual_neutral",
            real_surface_shape="full_answer",
            surface_texture="plain",
        )
        self.assertEqual(slot, "")
        self.assertEqual(instruction, "")

    def test_all_writer_control_paths_are_free_of_finance_prompt_residue(self) -> None:
        module = configure_generator_backend(load_generator_backend(), self.config)
        base = module.CommentTask(
            local_task_id=1,
            local_parent_task_id=None,
            depth=0,
            branch_id=1,
            branch_goal="address one visible camera detail",
            visible_scope="seed",
            local_anchor="Sony autofocus behavior",
            comment_function="reaction",
            content_angle="fit_use_case",
            evidence_mode="none_assertion",
            story_mode="no_story",
            voice="casual_neutral",
            payload_type="low_info_reaction",
            length_bucket="short",
            speaker_role="side_observer",
            utterance_mode="direct_answer",
            surface_texture="plain",
            allow_first_person_frame=False,
            allow_uncertainty_frame=False,
            planner_intent="make one local observation about autofocus",
            must_not_do="Do not add a complete product review.",
            real_word_count=12,
            semantic_move="note one autofocus behavior",
            local_topic="autofocus behavior",
            reply_relation="answers_parent",
            stance="neutral",
            detail_focus="autofocus behavior",
            avoid_repeating="complete review",
            claim_key="autofocus_behavior",
            claim_family="technical_explanation",
            perspective_id="P01",
            domain_intent="compare autofocus behavior",
            opening_style="concrete detail first",
            context_aperture="title_only",
            tone_shape="neutral_fact",
        )
        seed = module.SeedPost(
            index=0,
            title="Sony autofocus question",
            body="Does subject tracking lag in low light?",
            content="Sony autofocus question\nDoes subject tracking lag in low light?",
            source_raw_post_id="x",
            real_num_comments=8,
            metadata={},
        )
        variants = []
        variants.extend(("payload_type", value) for value in module.PAYLOAD_TYPES)
        variants.extend(("speaker_role", value) for value in module.SPEAKER_ROLES)
        variants.extend(("utterance_mode", value) for value in module.UTTERANCE_MODES)
        variants.extend(("surface_texture", value) for value in module.SURFACE_TEXTURES)
        variants.extend(("tone_shape", value) for value in module.TONE_SHAPES)
        forbidden = re.compile(
            r"r/creditcards|credit[- ]card|credit limit|issuer|approval datapoint|"
            r"utilization|cashback|reward program|annual fee|balance transfer|"
            r"hard pull|soft pull|\bapr\b|\bcli\b|\bhuca\b|5/24|\brecon\b",
            flags=re.I,
        )
        prompts_seen = [module.SYSTEM_PROMPTS["gpt54_reddit_writer"]]
        for field, value in variants:
            task = module.finalize_rebalanced_task(replace(base, **{field: value}))
            prompts_seen.append(
                module.build_writer_prompt(
                    profile="gpt54_reddit_writer",
                    seed_post=seed,
                    task=task,
                    parent_comment=None,
                    previous_comments=[],
                )
            )
        for rendered in prompts_seen:
            self.assertIsNone(forbidden.search(rendered), rendered)

    def test_historical_snapshot_is_explicit_not_default(self) -> None:
        generalized = configure_generator_backend(
            load_generator_backend(profile=GENERALIZED_V2_PROFILE),
            self.config,
            profile=GENERALIZED_V2_PROFILE,
        )
        snapshot = configure_generator_backend(
            load_generator_backend(profile=CARD_SNAPSHOT_PROFILE),
            self.config,
            profile=CARD_SNAPSHOT_PROFILE,
        )
        self.assertEqual(
            generalized.GENERALIZED_CARD_PARITY["generator_profile"],
            GENERALIZED_V2_PROFILE,
        )
        self.assertEqual(
            snapshot.GENERALIZED_CARD_PARITY["generator_profile"],
            CARD_SNAPSHOT_PROFILE,
        )
        self.assertNotEqual(
            generalized.GENERALIZED_CARD_PARITY["backend_source"],
            snapshot.GENERALIZED_CARD_PARITY["backend_source"],
        )

    def test_generation_defaults_match_core_control_surface(self) -> None:
        script = load_script_module(
            "generalized_card_run_generate",
            Path(__file__).resolve().parents[1] / "scripts" / "run_generate.py",
        )
        args = script.build_parser().parse_args(["--tag", "test"])
        command = script._generator_command(
            args=args,
            config_raw_dir=Path("raw"),
            seed_pool=Path("seed.json"),
            generated_root=Path("generated"),
        )
        expected = {
            "--claim-key-budget": "1",
            "--claim-family-max-share": "0.18",
            "--claim-family-min-budget": "3",
            "--opening-reuse-budget": "1",
            "--opener-family-reuse-budget": "5",
            "--template-phrase-reuse-budget": "4",
            "--advisor-max-share": "0.28",
            "--question-max-share": "0.18",
            "--micro-target-share": "0.07",
            "--short-max-share": "0.18",
            "--social-noise-min-share": "0.18",
            "--gratitude-min-share": "0.12",
            "--context-dropout-rate": "0.42",
            "--context-jitter-rate": "0.32",
        }
        for flag, value in expected.items():
            self.assertEqual(command[command.index(flag) + 1], value)

    def test_evaluation_resume_checks_expected_data_size(self) -> None:
        script = load_script_module(
            "generalized_card_run_evaluate",
            Path(__file__).resolve().parents[1] / "scripts" / "run_evaluate.py",
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "run_00_sampled_reddit"
            run.mkdir()
            (run / "discussion.json").write_text(
                json.dumps({"posts": [{"id": 1}, {"id": 2}]}),
                encoding="utf-8",
            )
            scores = root / "scores.csv"
            scores.write_text("thread_id,value\n1,0.1\n2,0.2\n", encoding="utf-8")
            self.assertEqual(script._count_discussion_posts(root), 2)
            self.assertTrue(script._cleaned_complete(root, 2))
            self.assertEqual(script._csv_row_count(scores), 2)
            scores.write_text("thread_id,value\n1,0.1\n", encoding="utf-8")
            self.assertEqual(script._csv_row_count(scores), 1)

            (run / "discussion.json").write_text(
                json.dumps(
                    {
                        "posts": [
                            {
                                "id": 1,
                                "comments": [
                                    {
                                        "comment_id": 10,
                                        "parent_comment_id": None,
                                        "depth": 0,
                                        "replies": [
                                            {
                                                "comment_id": 11,
                                                "parent_comment_id": 10,
                                                "depth": 2,
                                                "replies": [],
                                            }
                                        ],
                                    }
                                ],
                            },
                            {"id": 2},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            self.assertFalse(script._cleaned_complete(root, 2))

    def test_evaluation_score_completion_requires_every_metric(self) -> None:
        script = load_script_module(
            "generalized_card_run_evaluate_metric_completeness",
            Path(__file__).resolve().parents[1] / "scripts" / "run_evaluate.py",
        )
        with tempfile.TemporaryDirectory() as directory:
            scores = Path(directory) / "scores.csv"
            complete = {metric: "0.1" for metric in script.REQUIRED_THREAD_METRICS}
            with scores.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(complete))
                writer.writeheader()
                writer.writerow(complete)
            self.assertTrue(script._score_csv_complete(scores, 1))

            incomplete = dict(complete)
            incomplete["semantic_mean_cosine"] = ""
            with scores.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(incomplete))
                writer.writeheader()
                writer.writerow(incomplete)
            self.assertFalse(script._score_csv_complete(scores, 1))

    def test_evaluation_prints_saved_results_on_resume(self) -> None:
        script = load_script_module(
            "generalized_card_run_evaluate_print",
            Path(__file__).resolve().parents[1] / "scripts" / "run_evaluate.py",
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "matched_seed_group_eval.json").write_text(
                json.dumps(
                    {
                        "passing_metric": {
                            "mwu_p_value": 0.2,
                            "ks_p_value": 0.3,
                            "cliffs_delta": 0.1,
                            "wasserstein_distance": 0.02,
                        },
                        "partial_metric": {
                            "mwu_p_value": 0.01,
                            "ks_p_value": 0.2,
                            "cliffs_delta": -0.1,
                            "wasserstein_distance": 0.03,
                        },
                        "failing_metric": {
                            "mwu_p_value": 0.01,
                            "ks_p_value": 0.02,
                            "cliffs_delta": 0.2,
                            "wasserstein_distance": 0.04,
                        },
                    }
                ),
                encoding="utf-8",
            )
            with patch("builtins.print") as print_mock:
                script._print_saved_matched_results(root, sample_size=10)
            rendered = "\n".join(
                str(call.args[0]) for call in print_mock.call_args_list
            )
            self.assertIn("PASS/PARTIAL/FAIL: 1/1/1", rendered)
            self.assertIn("passing_metric", rendered)

    def test_evaluation_allows_measurable_quality_warning(self) -> None:
        script = load_script_module(
            "generalized_card_run_evaluate_audit",
            Path(__file__).resolve().parents[1] / "scripts" / "run_evaluate.py",
        )
        report = {
            "evaluable": True,
            "healthy": False,
            "semantic_plan_collision_posts": 0,
            "overconcentrated_perspective_posts": 2,
        }
        with patch("builtins.print") as print_mock:
            script._enforce_evaluable_audit(report, audit_path=Path("audit.json"))
        rendered = "\n".join(str(call.args[0]) for call in print_mock.call_args_list)
        self.assertIn("evaluation-audit-warning", rendered)

    def test_evaluation_rejects_integrity_failure(self) -> None:
        script = load_script_module(
            "generalized_card_run_evaluate_integrity",
            Path(__file__).resolve().parents[1] / "scripts" / "run_evaluate.py",
        )
        with self.assertRaisesRegex(RuntimeError, "evaluation-integrity"):
            script._enforce_evaluable_audit(
                {"evaluable": False, "healthy": False},
                audit_path=Path("audit.json"),
            )

    def test_evaluation_stage_rejects_incomplete_command_output(self) -> None:
        script = load_script_module(
            "generalized_card_run_evaluate_postcondition",
            Path(__file__).resolve().parents[1] / "scripts" / "run_evaluate.py",
        )
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "incomplete"
            with self.assertRaisesRegex(RuntimeError, "without complete output"):
                script._derived_stage(
                    target=target,
                    complete=lambda: False,
                    resume=True,
                    command=[sys.executable, "-c", "pass"],
                )


class FocusedWriterPromptTest(unittest.TestCase):
    """The focused prompt may drop rule mass, never a grounding rule.

    The full prompt averaged 22,249 characters to produce a 56-word comment and
    only 945 of those were identical across slots, so its size was control count,
    not boilerplate. Rebuilding one 38-slot thread at 2,523 characters held
    within-thread diversity (self_bleu_4 0.0466 -> 0.0434, self_bertscore
    0.5279 -> 0.5226, both toward the thread's real 0.0362 / 0.494) while the
    converged frame went 2.6% -> 0%. A first version of the focused prompt
    dropped the factual-grounding rule and the polite register's first-person
    frame; these assertions are what caught that.
    """

    def setUp(self) -> None:
        self.config = load_domain_config("camera")

    def _prompt(
        self,
        mode: str,
        tone: str,
        route_lock: str = "own_words",
        license_mode: str = "off",
        story_mode: str = "no_story",
        opener_type: str = "",
        rhythm_profile: dict[str, Any] | None = None,
        previous_comments: list[dict[str, Any]] | None = None,
        **task_overrides: Any,
    ) -> str:
        from generalized_card.generation_distribution import (
            apply_planner_distribution_fields,
        )

        module = configure_generator_backend(load_generator_backend(), self.config)
        if rhythm_profile is not None:
            # Installed after configuration, not before: the backend reinstalls
            # the frozen profile from `GENERALIZED_DOMAIN_PROFILE` on every
            # `configure_generator_backend` call, so a profile set beforehand is
            # silently replaced. Same shape as the v97 arm-reread defect.
            from generalized_card.sentence_rhythm import set_active_rhythm_profile

            set_active_rhythm_profile(rhythm_profile)
        module.GENERALIZED_WRITER_PROMPT_MODE = mode
        module.GENERALIZED_WRITER_ROUTE_LOCK = route_lock
        module.GENERALIZED_OWN_FACT_LICENSE = license_mode
        task = module.CommentTask(
            local_task_id=1,
            local_parent_task_id=None,
            depth=0,
            branch_id=1,
            branch_goal="assess grip comfort",
            visible_scope="seed",
            local_anchor="grip comfort",
            comment_function="verdict_evaluation",
            content_angle="fit_use_case",
            evidence_mode="small_observation",
            story_mode=story_mode,
            voice="casual_neutral",
            payload_type="soft_helpful",
            length_bucket="long",
            speaker_role="datapoint_only",
            utterance_mode="one_datapoint",
            surface_texture="plain",
            allow_first_person_frame=True,
            allow_uncertainty_frame=False,
            planner_intent="one local verdict",
            must_not_do="",
            semantic_move="commit to a verdict on grip comfort over a long shoot",
            local_topic="grip comfort",
            reply_relation="answers_parent",
            stance="agree",
            detail_focus="grip comfort",
            avoid_repeating="complete review",
            claim_key="grip_verdict",
            claim_family="direct_answer",
            opening_style="verdict then the condition",
            context_aperture="full_seed",
            tone_shape="neutral_fact",
            real_word_count=140,
        )
        if task_overrides:
            task = replace(task, **task_overrides)
        task = apply_planner_distribution_fields(task, {"tone_class": tone})
        seed = module.SeedPost(
            index=0,
            title="Sony A7 IV grip question",
            body="Is the grip comfortable over a long shoot?",
            content="Sony A7 IV grip question\nIs the grip comfortable over a long shoot?",
            source_raw_post_id="x",
            real_num_comments=8,
            metadata={},
        )
        if opener_type:
            # The scheduled entry grammar reaches the Writer through the same
            # keyed registry the Planner writes into, so a test that wants one
            # has to put it there rather than on the task.
            from generalized_card.domain_claim import seed_claim_key

            module.GENERALIZED_OPENER_TYPES[(seed_claim_key(seed), 1)] = opener_type
        return module.build_writer_prompt(
            profile="gpt54_reddit_writer",
            seed_post=seed,
            task=module.finalize_rebalanced_task(task),
            parent_comment=None,
            previous_comments=list(previous_comments or []),
            recent_openings=[],
        )

    def test_entity_spread_reaches_both_writer_prompt_paths(self) -> None:
        """The v108 lesson: test a prompt fix through the real dispatch, both modes.

        v108 shipped an instruction into `_thread_memory` (the `full` ledger),
        never touched `_focused_thread_ledger`, and burned $1.19 on a gate where
        the arm fired 0/186 times because `focused` is the default every run has
        used (`docs/DECISIONS.md` G23). Both writer prompts now render the
        referent offer through one shared helper; this asserts it on both.

        The profile is supplied as a real profile file through
        `GENERALIZED_CARD_DOMAIN_PROFILE`, so this exercises the actual load and
        install path rather than injecting module globals.
        """

        import json
        import os
        import tempfile

        from generalized_card.entity_spread import REFERENT_CUE

        profile = {
            "entity_inventory": {
                "available": True,
                "terms": [{"term": f"XZ-{n}00"} for n in range(1, 9)],
            },
            "entity_spread_profile": {
                "available": True,
                "bands": {
                    band: {"mention_rate": 1.0, "distinct_per_comment": 1.0}
                    for band in ("tiny", "small", "medium", "large")
                },
            },
        }
        previous = os.environ.get("GENERALIZED_CARD_DOMAIN_PROFILE")
        previous_arm = os.environ.get("GENERALIZED_CARD_ENTITY_SPREAD")
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "profile.json")
            from generalized_card.domain_profile import profile_hash

            profile["profile_sha256"] = profile_hash(profile)
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(profile, handle)
            os.environ["GENERALIZED_CARD_DOMAIN_PROFILE"] = path
            try:
                for mode in ("focused", "full"):
                    os.environ["GENERALIZED_CARD_ENTITY_SPREAD"] = "off"
                    off = self._prompt(mode, "impolite")
                    self.assertNotIn(
                        REFERENT_CUE, off, f"{mode}: arm off must render nothing"
                    )

                    os.environ["GENERALIZED_CARD_ENTITY_SPREAD"] = "measured"
                    module = configure_generator_backend(
                        load_generator_backend(), self.config
                    )
                    module.GENERALIZED_ACTIVE_THREAD_COMMENTS = 60
                    on = self._prompt(mode, "impolite")
                    self.assertIn(
                        REFERENT_CUE,
                        on,
                        f"{mode}: arm on must reach this writer prompt",
                    )
                    self.assertIn("XZ-", on, f"{mode}: a referent must be offered")
            finally:
                for key, value in (
                    ("GENERALIZED_CARD_DOMAIN_PROFILE", previous),
                    ("GENERALIZED_CARD_ENTITY_SPREAD", previous_arm),
                ):
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value

    def test_entity_spread_cue_carries_no_domain_vocabulary(self) -> None:
        from generalized_card.entity_spread import REFERENT_CUE

        lowered = REFERENT_CUE.lower()
        for word in ("camera", "lens", "phone", "laptop", "headphone", "photo"):
            self.assertNotIn(word, lowered)

    def test_entity_spread_draw_is_deterministic_and_rate_respecting(self) -> None:
        from generalized_card.entity_spread import (
            set_entity_spread,
            slot_offers_referent,
        )

        half = {"available": True, "bands": {"medium": {"distinct_per_comment": 0.5}}}
        try:
            set_entity_spread("measured")
            first = [
                slot_offers_referent(half, slot_key=f"s:{i}", comment_count=60)
                for i in range(400)
            ]
            second = [
                slot_offers_referent(half, slot_key=f"s:{i}", comment_count=60)
                for i in range(400)
            ]
            self.assertEqual(first, second, "draw must be deterministic per slot")
            rate = sum(first) / len(first)
            self.assertTrue(0.42 < rate < 0.58, f"rate {rate} should track 0.5")

            set_entity_spread("off")
            self.assertFalse(
                any(
                    slot_offers_referent(half, slot_key=f"s:{i}", comment_count=60)
                    for i in range(50)
                ),
                "arm off must never draw",
            )
        finally:
            set_entity_spread("off")

    def test_length_fidelity_flags_only_a_band_change(self) -> None:
        from generalized_card.length_fidelity import (
            length_band_problem,
            set_length_fidelity,
        )

        profile = {
            "available": True,
            "cuts": [18.0, 30.0, 46.0, 85.0],
            "band_counts": {"0": 900.0, "1": 900.0, "2": 900.0, "3": 900.0, "4": 900.0},
        }
        task = SimpleNamespace(real_word_count=70)
        try:
            set_length_fidelity("off")
            self.assertEqual(
                length_band_problem("word " * 30, task, profile=profile),
                "",
                "arm off must register nothing",
            )
            set_length_fidelity("measured")
            self.assertEqual(
                length_band_problem("word " * 60, task, profile=profile),
                "",
                "60 words is band 3 like the assigned 70; no problem",
            )
            miss = length_band_problem("word " * 40, task, profile=profile)
            self.assertTrue(miss.startswith("length_band_mismatch:"), miss)
            self.assertIn("assigned 70w in band 3", miss)
            self.assertIn("[47-85]", miss)
            # The realized-too-long direction is caught as well; the v109 audit
            # measured 1.44x overshoot on the shortest slots.
            self.assertTrue(
                length_band_problem(
                    "word " * 30, SimpleNamespace(real_word_count=6), profile=profile
                ).startswith("length_band_mismatch:")
            )
        finally:
            set_length_fidelity("off")

    def test_length_fidelity_withholds_on_an_unmeasured_band(self) -> None:
        from generalized_card.length_fidelity import (
            build_length_fidelity_profile,
            length_band_problem,
            set_length_fidelity,
        )

        thin = {
            "available": True,
            "cuts": [18.0, 30.0, 46.0, 85.0],
            "band_counts": {"0": 900.0, "1": 900.0, "2": 900.0, "3": 5.0, "4": 900.0},
        }
        try:
            set_length_fidelity("measured")
            self.assertEqual(
                length_band_problem(
                    "word " * 40, SimpleNamespace(real_word_count=70), profile=thin
                ),
                "",
                "a band with too few reference comments must withhold, not default",
            )
            self.assertEqual(
                length_band_problem(
                    "word " * 40,
                    SimpleNamespace(real_word_count=70),
                    profile={"available": False},
                ),
                "",
            )
            self.assertFalse(
                build_length_fidelity_profile(
                    [{"comments": [{"body": "a b c"} for _ in range(20)]}]
                )["available"],
                "a corpus too thin to bin must report itself unavailable",
            )
        finally:
            set_length_fidelity("off")

    def test_development_scope_long_only_reproduces_v110_exactly(self) -> None:
        """The legacy arm must return the shipped beat budget at every size."""

        from generalized_card.long_form_planning import (
            expected_development_beats,
            set_development_scope,
        )

        try:
            set_development_scope("long_only")
            for words in (0, 1, 9, 20, 34, 35, 45, 60, 61, 80, 99, 100):
                self.assertEqual(expected_development_beats(words), 0, words)
            self.assertEqual(expected_development_beats(101), 5)
            self.assertEqual(expected_development_beats(120), 6)
            self.assertEqual(expected_development_beats(200), 10)
            self.assertEqual(expected_development_beats(845), 12)
        finally:
            set_development_scope("long_only")

    def test_development_scope_measured_differs_only_inside_35_to_100(self) -> None:
        """The arm is a scope extension, not a re-budget: everything else is equal."""

        from generalized_card.long_form_planning import (
            DEVELOPMENT_FLOOR_WORDS,
            expected_development_beats,
            set_development_scope,
        )

        try:
            set_development_scope("long_only")
            legacy = {words: expected_development_beats(words) for words in range(0, 900)}
            set_development_scope("measured")
            measured = {words: expected_development_beats(words) for words in range(0, 900)}
        finally:
            set_development_scope("long_only")
        differing = {w for w in legacy if legacy[w] != measured[w]}
        self.assertEqual(
            differing,
            set(range(DEVELOPMENT_FLOOR_WORDS, 101)),
            "the arm must change 35-100 assigned words and nothing else",
        )
        # inside the extended range the budget follows the measured 21 words/beat
        for words in (35, 45, 60, 80, 100):
            self.assertEqual(measured[words], max(2, round(words / 21.0)), words)
        # and it is monotone through the old boundary, which the legacy arm is not
        self.assertLessEqual(measured[100], measured[101])
        self.assertGreater(legacy[101] - legacy[100], 4)

    def test_development_scope_reaches_the_writer_length_cue_on_both_values(self) -> None:
        """Prove the arm changes the rendered cue, not merely the beat integer."""

        from dataclasses import dataclass

        from generalized_card.length_policy import local_move_scope_guidance
        from generalized_card.long_form_planning import set_development_scope

        @dataclass
        class _Slot:
            real_word_count: int
            development_plan: str = ""

        try:
            set_development_scope("long_only")
            legacy_short = local_move_scope_guidance(_Slot(45))
            legacy_mid = local_move_scope_guidance(_Slot(90))
            set_development_scope("measured")
            armed_short = local_move_scope_guidance(_Slot(45))
            armed_mid = local_move_scope_guidance(_Slot(90))
            unchanged_micro = local_move_scope_guidance(_Slot(20))
            set_development_scope("long_only")
            legacy_micro = local_move_scope_guidance(_Slot(20))
        finally:
            set_development_scope("long_only")

        self.assertIn("one narrow local move", legacy_short)
        self.assertIn("two or three connected beats", legacy_mid)
        self.assertIn("2 distinct, connected beats", armed_short)
        self.assertIn("4 distinct, connected beats", armed_mid)
        self.assertEqual(unchanged_micro, legacy_micro)

    def test_development_scope_stops_deleting_the_planner_beat_plan(self) -> None:
        """`reconcile_development_plan_capacity` wipes plans with no capacity."""

        from generalized_card.long_form_planning import (
            reconcile_development_plan_capacity,
            set_development_scope,
        )

        try:
            set_development_scope("long_only")
            plan = {"_slot_word_count": 80, "development_plan": "a || b || c"}
            self.assertIsNotNone(reconcile_development_plan_capacity(plan))
            self.assertEqual(plan["development_plan"], "")

            set_development_scope("measured")
            kept = {"_slot_word_count": 80, "development_plan": "a || b || c"}
            self.assertIsNone(reconcile_development_plan_capacity(kept))
            self.assertEqual(kept["development_plan"], "a || b || c")

            # below the floor the plan is still residue and must still be dropped
            below = {"_slot_word_count": 20, "development_plan": "a || b"}
            self.assertIsNotNone(reconcile_development_plan_capacity(below))
        finally:
            set_development_scope("long_only")

    def test_length_transfer_v97_arm_reproduces_the_shipped_asks(self) -> None:
        """The legacy arm must be byte-identical, including the clamp bounds."""

        import math

        from generalized_card.length_calibration import (
            MAX_ASK_MULTIPLIER,
            MIN_ASK_MULTIPLIER,
            WORD_TRANSFER_INTERCEPT,
            WORD_TRANSFER_SLOPE,
            calibrated_word_ask,
            set_length_transfer,
        )

        try:
            set_length_transfer("v97")
            for target in (1, 2, 3, 6, 12, 25, 40, 70, 100, 121, 180, 250, 400, 845):
                asked = math.exp(
                    (math.log(target) - WORD_TRANSFER_INTERCEPT) / WORD_TRANSFER_SLOPE
                )
                clamped = max(
                    MIN_ASK_MULTIPLIER, min(MAX_ASK_MULTIPLIER, asked / target)
                )
                self.assertEqual(
                    calibrated_word_ask(target),
                    max(1, int(round(target * clamped))),
                    f"legacy ask changed for target {target}",
                )
        finally:
            set_length_transfer("v97")

    def test_length_transfer_refit_lands_realized_on_target(self) -> None:
        """Inverting the refitted line must predict the target back, unclamped."""

        import math

        from generalized_card.length_calibration import (
            REFIT_MAX_ASK_MULTIPLIER,
            REFIT_MIN_ASK_MULTIPLIER,
            REFIT_TRANSFER_INTERCEPT,
            REFIT_TRANSFER_SLOPE,
            ask_multiplier,
            calibrated_word_ask,
            set_length_transfer,
        )

        try:
            set_length_transfer("refit")
            for target in (3, 6, 12, 25, 40, 70, 100, 121, 180, 250):
                exact = math.exp(
                    (math.log(target) - REFIT_TRANSFER_INTERCEPT) / REFIT_TRANSFER_SLOPE
                )
                # The inversion itself must be exact.
                self.assertAlmostEqual(
                    math.exp(
                        REFIT_TRANSFER_INTERCEPT
                        + REFIT_TRANSFER_SLOPE * math.log(exact)
                    ),
                    target,
                    places=6,
                )
                # The shipped ask is that value rounded to whole words, so it may
                # differ by at most one word -- which is a visible relative error
                # only at the very short end.
                asked = calibrated_word_ask(target)
                self.assertLessEqual(
                    abs(asked - exact),
                    1.0,
                    f"target {target}: asked {asked} against exact {exact:.2f}",
                )
            # The clamp must not bind anywhere the fit covers, or the arm would
            # silently stop correcting exactly where the compression lives.
            for target in range(1, 251):
                self.assertLess(REFIT_MIN_ASK_MULTIPLIER, ask_multiplier(target))
                self.assertGreater(REFIT_MAX_ASK_MULTIPLIER, ask_multiplier(target))
            # The refit must ask for strictly more on a long slot and less on a
            # short one than the v97 constants did.
            long_refit, short_refit = calibrated_word_ask(121), calibrated_word_ask(6)
            set_length_transfer("v97")
            self.assertGreater(long_refit, calibrated_word_ask(121))
            self.assertLess(short_refit, calibrated_word_ask(6))
        finally:
            set_length_transfer("v97")

    def test_provider_ceiling_clears_the_refit_ask_and_is_a_legacy_no_op(self) -> None:
        from generalized_card.length_calibration import (
            calibrated_word_ask,
            set_length_transfer,
        )
        from generalized_card.length_policy import writer_provider_token_budget

        try:
            set_length_transfer("v97")
            for words in range(1, 101):
                self.assertEqual(
                    writer_provider_token_budget(
                        SimpleNamespace(real_word_count=words), configured_max=260
                    ),
                    260,
                    f"legacy ceiling changed at {words} words",
                )
            set_length_transfer("refit")
            task = SimpleNamespace(real_word_count=121)
            ceiling = writer_provider_token_budget(task, configured_max=260)
            self.assertGreater(
                ceiling,
                calibrated_word_ask(121) * 1.7,
                "the provider ceiling must clear the words the cue asks for",
            )
        finally:
            set_length_transfer("v97")

    def test_length_fidelity_bands_bound_the_long_tail(self) -> None:
        """The reason the arm uses deciles rather than quintiles.

        With quintiles, camera's top cut is 72 words and the band above it is
        open, so a slot assigned 100 words has no upper constraint -- exactly the
        50-100-word band that realizes at 0.82x. Deciles bound it.
        """

        from generalized_card.length_fidelity import (
            BAND_QUANTILES,
            band_of,
            build_length_fidelity_profile,
        )

        self.assertEqual(len(BAND_QUANTILES), 9, "the shipped band set is deciles")
        lengths = [2, 5, 9, 13, 18, 25, 34, 48, 70, 110, 180] * 60
        threads = [{"comments": [{"body": "w " * count} for count in lengths]}]
        profile = build_length_fidelity_profile(threads)
        self.assertTrue(profile["available"])
        cuts = profile["cuts"]
        self.assertEqual(len(cuts), 9)
        # A slot assigned near the top must not share a band with a much
        # shorter realization.
        self.assertNotEqual(
            band_of(180, cuts),
            band_of(70, cuts),
            "the long tail must be separated from the upper-middle band",
        )

    def test_length_fidelity_problem_is_soft_and_carries_no_domain_vocabulary(
        self,
    ) -> None:
        from generalized_card.length_fidelity import PROBLEM_PREFIX, retry_note
        from generalized_card.length_policy import is_soft_length_problem

        problem = f"{PROBLEM_PREFIX}40w in band 2, assigned 70w in band 3 [47-85]"
        self.assertTrue(
            is_soft_length_problem(problem),
            "a length-band miss must never be able to drop a matched slot",
        )
        note = retry_note(problem).lower()
        for word in ("camera", "canon", "lens", "phone", "laptop", "headphone", "photo"):
            self.assertNotIn(word, note)
        self.assertIn("assigned length", note)

    def test_focused_is_far_smaller_than_full(self) -> None:
        full = self._prompt("full", "impolite")
        focused = self._prompt("focused", "impolite")
        self.assertLess(len(focused), len(full) * 0.6)

    def test_focused_keeps_every_control_a_passing_metric_depends_on(self) -> None:
        focused = self._prompt("focused", "polite")
        # factual grounding, a hard failure
        self.assertIn("do not invent products", focused.lower())
        self.assertIn("only if it is visible above", focused)
        # the assigned register appears once rather than as a repeated contrast
        self.assertIn("Tone target selector: polite", focused)
        self.assertNotIn(">> polite", focused)
        self.assertIn("first-person positive frame is required here", focused)
        self.assertIn("does not bar the interpersonal", focused)
        # story mode drives mean_story_probability, which currently passes
        self.assertIn("Story realization", focused)
        # the length cue drives length_cv
        self.assertIn(f"roughly {calibrated_word_ask(140)} words", focused)

    def test_focused_keeps_the_planned_discourse_role_once(self) -> None:
        focused = self._prompt(
            "focused",
            "impolite",
            comment_function="reaction",
            payload_type="rant",
            speaker_role="ranter",
            voice="annoyed",
            evidence_mode="none_assertion",
            content_angle="risk_reliability_support",
            stance="hard_disagree",
            detail_focus="sticky shutter marker",
            domain_intent="vent about the failed repair marker",
            avoid_repeating="generic troubleshooting marker",
        )
        for row in (
            "- function: reaction",
            "- payload form: rant",
            "- speaker role: ranter",
            "- voice: annoyed",
            "- evidence basis: none assertion",
            "- content angle: risk reliability support",
            "- stance: hard_disagree",
            "- specific detail: sticky shutter marker",
            "- decision intent: vent about the failed repair marker",
            "- content to avoid: generic troubleshooting marker",
        ):
            self.assertEqual(focused.count(row), 1, row)
        self.assertIn("What kind of turn this is:", focused)

    def test_focused_relation_names_the_actual_visible_target(self) -> None:
        root = self._prompt("focused", "impolite", reply_relation="answers_parent")
        self.assertIn("- relation to post: answers_post", root)
        self.assertNotIn("- reply relation: answers_parent", root)

        reply = self._prompt(
            "focused",
            "impolite",
            local_parent_task_id=9,
            reply_relation="challenges_parent",
        )
        self.assertIn("- reply relation: challenges_parent", reply)
        self.assertNotIn("relation to post", reply)

    def test_focused_drops_the_blocks_no_metric_depends_on(self) -> None:
        focused = self._prompt("focused", "impolite")
        for dropped in (
            "Core metric guidance",
            "Core placeholder guidance",
            "Payload and matched-slot guidance",
            "One-shot semantic difference contract",
            "Planner intent:",
            "Thread-level distribution pressure",
        ):
            self.assertNotIn(dropped, focused, dropped)

    def test_full_mode_keeps_semantics_without_static_metric_boilerplate(self) -> None:
        full = self._prompt("full", "impolite")
        for kept in (
            "One-shot semantic difference contract",
            "Thread-level distribution pressure",
        ):
            self.assertIn(kept, full, kept)
        self.assertNotIn("Core metric guidance", full)
        self.assertNotIn("Core tone and discourse guidance", full)

    def test_semantic_coverage_nonrepeat_reaches_both_writer_prompt_paths(self) -> None:
        # The v108 gate shipped an arm that only touched `_thread_memory`
        # (the `full` path). `focused` is the default and every real
        # generation run in this project's history has used it -- the flag
        # never reached a single prompt on that gate. This is the exact
        # end-to-end check (through `configure_generator_backend` and
        # `build_writer_prompt`, not the helper function directly) that
        # would have caught it before spending, mirroring
        # `test_sentence_rhythm.WriterPromptTest`'s "reaches both paths"
        # convention for the same reason (v74's focused prompt once left
        # 106 of 522 slots on the old path).
        previous_comments = [
            {
                "content": "Earlier comment makes its own point at length.",
                "depth": 0,
                "semantic_move": "an earlier move",
                "decision_boundary": "an earlier boundary",
            }
        ]
        with patch.dict(
            os.environ, {"GENERALIZED_CARD_SEMANTIC_COVERAGE_NONREPEAT": "on"}
        ):
            focused = self._prompt(
                "focused", "impolite", previous_comments=previous_comments
            )
            full = self._prompt(
                "full", "impolite", previous_comments=previous_comments
            )
        self.assertIn(
            "Do not restate one of these already-covered points", focused
        )
        self.assertIn("Do not restate one of these already-covered points", full)

    def test_semantic_coverage_nonrepeat_off_reaches_neither_path(self) -> None:
        previous_comments = [
            {
                "content": "Earlier comment makes its own point at length.",
                "depth": 0,
                "semantic_move": "an earlier move",
                "decision_boundary": "an earlier boundary",
            }
        ]
        focused = self._prompt("focused", "impolite", previous_comments=previous_comments)
        full = self._prompt("full", "impolite", previous_comments=previous_comments)
        self.assertNotIn("Do not restate one of these already-covered points", focused)
        self.assertNotIn("Do not restate one of these already-covered points", full)


class WriterRouteLockTest(unittest.TestCase):
    """The Writer must realize the Planner's move, never transcribe it.

    Nothing asserted on this block before, and it regressed twice unnoticed.
    Longest contiguous shared word run between `semantic_move` and its own
    comment, share at 12 words or more over ~520 slots per run:

        v67 0.4%   v69 1.0%   v73 10.2%   v74 25.8%

    Restricted to comments of 25 words or more, v67 is 0.0% against v74's 34.7%,
    so a healthy run does not do this at all. Two changes produced it: the Writer
    was told "Say this, and only this" in front of a finished sentence, and the
    reply planner was asked for "a full sentence stating what this reply
    asserts". 19.3% of moves open with "I". Reply slots echoed at 25.1% against
    6.4% for root slots, whose schema always said "non-verbatim".
    """

    def setUp(self) -> None:
        self.config = load_domain_config("camera")

    def _writer_prompt(self, route_lock: str) -> str:
        helper = FocusedWriterPromptTest()
        helper.setUp()
        return helper._prompt("focused", "impolite", route_lock=route_lock)

    def test_own_words_does_not_tell_the_writer_to_say_the_move(self) -> None:
        prompt = self._writer_prompt("own_words")
        self.assertNotIn("Say this, and only this", prompt)
        self.assertIn("in your own words", prompt)
        # the counterweight v74 dropped, restored as an explicit rule
        self.assertIn("specification of what to say, never wording to", prompt)

    def test_say_only_reproduces_the_v74_wording(self) -> None:
        prompt = self._writer_prompt("say_only")
        self.assertIn("Say this, and only this", prompt)
        self.assertNotIn("specification of what to say", prompt)

    def _reply_planner_prompt(
        self,
        route_lock: str,
        *,
        domain_claim_mode: str = "planned",
    ) -> str:
        from generalized_card import reply_planning

        backend = SimpleNamespace(
            compact=lambda value, limit: str(value)[:limit],
            CLAIM_FAMILIES=("direct_answer", "tradeoff"),
            GENERALIZED_WRITER_ROUTE_LOCK=route_lock,
            GENERALIZED_DOMAIN_CLAIM_MODE=domain_claim_mode,
        )
        comments = [
            {
                "comment_id": "c1",
                "parent_id": None,
                "depth": 0,
                "body": "root turn body",
            },
            {"comment_id": "c2", "parent_id": "c1", "depth": 1, "body": "a reply body"},
        ]
        return reply_planning.render_direct_reply_planner_prompt(
            config=self.config,
            backend=backend,
            seed_post=SimpleNamespace(
                title="Camera question",
                body="Autofocus issue",
                content="Autofocus issue",
            ),
            comments=[comments[1]],
            all_comments=comments,
            sample_offset=1,
            prior_plans=[
                {
                    "sample_id": "S1",
                    "semantic_move": "name the autofocus tradeoff",
                    "decision_boundary": "whether autofocus is adequate",
                    "detail_focus": "autofocus",
                    "reply_delta_type": "root_turn",
                }
            ],
            slot_distribution="- S2: tone=impolite",
            reference_viewpoints=(
                "- R00001: source_topic=other camera; surface=full_answer; "
                "text=The adapter preserves electronic aperture control."
            ),
            claim_slots={2},
        )

    def test_reply_schema_stops_asking_for_a_finished_sentence(self) -> None:
        prompt = self._reply_planner_prompt("own_words")
        self.assertNotIn("a full sentence stating what this reply asserts", prompt)
        self.assertIn("one concrete but non-verbatim action for this reply", prompt)
        # the scale requirement is what stopped bare noun phrases; it must stay
        self.assertIn("not a bare noun phrase", prompt)

    def test_reply_schema_say_only_reproduces_the_v74_request(self) -> None:
        prompt = self._reply_planner_prompt("say_only")
        self.assertIn("a full sentence stating what this reply asserts", prompt)

    def test_domain_claim_off_does_not_plan_a_fact_the_writer_cannot_see(self) -> None:
        planned = self._reply_planner_prompt("own_words")
        claim_off = self._reply_planner_prompt(
            "own_words",
            domain_claim_mode="off",
        )
        self.assertIn("Give a substantive reply", planned)
        self.assertIn('"domain_claim": "none"', claim_off)
        self.assertIn("information the Writer will not receive", claim_off)
        self.assertNotIn("Give a substantive reply", claim_off)

    def test_selective_reply_claim_has_an_excluded_source_and_fixed_slot(self) -> None:
        prompt = self._reply_planner_prompt(
            "own_words",
            domain_claim_mode="selective",
        )
        self.assertIn("EVALUATION-EXCLUDED REFERENCE ROWS", prompt)
        self.assertIn("R00001", prompt)
        self.assertIn("Selective factual slots in this request: S2", prompt)
        self.assertIn("Every other slot returns ``domain_claim=none``", prompt)
        self.assertIn("Planner-only and never reaches the Writer", prompt)

    def test_deep_reply_excludes_the_full_ancestor_chain(self) -> None:
        from generalized_card import reply_planning

        backend = SimpleNamespace(
            compact=lambda value, limit: str(value)[:limit],
            CLAIM_FAMILIES=("direct_answer",),
            GENERALIZED_WRITER_ROUTE_LOCK="own_words",
            GENERALIZED_DOMAIN_CLAIM_MODE="off",
        )
        comments = [
            {"comment_id": "c1", "parent_id": None, "depth": 0, "body": "root"},
            {"comment_id": "c2", "parent_id": "c1", "depth": 1, "body": "reply"},
            {
                "comment_id": "c3",
                "parent_id": "c2",
                "depth": 2,
                "body": "a sufficiently developed direct reply body " * 4,
            },
        ]
        prompt = reply_planning.render_direct_reply_planner_prompt(
            config=self.config,
            backend=backend,
            seed_post=SimpleNamespace(
                title="Camera question",
                body="Autofocus issue",
                content="Autofocus issue",
            ),
            comments=[comments[2]],
            all_comments=comments,
            sample_offset=2,
            prior_plans=[
                {
                    "sample_id": "S1",
                    "semantic_move": "compare tracking modes",
                    "detail_focus": "subject tracking",
                },
                {
                    "sample_id": "S2",
                    "semantic_move": "test acquisition speed",
                    "decision_boundary": "first-frame lock",
                    "detail_focus": "acquisition speed",
                    "reply_delta_type": "operational_test",
                },
            ],
            slot_distribution="- S3: tone=neutral",
        )
        self.assertIn("Parent semantic move to exclude: test acquisition speed", prompt)
        self.assertIn("Ancestor coverage to exclude: S1", prompt)
        self.assertIn("subject tracking", prompt)
        self.assertIn("Differ from every object listed", prompt)

    def test_reply_story_rule_separates_synthetic_sequence_from_seed_facts(
        self,
    ) -> None:
        prompt = self._reply_planner_prompt("own_words")
        self.assertEqual(
            prompt.count(
                "Synthesize an ordinary, non-verifiable first-person sequence"
            ),
            1,
        )
        self.assertIn("externally checkable outcome", prompt)
        self.assertIn("A fact about this seed post still cannot be invented", prompt)


class OwnFactLicenseTest(unittest.TestCase):
    """A slot must never carry a permission and its revocation at once.

    Measured over the 522 rendered slots of v75, before the split:

        443 (84.9%)  "Do not invent products, specifications, prices,
                      measurements, dates, outcomes, policies, links, or
                      personal experiences."
        249 (47.7%)  "Equipment you may claim as your own, if this turn reports
                      personal experience: ..."
        170 (32.6%)  both, in the same prompt

    and every one of the 249 equipment blocks closed with "do not invent a
    specification, price, measurement, or test result for it", so a slot could
    name its gear and say nothing about it. Generated specifications per comment
    were 0.08 against 0.55 in the matched real threads, and novel brand or model
    tokens 6.6 per thread against 47.3.

    The license splits the ban by *what the fact is about*, and is gated on the
    same predicate as the equipment block so the permission and the facts it
    needs always arrive together.
    """

    def setUp(self) -> None:
        self.config = load_domain_config("camera")

    def _prompt(self, license_mode: str, *, story_mode: str = "no_story") -> str:
        helper = FocusedWriterPromptTest()
        helper.setUp()
        return helper._prompt(
            "focused",
            "impolite",
            license_mode=license_mode,
            story_mode=story_mode,
        )

    def _backend(self, license_mode: str):
        return SimpleNamespace(
            GENERALIZED_DOMAIN_PROFILE={},
            GENERALIZED_OWN_FACT_LICENSE=license_mode,
            compact=lambda text, limit: " ".join(str(text or "").split())[:limit],
        )

    def _task(self, **overrides):
        base = dict(
            allow_first_person_frame=True,
            evidence_mode="firsthand_experience",
            story_mode="no_story",
            payload_type="soft_helpful",
            local_task_id=1,
            concrete_anchors=(),
        )
        base.update(overrides)
        return SimpleNamespace(**base)

    # --- off is a true ablation control, not an approximation ---------------

    def test_off_reproduces_the_v75_blanket_ban_verbatim(self) -> None:
        from generalized_card.writer_grounding import story_fact_rule

        rule = story_fact_rule(self._task(), has_domain_claim=False, mode="off")
        self.assertEqual(
            rule,
            "Do not invent products, specifications, prices, measurements, "
            "dates, outcomes, policies, links, or personal experiences.",
        )

    def test_off_reproduces_the_v75_story_rule_verbatim(self) -> None:
        from generalized_card.writer_grounding import story_fact_rule

        rule = story_fact_rule(
            self._task(story_mode="specific_personal_story"),
            has_domain_claim=False,
            mode="off",
        )
        self.assertIn("This is a synthetic story slot", rule)
        self.assertIn("externally checkable outcome", rule)

    def test_off_keeps_the_blanket_ban_in_the_rendered_prompt(self) -> None:
        self.assertIn("or personal experiences", self._prompt("off"))

    def test_equipment_clause_tracks_the_license(self) -> None:
        from generalized_card.writer_grounding import equipment_closing_clause

        with self.assertRaisesRegex(ValueError, "explicit own-fact license"):
            equipment_closing_clause(mode="off")
        own = equipment_closing_clause(mode="own")
        self.assertNotIn("do not invent a specification", own)
        self.assertIn("what you paid, and how it held up", own)
        self.assertIn("Do not attribute it to the post", own)

    def test_unlicensed_prompt_never_offers_invented_equipment(self) -> None:
        from generalized_card import prompts

        backend = self._backend("off")
        backend.GENERALIZED_DOMAIN_PROFILE = {
            "entity_inventory": {
                "available": True,
                "terms": [{"term": "Invented Model 7"}],
            }
        }
        block = prompts._own_equipment_block(backend, self._task())
        self.assertEqual(block, "")

    # --- own removes the contradiction on every writer path -----------------

    def test_own_licenses_the_speakers_own_facts(self) -> None:
        prompt = self._prompt("own")
        self.assertIn("are yours to state", prompt)
        self.assertNotIn("or personal experiences", prompt)

    def test_own_still_grounds_the_product_under_discussion(self) -> None:
        """The half that must not move. A wrong spec here is a catchable error,
        and one shared invented fact is what made `domain_claim` a regression."""

        prompt = self._prompt("own", story_mode="specific_personal_story")
        self.assertIn("product under discussion", prompt)
        self.assertIn("unless it is visible above", prompt)

    def test_own_gives_a_story_its_consequence_back(self) -> None:
        from generalized_card.writer_grounding import story_fact_rule

        rule = story_fact_rule(
            self._task(story_mode="specific_personal_story"),
            has_domain_claim=False,
            mode="own",
        )
        self.assertIn("what came of it", rule)
        self.assertNotIn("externally checkable outcome", rule)

    def test_own_does_not_license_a_slot_with_no_first_person_frame(self) -> None:
        """Licensing own facts on a slot barred from a first-person frame would
        replace one contradiction with another."""

        from generalized_card.writer_grounding import licensed_for

        task = self._task(
            allow_first_person_frame=False,
            evidence_mode="small_observation",
            story_mode="no_story",
            payload_type="soft_helpful",
        )
        self.assertFalse(licensed_for(self._backend("own"), task))
        self.assertTrue(licensed_for(self._backend("own"), self._task()))

    def test_the_license_gate_matches_the_equipment_gate(self) -> None:
        """One predicate, so the permission and the facts never drift apart."""

        from generalized_card import prompts
        from generalized_card.writer_grounding import (
            first_person_experience_slot,
            licensed_for,
        )

        backend = self._backend("own")
        for task in (
            self._task(),
            self._task(
                allow_first_person_frame=False, evidence_mode="small_observation"
            ),
            self._task(
                allow_first_person_frame=False,
                evidence_mode="small_observation",
                story_mode="specific_personal_story",
            ),
        ):
            self.assertEqual(
                licensed_for(backend, task),
                first_person_experience_slot(task),
            )
            self.assertEqual(
                bool(prompts._first_person_experience_slot(task)),
                first_person_experience_slot(task),
            )

    # --- every path, not 80% of them ----------------------------------------

    def test_both_entity_rules_carry_the_split(self) -> None:
        """v74 applied its fix to only 416 of 522 slots because the low-info
        path was never converted, and the release could not be attributed."""

        from generalized_card import prompts

        backend = self._backend("own")
        task = self._task()
        for renderer in (
            prompts._focused_path_entity_rule,
            prompts._full_path_entity_rule,
        ):
            rule = renderer(backend, task)
            self.assertIn("your own gear and your own history", rule)
            self.assertIn("what is visible above", rule)

    def test_system_prompt_sentence_is_empty_when_off(self) -> None:
        from generalized_card.writer_grounding import system_prompt_fact_sentence

        self.assertEqual(system_prompt_fact_sentence(mode="off"), "")
        own = system_prompt_fact_sentence(mode="own")
        self.assertIn("may explicitly license facts about the speaker's own", own)
        self.assertIn("only for that turn", own)
        self.assertIn("otherwise do not invent personal history", own)


class NamedConcretenessLicenseTest(unittest.TestCase):
    """The correction to `own`, which run v76b refuted.

    v76b measured the personal-history license on seed 8 and moved concreteness
    the wrong way: 0.05 -> 0.02 specification tokens per comment against a real
    0.54, and 0.083 -> 0.024 on the licensed slots themselves. Two measurements
    say why. Across the ten matched real threads, 78 of 114 spec-carrying
    comments (68%) contain no first-person frame at all, so the gate selected the
    wrong slots; and replacing a vague blanket ban with an explicit "about the
    product under discussion, name only what is visible" sharpened a prohibition
    on exactly the detail real comments are full of.

    `named` is also stated without domain vocabulary, because specification-shaped
    tokens are thread-dependent -- 0% of comments in seed 1, 64% in seed 5. What
    holds on all ten threads is quantities (real 12.3x generated) and proper
    nouns (real 1.85x).
    """

    def _backend(self, mode: str):
        return SimpleNamespace(
            GENERALIZED_DOMAIN_PROFILE={},
            GENERALIZED_OWN_FACT_LICENSE=mode,
            compact=lambda text, limit: " ".join(str(text or "").split())[:limit],
        )

    def _task(self, **overrides):
        base = dict(
            allow_first_person_frame=False,
            evidence_mode="small_observation",
            story_mode="no_story",
            payload_type="soft_helpful",
            length_bucket="long",
            real_word_count=120,
            local_task_id=1,
            concrete_anchors=(),
        )
        base.update(overrides)
        return SimpleNamespace(**base)

    def test_named_licenses_a_slot_with_no_personal_frame(self) -> None:
        """The case `own` got wrong: 68% of real concrete comments look like
        this one."""

        from generalized_card.writer_grounding import slot_license

        task = self._task()
        self.assertEqual(slot_license(self._backend("own"), task), "off")
        self.assertEqual(slot_license(self._backend("named"), task), "named")

    def test_named_does_not_license_a_slot_with_no_room(self) -> None:
        """`missing_concrete_anchor` fired 84 times against slots that could not
        satisfy it."""

        from generalized_card.writer_grounding import slot_license

        for task in (
            self._task(real_word_count=6, length_bucket="micro"),
            self._task(real_word_count=9, length_bucket="short"),
        ):
            self.assertEqual(slot_license(self._backend("named"), task), "off")

    def test_the_named_rule_carries_no_domain_vocabulary(self) -> None:
        """A camera word here is a domain-transfer bug: the same rule has to
        work for news, sport, or audio."""

        from generalized_card.writer_grounding import (
            NAMED_ENTITY_RULE,
            story_fact_rule,
            system_prompt_fact_sentence,
        )

        texts = [
            NAMED_ENTITY_RULE,
            system_prompt_fact_sentence(mode="named"),
            story_fact_rule(self._task(), has_domain_claim=False, mode="named"),
            story_fact_rule(
                self._task(story_mode="specific_personal_story"),
                has_domain_claim=False,
                mode="named",
            ),
        ]
        banned = (
            "gear",
            "camera",
            "lens",
            "shot",
            "shoot",
            "photo",
            "iso",
            "aperture",
            "specification",
            "megapixel",
            "sensor",
            "product",
        )
        for text in texts:
            for word in banned:
                self.assertNotIn(word, text.lower(), f"{word!r} in {text!r}")

    def test_named_guards_convergence_rather_than_detail(self) -> None:
        """v71 injected one planned fact into 508 of 522 comments and produced
        157 extra semantic-overlap flags. Real concreteness is the opposite
        shape: 104 quantity tokens over 44 distinct values in one thread."""

        from generalized_card.writer_grounding import NAMED_ENTITY_RULE

        self.assertIn("Repeating its name is normal", NAMED_ENTITY_RULE)
        self.assertIn("do not repeat another comment's same fact", NAMED_ENTITY_RULE)
        self.assertNotIn("do not repeat a name", NAMED_ENTITY_RULE)
        self.assertIn("consistent with", NAMED_ENTITY_RULE)

    def test_named_personal_slot_receives_grounded_equipment_options(self) -> None:
        from generalized_card import prompts

        backend = self._backend("named")
        backend.GENERALIZED_DOMAIN_PROFILE = {
            "entity_inventory": {
                "available": True,
                "terms": [
                    {"term": "Sony a7III"},
                    {"term": "Fujifilm X-T3"},
                ],
            }
        }
        task = self._task(
            allow_first_person_frame=True,
            evidence_mode="firsthand_experience",
            story_mode="specific_personal_story",
        )
        rendered = prompts._own_equipment_block(backend, task)
        self.assertIn("Equipment you may claim as your own", rendered)
        self.assertIn("Sony a7III", rendered)
        self.assertEqual(
            prompts._own_equipment_block(
                backend,
                task,
                has_domain_claim=True,
            ),
            "",
        )

    def test_all_three_modes_stay_distinct_and_off_is_unchanged(self) -> None:
        from generalized_card.writer_grounding import entity_naming_rule

        rules = {
            mode: entity_naming_rule(mode=mode) for mode in ("off", "own", "named")
        }
        self.assertEqual(len(set(rules.values())), 3)
        self.assertEqual(
            rules["off"],
            "Name a product, model, or number only if it is visible above.",
        )

    def test_named_system_permission_is_explicitly_slot_gated(self) -> None:
        with patch.dict(
            os.environ,
            {"GENERALIZED_CARD_OWN_FACT_LICENSE": "named"},
        ):
            module = configure_generator_backend(
                load_generator_backend(),
                load_domain_config("camera"),
            )
        system_prompt = module.SYSTEM_PROMPTS["gpt54_reddit_writer"]
        self.assertIn("may explicitly license particulars not visible", system_prompt)
        self.assertIn("exception to the preceding visibility rule", system_prompt)
        self.assertIn("otherwise do not invent a name or amount", system_prompt)
        self.assertNotIn(
            "name the particular things you mean and give amounts",
            system_prompt,
        )

    def test_named_user_permission_reaches_only_substantive_slots(self) -> None:
        helper = FocusedWriterPromptTest()
        helper.setUp()
        substantive = helper._prompt("focused", "neutral", license_mode="named")
        micro = helper._prompt(
            "focused",
            "neutral",
            license_mode="named",
            real_word_count=4,
            length_bucket="micro",
            payload_type="low_info_reaction",
            comment_function="reaction",
            utterance_mode="fragment_only",
        )
        named_permission = "name the specific things you mean and give the amounts"
        self.assertIn(named_permission, substantive)
        self.assertNotIn(named_permission, micro)
        self.assertIn(
            "Named entities and numbers may appear only when visible",
            micro,
        )


class SpeakerRosterTest(unittest.TestCase):
    """A thread is people, not slots.

    `run_sampled_reddit_generator.py` built the author name as a pure function of
    the slot index, so every comment came from a different person: 186 comments,
    186 one-shot authors. The matched real threads are 559 comments from 265
    named participants -- 2.11 each -- with 68% of comment mass written by
    someone who speaks more than once, and seed 8's busiest author writing 10.

    Naming 186 users does not make 186 voices. It makes one voice wearing 186
    name tags, which is what `self_bertscore`'s near-uniform +0.033 overshoot on
    9 of 10 threads looks like.

    The structure is a join, not a new sampling policy: `real_sample_id` already
    binds each slot to one matched real comment, and that comment has an author.
    Verified on seed 8 -- `real_word_count` agrees for 186 of 186 slots.
    """

    def _rows(self):
        # Shape taken from a real thread: one prolific speaker, one OP, two
        # deleted accounts, and singletons.
        return [
            {"author": "alice", "is_submitter": True, "body": "a"},
            {"author": "bob", "is_submitter": False, "body": "b"},
            {"author": "[deleted]", "is_submitter": False, "body": "c"},
            {"author": "alice", "is_submitter": False, "body": "d"},
            {"author": "carol", "is_submitter": False, "body": "e"},
            {"author": "[deleted]", "is_submitter": False, "body": "f"},
            {"author": "bob", "is_submitter": False, "body": "g"},
            {"author": "alice", "is_submitter": False, "body": "h"},
        ]

    def _roster(self):
        from generalized_card.speaker_roster import build_speaker_roster

        return build_speaker_roster(self._rows())

    def test_slots_group_into_the_people_who_wrote_them(self) -> None:
        roster = self._roster()
        alice = roster.speaker_for(1)
        self.assertEqual(alice.slot_ids, (1, 4, 8))
        self.assertEqual(roster.speaker_for(4).speaker_id, alice.speaker_id)
        self.assertEqual(roster.speaker_for(8).speaker_id, alice.speaker_id)
        self.assertEqual(roster.speaker_for(2).slot_ids, (2, 7))

    def test_the_op_is_one_person_across_their_turns(self) -> None:
        roster = self._roster()
        self.assertTrue(roster.speaker_for(1).is_op)
        # is_submitter is only set on their first row; the speaker still owns it
        self.assertTrue(roster.speaker_for(8).is_op)
        self.assertFalse(roster.speaker_for(2).is_op)

    def test_deleted_accounts_never_merge_into_one_prolific_speaker(self) -> None:
        roster = self._roster()
        first, second = roster.speaker_for(3), roster.speaker_for(6)
        self.assertNotEqual(first.speaker_id, second.speaker_id)
        self.assertEqual(first.slot_ids, (3,))
        self.assertEqual(second.slot_ids, (6,))
        self.assertTrue(first.anonymous and second.anonymous)
        self.assertEqual(roster.summary()["anonymous_speaker_count"], 2)
        self.assertEqual(roster.summary()["named_speaker_count"], 3)

    def test_speaker_structure_carries_no_invented_biography(self) -> None:
        roster = self._roster()
        for speaker in roster.speakers:
            for field in ("display_name", "kit", "tenure", "use_case"):
                self.assertFalse(hasattr(speaker, field), field)

    def test_earlier_slots_are_only_the_ones_already_written(self) -> None:
        roster = self._roster()
        self.assertEqual(roster.earlier_slots(1), ())
        self.assertEqual(roster.earlier_slots(4), (1,))
        self.assertEqual(roster.earlier_slots(8), (1, 4))

    def test_the_real_author_string_never_reaches_a_speaker(self) -> None:
        """Only the participation structure crosses the boundary, never the
        identity itself."""

        import dataclasses

        roster = self._roster()
        names = {row["author"] for row in self._rows()}
        for speaker in roster.speakers:
            for value in dataclasses.astuple(speaker):
                if isinstance(value, str):
                    self.assertNotIn(value, names)
                    self.assertNotIn(value.lower(), {n.lower() for n in names})

    def test_active_expander_assigns_matched_speaker_ids(self) -> None:
        """Exercise the configured wrapper, not only the roster helper."""

        with patch.dict(
            os.environ,
            {"GENERALIZED_CARD_SPEAKER_IDENTITY": "matched"},
        ):
            module = configure_generator_backend(
                load_generator_backend(), load_domain_config("camera")
            )
        comments = [
            {
                **row,
                "comment_id": f"c{index}",
                "post_id": "p1",
                "parent_id": "t3_p1",
                "depth": 0,
            }
            for index, row in enumerate(self._rows(), start=1)
        ]
        plans = {
            index: {
                "branch_id": "1",
                "semantic_move": f"make distinct local point {index}",
                "local_topic": f"topic {index}",
                "detail_focus": f"detail {index}",
                "domain_intent": f"discuss point {index}",
                "payload_type": "bare_answer",
                "comment_function": "verdict_evaluation",
                "story_mode": "no_story",
                "evidence_mode": "none_assertion",
                "speaker_role": "side_observer",
                "voice": "casual_neutral",
            }
            for index in range(1, len(comments) + 1)
        }
        tasks = module.expand_matched_real_sample_to_tasks(
            branches=[
                module.BranchPlan(
                    branch_id=1,
                    anchor_quote="visible topic",
                    anchor_source="seed",
                    detour_type="none",
                    branch_goal="discuss one local point",
                    allowed_functions=("verdict_evaluation",),
                    evidence_modes=("none_assertion",),
                    tone_palette=("neutral",),
                    story_modes=("no_story",),
                    content_angles=("fit_use_case",),
                )
            ],
            target=module.ThreadTarget(
                target_comments=len(comments),
                top_level_comments=len(comments),
                max_depth_goal=0,
                shape_label="flat",
                length_mix_note="matched",
            ),
            seed_post=module.SeedPost(
                index=0,
                title="Visible topic",
                body="One visible question",
                content="Visible topic",
                source_raw_post_id="p1",
                real_num_comments=len(comments),
                metadata={},
            ),
            matched_real_thread={"comments": comments},
            matched_real_comments=0,
            comment_plans=plans,
            rng=random.Random(7),
        )
        self.assertEqual(len(tasks), len(comments))
        self.assertEqual(tasks[0].speaker_id, tasks[3].speaker_id)
        self.assertEqual(tasks[0].speaker_id, tasks[7].speaker_id)
        self.assertEqual(tasks[1].speaker_id, tasks[6].speaker_id)
        self.assertNotEqual(tasks[0].speaker_id, tasks[1].speaker_id)
        self.assertEqual(
            module.GENERALIZED_ACTIVE_SPEAKER_ROSTER.summary()["slot_count"],
            len(comments),
        )

    def test_off_yields_no_roster_and_no_speaker_block(self) -> None:
        from generalized_card import prompts
        from generalized_card.speaker_roster import EMPTY_ROSTER

        backend = SimpleNamespace(
            GENERALIZED_ACTIVE_SPEAKER_ROSTER=EMPTY_ROSTER,
            compact=lambda text, limit: str(text)[:limit],
        )
        task = SimpleNamespace(real_sample_id=1)
        self.assertIsNone(prompts._speaker_for_task(backend, task))
        self.assertEqual(prompts._speaker_identity_block(backend, task, []), "")

    def test_the_block_shows_a_speaker_their_own_earlier_turns(self) -> None:
        from generalized_card import prompts

        roster = self._roster()
        backend = SimpleNamespace(
            GENERALIZED_ACTIVE_SPEAKER_ROSTER=roster,
            compact=lambda text, limit: str(text)[:limit],
        )
        alice = roster.speaker_for(1)
        previous = [
            {"speaker_id": alice.speaker_id, "content": "my first take on this"},
            {"speaker_id": roster.speaker_for(2).speaker_id, "content": "someone else"},
        ]
        block = prompts._speaker_identity_block(
            backend, SimpleNamespace(real_sample_id=4), previous
        )
        self.assertIn("my first take on this", block)
        self.assertNotIn("someone else", block)
        self.assertIn("Same participant", block)
        self.assertIn("follow this turn's assigned voice and affect", block)
        self.assertIn("You are the person who wrote the post", block)

    def test_speaker_id_survives_the_surface_rebalancer(self) -> None:
        """`semantic_move` was lost in 347 of 347 reply slots because it was not
        listed as an invariant. A field the surface pass can reach but does not
        own is a field that disappears."""

        from generalized_card.task_distribution import PLANNER_AND_SLOT_INVARIANTS

        self.assertIn("speaker_id", PLANNER_AND_SLOT_INVARIANTS)


class MicroReactionShapeTest(unittest.TestCase):
    """A forced micro reaction must not collide itself into a dropped comment.

    The pool was 6 strings indexed by `local_task_id`. One v74 thread held 10
    micro slots, so tasks 26, 32 and 116 all resolved to "This"; the first was
    kept and the other two raised `exact_duplicate` on every repair round,
    because local repair never changes the task id. Both were dropped after the
    budget, which is two comments lost to a fixable defect.
    """

    def _shaped(self, module, task_id: int, text: str) -> str:
        task = SimpleNamespace(
            real_surface_shape="micro_reaction",
            local_task_id=task_id,
            surface_texture="plain",
            utterance_mode="one_datapoint",
        )
        return module.shape_writer_text_for_task(text, task)

    def test_a_fresh_candidate_can_escape_a_collision(self) -> None:
        module = load_generator_backend()
        first = self._shaped(
            module, 32, "a long candidate that must be cut down to size"
        )
        second = self._shaped(
            module, 32, "a different long candidate needing the same cut"
        )
        self.assertNotEqual(
            first,
            second,
            "same task id must not force the same fallback, or repair cannot escape",
        )

    def test_pool_is_wider_than_the_worst_observed_thread(self) -> None:
        module = load_generator_backend()
        produced = {
            self._shaped(
                module, task_id, f"candidate number {task_id} is far too long here"
            )
            for task_id in range(40)
        }
        # v74's worst thread had 10 micro slots
        self.assertGreaterEqual(len(produced), 10)


if __name__ == "__main__":
    unittest.main()


class PlanMoveLedgerTest(unittest.TestCase):
    """v124: the spent-move ledger must be off by default and reach the repair.

    E12 cost a paid run because an arm recorded itself ON in `run_config.json`
    while rendering into zero prompts -- verified at the time against a
    hand-built dict that did not match the real data. These assertions load
    REAL planner slots off disk for that reason.
    """

    def _real_plans(self) -> dict:
        run = (
            REPO_ROOT
            / "artifacts/generalized_card/runs/v122_writer_retries_n10_20260828_v1"
        )
        paths = sorted(run.glob("generated/run_*_sampled_reddit/discussion.json"))
        if not paths:
            self.skipTest("v122 artifact not present")
        plans: dict[int, dict] = {}
        data = json.loads(paths[0].read_text(encoding="utf-8"))
        for post in data.get("posts", []):
            stack = list(post.get("comments") or [])
            while stack:
                node = stack.pop(0)
                plans[len(plans)] = dict(node)
                stack.extend(node.get("replies") or [])
            break
        return plans

    def test_ledger_is_off_by_default_and_renders_when_armed(self) -> None:
        from generalized_card.planning_quality import (
            PLAN_MOVE_LEDGER_MODE,
            evaluate_plan_batch,
            set_plan_move_ledger,
        )

        self.assertEqual(PLAN_MOVE_LEDGER_MODE, "off")
        report = evaluate_plan_batch(self._real_plans())
        # The ledger is always collected; only its rendering is armed.
        self.assertGreater(len(report.spent_moves), 20)
        try:
            set_plan_move_ledger("off")
            self.assertEqual(report.spent_move_block(), "")
            set_plan_move_ledger("spent_moves")
            block = report.spent_move_block()
            self.assertIn("already spent the following semantic moves", block)
            # A concrete instruction, per E4 -- not a bare category.
            self.assertIn("not on this list", block)
            # The block shows the most RECENT moves: a repair competes with what
            # the thread just said, and an unbounded ledger would crowd the
            # prompt on a 186-comment thread. Assert on the newest, and assert
            # the elision is declared rather than silent.
            self.assertIn(report.spent_moves[-1][:40], block)
            if len(report.spent_moves) > 24:
                self.assertIn("earlier move(s)", block)
        finally:
            set_plan_move_ledger("off")

    def test_unknown_mode_is_rejected(self) -> None:
        from generalized_card.planning_quality import set_plan_move_ledger

        with self.assertRaises(ValueError):
            set_plan_move_ledger("on")


class OutsiderQuotaTest(unittest.TestCase):
    """v125: the topical-outsider quota must be off by default and reach the prompt."""

    def test_quota_is_off_by_default_and_names_concrete_channels(self) -> None:
        from generalized_card.planning_quality import (
            OUTSIDER_QUOTA_MODE,
            outsider_quota_block,
            set_outsider_quota,
        )

        self.assertEqual(OUTSIDER_QUOTA_MODE, "off")
        self.assertEqual(outsider_quota_block(45), "")
        try:
            set_outsider_quota("measured")
            block = outsider_quota_block(45)
            # Named channels, not a category -- E4 prices a category at 0.23.
            for channel in ("offtopic_noise", "joke", "side_tangent"):
                self.assertIn(channel, block)
            # The long-outsider requirement is the one category we ship at zero.
            self.assertIn("LONG slot", block)
            # And it must not be satisfiable with the thing we over-produce.
            self.assertIn("thanks", block)
            # Sized against the BATCH the Planner actually sees, not the thread:
            # v125's first run asked a 186-slot thread for 22 outsiders inside a
            # call that could place 8, and an unsatisfiable quota is discarded
            # whole (compliance 0.5% there against 14.3% on a 7-slot thread).
            self.assertIn("of these 8 slots", outsider_quota_block(8))
            # Too small, or a quota that would eat the batch: no instruction
            # rather than an impossible one.
            self.assertEqual(outsider_quota_block(3), "")
            self.assertEqual(outsider_quota_block(1), "")
        finally:
            set_outsider_quota("off")

    def test_quota_reaches_the_rendered_planner_prompt(self) -> None:
        """G23/G41: a computed block that never interpolates is a silent no-op."""

        import inspect

        from generalized_card import prompts

        source = inspect.getsource(prompts.comment_planner_prompt)
        self.assertIn("outsider_quota_block(", source)
        self.assertIn("{outsider_block}", source)

    def test_unknown_mode_is_rejected(self) -> None:
        from generalized_card.planning_quality import set_outsider_quota

        with self.assertRaises(ValueError):
            set_outsider_quota("on")


def test_writer_temperature_legacy_sends_no_temperature_for_gpt5():
    """`legacy` must reproduce v128: gpt-5* writer calls carry no temperature."""

    from generalized_card import backend

    backend.set_writer_temperature("legacy")
    try:
        kwargs = backend._completion_kwargs(
            model="gpt-5.4-mini",
            messages=[],
            temperature=0.82,
            max_tokens=110,
            response_format_json=False,
            extra_body=None,
        )
        assert "temperature" not in kwargs
        assert kwargs["max_completion_tokens"] == 366
    finally:
        backend.set_writer_temperature("legacy")


def test_writer_temperature_arm_never_touches_the_planner():
    """The arm gates on response_format_json, which only the planner sets."""

    from generalized_card import backend

    backend.set_writer_temperature("1.3")
    try:
        writer = backend._completion_kwargs(
            model="gpt-5.4-mini",
            messages=[],
            temperature=0.82,
            max_tokens=110,
            response_format_json=False,
            extra_body=None,
        )
        planner = backend._completion_kwargs(
            model="gpt-5.4-mini",
            messages=[],
            temperature=0.30,
            max_tokens=110,
            response_format_json=True,
            extra_body=None,
        )
        assert writer["temperature"] == 1.3
        assert "temperature" not in planner
    finally:
        backend.set_writer_temperature("legacy")


def test_writer_temperature_schedule_honours_the_per_slot_value():
    """`schedule` passes through what `writer_temperature(task)` computed."""

    from generalized_card import backend

    backend.set_writer_temperature("schedule")
    try:
        for scheduled in (0.82, 0.88, 0.95, 1.08):
            kwargs = backend._completion_kwargs(
                model="gpt-5.4-mini",
                messages=[],
                temperature=scheduled,
                max_tokens=110,
                response_format_json=False,
                extra_body=None,
            )
            assert kwargs["temperature"] == scheduled
    finally:
        backend.set_writer_temperature("legacy")


def test_writer_temperature_rejects_out_of_range_and_non_numeric():
    from generalized_card import backend

    try:
        for bad in ("3.0", "-0.5", "hot"):
            try:
                backend.set_writer_temperature(bad)
            except ValueError:
                continue
            raise AssertionError(f"accepted invalid writer temperature: {bad}")
    finally:
        backend.set_writer_temperature("legacy")


def _persona_runtime(mode="matraix-full"):
    from pathlib import Path

    from generalized_card.persona_bridge import build_runtime

    root = Path(__file__).resolve().parents[2] / "third_party" / "MatrAIx-Persona-8B"
    if not root.is_dir():
        import pytest

        pytest.skip("MatrAIx checkout not present")
    return build_runtime(
        mode=mode,
        matraix_root=root,
        dataset_dir=root / "persona" / "datasets" / "matraix-persona-dev-sample",
        assignment_seed=42,
        expertise_dimensions=("fam_photography", "ind_consumer_electronics"),
    )


def test_persona_speaker_key_is_stable_across_a_speakers_slots():
    """One person, one voice: every slot a speaker holds gets the same persona."""

    runtime = _persona_runtime()
    tasks = [
        {"local_task_id": i, "speaker_role": role, "voice": voice, "tone_shape": tone}
        for i, (role, voice, tone) in enumerate(
            [
                ("advisor", "blunt", "disagree"),
                ("confused_asker", "uncertain", "uncertain"),
                ("gratitude_reply", "grateful", "polite"),
                ("jokester", "sarcastic", ""),
            ]
        )
    ]
    assigned = {
        runtime.assign(seed_index=3, task=task, speaker_id="S007").persona_id
        for task in tasks
    }
    assert len(assigned) == 1, "a speaker must keep one persona across differing roles"


def test_persona_without_a_speaker_still_varies_per_slot():
    """`--speaker-identity off` leaves the legacy per-slot behaviour intact."""

    runtime = _persona_runtime()
    assigned = {
        runtime.assign(
            seed_index=3, task={"local_task_id": i, "speaker_role": "advisor"}, speaker_id=""
        ).persona_id
        for i in range(12)
    }
    assert len(assigned) > 1


def test_persona_speaker_key_separates_different_speakers():
    runtime = _persona_runtime()
    task = {"local_task_id": 1, "speaker_role": "advisor"}
    assigned = {
        runtime.assign(seed_index=3, task=task, speaker_id=f"S{i:03d}").persona_id
        for i in range(1, 25)
    }
    assert len(assigned) >= 12, f"only {len(assigned)} personas over 24 speakers"


def test_persona_system_prompts_stay_under_the_dilution_cap():
    from generalized_card.persona_bridge import _MAX_SYSTEM_PROMPT_CHARS

    runtime = _persona_runtime()
    longest = max(
        len(runtime.assignment_for_id(p.persona_id).system_prompt)
        for p in runtime._eligible
    )
    assert longest <= _MAX_SYSTEM_PROMPT_CHARS
    assert len(runtime._eligible) >= 100
    assert all(
        "## Who you are" in runtime.assignment_for_id(p.persona_id).system_prompt
        for p in runtime._eligible
    )


def test_speaker_id_recovered_from_generated_author():
    from generalized_card.persona_bridge import _speaker_id_from_author

    assert _speaker_id_from_author("sampled_user_0_0_S001") == "S001"
    assert _speaker_id_from_author("sampled_user_12_3_S042") == "S042"
    for absent in ("", None, "weird_name"):
        assert _speaker_id_from_author(absent) == ""


def test_recurring_phrase_ledger_off_reproduces_v133():
    from generalized_card.semantic_realization import (
        recurring_function_phrases,
        set_recurring_phrase_ledger,
    )

    set_recurring_phrase_ledger("off")
    try:
        comments = [{"content": "this is the one and the other is the same"}] * 6
        assert recurring_function_phrases(comments) == []
    finally:
        set_recurring_phrase_ledger("off")


def test_recurring_phrase_ledger_lists_only_reused_function_pairs():
    from generalized_card.semantic_realization import (
        recurring_function_phrases,
        set_recurring_phrase_ledger,
    )

    set_recurring_phrase_ledger("3")
    try:
        comments = [
            {"content": "the ricoh is the one I would get"},
            {"content": "the ricoh is the body that fits"},
            {"content": "the ricoh is the pick for me"},
            {"content": "a different sentence entirely here"},
        ]
        listed = recurring_function_phrases(comments)
        assert "is the" in listed, listed
        # `the ricoh` recurs just as often and must NOT be listed: real threads
        # reuse their topic nouns, so suppressing them moves away from real.
        assert "the ricoh" not in listed, listed
        assert all(len(v.split()) == 2 for v in listed)
    finally:
        set_recurring_phrase_ledger("off")


def test_recurring_phrase_ledger_respects_the_minimum_and_the_cap():
    from generalized_card.semantic_realization import (
        RECURRING_PHRASE_LIMIT,
        recurring_function_phrases,
        set_recurring_phrase_ledger,
    )

    twice = [{"content": "it is the same"}, {"content": "it is the other"}]
    set_recurring_phrase_ledger("3")
    try:
        assert recurring_function_phrases(twice) == []
        set_recurring_phrase_ledger("2")
        assert "is the" in recurring_function_phrases(twice)
        many = [
            {"content": " ".join(f"{a} {b}" for a, b in zip(
                "the a is of to in on at for with from by".split(),
                "the a is of to in on at for with from by".split()))}
        ] * 5
        assert len(recurring_function_phrases(many)) <= RECURRING_PHRASE_LIMIT
    finally:
        set_recurring_phrase_ledger("off")


def test_recurring_phrase_ledger_rejects_a_meaningless_minimum():
    from generalized_card.semantic_realization import set_recurring_phrase_ledger

    try:
        for bad in ("1", "0", "-2"):
            try:
                set_recurring_phrase_ledger(bad)
            except ValueError:
                continue
            raise AssertionError(f"accepted {bad}")
    finally:
        set_recurring_phrase_ledger("off")


def test_planner_residue_guard_accepts_real_product_model_names():
    """A camera called S5 or X-S20 is not planner skeleton residue.

    The `(?:P\\d{2}|S\\d+|B\\d+)` control-id pattern reads the Fujifilm X-S20
    and the Panasonic Lumix S5 as slot ids. Restricting the whitelist to
    anchors tagged `(seed)` meant a model name arriving as `(planner)` or
    `(local)` failed six writer retries and three post retries and killed a
    whole run at $1.50 with five threads produced.
    """
    import types

    module = types.SimpleNamespace(GENERALIZED_DOMAIN_PROFILE={"perspectives": []})
    from generalized_card import backend as _backend
    check = _backend._planner_residue_check(module, lambda text, task: False)
    task = types.SimpleNamespace(
        concrete_anchors=("X-S20 (planner)", "Lumix S5 (local)")
    )
    bare = types.SimpleNamespace(concrete_anchors=())

    # product names, whatever tag the anchor carried
    assert not check("way smaller than a Fuji X-S20, honestly", task)
    assert not check("the linked S5 II setup solves a different problem", task)
    # hyphen shape is product shape even with no anchor at all
    assert not check("panasonic-lumix-dc-s5-ii-mirrorless", bare)
    assert not check("a Fuji X-S20 body", bare)
    # a bare slot id with nothing to justify it is still residue
    assert check("cover S20 in this reply", bare)
    assert check("branch B3 owns the price angle", bare)


def test_audit_control_id_leak_ignores_urls_and_product_names():
    """A Canon URL containing `rf-s18-45mm` is not a leaked planner slot id.

    The same `(?:P\\d{2}|S\\d+|B\\d+)` pattern that misread "Fujifilm X-S20"
    in the writer guard also fires inside
    `.../refurbished-eos-r50-rf-s18-45mm-...`, which failed the
    evaluation-integrity audit on a finished 40-thread run and blocked its
    evaluation outright.
    """
    from generalized_card.audit import _is_product_shaped
    from generalized_card.prompts import INTERNAL_CONTROL_ID_RE

    def only(text):
        m = list(INTERNAL_CONTROL_ID_RE.finditer(text))
        assert m, text
        return text, m[0]

    # inside a URL
    assert _is_product_shaped(*only("see https://www.usa.canon.com/p/rf-s18-45mm-kit here"))
    # hyphenated product model
    assert _is_product_shaped(*only("way smaller than a Fuji X-S20"))
    # a bare slot id in prose is still a leak
    assert not _is_product_shaped(*only("cover S20 in this reply"))
    assert not _is_product_shaped(*only("branch B3 owns the price angle"))


def test_deepseek_counts_as_a_reasoning_model_for_the_token_reserve():
    """DeepSeek v4 spends reasoning tokens out of the same completion budget.

    It returns `reasoning_content` beside `content` and charges both against
    the cap, so a 260-token writer call returns `finish_reason=length` with an
    empty body -- the exact gpt-5 failure the reserve branch exists for. Before
    this, `--gpt5-reasoning-token-reserve` was silently inert for deepseek and
    a smoke run lost every post to empty completions.
    """
    from generalized_card.backend import _uses_max_completion_tokens

    assert _uses_max_completion_tokens("deepseek-v4-flash")
    assert _uses_max_completion_tokens("deepseek-v4-pro")
    assert _uses_max_completion_tokens("gpt-5.4-mini")
    assert not _uses_max_completion_tokens("gpt-4o-mini")
    assert not _uses_max_completion_tokens("qwen-plus")


def test_reasoning_content_fallback_rejects_deliberation():
    """A reasoning scratchpad must not be persisted as a comment.

    DeepSeek v4 sometimes writes the answer into `reasoning_content` and
    sometimes writes only its planning there. v146 persisted 4,099 words of
    "The user wants me to write a Reddit comment. Let me parse the
    instructions..." ending in "This is my final answer" -- which also leaked
    planner-internal register rules into the corpus. Real comments must still
    pass through.
    """
    from generalized_card.backend import _reads_as_deliberation

    assert _reads_as_deliberation(
        "The user wants me to write a Reddit comment. Let me parse the instructions."
    )
    assert _reads_as_deliberation("Okay, so the plan allows DSLR to be named here.")
    assert _reads_as_deliberation("word " * 401)
    assert not _reads_as_deliberation(
        "Weeks on my DSLR. Does the shutter get pressed once the novelty's gone?"
    )
    assert not _reads_as_deliberation(
        "I shot terns at the harbor mouth and the focus box just stayed on the bird."
    )


def test_audit_control_id_leak_whitelists_anchor_tokens():
    """A token in the slot's own anchors is a product name, not a planner id.

    `The Lumix S9` reached a v137ds slot as the anchor `The Lumix S9 (planner)`
    and still failed the evaluation-integrity audit as slot S9, blocking a
    finished run. Anchors are drawn from the seed post and the matched real
    comments and are never control ids, so any id-shaped token appearing in
    them is safe. The writer guard already used this rule; the audit did not.
    """
    from generalized_card.prompts import INTERNAL_CONTROL_ID_RE

    anchors = ["The Lumix S9 (planner)", "a7R (planner)"]
    anchor_ids = {
        m.group(0).upper()
        for a in anchors
        for m in INTERNAL_CONTROL_ID_RE.finditer(str(a))
    }
    assert "S9" in anchor_ids
    # a slot id the anchors do not justify is still a leak
    assert "S20" not in anchor_ids


def test_length_ceiling_is_off_by_default_and_withholds_without_a_profile():
    """The arm reproduces the previous release until it is switched on."""

    from generalized_card import length_fidelity as lf

    lf.set_length_ceiling("off")
    lf.set_active_length_fidelity_profile({"available": True, "tail_cut": 300.0, "tail_count": 143.0})
    task = SimpleNamespace(real_word_count=40)
    assert lf.length_ceiling_problem("word " * 900, task) == ""

    # On, but the domain has no measured tail: withhold rather than gate.
    lf.set_length_ceiling("measured")
    lf.set_active_length_fidelity_profile({"available": True, "cuts": [7.0, 12.0]})
    assert lf.length_ceiling_problem("word " * 900, task) == ""

    # On, but the tail rests on too few reference comments.
    lf.set_active_length_fidelity_profile(
        {"available": True, "tail_cut": 300.0, "tail_count": 12.0}
    )
    assert lf.length_ceiling_problem("word " * 900, task) == ""
    lf.set_length_ceiling("off")


def test_length_ceiling_fires_on_the_tail_regardless_of_the_assigned_band():
    """A tail overshoot sits inside its own assigned band, which is why the band
    check cannot see it and this one must be separate."""

    from generalized_card import length_fidelity as lf

    lf.set_length_ceiling("measured")
    lf.set_active_length_fidelity_profile(
        {
            "available": True,
            "cuts": [7.0, 12.0, 18.0, 24.0, 31.0, 39.0, 52.0, 72.0, 108.0],
            "band_counts": {str(i): 1500.0 for i in range(11)},
            "tail_cut": 300.0,
            "tail_count": 143.0,
        }
    )
    # Assigned 150 and realized 523: both land in the open top band, so the band
    # rule is silent while the ceiling rule fires.
    task = SimpleNamespace(real_word_count=150)
    assert lf.length_band_problem("word " * 523, task) == ""
    problem = lf.length_ceiling_problem("word " * 523, task)
    assert problem.startswith(lf.CEILING_PREFIX)
    assert "523w" in problem and "300w" in problem

    # Under the cut, nothing fires even for a badly missed target.
    assert lf.length_ceiling_problem("word " * 299, task) == ""

    # A slot assigned past the cut is NOT waived -- those slots are the largest
    # contributors to the thread's length spread.
    assert lf.length_ceiling_problem("word " * 523, SimpleNamespace(real_word_count=400))
    lf.set_length_ceiling("off")


def test_length_ceiling_problem_is_soft_and_carries_a_length_only_retry_note():
    """Soft keeps a matched structural slot from ever being dropped."""

    from generalized_card import length_fidelity as lf
    from generalized_card.length_policy import is_soft_length_problem

    problem = f"{lf.CEILING_PREFIX}523w past the 300w ceiling"
    assert is_soft_length_problem(problem)
    note = lf.ceiling_retry_note(problem)
    assert "523w" in note
    # It must name no content, or it teaches the Writer a shared phrasing (G37).
    for word in ("camera", "lens", "product", "topic", "point about"):
        assert word not in note.lower()


def test_length_fidelity_profile_records_the_tail_cut():
    """The profile is one measurement of the domain, not one per arm."""

    from generalized_card import length_fidelity as lf

    # p99 leaves 1% above it by construction, so a supported tail needs a corpus
    # of at least MIN_TAIL_COMMENTS * 100 comments. Camera's excluded corpus has
    # 14,304 and puts 143 above its 300-word cut.
    lengths = [1 + (i % 120) for i in range(5000)] + [400 + i for i in range(100)]
    threads = [{"comments": [{"body": "word " * n} for n in lengths]}]
    profile = lf.build_length_fidelity_profile(threads)
    assert profile["available"] is True
    assert profile["tail_cut"] > 0
    assert profile["tail_count"] >= lf.MIN_TAIL_COMMENTS
    assert lf.active_tail_cut(profile) == profile["tail_cut"]

    # A lumpy distribution whose p99 IS its maximum leaves nothing above the cut.
    # The contract is to withhold the check, not to gate on an unsupported tail.
    lumpy = [{"comments": [{"body": "word " * n} for n in ([5] * 400 + [50] * 400 + [500] * 60)]}]
    thin = lf.build_length_fidelity_profile(lumpy)
    assert thin["tail_count"] < lf.MIN_TAIL_COMMENTS
    assert lf.active_tail_cut(thin) == 0.0


def test_build_seed_pool_can_hold_out_the_evaluation_threads(tmp_path):
    """A calibration pool must share no thread with any evaluation pool."""

    import json

    from generalized_card.data import build_seed_pool
    from generalized_card.domain import load_domain_config

    config = load_domain_config("camera")
    first = build_seed_pool(config, tmp_path / "a.json", count=12, seed=11)
    keys = {
        (row["source_product_dir"], row["source_raw_post_id"])
        for row in first["seed_posts"]
    }
    second = build_seed_pool(config, tmp_path / "b.json", count=12, seed=11, exclude_keys=keys)
    other = {
        (row["source_product_dir"], row["source_raw_post_id"])
        for row in second["seed_posts"]
    }
    assert not (keys & other)
    assert second["meta"]["excluded_threads"] == len(keys)

    # Default None leaves sampling identical, so every existing caller is unchanged.
    repeat = build_seed_pool(config, tmp_path / "c.json", count=12, seed=11)
    assert [row["source_raw_post_id"] for row in repeat["seed_posts"]] == [
        row["source_raw_post_id"] for row in first["seed_posts"]
    ]
    assert repeat["meta"]["excluded_threads"] == 0
    assert json.loads((tmp_path / "c.json").read_text())["seed_posts"]


def test_length_ceiling_redraw_task_names_length_and_not_content():
    """The ceiling re-draw must not be the hard-recovery task.

    Hard recovery opens with "could not be stored", which is false for an
    over-long comment, and echoes the failed candidate -- for a 500-word
    overshoot that would dominate the prompt.
    """

    from dataclasses import dataclass

    from generalized_card import length_fidelity as lf
    from generalized_card.writer_quality import writer_length_ceiling_task

    @dataclass
    class Task:
        planner_intent: str = "Fill matched real sample slot S3: real_words=40."
        must_not_do: str = ""

    problem = f"{lf.CEILING_PREFIX}523w past the 300w ceiling"
    out = writer_length_ceiling_task(Task(), problem=problem)
    assert "523w" in out.planner_intent and "300w" in out.planner_intent
    assert "could not be stored" not in out.planner_intent
    assert out.planner_intent.startswith("Fill matched real sample slot S3")
    for word in ("camera", "lens", "product", "topic"):
        assert word not in out.planner_intent.lower()
    assert writer_length_ceiling_task(None, problem=problem) is None


def test_length_ceiling_problems_selects_only_ceiling_misses():
    """The re-draw loop must not fire on a band miss or any other soft problem."""

    from generalized_card import length_fidelity as lf

    problems = [
        "template_phrase_reused",
        f"{lf.PROBLEM_PREFIX}12w in band 1, assigned 40w in band 6 [32-39]",
        f"{lf.CEILING_PREFIX}523w past the 300w ceiling",
    ]
    picked = lf.length_ceiling_problems(problems)
    assert picked == [f"{lf.CEILING_PREFIX}523w past the 300w ceiling"]
    assert lf.length_ceiling_problems([]) == []
    assert lf.length_ceiling_problems(None) == []
    assert lf.length_ceiling_problems(["template_phrase_reused"]) == []


def test_length_ceiling_is_soft_so_an_exhausted_redraw_still_stores_the_slot():
    """ORIENTATION s4: a matched structural slot must never be dropped."""

    from generalized_card import length_fidelity as lf
    from generalized_card.writer_quality import hard_realization_problems

    problem = f"{lf.CEILING_PREFIX}523w past the 300w ceiling"
    # Not hard -> the recovery-exhausted path stores the text instead of skipping.
    assert hard_realization_problems([problem]) == []


class WriterPlanFieldsTest(unittest.TestCase):
    """v157: withholding the Planner's own prose from the Writer.

    The arm exists because three of the six meaning-bearing things the Writer
    sees are finished sentences the Planner wrote, and one thread's sentences
    resemble each other. Measured on v156's 364 stored Writer prompts, any
    subset containing `semantic_move` or `domain_intent` prices at plan cosine
    ~0.31 while `content_angle` + `detail_focus` reaches 0.2173 against the
    0.2310 that the run's own realization function says real needs.
    """

    def test_full_is_the_default_and_hides_nothing(self) -> None:
        from generalized_card import writer_plan_fields as wpf

        self.assertEqual(wpf.WRITER_PLAN_FIELDS_MODE, "full")
        self.assertFalse(wpf.active())
        for field in ("semantic_move", "decision_boundary", "domain_intent",
                      "content_angle", "detail_focus", "stance"):
            self.assertFalse(wpf.hidden(field), field)

    def test_angle_detail_hides_exactly_the_three_prose_fields(self) -> None:
        from generalized_card import writer_plan_fields as wpf

        try:
            wpf.set_writer_plan_fields("angle_detail")
            self.assertTrue(wpf.active())
            for field in ("semantic_move", "decision_boundary", "domain_intent"):
                self.assertTrue(wpf.hidden(field), field)
            # The two dispersing fields, and every non-semantic control, stay:
            # this arm must not become a general "tell the Writer less" change.
            for field in ("content_angle", "detail_focus", "stance", "voice",
                          "evidence_mode", "speaker_role", "reply_relation",
                          "avoid_repeating", "opening_style", "reply_delta"):
                self.assertFalse(wpf.hidden(field), field)
        finally:
            wpf.set_writer_plan_fields("full")

    def test_substitute_names_the_detail_and_hands_back_the_point(self) -> None:
        from generalized_card import writer_plan_fields as wpf

        rows = wpf.substitute_route_lock("shirt as social statement")
        self.assertIn("shirt as social statement", rows[0])
        # The block has a header in three Writer templates, so it cannot be empty.
        self.assertTrue(all(row.startswith("- ") for row in rows))
        self.assertTrue(any("Decide for yourself" in row for row in rows))
        # An empty detail must still leave an instruction rather than a bare header.
        self.assertTrue(wpf.substitute_route_lock(""))

    def test_unknown_mode_fails_at_configuration_time(self) -> None:
        from generalized_card import writer_plan_fields as wpf

        with self.assertRaises(ValueError):
            wpf.set_writer_plan_fields("angle-detail")
        self.assertEqual(wpf.WRITER_PLAN_FIELDS_MODE, "full")

    def test_semantic_contract_path_honours_the_arm(self) -> None:
        """The second Writer path must not leak a field the first one hides."""

        from generalized_card import writer_plan_fields as wpf
        from generalized_card.semantic_realization import semantic_contract_values

        class _Task:
            semantic_move = "Reject the shirt as a loud virtue signal."
            detail_focus = "shirt as social statement"
            domain_intent = "Push back on the gesture itself."
            decision_boundary = ""
            stance = "disagree"
            reply_relation = ""
            evidence_mode = "none_assertion"
            owned_decision_subject = ""
            forbidden_decision_subjects = ""
            reply_delta = ""
            reply_delta_type = ""
            reply_novelty_anchor = ""
            parent_semantic_move = ""
            parent_decision_boundary = ""
            branch_exclusion = ""
            development_plan = ""
            opening_style = ""
            avoid_repeating = ""
            comment_function = "verdict_evaluation"
            payload_type = "rant"

        labels = lambda: {label for label, _ in semantic_contract_values(_Task())}
        self.assertIn("required contribution", labels())
        self.assertIn("decision intent", labels())
        try:
            wpf.set_writer_plan_fields("angle_detail")
            after = labels()
            self.assertNotIn("required contribution", after)
            self.assertNotIn("decision intent", after)
            self.assertIn("local detail", after)
            self.assertIn("stance", after)
        finally:
            wpf.set_writer_plan_fields("full")
