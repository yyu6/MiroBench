from __future__ import annotations

import sys
import unittest
import random
import numpy as np
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
EVALUATION_DIR = REPO_ROOT / "scripts" / "evaluation"
if str(EVALUATION_DIR) not in sys.path:
    sys.path.insert(0, str(EVALUATION_DIR))

from score_thread_self_bleu import (  # noqa: E402
    symmetric_pair_bleu as evaluator_symmetric_pair_bleu,
    tokenize as evaluator_tokenize,
)

from generalized_card.generation_distribution import (  # noqa: E402
    _tone_instruction,
    allocate_story_and_affect,
    enrich_distribution_plan_fields,
    render_planner_distribution_target,
    select_thread_template,
    set_social_contract_coherence,
)
from generalized_card.planner_distribution import (  # noqa: E402
    apply_slot_distribution_schedule,
    build_slot_distribution_schedule,
    render_slot_distribution_schedule,
)
from generalized_card.planning_quality import (  # noqa: E402
    evaluate_plan_batch,
    social_contract_problem,
    surface_capacity_problem,
)
from generalized_card.branch_routing import (  # noqa: E402
    required_branch_count,
    render_branch_requirements,
    root_branch_schedule,
)
from generalized_card.generation_diversity import (  # noqa: E402
    build_thread_distribution_target,
    semantic_distribution_problem,
    semantic_thread_diagnostics,
)
from generalized_card.lexical_quality import (  # noqa: E402
    build_lexical_calibration,
    lexical_overlap_problem,
    symmetric_pair_bleu,
    thread_mean_pair_bleu,
    tokenize,
)
from generalized_card.semantic_realization import (  # noqa: E402
    opening_route_counts,
    repeated_phrase_counts,
    semantic_contract_values,
    semantic_coverage_entries,
    used_sentence_routes,
)
from generalized_card.task_distribution import restore_planner_task_contract  # noqa: E402


@dataclass(frozen=True)
class Task:
    local_task_id: int
    local_parent_task_id: int | None = None
    length_bucket: str = "medium"
    story_mode: str = "no_story"
    story_instruction: str = ""
    affect_role: str = ""
    affect_instruction: str = ""
    distribution_assignment: str = ""
    payload_type: str = "soft_helpful"
    comment_function: str = "explanation_analysis"
    evidence_mode: str = "small_observation"
    speaker_role: str = "side_observer"
    utterance_mode: str = "small_observation"
    tone_shape: str = "neutral_fact"
    tone_target: str = "neutral_local"
    tone_target_instruction: str = ""
    voice: str = "casual_neutral"
    stance: str = "neutral"
    real_surface_shape: str = ""
    allow_first_person_frame: bool = False
    planner_intent: str = "preserve one local point"
    must_not_do: str = "Do not add a second claim."
    decision_boundary: str = ""


class StoryAffectDistributionTest(unittest.TestCase):
    def test_planner_contract_wins_over_legacy_surface_social_inference(self) -> None:
        @dataclass(frozen=True)
        class ContractTask:
            local_task_id: int = 1
            real_word_count: int = 58
            length_bucket: str = "short"
            story_mode: str = "no_story"
            payload_type: str = "low_info_reaction"
            comment_function: str = "reaction"
            content_angle: str = "fit_use_case"
            evidence_mode: str = "none_assertion"
            speaker_role: str = "gratitude_reply"
            voice: str = "grateful"
            reply_relation: str = "answers_parent"
            stance: str = "agree"
            detail_focus: str = "local detail"
            claim_key: str = "old_claim"
            claim_family: str = "direct_answer"
            perspective_id: str = "P12"
            domain_intent: str = "acknowledge"
            decision_boundary: str = "old boundary"
            opening_style: str = "thanks"
            context_aperture: str = "parent_only"
            utterance_mode: str = "op_followup"
            allow_first_person_frame: bool = False
            allow_uncertainty_frame: bool = False

        class Core:
            @staticmethod
            def length_bucket_for_payload(*, payload_type, word_count):
                return f"{payload_type}:{word_count}"

            @staticmethod
            def infer_utterance_mode(**kwargs):
                return f"{kwargs['speaker_role']}:{kwargs['payload_type']}"

        inherited = ContractTask()
        restored = restore_planner_task_contract(
            inherited,
            {
                "payload_type": "personal_story",
                "comment_function": "personal_datapoint",
                "evidence_mode": "firsthand_experience",
                "story_mode": "specific_personal_story",
                "speaker_role": "datapoint_only",
                "voice": "casual_neutral",
                "perspective_id": "P10",
                "claim_key": "one_local_experience",
                "decision_boundary": "observed outcome rather than purchase advice",
            },
            core=Core(),
        )
        self.assertEqual(restored.payload_type, "personal_story")
        self.assertEqual(restored.story_mode, "specific_personal_story")
        self.assertEqual(restored.speaker_role, "datapoint_only")
        self.assertEqual(restored.length_bucket, "personal_story:58")
        self.assertEqual(restored.utterance_mode, "datapoint_only:personal_story")
        self.assertTrue(restored.allow_first_person_frame)
    def test_root_branch_schedule_keeps_replies_with_their_root(self) -> None:
        rows = [
            {"id": "a", "parent_id": "t3_post"},
            {"id": "b", "parent_id": "t1_a"},
            {"id": "c", "parent_id": "t3_post"},
            {"id": "d", "parent_id": "t1_b"},
        ]
        self.assertEqual(required_branch_count(rows), 3)
        self.assertEqual(
            root_branch_schedule(rows, branch_ids=[1, 2, 3]),
            {1: 1, 2: 1, 3: 2, 4: 1},
        )

    def test_root_branch_count_and_schedule_cover_every_independent_root(self) -> None:
        rows = [
            {"id": f"root-{index}", "parent_id": "t3_post"}
            for index in range(1, 21)
        ]
        branch_ids = list(range(1, 21))
        self.assertEqual(required_branch_count(rows), 9)
        self.assertEqual(
            root_branch_schedule(rows, branch_ids=branch_ids),
            {index: index for index in range(1, 21)},
        )

    def test_branch_requirements_make_sibling_group_visible_to_planner(self) -> None:
        rendered = render_branch_requirements(
            {1: 1, 2: 1, 3: 1, 4: 2},
            sample_ids=[1, 2, 3, 4],
            parent_slots={2: 1, 3: 1},
        )
        self.assertIn("sibling_group=S2,S3", rendered)
        self.assertIn("sibling_turn=1/2", rendered)
        self.assertIn("sibling_turn=2/2", rendered)

    def test_allocator_audits_without_rewriting_completed_plans(self) -> None:
        tasks = [Task(local_task_id=index) for index in range(1, 11)]
        calibration = {
            "available": True,
            "reference_thread_count": 1,
            "templates_by_size": {
                "small": [
                    {
                        "comment_count": 10,
                        "story_count": 1,
                        "story_rate": 0.1,
                        "dominant_emotion_counts": {
                            "neutral": 7,
                            "approval": 2,
                            "curiosity": 1,
                        },
                    }
                ]
            },
        }
        revised, report = allocate_story_and_affect(
            tasks,
            personal_min_share=0.30,
            calibration=calibration,
            rng=random.Random(42),
        )
        self.assertEqual(revised, tasks)
        self.assertEqual(report["target_story_slots"], 1)
        self.assertEqual(report["story_slots_after"], 0)
        self.assertFalse(report["story_target_met"])
        self.assertEqual(report["affect_target_counts"]["neutral"], 7)
        self.assertEqual(
            report["policy"],
            "audit_planner_template_contract_without_post_planner_reassignment",
        )
        self.assertTrue(report["reference_template_used"])

    def test_allocator_preserves_existing_planner_story_labels(self) -> None:
        tasks = [Task(local_task_id=index) for index in range(1, 11)]
        tasks[0] = Task(
            local_task_id=1,
            story_mode="specific_personal_story",
            story_instruction="Planner supplied story",
            evidence_mode="firsthand_experience",
            payload_type="personal_story",
            comment_function="personal_datapoint",
        )
        revised, report = allocate_story_and_affect(tasks, personal_min_share=0.30)
        self.assertEqual(report["target_story_slots"], 3)
        self.assertEqual(report["story_slots_before"], 1)
        self.assertEqual(report["story_slots_after"], 1)
        self.assertFalse(report["story_target_met"])
        self.assertEqual(revised[0].story_instruction, "Planner supplied story")
        self.assertTrue(all(not task.affect_role for task in revised))

    def test_slot_schedule_assigns_exact_template_labels_before_planning(self) -> None:
        comments = [
            {"body": "word " * (12 + index * 4), "depth": index % 3}
            for index in range(10)
        ]
        template = {
            "comment_count": 10,
            "story_count": 2,
            "story_rate": 0.2,
            "polite_rate": 0.2,
            "impolite_rate": 0.4,
            "neutral_rate": 0.1,
            "dominant_emotion_counts": {"neutral": 7, "approval": 2, "curiosity": 1},
        }
        schedule = build_slot_distribution_schedule(
            template=template,
            comments=comments,
            total_comments=10,
        )
        plans = {
            index: {"story_mode": "specific_personal_story", "tone_class": "", "affect_role": ""}
            for index in range(1, 11)
        }
        apply_slot_distribution_schedule(plans, schedule)
        tone_counts: dict[str, int] = {}
        for plan in plans.values():
            tone = plan["tone_class"]
            tone_counts[tone] = tone_counts.get(tone, 0) + 1
        # The template's three reported rates sum to 0.7. The remaining 0.3 is
        # the classifier's fourth class, so it is planned as somewhat_polite
        # rather than renormalized onto the reported three and inflating them.
        self.assertEqual(
            schedule["targets"]["tone_counts"],
            {"polite": 2, "somewhat_polite": 3, "impolite": 4, "neutral": 1},
        )
        self.assertEqual(tone_counts, schedule["targets"]["tone_counts"])
        self.assertEqual(
            sum(plan["story_mode"] != "no_story" for plan in plans.values()),
            schedule["targets"]["story_slots"],
        )
        self.assertEqual(
            sum(plan["story_mode"] == "no_story" for plan in plans.values()),
            10 - schedule["targets"]["story_slots"],
        )

    def test_measured_fourth_class_rate_is_used_when_present(self) -> None:
        comments = [
            {"body": "word " * (12 + index * 4), "depth": index % 3}
            for index in range(10)
        ]
        schedule = build_slot_distribution_schedule(
            template={
                "comment_count": 10,
                "story_count": 0,
                "polite_rate": 0.3,
                "impolite_rate": 0.4,
                "neutral_rate": 0.2,
                "somewhat_polite_rate": 0.1,
            },
            comments=comments,
            total_comments=10,
        )
        self.assertEqual(
            schedule["targets"]["tone_counts"],
            {"polite": 3, "somewhat_polite": 1, "impolite": 4, "neutral": 2},
        )

    def test_polite_is_scheduled_onto_the_longer_slots(self) -> None:
        # The evaluation classifier's polite class concentrates in long real
        # comments, so the schedule must not route it onto the shortest slots.
        comments = [
            {"body": "word " * words, "depth": 1}
            for words in (4, 8, 15, 30, 60, 120)
        ]
        schedule = build_slot_distribution_schedule(
            template={
                "comment_count": 6,
                "story_count": 0,
                "polite_rate": 0.34,
                "impolite_rate": 0.33,
                "neutral_rate": 0.33,
                "somewhat_polite_rate": 0.0,
            },
            comments=comments,
            total_comments=6,
        )
        assignments = schedule["assignments"]
        polite_slots = [
            int(sample_id)
            for sample_id, value in assignments.items()
            if value.get("tone_class") == "polite"
        ]
        neutral_slots = [
            int(sample_id)
            for sample_id, value in assignments.items()
            if value.get("tone_class") == "neutral"
        ]
        self.assertTrue(polite_slots)
        self.assertTrue(neutral_slots)
        # Slot ids follow the comment order, so a larger id is a longer slot.
        self.assertGreater(min(polite_slots), max(neutral_slots))

    def test_micro_slot_rejects_an_impossible_story_contract(self) -> None:
        problem = surface_capacity_problem(
            {
                "_slot_word_count": "2",
                "_slot_surface_label": "micro",
                "payload_type": "personal_story",
                "comment_function": "personal_datapoint",
                "story_mode": "specific_personal_story",
                "evidence_mode": "firsthand_experience",
            }
        )
        self.assertIn("cannot be realized in a micro reaction", problem)

    def test_slot_schedule_never_forces_an_incompatible_affect(self) -> None:
        template = {
            "comment_count": 4,
            "story_count": 0,
            "polite_rate": 0.0,
            "impolite_rate": 0.0,
            "neutral_rate": 1.0,
            "dominant_emotion_counts": {"gratitude": 4},
        }
        # Only S2 is structurally compatible with gratitude. The remaining
        # quota must remain a Planner-level target, not be forced into advice.
        schedule = build_slot_distribution_schedule(
            template=template,
            comments=[
                {"body": "word " * 50, "depth": 0},
                {"body": "thanks that is helpful", "depth": 1},
                {"body": "word " * 80, "depth": 0},
                {"body": "word " * 70, "depth": 0},
            ],
            total_comments=4,
        )
        assignments = schedule["assignments"]
        assigned = [
            sample_id
            for sample_id, values in assignments.items()
            if values.get("affect_role") == "gratitude"
        ]
        self.assertEqual(assigned, ["2"])
        self.assertEqual(schedule["unassigned_affect_labels"], ["gratitude"] * 3)
        rendered = render_slot_distribution_schedule(schedule, sample_ids=[1, 2, 3, 4])
        self.assertIn("unavailable template labels", rendered)
        self.assertIn("affect=gratitude", rendered)

    def test_gratitude_metadata_must_describe_a_social_reaction(self) -> None:
        self.assertIn(
            "requires a social reaction contract",
            social_contract_problem(
                {
                    "affect_role": "gratitude",
                    "speaker_role": "advisor",
                    "comment_function": "recommendation_advice",
                    "payload_type": "advice",
                }
            ),
        )
        self.assertEqual(
            "",
            social_contract_problem(
                {
                    "affect_role": "gratitude",
                    "speaker_role": "gratitude_reply",
                    "comment_function": "reaction",
                    "payload_type": "low_info_reaction",
                }
            ),
        )

    def test_forced_no_story_rejects_a_personal_story_payload(self) -> None:
        self.assertIn(
            "story_mode=no_story",
            social_contract_problem(
                {
                    "story_mode": "no_story",
                    "payload_type": "personal_story",
                    "evidence_mode": "firsthand_experience",
                    "tone_class": "neutral",
                }
            ),
        )

    def test_forced_no_story_rejects_firsthand_evidence_without_story_payload(self) -> None:
        problem = social_contract_problem(
            {
                "story_mode": "no_story",
                "payload_type": "fragment_datapoint",
                "comment_function": "personal_datapoint",
                "evidence_mode": "firsthand_experience",
            }
        )
        self.assertIn("evidence_mode=firsthand_experience", problem)

    def test_story_slot_requires_narrative_evidence_contract(self) -> None:
        problem = social_contract_problem(
            {
                "story_mode": "specific_personal_story",
                "payload_type": "advice",
                "comment_function": "recommendation_advice",
                "evidence_mode": "technical_or_policy_reasoning",
            }
        )
        self.assertIn("needs a coherent narrative-evidence plan", problem)
        self.assertEqual(
            "",
            social_contract_problem(
                {
                    "story_mode": "specific_personal_story",
                    "payload_type": "personal_story",
                    "comment_function": "personal_datapoint",
                    "evidence_mode": "firsthand_experience",
                }
            ),
        )

    def test_polite_label_requires_a_coherent_social_move(self) -> None:
        self.assertIn(
            "tone_class=polite",
            social_contract_problem(
                {
                    "story_mode": "no_story",
                    "payload_type": "soft_helpful",
                    "tone_class": "polite",
                    "stance": "agree",
                    "speaker_role": "advisor",
                    "comment_function": "explanation_analysis",
                }
            ),
        )
        self.assertEqual(
            "",
            social_contract_problem(
                {
                    "story_mode": "no_story",
                    "payload_type": "fragment_datapoint",
                    "tone_class": "polite",
                    "stance": "agree",
                    "speaker_role": "datapoint_only",
                    "comment_function": "personal_datapoint",
                }
            ),
        )

    def test_social_contract_ablation_restores_pre_v80_behavior(self) -> None:
        contradictory = {
            1: {
                "sample_id": 1,
                "story_mode": "no_story",
                "payload_type": "personal_story",
                "tone_class": "neutral",
                "comment_function": "personal_datapoint",
                "semantic_move": "state one narrow firsthand observation",
                "local_topic": "the visible condition",
                "detail_focus": "one result",
            }
        }
        report = evaluate_plan_batch(
            contradictory,
            enforce_social_contract=False,
            max_perspective_share=1.0,
        )
        self.assertNotIn(
            "social_contract_conflict",
            {issue.code for issue in report.issues},
        )
        legacy_affect = evaluate_plan_batch(
            {
                1: {
                    **contradictory[1],
                    "payload_type": "advice",
                    "story_mode": "no_story",
                    "affect_role": "gratitude",
                    "speaker_role": "advisor",
                    "comment_function": "recommendation_advice",
                }
            },
            enforce_social_contract=False,
            max_perspective_share=1.0,
        )
        self.assertIn(
            "social_contract_conflict",
            {issue.code for issue in legacy_affect.issues},
        )

        try:
            set_social_contract_coherence("off")
            legacy = _tone_instruction("polite")
        finally:
            set_social_contract_coherence("on")
        self.assertIn("Do not hedge", legacy)
        self.assertIn("A warm turn needs room", legacy)

    def test_template_selection_and_planner_brief_are_deterministic(self) -> None:
        calibration = {
            "available": True,
            "templates_by_size": {
                "small": [
                    {"comment_count": 10, "story_count": 1, "neutral_rate": 1.0},
                    {"comment_count": 10, "story_count": 3, "neutral_rate": 1.0},
                ]
            },
        }
        first = select_thread_template(calibration, comment_count=10, seed_key="seed-a")
        second = select_thread_template(calibration, comment_count=10, seed_key="seed-a")
        self.assertEqual(first, second)
        brief = render_planner_distribution_target(
            first,
            total_comments=10,
            prior_plans=[
                {
                    "story_mode": "no_story",
                    "tone_class": "neutral",
                    "affect_role": "neutral",
                }
            ],
        )
        self.assertIn("slots remaining before this batch: 9", brief)
        self.assertIn("tone_class exact counts", brief)

    def test_distribution_fields_survive_shared_plan_normalization(self) -> None:
        normalized = {1: {"story_mode": "no_story"}}
        enriched = enrich_distribution_plan_fields(
            {
                "comment_plans": [
                    {
                        "sample_id": 1,
                        "tone_class": "polite",
                        "affect_role": "curiosity",
                    }
                ]
            },
            normalized,
        )
        self.assertEqual(enriched[1]["tone_class"], "polite")
        self.assertEqual(enriched[1]["affect_role"], "curiosity")


class LexicalQualityTest(unittest.TestCase):
    def test_decision_boundary_survives_writer_semantic_memory(self) -> None:
        task = Task(
            local_task_id=1,
            decision_boundary="whether the one-off use justifies buying instead of renting",
        )
        self.assertIn(
            ("decision boundary", task.decision_boundary),
            semantic_contract_values(task),
        )
        coverage = semantic_coverage_entries(
            [{"semantic_move": "reduce commitment", "decision_boundary": task.decision_boundary}]
        )
        self.assertIn("boundary=whether the one-off use", coverage[0])

    def test_dynamic_thread_ledger_reports_repeated_openings_and_phrases(self) -> None:
        comments = [
            {"content": "That's the part I would check before buying."},
            {"content": "That's the part I would compare in person."},
            {"content": "For me that's the part I would check again."},
        ]
        self.assertEqual(opening_route_counts(comments)["that's the part"], 2)
        self.assertEqual(
            repeated_phrase_counts(comments)["that's the part i"],
            3,
        )
        routes = used_sentence_routes(comments)
        # Reused routes rank first and carry their count, so an entrenched route
        # cannot be pushed out of the ledger by more recent one-off routes.
        self.assertEqual(routes[0], "that's the part i (used 2x)")
        self.assertIn("for me that's the", routes)

    def test_used_routes_rank_by_reuse_not_recency(self) -> None:
        entrenched = {"content": "That is the recurring route here."}
        comments = [entrenched, entrenched] + [
            {"content": f"Distinct opening number {index} follows here."}
            for index in range(30)
        ]
        routes = used_sentence_routes(comments, limit=5)
        self.assertTrue(
            any(route.startswith("that is the recurring") for route in routes),
            routes,
        )
        self.assertIn("(used 2x)", routes[0])

    def test_dynamic_route_ledger_includes_comma_led_clauses(self) -> None:
        routes = used_sentence_routes(
            [
                {
                    "content": (
                        "Around an ice show, the real question is whether reach "
                        "or low light matters more."
                    )
                }
            ]
        )
        self.assertIn("the real question is", routes)

    def test_dynamic_route_ledger_can_cover_an_entire_large_thread(self) -> None:
        comments = [
            {"content": f"Route number {index} starts differently here."}
            for index in range(60)
        ]
        routes = used_sentence_routes(comments, limit=120)
        self.assertEqual(len(routes), 60)
        self.assertIn("route number 0 starts", routes)
        self.assertIn("route number 59 starts", routes)

    def test_bleu_implementation_matches_evaluator(self) -> None:
        left = "I kept the smaller lens because it fits the bag."
        right = "I kept a small prime in the bag for the same reason."
        self.assertEqual(tokenize(left), evaluator_tokenize(left))
        self.assertAlmostEqual(
            symmetric_pair_bleu(tokenize(left), tokenize(right), 4),
            evaluator_symmetric_pair_bleu(
                evaluator_tokenize(left),
                evaluator_tokenize(right),
                4,
            ),
            places=12,
        )

    def test_calibrated_guard_covers_short_and_long_comments(self) -> None:
        calibration = {
            "thresholds": {
                "micro": 0.40,
                "short": 0.35,
                "medium": 0.30,
                "long": 0.25,
                "very_long": 0.25,
            }
        }
        short_problem = lexical_overlap_problem(
            text="same tiny reply",
            previous_texts=["same tiny reply"],
            calibration=calibration,
        )
        long_problem = lexical_overlap_problem(
            text="The smaller body fits my bag, but the lens choice still matters more for low light.",
            previous_texts=[
                "The smaller body fits my bag, but the lens choice still matters more for low light."
            ],
            calibration=calibration,
        )
        distinct = lexical_overlap_problem(
            text="That crop looks like sharpening to me.",
            previous_texts=["Battery life was the deciding factor."],
            calibration=calibration,
        )
        self.assertTrue(short_problem.startswith("lexical_overlap_high:"))
        self.assertTrue(long_problem.startswith("lexical_overlap_high:"))
        self.assertEqual(distinct, "")

    def test_prefix_mean_guard_matches_final_evaluator_objective(self) -> None:
        real_threads = [
            {
                "comments": [
                    {"body": "One narrow observation about focus."},
                    {"body": "Battery handling was the deciding detail."},
                    {"body": "The smaller body fit a travel bag."},
                ]
            }
            for _ in range(25)
        ]
        calibration = build_lexical_calibration(real_threads, quantile=0.90)
        repeated = lexical_overlap_problem(
            text="One narrow observation about focus.",
            previous_texts=[
                "One narrow observation about focus.",
                "One narrow observation about focus.",
            ],
            calibration=calibration,
        )
        improving = lexical_overlap_problem(
            text="A used prime made more sense for that budget.",
            previous_texts=[
                "One narrow observation about focus.",
                "One narrow observation about focus.",
            ],
            calibration=calibration,
        )
        self.assertTrue(repeated.startswith("lexical_overlap_high:"))
        self.assertEqual(improving, "")
        self.assertAlmostEqual(
            thread_mean_pair_bleu(
                [
                    "One narrow observation about focus.",
                    "Battery handling was the deciding detail.",
                ]
            ),
            evaluator_symmetric_pair_bleu(
                evaluator_tokenize("One narrow observation about focus."),
                evaluator_tokenize("Battery handling was the deciding detail."),
                4,
            ),
        )

    def test_selected_real_template_guides_lexical_target_and_feedback(self) -> None:
        calibration = {
            "prefix_mean_upper": {"tiny": 0.20, "small": 0.20},
            "prefix_mean_lower": {"tiny": 0.01, "small": 0.01},
            "prefix_mean_median": {"tiny": 0.04, "small": 0.04},
        }
        target = {
            "comment_count": 3,
            "self_bleu_4": 0.03,
            "metric_bands": {
                "self_bleu_4": {"q10": 0.02, "q90": 0.05}
            },
        }
        problem = lexical_overlap_problem(
            text="The smaller body fits my bag and the smaller lens fits my bag.",
            previous_texts=[
                "The smaller body fits my bag and the smaller lens fits my bag.",
                "The smaller body fits my bag and the smaller lens fits my bag.",
            ],
            calibration=calibration,
            thread_target=target,
        )
        self.assertTrue(problem.startswith("lexical_overlap_high:"))
        self.assertIn("shared=", problem)
        self.assertIn("nearest=", problem)


class JointDistributionControlTest(unittest.TestCase):
    class FakeIndex:
        def __init__(self, vectors: dict[str, list[float]]) -> None:
            self.vectors = vectors

        def encode_texts(self, texts):
            return [np.asarray(self.vectors[text], dtype=float) for text in texts]

    def test_target_uses_only_heldout_aggregate_metrics(self) -> None:
        target = build_thread_distribution_target(
            {
                "comment_count": 10,
                "self_bleu_4": 0.03,
                "semantic_mean_cosine": 0.31,
            },
            {
                "metric_bands_by_size": {
                    "small": {
                        "self_bleu_4": {"q10": 0.02, "q90": 0.05},
                        "semantic_mean_cosine": {"q10": 0.25, "q90": 0.36},
                    }
                }
            },
        )
        self.assertEqual(target["semantic_mean_cosine"], 0.31)
        self.assertFalse(target["raw_text_included"])
        self.assertNotIn("text", target)

    def test_semantic_guard_matches_pairwise_cosine_and_reports_nearest(self) -> None:
        index = self.FakeIndex(
            {
                "prior one": [1.0, 0.0],
                "prior two": [0.8, 0.6],
                "paraphrase": [1.0, 0.0],
            }
        )
        diagnostics = semantic_thread_diagnostics(
            text="paraphrase",
            previous_texts=["prior one", "prior two"],
            thread_target={
                "comment_count": 3,
                "semantic_mean_cosine": 0.30,
                "metric_bands": {
                    "semantic_mean_cosine": {"q10": 0.25, "q90": 0.35}
                },
            },
            semantic_index=index,
        )
        self.assertTrue(diagnostics["available"])
        self.assertGreater(diagnostics["proposed_mean"], 0.8)
        self.assertEqual(diagnostics["nearest"][0]["text"], "prior one")
        self.assertTrue(
            semantic_distribution_problem(diagnostics).startswith(
                "semantic_overlap_high:"
            )
        )

    def test_semantic_guard_detects_candidate_below_real_coherence_band(self) -> None:
        index = self.FakeIndex(
            {
                "prior one": [1.0, 0.0],
                "prior two": [0.8, 0.6],
                "unrelated": [-1.0, 0.0],
            }
        )
        diagnostics = semantic_thread_diagnostics(
            text="unrelated",
            previous_texts=["prior one", "prior two"],
            thread_target={
                "comment_count": 3,
                "semantic_mean_cosine": 0.30,
                "metric_bands": {
                    "semantic_mean_cosine": {"q10": 0.25, "q90": 0.35}
                },
            },
            semantic_index=index,
        )
        self.assertLess(diagnostics["proposed_mean"], diagnostics["lower"])
        self.assertTrue(
            semantic_distribution_problem(diagnostics).startswith(
                "semantic_overlap_low:"
            )
        )


if __name__ == "__main__":
    unittest.main()


class AffectInstructionRegisterTest(unittest.TestCase):
    """Affect instructions must ask for the emotion, not suppress its markers.

    Measured over v72: every non-neutral affect realized at 0-23% while neutral
    realized at 48-80%, and that held for perfectly tone-compatible pairs
    (impolite+annoyance 0/12, polite+admiration 0/30), so it was never a tone
    conflict. The instruction set was: express this emotion "but not with hype",
    "without exclamation marks", "not broad praise". GoEmotions keys on exactly
    those markers, so the set described `neutral`.
    """

    SUPPRESSORS = (
        "do not add broad praise",
        "avoid hype",
        "without hype",
        "no exclamation",
        "extra exclamation",
        "exclamation stacking",
        "realize it once",
    )

    def test_no_instruction_suppresses_the_markers_the_metric_reads(self) -> None:
        from generalized_card.generation_distribution import AFFECT_INSTRUCTIONS

        for role, text in AFFECT_INSTRUCTIONS.items():
            lowered = text.lower()
            for phrase in self.SUPPRESSORS:
                self.assertNotIn(phrase, lowered, f"{role}: {phrase}")

    def test_grounding_constraints_are_kept(self) -> None:
        """Negative clauses that protect factual grounding must survive."""

        from generalized_card.generation_distribution import AFFECT_INSTRUCTIONS

        self.assertIn("without inventing a loss", AFFECT_INSTRUCTIONS["sadness"].lower())
        self.assertIn("without inventing a purchase", AFFECT_INSTRUCTIONS["desire"].lower())
        self.assertIn("without promising an outcome", AFFECT_INSTRUCTIONS["optimism"].lower())

    def test_amusement_allows_an_unforced_laughter_surface(self) -> None:
        from generalized_card.generation_distribution import AFFECT_INSTRUCTIONS

        self.assertIn(
            "natural laughter token is allowed",
            AFFECT_INSTRUCTIONS["amusement"],
        )
