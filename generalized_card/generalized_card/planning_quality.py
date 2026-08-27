from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable, Iterable

from .long_form_planning import development_plan_problem
from .reply_planning import (
    CRITICAL_REPLY_DELTA_TYPES,
    SOCIAL_REPLY_DELTA_TYPES,
    SUPPORTIVE_REPLY_DELTA_TYPES,
)


# Every increment that must name a concrete new object. Kept as a derived set so
# adding a delta type in one place cannot leave this validator rejecting it.
SUBSTANTIVE_REPLY_DELTA_TYPES = frozenset(
    CRITICAL_REPLY_DELTA_TYPES + SUPPORTIVE_REPLY_DELTA_TYPES
)


# These are domain-neutral decision lenses. Topic facets such as a product,
# feature, team, event, or policy remain local topics rather than perspectives.
UNIVERSAL_VIEWPOINTS: tuple[dict[str, Any], ...] = (
    {
        "perspective_id": "P01",
        "label": "needs and constraints",
        "axis": "fit",
        "decision_question": "Which stated need or constraint changes the answer?",
    },
    {
        "perspective_id": "P02",
        "label": "tradeoff and priority",
        "axis": "tradeoff",
        "decision_question": "Which benefit is gained and what is given up?",
    },
    {
        "perspective_id": "P03",
        "label": "cost and value",
        "axis": "value",
        "decision_question": "Is the extra cost, effort, or commitment justified?",
    },
    {
        "perspective_id": "P04",
        "label": "quality and performance",
        "axis": "performance",
        "decision_question": "Which observable quality or performance difference matters?",
    },
    {
        "perspective_id": "P05",
        "label": "reliability and risk",
        "axis": "risk",
        "decision_question": "What can fail, vary, or create downstream risk?",
    },
    {
        "perspective_id": "P06",
        "label": "workflow and usability",
        "axis": "workflow",
        "decision_question": "How does the choice affect ordinary use or workflow?",
    },
    {
        "perspective_id": "P07",
        "label": "compatibility and ecosystem",
        "axis": "compatibility",
        "decision_question": "What must work with an existing setup, habit, or community?",
    },
    {
        "perspective_id": "P08",
        "label": "cause and troubleshooting",
        "axis": "diagnosis",
        "decision_question": "What observation would separate likely causes or fixes?",
    },
    {
        "perspective_id": "P09",
        "label": "timing and availability",
        "axis": "timing",
        "decision_question": "Does timing, access, or waiting change the decision?",
    },
    {
        "perspective_id": "P10",
        "label": "firsthand outcome",
        "axis": "experience",
        "decision_question": "What narrow firsthand outcome or counterexample is relevant?",
    },
    {
        "perspective_id": "P11",
        "label": "evidence and uncertainty",
        "axis": "evidence",
        "decision_question": "How strong is the evidence and what remains uncertain?",
    },
    {
        "perspective_id": "P12",
        "label": "social and community context",
        "axis": "social",
        "decision_question": "What social norm, audience, or conversational role matters?",
    },
)

TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?", re.I)
STOPWORDS = {
    "a",
    "about",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "because",
    "been",
    "but",
    "by",
    "can",
    "comment",
    "concrete",
    "detail",
    "do",
    "does",
    "for",
    "from",
    "generated",
    "give",
    "has",
    "have",
    "how",
    "in",
    "into",
    "is",
    "it",
    "local",
    "make",
    "mention",
    "move",
    "of",
    "on",
    "one",
    "or",
    "parent",
    "point",
    "post",
    "reply",
    "seed",
    "should",
    "that",
    "the",
    "their",
    "this",
    "to",
    "use",
    "visible",
    "what",
    "when",
    "where",
    "which",
    "with",
    "would",
}
NON_SUBSTANTIVE_PAYLOADS = {
    "low_info_reaction",
    "joke",
    "meta_or_template",
    "side_tangent",
}
NON_SUBSTANTIVE_FUNCTIONS = {"offtopic_noise"}
LOW_INFORMATION_PAYLOADS = {
    "low_info_reaction",
    "joke",
    "meta_or_template",
    "narrow_question",
}
DEPENDENT_RELATIONS = {
    "answers_parent",
    "challenges_parent",
    "asks_narrow_followup",
    "corrects_detail",
}
SEMANTIC_FIELDS = (
    "semantic_move",
    "local_topic",
    "detail_focus",
    "domain_intent",
    "decision_boundary",
    "reply_delta",
    "reply_novelty_anchor",
    "development_plan",
    "actor_participation_goal",
    "actor_attention_focus",
)
CONTROL_FIELDS = (
    "perspective_id",
    "content_angle",
    "claim_family",
    "comment_function",
    "reply_relation",
    "stance",
    "evidence_mode",
)
# The similarity bar a reply's novelty anchor must clear against every
# ancestor already in its branch, not only its immediate parent (see
# `novelty_scope` on `reply_increment_problem`). Reused unmodified from the
# parent-only check so no new threshold is introduced.
REPLY_NOVELTY_SIMILARITY_THRESHOLD = 0.76
NON_REPAIRABLE_ISSUES = frozenset(
    {
        # Reusing a reference is useful audit context, not proof that the two
        # semantic plans collide.
        "duplicate_reference",
        # The structural root-branch schedule owns perspective_id and rewrites
        # it before every evaluation, so a slot-local Planner retry cannot
        # change the thread-level concentration.
        "perspective_concentration",
    }
)

# These conflicts receive repair priority because the Writer would otherwise
# have to choose between controls in the same slot. They are not terminal:
# schema completeness is enforced separately, while residual content quality
# is persisted for Writer realization and final evaluation.
BLOCKING_PLAN_ISSUES = frozenset(
    {
        "social_contract_conflict",
        "surface_density_conflict",
        "surface_capacity_conflict",
    }
)


@dataclass(frozen=True)
class PlanQualityIssue:
    code: str
    sample_id: int
    message: str
    other_sample_id: int | None = None


@dataclass(frozen=True)
class PlanQualityReport:
    issues: tuple[PlanQualityIssue, ...]
    substantive_count: int
    thread_substantive_count: int
    colliding_samples: tuple[int, ...]
    collision_rate: float
    dominant_perspective: str
    dominant_perspective_share: float
    issue_score: float
    # Every semantic move already spent in this thread, in plan order. G96: the
    # repair loop gave up on 111 slot instances in v122 because it was asked to
    # "change the decision lens" -- a category, which E4 prices at 0.23
    # compliance -- while never being shown which lenses the thread had already
    # used. Carrying the ledger is what turns that into a concrete instruction.
    spent_moves: tuple[str, ...] = ()

    @property
    def healthy(self) -> bool:
        return not self.repair_issues

    @property
    def repair_issues(self) -> tuple[PlanQualityIssue, ...]:
        return tuple(
            issue
            for issue in self.issues
            if issue.code not in NON_REPAIRABLE_ISSUES
        )

    @property
    def blocking_issues(self) -> tuple[PlanQualityIssue, ...]:
        return tuple(
            issue for issue in self.repair_issues if issue.code in BLOCKING_PLAN_ISSUES
        )

    @property
    def repair_rank(self) -> tuple[int, float]:
        """Order candidates by realizability first, then aggregate quality."""

        return (len(self.blocking_issues), self.issue_score)

    def spent_move_block(self, *, limit: int = 24) -> str:
        """Name the moves this thread has already spent, so a repair can avoid them.

        G96: the repair loop surrendered on 111 v122 slot instances while being
        told only to "change the decision lens, stance, evidence role, or local
        detail". That is a category (E4: 0.23 compliance) and it never says what
        is already taken, so the Planner re-rolls from the same small vocabulary
        -- G94 measured that vocabulary at 72% rejection under greedy dedup at
        cosine 0.45. This block converts the instruction into a concrete one.
        """

        # Gate here as well as at the call site. A rendering method that ignores
        # its own arm is one refactor away from leaking into the legacy path,
        # and E12 cost a paid run to exactly that class of mistake.
        if not plan_move_ledger_enabled() or not self.spent_moves:
            return ""
        shown = self.spent_moves[-limit:]
        rows = [
            "This thread has already spent the following semantic moves. The "
            "repaired slot must make a move that is not on this list and is not "
            "a rewording of one:",
        ]
        rows.extend(f"- {move}" for move in shown)
        if len(self.spent_moves) > limit:
            rows.append(
                f"- (plus {len(self.spent_moves) - limit} earlier move(s); do not "
                "repeat those either)"
            )
        rows.append(
            "Name the new move's decision lens explicitly and make it one this "
            "thread has not used. Do not reword a listed move, and do not repair "
            "by changing only claim_key, entity names, or numbers."
        )
        return "\n".join(rows)

    def feedback(
        self,
        *,
        repair_attempt: int = 1,
        sample_ids: Iterable[int] | None = None,
        limit: int = 14,
    ) -> str:
        requested = (
            {int(sample_id) for sample_id in sample_ids}
            if sample_ids is not None
            else None
        )
        issues = [
            issue
            for issue in self.repair_issues
            if requested is None or issue.sample_id in requested
        ]
        if not issues:
            return ""
        rows = [
            "The listed plan slot failed the domain-neutral plan-quality check. "
            "Regenerate only the displayed S# and repair every listed item:"
        ]
        for issue in issues[:limit]:
            rows.append(f"- S{issue.sample_id} [{issue.code}]: {issue.message}")
        if len(issues) > limit:
            rows.append(
                f"- Repair the remaining {len(issues) - limit} issue(s) by the same rule."
            )
        rows.extend(
            [
                "- Return exactly one plan for the displayed S#; do not rewrite any healthy slot.",
                "- Keep slot structure and information density unchanged.",
                "- For a surface_density_conflict, keep the semantic move and choose a payload, role, and sentence route that can carry the displayed ordinary or long slot.",
                "- For long_form_capacity, keep the semantic move and add distinct connected development beats; do not pad it with paraphrases or unrelated claims.",
                "- For a semantic collision, change the decision lens, stance, evidence role, or local detail; changing only claim_key wording is not a repair.",
                "- Use each displayed R# at most once for a substantive, non-dependent comment.",
            ]
        )
        strategies = (
            "Choose a different decision lens and a different seed-grounded local detail.",
            "Change the discourse function, stance, and evidence role while preserving the slot's surface density.",
            "Use a different reference pattern or a genuinely different seed-local/social move; avoid the conflicting plan's recommendation path.",
        )
        rows.append(
            "- Repair strategy for this attempt: "
            + strategies[max(0, repair_attempt - 1) % len(strategies)]
        )
        return "\n".join(rows)

    def to_dict(self) -> dict[str, Any]:
        return {
            "healthy": self.healthy,
            "blocking_issue_count": len(self.blocking_issues),
            "warning_count": len(self.issues) - len(self.repair_issues),
            "substantive_count": self.substantive_count,
            "thread_substantive_count": self.thread_substantive_count,
            "colliding_samples": list(self.colliding_samples),
            "collision_rate": round(self.collision_rate, 6),
            "dominant_perspective": self.dominant_perspective,
            "dominant_perspective_share": round(self.dominant_perspective_share, 6),
            "issue_score": round(self.issue_score, 6),
            "issues": [
                {
                    "code": issue.code,
                    "sample_id": issue.sample_id,
                    "other_sample_id": issue.other_sample_id,
                    "message": issue.message,
                }
                for issue in self.issues
            ],
        }


def universal_viewpoints(*, limit: int = 32) -> list[dict[str, Any]]:
    return [
        dict(item, source="universal_decision_lens")
        for item in UNIVERSAL_VIEWPOINTS[:limit]
    ]


def evaluate_plan_batch(
    plans: dict[int, dict[str, Any]],
    *,
    prior_plans: Iterable[dict[str, Any]] = (),
    similarity_threshold: float = 0.72,
    embedding_similarity_threshold: float = 0.82,
    semantic_similarity: Callable[[dict[str, Any], dict[str, Any]], float] | None = None,
    max_perspective_share: float = 0.34,
    require_reply_novelty: bool = False,
    reply_novelty_scope: str = "parent_only",
    enforce_social_contract: bool = True,
) -> PlanQualityReport:
    prior = [dict(row) for row in prior_plans if isinstance(row, dict)]
    current = [
        dict(plan, sample_id=int(sample_id))
        for sample_id, plan in sorted(plans.items())
    ]
    issues: list[PlanQualityIssue] = []
    colliding: set[int] = set()

    substantive = [plan for plan in current if is_substantive_plan(plan)]
    seen: list[dict[str, Any]] = list(prior)
    for plan in current:
        sample_id = _sample_id(plan)
        surface_problem = surface_density_problem(plan)
        if surface_problem:
            issues.append(
                PlanQualityIssue(
                    code="surface_density_conflict",
                    sample_id=sample_id,
                    message=surface_problem,
                )
            )
        capacity_problem = surface_capacity_problem(plan)
        if capacity_problem:
            issues.append(
                PlanQualityIssue(
                    code="surface_capacity_conflict",
                    sample_id=sample_id,
                    message=capacity_problem,
                )
            )
        development_problem = development_plan_problem(plan)
        if development_problem:
            issues.append(
                PlanQualityIssue(
                    code="long_form_capacity",
                    sample_id=sample_id,
                    message=development_problem,
                )
            )
        social_problem = social_contract_problem(
            plan,
            enforce_coherence=enforce_social_contract,
        )
        if social_problem:
            issues.append(
                PlanQualityIssue(
                    code="social_contract_conflict",
                    sample_id=sample_id,
                    message=social_problem,
                )
            )
        tone_problem = tone_register_problem(
            plan,
            enforce_coherence=enforce_social_contract,
        )
        if tone_problem:
            issues.append(
                PlanQualityIssue(
                    code="tone_role_mismatch",
                    sample_id=sample_id,
                    message=tone_problem,
                )
            )
        reply_problem = reply_increment_problem(
            plan,
            parent_plans=seen,
            semantic_similarity=semantic_similarity,
            required=require_reply_novelty,
            novelty_scope=reply_novelty_scope,
        )
        if reply_problem:
            issues.append(
                PlanQualityIssue(
                    code="reply_increment_conflict",
                    sample_id=sample_id,
                    message=reply_problem,
                    other_sample_id=_parent_sample_id(plan) or None,
                )
            )
        branch_problem = branch_goal_problem(
            plan,
            semantic_similarity=semantic_similarity,
        )
        if branch_problem:
            issues.append(
                PlanQualityIssue(
                    code="branch_goal_conflict",
                    sample_id=sample_id,
                    message=branch_problem,
                )
            )
        if is_substantive_plan(plan):
            best: tuple[float, dict[str, Any], str] | None = None
            for other in seen:
                if not is_substantive_plan(other):
                    continue
                reason, score = collision_reason(
                    plan,
                    other,
                    similarity_threshold=similarity_threshold,
                    embedding_similarity_threshold=embedding_similarity_threshold,
                    semantic_similarity=semantic_similarity,
                )
                if reason and (best is None or score > best[0]):
                    best = (score, other, reason)
            if best is not None:
                score, other, reason = best
                other_id = _sample_id(other)
                issues.append(
                    PlanQualityIssue(
                        code=reason,
                        sample_id=sample_id,
                        other_sample_id=other_id,
                        message=(
                            f"collides with S{other_id}; semantic similarity={score:.3f}. "
                            "Assign a materially different local move rather than renaming the claim."
                        ),
                    )
                )
                if reason != "duplicate_reference":
                    colliding.add(sample_id)
        seen.append(plan)

    thread_substantive = [
        plan for plan in prior if is_substantive_plan(plan)
    ] + substantive
    perspective_counts = Counter(
        str(plan.get("perspective_id") or "seed_local").strip().upper()
        for plan in thread_substantive
    )
    dominant = ""
    dominant_share = 0.0
    if perspective_counts:
        dominant, dominant_count = perspective_counts.most_common(1)[0]
        dominant_share = dominant_count / len(thread_substantive)
        if (
            substantive
            and len(thread_substantive) >= 8
            and dominant_share > max_perspective_share
        ):
            current_dominant = [
                plan
                for plan in substantive
                if str(plan.get("perspective_id") or "seed_local").strip().upper()
                == dominant
            ]
            sample_id = _sample_id(
                current_dominant[-1] if current_dominant else substantive[-1]
            )
            issues.append(
                PlanQualityIssue(
                    code="perspective_concentration",
                    sample_id=sample_id,
                    message=(
                        f"{dominant} covers {dominant_share:.1%} of substantive plans; "
                        f"target at most {max_perspective_share:.1%}. Reassign only plans whose decision lens is genuinely different."
                    ),
                )
            )
    # A tail batch may contain only a few slots. Using the current batch as the
    # denominator makes one collision look arbitrarily severe depending on
    # batch boundaries. The configured threshold is a thread-level contract.
    collision_rate = len(colliding) / max(1, len(thread_substantive))
    issue_score = (
        10.0 * len(colliding)
        + 9.0 * sum(issue.code == "surface_density_conflict" for issue in issues)
        + 9.0 * sum(issue.code == "surface_capacity_conflict" for issue in issues)
        + 6.0 * sum(issue.code == "long_form_capacity" for issue in issues)
        + 8.0 * sum(issue.code == "social_contract_conflict" for issue in issues)
        + 1.0 * sum(issue.code == "tone_role_mismatch" for issue in issues)
        + 8.0 * sum(issue.code == "reply_increment_conflict" for issue in issues)
        + 8.0 * sum(issue.code == "branch_goal_conflict" for issue in issues)
        + 2.0 * sum(issue.code == "duplicate_reference" for issue in issues)
        + max(0.0, dominant_share - max_perspective_share)
        * max(1, len(thread_substantive))
        * 4.0
    )
    return PlanQualityReport(
        issues=tuple(issues),
        substantive_count=len(substantive),
        thread_substantive_count=len(thread_substantive),
        colliding_samples=tuple(sorted(colliding)),
        collision_rate=collision_rate,
        dominant_perspective=dominant,
        dominant_perspective_share=dominant_share,
        issue_score=issue_score,
        spent_moves=_spent_moves(prior, current),
    )


def _spent_moves(
    prior: list[dict[str, Any]], current: list[dict[str, Any]]
) -> tuple[str, ...]:
    """Semantic moves already committed in this thread, deduplicated, in order.

    Prior slots come first because they are already fixed; the current batch's
    healthy slots are equally unavailable to a repair. Trimmed per entry so a
    long thread's ledger stays readable in a prompt.
    """

    out: list[str] = []
    seen: set[str] = set()
    for row in (*prior, *current):
        move = _normalized_value(row.get("semantic_move"))
        if not move or move in seen:
            continue
        seen.add(move)
        text = str(row.get("semantic_move") or "").strip()
        out.append(text[:110])
    return tuple(out)


def collision_reason(
    current: dict[str, Any],
    other: dict[str, Any],
    *,
    similarity_threshold: float,
    embedding_similarity_threshold: float = 0.82,
    semantic_similarity: Callable[[dict[str, Any], dict[str, Any]], float] | None = None,
) -> tuple[str, float]:
    current_claim = _normalized_value(current.get("claim_key"))
    other_claim = _normalized_value(other.get("claim_key"))
    if (
        current_claim
        and current_claim not in {"local_claim", "miscellaneous"}
        and current_claim == other_claim
    ):
        return "duplicate_claim", 1.0
    if _dependent_variation(current, other):
        return "", 0.0
    lexical_similarity = plan_similarity(current, other)
    embedding_similarity = (
        float(semantic_similarity(current, other))
        if semantic_similarity is not None
        else 0.0
    )
    similarity = max(lexical_similarity, embedding_similarity)
    if (
        lexical_similarity >= similarity_threshold
        or embedding_similarity >= embedding_similarity_threshold
    ):
        return "semantic_collision", similarity
    current_reference = _reference_id(current)
    other_reference = _reference_id(other)
    if current_reference and current_reference == other_reference:
        # Reusing a reference is useful Planner feedback, but it is not itself
        # proof that two planned comments have the same meaning.
        return "duplicate_reference", similarity
    return "", similarity


def plan_similarity(left: dict[str, Any], right: dict[str, Any]) -> float:
    left_tokens = semantic_tokens(left)
    right_tokens = semantic_tokens(right)
    union = left_tokens | right_tokens
    lexical = len(left_tokens & right_tokens) / max(1, len(union))
    comparable = [
        field for field in CONTROL_FIELDS if left.get(field) and right.get(field)
    ]
    control = sum(
        _normalized_value(left.get(field)) == _normalized_value(right.get(field))
        for field in comparable
    ) / max(1, len(comparable))
    return 0.78 * lexical + 0.22 * control


def semantic_tokens(plan: dict[str, Any]) -> set[str]:
    tokens: set[str] = set()
    for field in SEMANTIC_FIELDS:
        for token in TOKEN_RE.findall(str(plan.get(field) or "").lower()):
            normalized = _stem(token)
            if len(normalized) >= 3 and normalized not in STOPWORDS:
                tokens.add(normalized)
    return tokens


def plan_semantic_text(plan: dict[str, Any]) -> str:
    """Render only meaning-bearing plan fields for embedding comparison."""

    return " | ".join(
        str(plan.get(field) or "").strip()
        for field in SEMANTIC_FIELDS
        if str(plan.get(field) or "").strip()
    )


class PlanSemanticIndex:
    """Lazy, cached sentence-embedding index for domain-neutral plan checks."""

    def __init__(self, *, model_name: str, device: str = "cpu") -> None:
        self.model_name = model_name
        self.device = device
        self._model: Any | None = None
        self._vectors: dict[str, Any] = {}

    def prepare(self, plans: Iterable[dict[str, Any]]) -> None:
        texts = [plan_semantic_text(plan) for plan in plans]
        self.encode_texts(texts)

    def encode_texts(self, texts: Iterable[str]) -> list[Any]:
        """Encode arbitrary comment text with the evaluator's shared model."""

        ordered = [str(text or "").strip() for text in texts]
        missing = list(
            dict.fromkeys(
                text for text in ordered if text and text not in self._vectors
            )
        )
        if not missing:
            return [self._vectors[text] for text in ordered if text]
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:  # pragma: no cover - runtime dependency guard
                raise RuntimeError(
                    "Plan semantic quality requires sentence-transformers. "
                    "Install it or explicitly disable embedding plan quality."
                ) from exc
            self._model = SentenceTransformer(self.model_name, device=self.device)
            print(
                f"[plan-semantic-model] model={self.model_name} device={self._model.device}",
                flush=True,
            )
        vectors = self._model.encode(
            missing,
            batch_size=min(64, max(1, len(missing))),
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        self._vectors.update(zip(missing, vectors, strict=True))
        return [self._vectors[text] for text in ordered if text]

    def similarity(self, left: dict[str, Any], right: dict[str, Any]) -> float:
        left_text = plan_semantic_text(left)
        right_text = plan_semantic_text(right)
        if not left_text or not right_text:
            return 0.0
        left_vector = self._vectors.get(left_text)
        right_vector = self._vectors.get(right_text)
        if left_vector is None or right_vector is None:
            self.prepare((left, right))
            left_vector = self._vectors[left_text]
            right_vector = self._vectors[right_text]
        return float(left_vector @ right_vector)


def is_substantive_plan(plan: dict[str, Any]) -> bool:
    payload = _normalized_value(plan.get("payload_type"))
    function = _normalized_value(plan.get("comment_function"))
    return (
        payload not in NON_SUBSTANTIVE_PAYLOADS
        and function not in NON_SUBSTANTIVE_FUNCTIONS
    )


def social_contract_problem(
    plan: dict[str, Any],
    *,
    enforce_coherence: bool = True,
) -> str:
    """Reject social labels that contradict the rest of the planned turn.

    This inspects Planner metadata only. It deliberately does not use wording,
    entities, or domain-specific semantic rules.
    """

    problems: list[str] = []
    story = _normalized_value(plan.get("story_mode")) or "no_story"
    payload = _normalized_value(plan.get("payload_type"))
    function = _normalized_value(plan.get("comment_function"))
    evidence = _normalized_value(plan.get("evidence_mode"))
    if enforce_coherence and story == "no_story":
        conflicts = []
        if payload == "personal_story":
            conflicts.append("payload_type=personal_story")
        if evidence == "firsthand_experience":
            conflicts.append("evidence_mode=firsthand_experience")
        if conflicts:
            problems.append(
                "story_mode=no_story conflicts with "
                + " and ".join(conflicts)
                + "; preserve the local point as a present-state appraisal, "
                "small observation, or assertion without a past action, event, "
                "before/after change, or temporal sequence"
            )
    elif enforce_coherence and story:
        required = []
        if payload not in {"personal_story", "fragment_datapoint"}:
            required.append("payload_type=personal_story or fragment_datapoint")
        if function != "personal_datapoint":
            required.append("comment_function=personal_datapoint")
        if evidence != "firsthand_experience":
            required.append("evidence_mode=firsthand_experience")
        if required:
            problems.append(
                f"story_mode={story} needs a coherent narrative-evidence plan: "
                + ", ".join(required)
                + "; keep the scheduled story label and repair the surrounding "
                "role rather than writing advice, analysis, or a bare verdict"
            )

    role = _normalized_value(plan.get("speaker_role"))
    affect = _normalized_value(plan.get("affect_role"))
    reply_delta_type = _normalized_value(plan.get("reply_delta_type"))
    social_close = role == "gratitude_reply" or reply_delta_type == "social_close"
    social_reaction_contract = (
        affect in {"gratitude", "relief"}
        and role == "gratitude_reply"
        and function == "reaction"
        and payload
        in {
            "low_info_reaction",
            "bare_answer",
            "soft_helpful",
            "fragment_datapoint",
        }
        and story == "no_story"
    )
    if affect in {"gratitude", "relief"} and not social_reaction_contract:
        problems.append(
            f"affect_role={affect} requires a social reaction contract: a no-story "
            "reaction with "
            "speaker_role=gratitude_reply and an acknowledgement-capable "
            f"payload; current role={role or 'unset'}, function={function or 'unset'}, "
            f"payload={payload or 'unset'}, story={story or 'unset'}"
        )
    elif social_close and not social_reaction_contract:
        problems.append(
            "speaker_role=gratitude_reply or reply_delta_type=social_close requires "
            "the matching social reaction contract: affect_role=gratitude or relief, "
            "comment_function=reaction, an acknowledgement-capable payload, "
            f"and story_mode=no_story; current affect={affect or 'unset'}, "
            f"role={role or 'unset'}, function={function or 'unset'}, "
            f"payload={payload or 'unset'}, story={story or 'unset'}"
        )
    return "; ".join(problems)


def tone_register_problem(
    plan: dict[str, Any],
    *,
    enforce_coherence: bool = True,
) -> str:
    """Diagnose a weak tone/function pairing without making it semantic truth.

    Polite-Guard classifies realized text; it does not define the comment's
    function. This remains useful Planner feedback because routing every warm
    slot through advice recreates a customer-support register, but an unresolved
    mismatch must not abort a large thread before the Writer can realize tone.
    """

    if not enforce_coherence or _normalized_value(plan.get("tone_class")) != "polite":
        return ""
    stance = _normalized_value(plan.get("stance"))
    role = _normalized_value(plan.get("speaker_role"))
    function = _normalized_value(plan.get("comment_function"))
    allowed_roles = {
        "datapoint_only",
        "op_followup",
        "gratitude_reply",
        "side_observer",
    }
    allowed_functions = {
        "personal_datapoint",
        "reaction",
        "verdict_evaluation",
    }
    if stance == "agree" and role in allowed_roles and function in allowed_functions:
        return ""
    return (
        "tone_class=polite is more naturally realized as agreement, a personal "
        "datapoint, reaction, or positive verdict than as advice, correction, "
        "or abstract analysis; preserve the local contribution, but avoid a "
        "customer-support role"
    )


def reply_increment_problem(
    plan: dict[str, Any],
    *,
    parent_plans: Iterable[dict[str, Any]],
    semantic_similarity: Callable[[dict[str, Any], dict[str, Any]], float] | None,
    required: bool,
    novelty_scope: str = "parent_only",
) -> str:
    """Verify that a direct reply carries an irreducible parent-local delta.

    This inspects Planner metadata, not generated wording. It permits natural
    social closes, but otherwise requires a concrete anchor that represents a
    new test, observation, consequence, exception, or evidence requirement.

    `novelty_scope="parent_only"` (legacy, default) checks the anchor against
    only the immediate parent's plan. A reply chain can clear that bar at
    every hop while collectively drifting back onto a claim made several hops
    higher up -- measured on the v103 N=10 artifact as a `self_bertscore_mean_f1`
    excess that grows from +0.0004 at reply depth 1-2 to +0.0432 at depth 7+
    (`generalized_card/analysis/bertscore_pair_diagnosis.py depth`).
    `novelty_scope="chain"` checks the same probe against every ancestor
    already in the thread's plan ledger instead, at the same threshold.
    """

    parent_plans = list(parent_plans)
    parent_id = _parent_sample_id(plan)
    if parent_id <= 0:
        return ""
    parent = next(
        (
            row
            for row in parent_plans
            if _sample_id(row) == parent_id
        ),
        None,
    )
    if parent is None:
        return ""
    affect = _normalized_value(plan.get("affect_role"))
    role = _normalized_value(plan.get("speaker_role"))
    function = _normalized_value(plan.get("comment_function"))
    if affect in {"gratitude", "relief"} or (
        role == "gratitude_reply" and function == "reaction"
    ):
        return ""
    delta_type = _normalized_value(plan.get("reply_delta_type"))
    novelty = " ".join(str(plan.get("reply_novelty_anchor") or "").split())
    if delta_type in SOCIAL_REPLY_DELTA_TYPES:
        # A social close is a legitimate increment that carries no new claim, so
        # it has no novelty anchor to check.
        return ""
    if required and delta_type not in SUBSTANTIVE_REPLY_DELTA_TYPES:
        return (
            "direct reply needs one valid reply_delta_type and a parent-local "
            "increment, not a generic agreement or paraphrase"
        )
    if required and not novelty:
        return (
            "direct reply is missing reply_novelty_anchor; name one new "
            "observation, test, consequence, threshold, exception, or evidence requirement"
        )
    if not novelty or semantic_similarity is None:
        return ""

    scope = (
        "chain"
        if str(novelty_scope or "parent_only").strip().lower() == "chain"
        else "parent_only"
    )
    if scope == "parent_only":
        # Byte-for-byte legacy: the anchor phrase alone against the parent's
        # plan. Kept exactly as it shipped, including the probe asymmetry --
        # see `novelty_probe` below for why that asymmetry makes this check
        # nearly never fire, which is precisely why `"chain"` does not reuse
        # it (`docs/DECISIONS.md` G3, `analysis/reply_novelty_chain_diagnosis.py`).
        ancestors = [parent]
        novelty_probe = {"semantic_move": novelty}

        def ancestor_probe(ancestor: dict[str, Any]) -> dict[str, Any]:
            return {
                "semantic_move": ancestor.get("semantic_move") or "",
                "decision_boundary": ancestor.get("decision_boundary") or "",
                "detail_focus": ancestor.get("detail_focus") or "",
            }
    else:
        # A short anchor phrase compared against a longer compound ancestor
        # probe suppresses cosine similarity regardless of content -- measured
        # on the v103 artifact, a chain that a human reading the text sees as
        # restating one claim six times over never crossed 0.4 similarity this
        # way. Comparing same-shape probes (the reply's own plan against each
        # ancestor's) is what actually surfaces the restatement: the same
        # chain scores 0.73-0.92 hop to hop this way, against 0.22-0.62 for
        # genuinely unrelated branches in the same artifact.
        ancestors = _ancestor_chain(plan, parent_plans) or [parent]
        novelty_probe = {
            "semantic_move": plan.get("semantic_move") or "",
            "decision_boundary": plan.get("decision_boundary") or "",
            "detail_focus": plan.get("detail_focus") or "",
        }

        def ancestor_probe(ancestor: dict[str, Any]) -> dict[str, Any]:
            return {
                "semantic_move": ancestor.get("semantic_move") or "",
                "decision_boundary": ancestor.get("decision_boundary") or "",
                "detail_focus": ancestor.get("detail_focus") or "",
            }

    worst_score = 0.0
    worst_ancestor_id = parent_id
    for ancestor in ancestors:
        score = float(semantic_similarity(novelty_probe, ancestor_probe(ancestor)))
        if score > worst_score:
            worst_score, worst_ancestor_id = score, _sample_id(ancestor) or parent_id

    if worst_score >= REPLY_NOVELTY_SIMILARITY_THRESHOLD:
        where = (
            "the parent plan"
            if worst_ancestor_id == parent_id
            else f"an earlier ancestor (S{worst_ancestor_id})"
        )
        return (
            f"reply_novelty_anchor is too close to {where} "
            f"(semantic similarity={worst_score:.3f}); use a different test, outcome, "
            "exception, or evidence requirement"
        )
    return ""


def branch_goal_problem(
    plan: dict[str, Any],
    *,
    semantic_similarity: Callable[[dict[str, Any], dict[str, Any]], float] | None,
) -> str:
    """Require a substantive slot to develop its assigned root decision axis."""

    if (
        not is_substantive_plan(plan)
        or semantic_similarity is None
        or _parent_sample_id(plan) > 0
    ):
        return ""
    goal = " ".join(str(plan.get("_required_branch_goal") or "").split())
    if not goal:
        return ""
    probe = {"semantic_move": goal, "local_topic": goal}
    score = float(semantic_similarity(plan, probe))
    if score >= 0.43:
        return ""
    return (
        f"semantic plan does not develop required branch goal "
        f"(similarity={score:.3f}); change the local decision axis, not branch_id"
    )


def surface_density_problem(plan: dict[str, Any]) -> str:
    """Detect a Planner control that would collapse a substantive slot."""

    try:
        words = int(plan.get("_slot_word_count") or 0)
    except (TypeError, ValueError):
        words = 0
    surface = _normalized_value(plan.get("_slot_surface_label"))
    payload = _normalized_value(plan.get("payload_type"))
    if (
        words >= 35
        and surface in {"ordinary_turn", "long_turn"}
        and payload in LOW_INFORMATION_PAYLOADS
    ):
        return (
            f"the anonymous slot has {words} words and surface={surface}, but "
            f"payload={payload} would force it into a short whole-comment mode; "
            "keep the planned local contribution and use a substantive payload and one-use sentence route"
        )
    return ""


def surface_capacity_problem(plan: dict[str, Any]) -> str:
    """Detect a semantic plan that cannot fit the anonymous slot shape.

    This checks only Planner labels plus anonymous word count/surface shape.
    It is deliberately not a wording heuristic and does not prescribe a
    generated length.  Its purpose is to prevent impossible plan contracts,
    such as assigning a personal story to a two-word reaction slot.
    """

    try:
        words = int(plan.get("_slot_word_count") or 0)
    except (TypeError, ValueError):
        words = 0
    surface = _normalized_value(plan.get("_slot_surface_label"))
    payload = _normalized_value(plan.get("payload_type"))
    function = _normalized_value(plan.get("comment_function"))
    story = _normalized_value(plan.get("story_mode"))
    evidence = _normalized_value(plan.get("evidence_mode"))
    impossible = (
        story not in {"", "no_story"}
        or payload in {"personal_story", "advice"}
        or function in {"personal_datapoint", "recommendation_advice", "explanation_analysis"}
        or evidence == "firsthand_experience"
    )
    if 0 < words <= 5 and surface == "micro" and impossible:
        return (
            f"the anonymous slot has {words} words and surface={surface}, but "
            "the assigned story/advice/datapoint contract cannot be realized in "
            "a micro reaction; use no_story with one reaction, fragment, bare "
            "acknowledgement, joke, or narrow question"
        )
    if 0 < words <= 12 and surface in {"short_turn", "short_question"} and story not in {"", "no_story"}:
        return (
            f"the anonymous slot has {words} words and surface={surface}, but "
            "story_mode requires more connected information than this short "
            "slot can carry; use no_story and one narrow local move"
        )
    return ""


def ledger_entry(sample_id: int, plan: dict[str, Any]) -> dict[str, Any]:
    fields = (
        "reference_id",
        "claim_key",
        "perspective_id",
        "reply_relation",
        "semantic_move",
        "local_topic",
        "detail_focus",
        "domain_intent",
        "stance",
        "evidence_mode",
        "payload_type",
        "comment_function",
        "content_angle",
        "story_mode",
        "tone_class",
        "affect_role",
        "opening_style",
        "development_plan",
        "decision_boundary",
        "reply_delta",
        "reply_delta_type",
        "reply_novelty_anchor",
        "parent_semantic_move",
        "parent_decision_boundary",
        "branch_exclusion",
        "owned_decision_subject",
        "claim_family",
        "branch_id",
        "parent_sample_id",
        "real_parent_sample_id",
        "local_parent_task_id",
        "actor_participant_key",
        "actor_knowledge_boundary",
        "actor_participation_goal",
        "actor_evidence_access",
        "actor_attention_focus",
        "actor_interaction_tendency",
        "actor_context_visibility",
        "actor_realization_route",
        "actor_source",
    )
    return {
        "sample_id": int(sample_id),
        **{field: str(plan.get(field) or "") for field in fields},
    }


def _dependent_variation(current: dict[str, Any], other: dict[str, Any]) -> bool:
    relation = _normalized_value(current.get("reply_relation"))
    if relation not in DEPENDENT_RELATIONS:
        return False
    parent_id = _parent_sample_id(current)
    if parent_id <= 0 or parent_id != _sample_id(other):
        return False
    # A reply naturally has a different relation from its parent, so relation
    # alone must not exempt a restatement from semantic collision checks.
    changed = sum(
        _normalized_value(current.get(field)) != _normalized_value(other.get(field))
        for field in ("stance", "evidence_mode", "detail_focus")
    )
    return changed >= 3


def _parent_sample_id(plan: dict[str, Any]) -> int:
    for field in (
        "parent_sample_id",
        "real_parent_sample_id",
        "local_parent_task_id",
    ):
        try:
            value = int(plan.get(field) or 0)
        except (TypeError, ValueError):
            continue
        if value > 0:
            return value
    return 0


def _ancestor_chain(
    plan: dict[str, Any],
    parent_plans: list[dict[str, Any]],
    *,
    max_depth: int = 64,
) -> list[dict[str, Any]]:
    """Walk parent_id links back through the in-thread ledger, immediate
    parent first.

    `parent_plans` is the full already-generated ledger for this thread
    (`evaluate_plan_batch`'s `seen`), which by construction always contains
    every ancestor of `plan` before `plan` itself is evaluated: `seen` is
    appended to only after each plan's checks run, and plans are processed in
    ascending `sample_id` order while a reply's parent always has a smaller
    `sample_id` than the reply itself. `max_depth` is a defensive bound, not a
    design choice -- real threads never approach it.
    """

    by_id = {_sample_id(row): row for row in parent_plans if _sample_id(row) > 0}
    chain: list[dict[str, Any]] = []
    visited = {_sample_id(plan)}
    current_id = _parent_sample_id(plan)
    while current_id > 0 and current_id not in visited and len(chain) < max_depth:
        ancestor = by_id.get(current_id)
        if ancestor is None:
            break
        chain.append(ancestor)
        visited.add(current_id)
        current_id = _parent_sample_id(ancestor)
    return chain


def _reference_id(plan: dict[str, Any]) -> str:
    value = str(plan.get("reference_id") or "").strip().upper()
    return "" if value in {"", "NONE", "NULL", "N/A"} else value


def _sample_id(plan: dict[str, Any]) -> int:
    try:
        return int(plan.get("sample_id") or 0)
    except (TypeError, ValueError):
        return 0


def _normalized_value(value: Any) -> str:
    return "_".join(TOKEN_RE.findall(str(value or "").lower()))


def _stem(token: str) -> str:
    value = token.lower().strip("'")
    for suffix in (
        "ization",
        "ational",
        "fulness",
        "iveness",
        "ments",
        "ment",
        "ing",
        "ers",
        "ies",
        "ed",
        "es",
        "s",
    ):
        if value.endswith(suffix) and len(value) - len(suffix) >= 4:
            return value[: -len(suffix)]
    return value


# --------------------------------------------------------------------------- #
# v124 arm: the spent-move ledger
# --------------------------------------------------------------------------- #
# G96 established that the Planner detects a semantic collision, attempts a
# repair, fails, and ships the slot anyway -- 22 warnings covering 111 slot
# instances in v122, with collision_rate at surrender reaching 0.667. G88 showed
# that raising a retry budget is not the fix (its own objective moved -0.0030 at
# 55% accuracy). The defect is the instruction: it names a category, which E4
# prices at 0.23 compliance against ~1.0 for a concrete token, and it never
# tells the Planner which moves the thread has already spent.
#
# "off" reproduces v122 byte-for-byte: no ledger is rendered anywhere.
PLAN_MOVE_LEDGER_MODE = "off"


def set_plan_move_ledger(mode: str) -> None:
    global PLAN_MOVE_LEDGER_MODE
    value = str(mode or "off").strip().lower()
    if value not in {"off", "spent_moves"}:
        raise ValueError(
            f"unknown plan-move-ledger mode {mode!r}; expected off|spent_moves"
        )
    PLAN_MOVE_LEDGER_MODE = value


def plan_move_ledger_enabled() -> bool:
    return PLAN_MOVE_LEDGER_MODE == "spent_moves"


# --------------------------------------------------------------------------- #
# v125 arm: the topical-outsider quota
# --------------------------------------------------------------------------- #
# G97 measured that the gap is entirely in the LOW tail of the pairwise cosine
# distribution -- p90 is +0.008 but p1 is +0.038, and pairs below cosine 0 are
# 8.09% of real against 3.30% of ours. Per comment, affinity < 0.10 is 11.43%
# real against 4.14% generated, a 2.8x deficit.
#
# It is not a length problem. Word counts already match. But the off-topic rate
# collapses as length grows: at 1-10 words we match real (37.2% vs 36.7%), at
# 61+ words real is 3.4% and we are 0.8%. Of real's low-affinity comments 6.1%
# are >= 40 words; of ours, ZERO. We already write short throwaway lines. We
# never write a long, substantive comment about something else.
#
# The channels already exist in the Planner schema -- `offtopic_noise`,
# `side_tangent`, `joke`, `link_quote_reference` -- and `offtopic_noise` was
# chosen 0 times in 532 v122 slots. The old `--social-noise-min-share` cannot
# do this: `rebalance_card_surfaces` accepts every share argument and runs
# `del kwargs` by design, because the calibrated Planner owns these controls.
# So the quota has to be stated to the Planner.
#
# G97 also recorded the shape this must NOT take: real threads do not drift
# with ordinal position (deciles 0.343 ... 0.345, flat) and our depth curve
# already matches real's almost exactly. A positional "you may leave the topic
# later" rule would move a trend that is already correct. The quota is per-slot.
OUTSIDER_QUOTA_MODE = "off"

# Real rates from 424 evaluation-excluded camera threads (G97). Held as the
# measured target, not a hand-picked number: 13.1% of real comments have
# OP-cosine < 0.10, and the deficit is concentrated in the long tail, so the long
# share is stated separately.
OUTSIDER_SHARE = 0.12
OUTSIDER_LONG_SHARE = 0.30


def set_outsider_quota(mode: str) -> None:
    global OUTSIDER_QUOTA_MODE
    value = str(mode or "off").strip().lower()
    if value not in {"off", "measured"}:
        raise ValueError(
            f"unknown outsider-quota mode {mode!r}; expected off|measured"
        )
    OUTSIDER_QUOTA_MODE = value


def outsider_quota_enabled() -> bool:
    return OUTSIDER_QUOTA_MODE == "measured"


def outsider_quota_block(slot_count: int) -> str:
    """Ask the Planner for a measured share of comments that leave the topic.

    Named channels rather than a category, per E4: naming a concrete move buys
    ~1.0 compliance where naming a category buys 0.23. `offtopic_noise` has been
    in the schema all along and was selected zero times.
    """

    # `slot_count` is the slots in THIS Planner call, not the thread. A quota
    # larger than the batch is unsatisfiable and gets ignored wholesale, which
    # is what v125's first run measured. Round stochastically-free: a batch of 8
    # at 12% asks for 1, which is the honest per-batch expression of the rate.
    if not outsider_quota_enabled() or slot_count < 4:
        return ""
    target = max(1, round(slot_count * OUTSIDER_SHARE))
    if target > slot_count // 2:
        return ""
    long_target = max(1, round(target * OUTSIDER_LONG_SHARE))
    return "\n".join(
        [
            "TOPICAL OUTSIDER QUOTA (measured against real threads in this domain):",
            f"- Exactly {target} of these {slot_count} slots must NOT answer the "
            "post's question at all. Give each one `comment_function`: "
            "`offtopic_noise`, or `payload_type`: `joke`, `side_tangent`, or "
            "`meta_or_template`.",
            f"- At least {long_target} of those {target} must be a LONG slot "
            "(`length_bucket`: `long` or `very_long`) -- a full, seriously argued "
            "comment about a different subject: an off-domain technical "
            "explainer, a process or policy explainer, an extended personal "
            "anecdote, or a reply about another commenter rather than the post. "
            "Real threads carry these; a short throwaway line does not count and "
            "neither does a thank-you.",
            "- An outsider slot is exempt from its branch's owned subject and "
            "from `forbidden_decision_subjects`: it is allowed to be irrelevant.",
            "- Do not satisfy this quota with acknowledgements, agreement, or "
            "thanks. Those are already over-produced.",
        ]
    )
