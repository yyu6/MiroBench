from __future__ import annotations

from dataclasses import dataclass
from typing import Any

@dataclass(frozen=True)
class SeedPost:
    index: int
    title: str
    body: str
    content: str
    source_raw_post_id: str
    real_num_comments: int
    metadata: dict[str, Any]

@dataclass(frozen=True)
class ThreadTarget:
    target_comments: int
    top_level_comments: int
    max_depth_goal: int
    shape_label: str
    length_mix_note: str

@dataclass(frozen=True)
class BranchPlan:
    branch_id: int
    anchor_quote: str
    anchor_source: str
    detour_type: str
    branch_goal: str
    allowed_functions: tuple[str, ...]
    evidence_modes: tuple[str, ...]
    tone_palette: tuple[str, ...]
    story_modes: tuple[str, ...]
    content_angles: tuple[str, ...]
    perspective_id: str = "seed_local"
    decision_boundary: str = ""
    branch_exclusion: str = ""
    # A concrete, seed-derived condition owned by this root chain.  It is
    # distinct from a reusable P## reasoning lens and keeps parallel branches
    # from rephrasing the same high-salience comparison.
    owned_decision_subject: str = ""

@dataclass(frozen=True)
class CommentTask:
    local_task_id: int
    local_parent_task_id: int | None
    depth: int
    branch_id: int
    branch_goal: str
    visible_scope: str
    local_anchor: str
    comment_function: str
    content_angle: str
    evidence_mode: str
    story_mode: str
    voice: str
    payload_type: str
    length_bucket: str
    speaker_role: str
    utterance_mode: str
    surface_texture: str
    allow_first_person_frame: bool
    allow_uncertainty_frame: bool
    planner_intent: str
    must_not_do: str
    real_sample_id: int | None = None
    real_parent_sample_id: int | None = None
    real_word_count: int | None = None
    semantic_move: str = ""
    local_topic: str = ""
    reply_relation: str = ""
    stance: str = ""
    detail_focus: str = ""
    avoid_repeating: str = ""
    claim_key: str = ""
    claim_family: str = ""
    perspective_id: str = ""
    domain_intent: str = ""
    decision_boundary: str = ""
    branch_exclusion: str = ""
    owned_decision_subject: str = ""
    forbidden_decision_subjects: str = ""
    reply_delta: str = ""
    reply_delta_type: str = ""
    reply_novelty_anchor: str = ""
    parent_semantic_move: str = ""
    parent_decision_boundary: str = ""
    opening_style: str = ""
    development_plan: str = ""
    context_aperture: str = ""
    context_transform: str = "normal"
    real_surface_shape: str = ""
    surface_skeleton: str = ""
    surface_instruction: str = ""
    real_tone_slot: str = ""
    real_tone_instruction: str = ""
    # Retained only so historical generation records still deserialize. The
    # current Planner and Writer policy no longer assigns or consumes overlays.
    tone_overlay_slot: str = ""
    tone_overlay_instruction: str = ""
    tone_target: str = ""
    tone_target_instruction: str = ""
    story_instruction: str = ""
    affect_role: str = ""
    affect_instruction: str = ""
    distribution_assignment: str = ""
    tone_shape: str = ""
    # Which participant holds this slot. Empty reproduces the pre-v77 behaviour,
    # where the author name was a pure function of the slot index and no two
    # comments in a thread shared a speaker.
    speaker_id: str = ""
    concrete_anchors: tuple[str, ...] = ()
