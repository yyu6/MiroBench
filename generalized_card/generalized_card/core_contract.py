from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Iterable

from .domain import REPO_ROOT


CORE_POLICY_VERSION = "card-paper-v37-domain-neutral-profile-v4-20260807"
REVISION_CORE_POLICY_VERSION = (
    "generalized-card-revision-v7-bidirectional-exact-no-fail-20260811"
)
HISTORICAL_REVISION_POLICY_VERSIONS = {
    "generalized-card-revision-v6-dynamic-coverage-history-20260807",
}
GENERALIZED_V2_GENERATION_POLICY_VERSION = (
    "generalized-card-v2-affirmative-affect-uncapped-slot-shape-v73-20260814"
)
HISTORICAL_GENERATION_POLICY_VERSIONS = {
    "generalized-v2": {
        "generalized-card-v2-frozen-domain-profile-v2-20260807",
        "generalized-card-v2-frozen-domain-profile-v3-tree-metadata-20260807",
        "generalized-card-v2-decision-lens-plan-quality-v9-20260809",
        "generalized-card-v2-embedding-plan-quality-anchor-budget-v10-20260809",
        "generalized-card-v2-targeted-plan-repair-v11-20260809",
        "generalized-card-v2-bounded-targeted-plan-repair-v12-20260809",
        "generalized-card-v2-recoverable-targeted-plan-repair-v13-20260810",
        "generalized-card-v2-story-affect-bleu-calibration-v14-20260810",
        "generalized-card-v2-heldout-distribution-writer-selection-v15-20260810",
        "generalized-card-v2-joint-lexical-semantic-writer-control-v16-20260810",
        "generalized-card-v2-local-repair-text-ledger-v17-20260810",
        "generalized-card-v2-completion-aware-rotating-local-repair-v18-20260811",
        "generalized-card-v2-short-safe-completion-repair-v19-20260811",
        "generalized-card-v2-lossless-slot-recovery-v20-20260811",
        "generalized-card-v2-bounded-slot-selection-v21-20260811",
        "generalized-card-v2-domain-derived-actor-single-stage-v22-20260811",
        "generalized-card-v2-domain-actor-matched-length-v23-20260811",
        "generalized-card-v2-domain-actor-soft-length-v24-20260811",
        "generalized-card-v2-single-stage-diagnostic-guards-v25-20260811",
        "generalized-card-v2-one-shot-semantic-contract-v26-20260811",
        "generalized-card-v2-joint-planner-distribution-v27-20260811",
        "generalized-card-v2-three-tone-joint-planner-v28-20260811",
        "generalized-card-v2-bounded-structural-completion-v29-20260811",
        "generalized-card-v2-first-pass-short-ledger-hard-completion-v30-20260811",
        "generalized-card-v2-v12-first-pass-no-structural-backfill-v31-20260812",
        "generalized-card-v2-substantive-surface-contract-v32-20260812",
        "generalized-card-v2-card-surface-invariant-controls-v33-20260812",
        "generalized-card-v2-one-shot-content-depth-v34-20260812",
        "generalized-card-v2-structural-complete-planner-owned-v35-20260812",
        "generalized-card-v2-planner-slot-contract-v36-20260813",
        "generalized-card-v2-boundary-safe-slot-contract-v37-20260813",
        "generalized-card-v2-sequential-semantic-social-contract-v38-20260813",
        "generalized-card-v2-root-branch-first-pass-contract-v39-20260813",
        "generalized-card-v2-planner-authority-first-pass-contract-v42-20260813",
        "generalized-card-v2-branch-subject-reply-delta-first-pass-v43-20260813",
        "generalized-card-v2-parent-delta-first-pass-v44-20260813",
        "generalized-card-v2-parent-delta-normalized-first-pass-v45-20260813",
        "generalized-card-v2-focused-smoke-parent-delta-v46-20260813",
        "generalized-card-v2-root-complete-branch-coverage-v47-20260813",
        "generalized-card-v2-sibling-coherent-first-pass-v48-20260813",
        "generalized-card-v2-compact-highfanout-root-plan-v54-20260813",
        "generalized-card-v2-schema-safe-comment-plan-ids-v55-20260813",
        "generalized-card-v2-topology-parent-delta-v56-20260813",
        "generalized-card-v2-lossless-planner-contract-v57-20260813",
        "generalized-card-v2-first-pass-route-lock-v58-20260813",
        "generalized-card-v2-parent-novelty-contract-v59-20260813",
        "generalized-card-v2-novelty-anchor-primary-v60-20260813",
        "generalized-card-v2-reply-discourse-contract-v61-20260813",
        "generalized-card-v2-compact-parent-local-reply-planner-v62-20260813",
        "generalized-card-v2-axis-budget-parent-local-reply-planner-v63-20260813",
        "generalized-card-v2-calibrated-tone-register-length-scale-v64-20260813",
        "generalized-card-v2-tone-compatible-increments-reply-development-v65-20260813",
        "generalized-card-v2-heldout-entity-inventory-route-ledger-v66-20260813",
        "generalized-card-v2-bounded-thread-blackboard-v67-20260813",
        "generalized-card-v2-domain-claim-entity-generalized-v68-20260813",
        "generalized-card-v2-scheduled-opener-grammar-v69-20260813",
        "generalized-card-v2-domain-claim-field-survival-v70-20260813",
        "generalized-card-v2-planner-owned-reply-move-single-parent-exclusion-v71-20260813",
    },
}

# These are the exact CARD implementations that the generalized adapters are
# allowed to wrap. A changed hash means the algorithm changed outside the
# domain adapter and must be reviewed before another paper run.
CORE_FILES = {
    "generator": (
        "artifacts/pipeline_snapshots/v37_gpt_writer_selfbleu_3rounds_20260704/source_snapshots/generator_v37_surface_tone_balanced.py",
        "a9fd440e723d712841ec14b05b7cf717338847bda7b14523998f302fb85b1e47",
    ),
    # The generalized-v2 engine is a package: the facade holds the pipeline the
    # adapter patches, and each engine module holds one concern. Every file is
    # pinned, so a change anywhere in the engine is still reviewed.
    "generator_generalized_v2": (
        "scripts/sampling_generator/run_sampled_reddit_generator.py",
        "3b544b0a593acb4ece6c616b21dbdfe576ac6a77617b98e4de10c98bfd93865b",
    ),
    "engine_vocabulary": (
        "scripts/sampling_generator/engine/vocabulary.py",
        "0d9ef802413191975214eefeb117e6415b1f7b3876e86f6f0b3745d4815cead8",
    ),
    "engine_model": (
        "scripts/sampling_generator/engine/model.py",
        "cf284ea168c12f5527316534495004788bbdc4ac27688baa478a890ac9c3aa7c",
    ),
    "engine_util": (
        "scripts/sampling_generator/engine/util.py",
        "90d59a2ce1e1a241120f58d0ff5f2f7cb7611e5df6b1b48e3fdda0280ff59553",
    ),
    "engine_cli": (
        "scripts/sampling_generator/engine/cli.py",
        "593aac8ee133982035f45b8b481f57a93a98789b81223ac9862aafe86a61330b",
    ),
    "engine_thread_structure": (
        "scripts/sampling_generator/engine/thread_structure.py",
        "0113eb2caa85224ce030284c171c51dfa5310b0be873fd01329b4368e9c533a3",
    ),
    "engine_slot_inference": (
        "scripts/sampling_generator/engine/slot_inference.py",
        "e0c116f4384945ec885155897a50d16898090d7ced992528dc82b578a6cde961",
    ),
    "engine_context_policy": (
        "scripts/sampling_generator/engine/context_policy.py",
        "ed931efd82bb0bfa3e916414763aa8e2ad0f02d13bd9002094ae49435cc2701f",
    ),
    "engine_anchors": (
        "scripts/sampling_generator/engine/anchors.py",
        "1f0ea6fcf35f351fa4923ad529b2060aa554b36e7d9b7327db0980ee21e678a0",
    ),
    "engine_parent_alignment": (
        "scripts/sampling_generator/engine/parent_alignment.py",
        "628717a58ae0bc823bf700666a8f9a593f3212eaa0246aa13fa1dd7dffec3d13",
    ),
    "engine_writer_request": (
        "scripts/sampling_generator/engine/writer_request.py",
        "196d18e654d84be4c5eeb61b5fecdc24149233cd12f10fffbfce56c81fc5f111",
    ),
    "engine_writer_validation": (
        "scripts/sampling_generator/engine/writer_validation.py",
        "d850f7e98c9c4896d361875d26dcbb4a9b4472c04af3e3412b63194322d51796",
    ),
    "engine_persistence": (
        "scripts/sampling_generator/engine/persistence.py",
        "b020a0db57873b7674c731128af20b2a501c0dacfab94fa8b338d4564b93f214",
    ),
    "generator_adapter": (
        "generalized_card/generalized_card/backend.py",
        "f3672f1f2e3cea0764eff684d89be7c2956daabc49fa4517d32e32d501e4cd96",
    ),
    "actor_conditioning": (
        "generalized_card/generalized_card/actor_conditioning.py",
        "476241a346f8a73c31998f5ab315fe6cf9581b37f5be4418fa7ccb42ae5df12f",
    ),
    "planner_schema": (
        "generalized_card/generalized_card/planner_schema.py",
        "2b5f24f1f655a03bfef60294779e59db3f1501f4c46952e445b51928d26e9aaf",
    ),
    "generation_runner": (
        "generalized_card/scripts/run_generate.py",
        "89540a8402f6a8d44af6b0654a4b421219989d1b83a2eddb4778f9c7258d5d36",
    ),
    "domain_profile": (
        "generalized_card/generalized_card/domain_profile.py",
        "c54da9afe5b14853dfa7384159c6bb7d1dee10696583742d1def29c8ca903817",
    ),
    "viewpoint_bank": (
        "generalized_card/generalized_card/viewpoint_bank.py",
        "baf4ede032bf67642da2026ed315b2c8b8af41fb0c8051b0b71293362cd9b2a6",
    ),
    "planning_quality": (
        "generalized_card/generalized_card/planning_quality.py",
        "ef20f9c385e2be4defb43a0bd4173556058c7377a9b111c73b6d76b85bed12c8",
    ),
    "generation_distribution": (
        "generalized_card/generalized_card/generation_distribution.py",
        "5f1ed653cf8a19305d9aea9db0c355ede39aeaaa1d0590525c506ed765d71113",
    ),
    "planner_distribution": (
        "generalized_card/generalized_card/planner_distribution.py",
        "adf2377d9e80849271f80752b493583d210a43bc2f1f885c31cf1fc8bf58c6f1",
    ),
    "branch_routing": (
        "generalized_card/generalized_card/branch_routing.py",
        "fea4e44a5a751a954f5fe33f43d6dbd0c6a28d4fecbc150e85c1cf9961a8f80a",
    ),
    "task_distribution": (
        "generalized_card/generalized_card/task_distribution.py",
        "97818b96dd51542cee57080745c48bf0fdb19812eadee1081612689e8db6a98c",
    ),
    "first_pass_policy": (
        "generalized_card/generalized_card/first_pass_policy.py",
        "3b55eaf838c8b74b153e3eed86a90e5f22e02da239981f31941ac920ec04131c",
    ),
    "domain_claim": (
        "generalized_card/generalized_card/domain_claim.py",
        "65362fdac664c858c6883e872be447219352a754a2218c86928ba14e3b2dc111",
    ),
    "opener_profile": (
        "generalized_card/generalized_card/opener_profile.py",
        "bc907859b6473e1c22ee063c3bdf93563e92188c8594f6c6e72f60cdd6558027",
    ),
    "entity_inventory": (
        "generalized_card/generalized_card/entity_inventory.py",
        "0157a4d67d2b21123848996818a2d9b76a46cbe09af13017905c617be9e50cff",
    ),
    "lexical_quality": (
        "generalized_card/generalized_card/lexical_quality.py",
        "62137cdbf572c37ea4a7bec53eee92e089e3636766e4be8e798b18bf7ba6aeb7",
    ),
    "reference_metric_calibration": (
        "generalized_card/generalized_card/reference_metric_calibration.py",
        "820382c58fa0b3efccd11e2b5ccee8968bd4ca226395869ceb8471e688ed000c",
    ),
    "generation_diversity": (
        "generalized_card/generalized_card/generation_diversity.py",
        "b6bb4afd47c1d67f115688654531c131b3dfac33560d8b84248297e490bcf73e",
    ),
    "writer_quality": (
        "generalized_card/generalized_card/writer_quality.py",
        "863149ef6debd5bba15e9fdc4246008cff909f565ad0272201781eeac19f2b44",
    ),
    "semantic_realization": (
        "generalized_card/generalized_card/semantic_realization.py",
        "b554682b4c93b8a2bc350b8e8a987812bc0cc65f24c0af7e9556fd4e9b22dba6",
    ),
    "length_policy": (
        "generalized_card/generalized_card/length_policy.py",
        "fa6c3db16af19542efca8158fd3acfe0d605a52b5654cd17d7000e7f0f3e28e1",
    ),
    "long_form_planning": (
        "generalized_card/generalized_card/long_form_planning.py",
        "39626ffefe85a141c31baaec5ef034f39988491f3d1a6a1ca3a72e0f070bfa80",
    ),
    "surface_contract": (
        "generalized_card/generalized_card/surface_contract.py",
        "c6b92fb3acb8d47ff24bf952efc1ff26f10fccc06ca42ebd2032c3b0e0d7ccde",
    ),
    "generation_harness": (
        "generalized_card/scripts/run_generation_harness.py",
        "e88b7f0bd7d0317795eebea4582f365562c9b6b377c1458f9ae14202d13a4681",
    ),
    "selfbleu_controller": (
        "scripts/run_metric_revision_controller.py",
        "f79e8971cc311781e48b508f2ad61326436d293211c735a7e8a122d5a0f78059",
    ),
    "generalized_selfbleu_controller": (
        "generalized_card/scripts/run_selfbleu_revision_controller.py",
        "2c3d6f1a70e026c98300ce99ad26af527eacda517a31b4c11e8565bdece54212",
    ),
    "selfbleu_reviser": (
        "scripts/postprocess_selfbleu_lexical_reviser.py",
        "d7554465705793dfe4b9cd688a67deb68c4332b7c9e93fa86d8c07ed575c36d1",
    ),
    "tone_controller": (
        "scripts/run_tone_revision_controller.py",
        "cad9eebf97d869eedb7961971345e26b2eefe23dc1b0cfbdd3c264a490bed623",
    ),
    "tone_reviser": (
        "scripts/postprocess_tone_calibrated_reviser.py",
        "2337f2018bae33d205a6b4be30b50e49f569b5a59a6a9a7c189f9e281d08de1a",
    ),
    "selfbert_reviser": (
        "scripts/postprocess_selfbert_discourse_diversify.py",
        "01d7213aa438141c681a54b3c859246aab17cbf43adc09aa0ca69ce0436e6f3e",
    ),
    "story_structure_controller": (
        "scripts/run_story_structure_revision_controller.py",
        "e2fa09d99137a1d652d75b82ce9fb3bfde3917c5ecadb8f7e48c8fec4ef86803",
    ),
    "story_reviser": (
        "scripts/postprocess_story_probability_reviser.py",
        "36e9da06a927b74900c84922fcf82ee7c0a9068e50f4fa3e17fd9537effb8693",
    ),
    "structure_reviser": (
        "scripts/postprocess_structure_reviser.py",
        "d8ffc907fb908f7cc59540780d5118e523ef887baf8509a57fc8b25fb0a69868",
    ),
    "revision_memory": (
        "scripts/revision_memory.py",
        "3cba826b294c70cca5782d47c85adbac0ed609cb212294e83e894fdcfa801605",
    ),
    "revision_orchestrator": (
        "generalized_card/scripts/run_full_revise.py",
        "c41c5faca7f15fa5e684ba0a6949ac6106ea9a3f06bc2c87b4267a52bcf11a7d",
    ),
    "revision_stage_runner": (
        "generalized_card/scripts/run_revise.py",
        "7bf8d132d0282151fb4502f7885c8aebcc4657f7bf97c2c006f90375a7ea0cbd",
    ),
    "reviser_adapter": (
        "generalized_card/generalized_card/reviser_backend.py",
        "68e086b169a0cc76578f144bb659217ef29f712bd2e9f79243659d7ef670b123",
    ),
    "domain_prompt_adapter": (
        "generalized_card/generalized_card/prompts.py",
        "288f80bf90b5044ac04d9d3c6b859791fbac6cd5d07fbf765c126cd1dc6cac89",
    ),
    "reply_planning": (
        "generalized_card/generalized_card/reply_planning.py",
        "7e943afdf4439734f3f464d32bbc8f709dcd0ec0b2f65665eb4d29535664e49e",
    ),
    "persona_bridge": (
        "generalized_card/generalized_card/persona_bridge.py",
        "7dbad1b96b6e69ae92a8502621b02dda67391e3cbcccc63ba592c603bd336086",
    ),
    "selfbert_controller": (
        "generalized_card/scripts/run_selfbert_revision_controller.py",
        "52318d35f2be38c22d95756931badbecb3bb1873f80c8155e7059053862fd714",
    ),
    "text_metric_controller": (
        "generalized_card/scripts/run_text_metric_revision_controller.py",
        "db7a4c263f2b9921f30eefc02001cc3be751e01ebdd42917b555a2c07c859f9f",
    ),
    "text_metric_reviser": (
        "generalized_card/generalized_card/text_metric_reviser.py",
        "61993555b03326ed07ab5f2a76ea4995a13a4cfe584721b9bd9670ebf0ba478f",
    ),
    "selfbleu_backend": (
        "generalized_card/scripts/run_selfbleu_reviser_backend.py",
        "6e4e050f554056c29b5ac87f3a21cf82699725344ad5b07d6fdc7deef0c7f672",
    ),
    "selfbert_backend": (
        "generalized_card/scripts/run_selfbert_reviser_backend.py",
        "e72a2705e212a4c93ad0003b2476f72b59a806697fa7d092817c2b687092838b",
    ),
    "tone_backend": (
        "generalized_card/scripts/run_tone_reviser_backend.py",
        "a17710e8df9b89cae8047fb03dc22f27229a6f98b8faa88a4a03382e4335b27f",
    ),
    "story_backend": (
        "generalized_card/scripts/run_story_reviser_backend.py",
        "c4042d3492b5da73a4c187e307e1642f1965464131e8d16a07bd2152380d3284",
    ),
    "structure_backend": (
        "generalized_card/scripts/run_structure_reviser_backend.py",
        "ff76f1f8cc1d665cda607b189015e9a694de4856724e5479a46243c3f909b5e4",
    ),
    "distribution_diagnostics": (
        "scripts/distribution_diagnostics.py",
        "665198075cea4c7c4f5760d399ef8cf54afcf96a0d4a0eb00461cd78b3e4a0c5",
    ),
    "output_audit": (
        "generalized_card/generalized_card/audit.py",
        "ea2a17a9260eebc842ec519e0c3e6a65f1bcf9bd665b3f4583a99cff7cc710f4",
    ),
    "evaluation_runner": (
        "generalized_card/scripts/run_evaluate.py",
        "74c5ab6f6d09cbbef425350930e7a3d485de8f85b7879e40cba2b3f278a58d17",
    ),
    "parity_auditor": (
        "generalized_card/scripts/audit_core_parity.py",
        "24029279d48f1cd7f4170dfd958c41141085623b20e45754f48e9e9e5f5cd63f",
    ),
    "cleanup": (
        "scripts/postprocess_generated_discussions_gpt_cleanup.py",
        "05d035cc0792f35ad8fa01564564fb882c7689505151f45654abe88321213ffe",
    ),
    "score_runner": (
        "scripts/evaluation/score_sampled_generated_runs.py",
        "25381c0ffe2bb386059df8b8837673b87e0795f738c39abee3c4704eaf8c3766",
    ),
    "metric_runner": (
        "calibration/runner.py",
        "375c1adfeb7a73b944cff99609ee73bb7fc19927026acef50b229fb61248ebfc",
    ),
    "matched_evaluator": (
        "scripts/evaluate_matched_seed_group.py",
        "b09d77e5534457df0e5bcfa7f1d3c460e8a003ce57c71cdd458c3c632dc1fa3b",
    ),
}


# The generalized-v2 generator is verified as a whole package, not one file.
GENERALIZED_V2_ENGINE_FILES = (
    "generator_generalized_v2",
    "engine_vocabulary",
    "engine_model",
    "engine_util",
    "engine_cli",
    "engine_thread_structure",
    "engine_slot_inference",
    "engine_context_policy",
    "engine_anchors",
    "engine_parent_alignment",
    "engine_writer_request",
    "engine_writer_validation",
    "engine_persistence",
)


def verify_core_contract(names: Iterable[str]) -> dict[str, dict[str, str]]:
    provenance: dict[str, dict[str, str]] = {}
    mismatches: list[str] = []
    for name in names:
        relative, expected = CORE_FILES[name]
        path = REPO_ROOT / relative
        actual = _sha256(path) if path.exists() else "missing"
        provenance[name] = {
            "path": str(path),
            "expected_sha256": expected,
            "actual_sha256": actual,
        }
        if actual != expected:
            mismatches.append(f"{name}:{relative}:{actual}")
    if mismatches and os.environ.get("GENERALIZED_CARD_ALLOW_CORE_DRIFT") != "1":
        raise RuntimeError(
            "CARD core contract mismatch. Review the changed core before running, "
            "or set GENERALIZED_CARD_ALLOW_CORE_DRIFT=1 only for an intentional "
            "non-paper experiment: " + "; ".join(mismatches)
        )
    return provenance


def verify_run_policy(
    run_config: dict[str, object],
    *,
    operation: str,
    allow_historical: bool = False,
) -> str:
    """Prevent old generalized artifacts from entering the current generation chain."""

    profile = str(run_config.get("generator_profile") or "card-snapshot")
    expected = (
        GENERALIZED_V2_GENERATION_POLICY_VERSION
        if profile == "generalized-v2"
        else CORE_POLICY_VERSION
    )
    actual = str(
        run_config.get("generator_policy_version")
        or run_config.get("card_core_policy_version")
        or "missing"
    )
    if actual == expected:
        return actual
    if allow_historical and actual in HISTORICAL_GENERATION_POLICY_VERSIONS.get(profile, set()):
        return actual
    if os.environ.get("GENERALIZED_CARD_ALLOW_LINEAGE_MISMATCH") == "1":
        return actual
    raise RuntimeError(
        f"Cannot {operation}: run policy is {actual!r}, but this code requires "
        f"{expected!r} for profile {profile!r}. Existing generated comments cannot be relabeled "
        "as CARD-core-parity output. Use a new tag and regenerate, or set "
        "GENERALIZED_CARD_ALLOW_LINEAGE_MISMATCH=1 only for an intentional "
        "non-paper audit."
    )


def verify_revision_policy(
    run_config: dict[str, object],
    *,
    operation: str,
) -> str:
    """Verify the reviser lineage without relabeling its generator lineage.

    Native versioned runs carry ``card_core_policy_version``. Revision-only
    workspaces instead carry ``revision_core_policy_version`` and retain the
    source generator policy separately. Both use the same pinned reviser core.
    """

    actual = str(
        run_config.get("revision_core_policy_version")
        or run_config.get("card_core_policy_version")
        or "missing"
    )
    if actual == REVISION_CORE_POLICY_VERSION:
        return actual
    if os.environ.get("GENERALIZED_CARD_ALLOW_LINEAGE_MISMATCH") == "1":
        return actual
    raise RuntimeError(
        f"Cannot {operation}: revision policy is {actual!r}, but this code "
        f"requires {REVISION_CORE_POLICY_VERSION!r}. Initialize an audited "
        "revision workspace rather than relabeling an older generator run."
    )


def upgrade_revision_policy_config(
    run_config: dict[str, object],
) -> dict[str, object]:
    """Upgrade reviser lineage while retaining the prior policy record."""

    actual = str(
        run_config.get("revision_core_policy_version")
        or run_config.get("card_core_policy_version")
        or "missing"
    )
    if actual == REVISION_CORE_POLICY_VERSION:
        return dict(run_config)
    if actual not in HISTORICAL_REVISION_POLICY_VERSIONS:
        raise RuntimeError(
            f"Cannot upgrade unknown revision policy {actual!r}; expected one of "
            f"{sorted(HISTORICAL_REVISION_POLICY_VERSIONS)!r}."
        )
    upgraded = dict(run_config)
    history = [
        str(value)
        for value in (upgraded.get("revision_policy_history") or [])
        if str(value)
    ]
    if actual not in history:
        history.append(actual)
    upgraded["revision_policy_history"] = history
    upgraded["revision_core_policy_version"] = REVISION_CORE_POLICY_VERSION
    return upgraded


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
