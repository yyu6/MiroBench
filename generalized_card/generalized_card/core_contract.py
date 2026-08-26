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
    "generalized-card-v2-drawn-parenthetical-count-v116-20260826"
)
HISTORICAL_GENERATION_POLICY_VERSIONS = {
    "generalized-v2": {
        # v115 shipped the inverted tone quota but still cued the literal word
        # "one" for every parenthetical aside, at every band. Never ran.
        "generalized-card-v2-inverted-tone-quota-v115-20260826",
        # v114 corrected v113's URL reader but never ran; the tone quota it
        # rendered was still the template's own rates, so the Writer's 0.854
        # impolite realization carried the output to 0.607 against real's 0.464.
        "generalized-card-v2-drawn-reference-link-v114-20260826",
        # v113 shipped the drawn link with a `\S+` URL reader that swallowed
        # Reddit's `[url](url)` markdown. Its N=10 gate is a paid artifact, so
        # the string is burned; v114 carries the corrected reader.
        "generalized-card-v2-drawn-reference-link-v113-20260825",
        "generalized-card-v2-development-scope-v112-20260825",
        # v111 shipped the `--development-scope` arm with only three of the four
        # capacity gates wired to it; the Planner's prose rule kept a hard-coded
        # 100. The N=50 paper run consumed the v111 string with the arm off, so
        # the string is burned and v112 carries the completed mechanism.
        "generalized-card-v2-development-scope-v111-20260825",
        "generalized-card-v2-refit-length-transfer-v110-20260824",
        "generalized-card-v2-entity-referent-spread-v109-20260824",
        "generalized-card-v2-semantic-coverage-nonrepeat-v108-20260823",
        "generalized-card-v2-verdict-close-check-guard-v107-20260822",
        "generalized-card-v2-digit-cue-quantifier-guard-v106-20260822",
        "generalized-card-v2-chain-scoped-reply-novelty-v105-20260822",
        "generalized-card-v2-evaluative-register-v104-20260821",
        "generalized-card-v2-stance-consistent-opening-v103-20260821",
        "generalized-card-v2-drawn-opening-move-v102-20260820",
        "generalized-card-v2-per-register-realization-v101-20260820",
        "generalized-card-v2-measured-closing-move-v100-20260820",
        "generalized-card-v2-drawn-register-realization-v99-20260820",
        "generalized-card-v2-drawn-typing-rhythm-length-calibration-v98-20260819",
        "generalized-card-v2-keyboard-surface-measured-joints-v97-20260819",
        "generalized-card-v2-selective-facts-ancestor-novelty-v96-20260818",
        "generalized-card-v2-nonfatal-compiled-plan-contract-v95-20260818",
        "generalized-card-v2-state-preserving-plan-repair-v94-20260818",
        "generalized-card-v2-root-reply-boundary-v93-20260818",
        "generalized-card-v2-lossless-domain-claim-off-v92-20260817",
        "generalized-card-v2-slot-gated-fact-license-v91-20260817",
        "generalized-card-v2-reply-story-grounding-v90-20260817",
        "generalized-card-v2-realizability-first-planner-v89-20260817",
        "generalized-card-v2-structural-speakers-grounding-v88-20260817",
        "generalized-card-v2-payload-safe-writer-routing-v87-20260817",
        "generalized-card-v2-root-relation-prompt-v86-20260817",
        "generalized-card-v2-auditable-plan-controls-v85-20260817",
        "generalized-card-v2-complete-writer-coverage-v84-20260817",
        "generalized-card-v2-matched-text-semantic-isolation-v83-20260817",
        "generalized-card-v2-focused-discourse-contract-v82-20260817",
        "generalized-card-v2-joint-story-affect-handoff-v81-20260817",
        "generalized-card-v2-planner-contract-coherence-v80-20260816",
        "generalized-card-v2-own-fact-license-v76-20260815",
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
        "generalized-card-v2-affirmative-affect-uncapped-slot-shape-v73-20260814",
        "generalized-card-v2-focused-writer-prompt-v74-20260814",
        "generalized-card-v2-writer-realizes-planner-move-v75-20260814",
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
        "6244cd391cf98bb915ebd4c4c5649a08b6917409274f22544e7e4bce662e9327",
    ),
    "engine_vocabulary": (
        "scripts/sampling_generator/engine/vocabulary.py",
        "0d9ef802413191975214eefeb117e6415b1f7b3876e86f6f0b3745d4815cead8",
    ),
    "engine_model": (
        "scripts/sampling_generator/engine/model.py",
        "b9782b7e3799a7e06bba8f99569261da89fb4b307c1672e2f6015bdbe84007e6",
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
        "9413920ec2f9d5454e80d0e02a591d755e08c6a5b44478a19861322a202d0cb5",
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
        "58284ce7c1e70b3169c47ef7ec0bb2ab5faa547cea5b2c936729271c165eb5e8",
    ),
    "data_boundary": (
        "generalized_card/generalized_card/data.py",
        "55510f6b3da3b81a2a21654d39f910947e241b3168eee54c042ecd88fa49145e",
    ),
    "domain_config": (
        "generalized_card/generalized_card/domain.py",
        "f41fb60aa316701249f2f795473fc8a53524206c39874a52e78f2c5355de185f",
    ),
    # The grounding rules were smeared across eight places in the prompt adapter
    # before v76 and disagreed with each other. Pinned so the one definition
    # cannot drift while every other pin still passes.
    # The thread's participation structure. Pinned for the same reason
    # `writer_grounding` is: it decides who is speaking, and a silent change
    # there would be invisible to every other pin.
    "speaker_roster": (
        "generalized_card/generalized_card/speaker_roster.py",
        "06ee9c36ebd4d8b489d1e9ea7a0c265873507ce47a0a6be0db3753f792c2ca0f",
    ),
    "writer_grounding": (
        "generalized_card/generalized_card/writer_grounding.py",
        "c6e9349e8f85269ef3e70a360de7df0945f706f30a62e60421319e8601fae2c9",
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
        "2bf3d51998cf96a756e0e5d65ce2044fb6fc1cfd6be1b7da4f3d439a16ea67fb",
    ),
    "generation_backend_runner": (
        "generalized_card/scripts/run_generator_backend.py",
        "49c60fcff99c7fb706d674b8f74c19a9dd43a5929a1bcff62970e3e53c21009d",
    ),
    "token_usage_tracker": (
        "scripts/token_usage_tracker.py",
        "296675ce356ad102de410a9187fa9c6ccd145570fdfd06ccc0382ec542f3ede7",
    ),
    "token_usage_summarizer": (
        "scripts/summarize_token_usage.py",
        "83c5edf0be01d7c6b5874a5faf2207af630f189c71a90c61a1dd51b14c873cef",
    ),
    "domain_profile": (
        "generalized_card/generalized_card/domain_profile.py",
        "aac42cf80b4bd1aef65a7fa155b68ee30c63bbb0f65f486638548d3a1e2985f7",
    ),
    "viewpoint_bank": (
        "generalized_card/generalized_card/viewpoint_bank.py",
        "27ea5d14192819fb495365c495db14e2d5d72be44240fb6bc75d3a895feb3027",
    ),
    "planning_quality": (
        "generalized_card/generalized_card/planning_quality.py",
        "029f96b6b6269ed6e1223ce828b1a20dfddaeaebcd26e895370faf163553bddd",
    ),
    "planner_contract": (
        "generalized_card/generalized_card/planner_contract.py",
        "6b657328a39545c7718e5fa0993702662c4a33e6bfec4c3ecf6c15821e5e307d",
    ),
    "plan_repair": (
        "generalized_card/generalized_card/plan_repair.py",
        "b87f62ac84df604b9d3e39613570176996b87238fdafc2b3b33ef1cabe0516ed",
    ),
    "generation_distribution": (
        "generalized_card/generalized_card/generation_distribution.py",
        "32a781b9236420d8fcb6683ac45c468e72127c600e6c96dacbfa21c5ab670ab7",
    ),
    "planner_distribution": (
        "generalized_card/generalized_card/planner_distribution.py",
        "8d8422c9ac0d1b28110ff89a28748d7bab9d9e72469c7c88593470c0c65a87dd",
    ),
    "branch_routing": (
        "generalized_card/generalized_card/branch_routing.py",
        "fea4e44a5a751a954f5fe33f43d6dbd0c6a28d4fecbc150e85c1cf9961a8f80a",
    ),
    "task_distribution": (
        "generalized_card/generalized_card/task_distribution.py",
        "0aa93173dd410670861cc65bb3028232b88987a3d2215073ec3e4fb4e6268250",
    ),
    "first_pass_policy": (
        "generalized_card/generalized_card/first_pass_policy.py",
        "5d29ed1e2fd6e48535f25d780c5854f792bbb4d09001dd8643535d5e877e4953",
    ),
    "domain_claim": (
        "generalized_card/generalized_card/domain_claim.py",
        "a75e16fb676a29a0276ded6d062d09a96ee52c38a2108d2b3da21cded7c5ec08",
    ),
    "opener_profile": (
        "generalized_card/generalized_card/opener_profile.py",
        "72914369ac04485a04a07712c13d2da78316d20bfd8a7d7b3af79a2e5000fb34",
    ),
    "entity_inventory": (
        "generalized_card/generalized_card/entity_inventory.py",
        "0157a4d67d2b21123848996818a2d9b76a46cbe09af13017905c617be9e50cff",
    ),
    "tone_realization": (
        "generalized_card/generalized_card/tone_realization.py",
        "9992076312c3ebe890bed6204ef25e14437c306e02697c5b077b7f1216be1609",
    ),
    "reference_link": (
        "generalized_card/generalized_card/reference_link.py",
        "25e92abb81e2fcda4828e7b78c99b4528f437881f9b6125812fd3687d29fde7f",
    ),
    "entity_spread": (
        "generalized_card/generalized_card/entity_spread.py",
        "fb376f912033d80bf8a7a0b01545f23259f1fed76f09ff540c3687cc5ebadefe",
    ),
    "length_fidelity": (
        "generalized_card/generalized_card/length_fidelity.py",
        "c975e10bbca23aef2a5fb1fed3121a67ed59852f4fcb162d5c6f03161f65f829",
    ),
    "lexical_quality": (
        "generalized_card/generalized_card/lexical_quality.py",
        "748ddea8a3042ceb09364089cf406075fe9c13abafd6886ad3a99a57f4e1b69e",
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
        "70bb0cbd226eb9135e4ec0699fb86f475d576bec5f8447e1a669b367b2d04e8c",
    ),
    "semantic_realization": (
        "generalized_card/generalized_card/semantic_realization.py",
        "2fc67ef1806fe5bf7524bb9ea431e74bc93a8378f036e7282a6b1fe239ba2c40",
    ),
    "length_policy": (
        "generalized_card/generalized_card/length_policy.py",
        "614298243867ae8ed01a5bfd5e5776d36e0b27fb69e461411616697d3990c66e",
    ),
    "long_form_planning": (
        "generalized_card/generalized_card/long_form_planning.py",
        "2b79463557435223b8dc9442559cf65f8feca71aebe7629b20e58cf22e909f5e",
    ),
    "tone_length_fit": (
        "generalized_card/generalized_card/tone_length_fit.py",
        "b816e9f033c890c99126b0b1e8c6550028d516c7eec179956726415ad6af10d1",
    ),
    "comment_structure": (
        "generalized_card/generalized_card/comment_structure.py",
        "a3887dd9b820ec2146652ad46e2dcd5802aed39442b99372ed3502ba466fa8e7",
    ),
    "surface_typography": (
        "generalized_card/generalized_card/surface_typography.py",
        "a329aeaa6ccaed2e12cab7df63f084c140b86aaec3baad8efac0bb1897a90653",
    ),
    "sentence_rhythm": (
        "generalized_card/generalized_card/sentence_rhythm.py",
        "e3955e85843cb55e55b2f149964ca4b26d88cd6e0b213fff476ea7280189e344",
    ),
    "story_scope": (
        "generalized_card/generalized_card/story_scope.py",
        "5e53af3d730da4a781b31ea8a7a5c53e188b7ee791306d18fbf1044c5596e22d",
    ),
    "length_calibration": (
        "generalized_card/generalized_card/length_calibration.py",
        "29199a8caeafeada14e94586910c32e29d87ac5397486c3ee75d54f175a70e7f",
    ),
    "closing_move": (
        "generalized_card/generalized_card/closing_move.py",
        "c3c39bfe3be458f1e1bf353c1a232c020f8b755de8023f5bd961aef30cca8554",
    ),
    "evaluative_register": (
        "generalized_card/generalized_card/evaluative_register.py",
        "cf3d97f92cdaa2f7b81830e847dcc21130e2285e6fa38d393d6964b1befb090f",
    ),
    "opening_move": (
        "generalized_card/generalized_card/opening_move.py",
        "1a7f5696812df6e1362ebc6e63146728f24768ef0b9413042fd102e58b1b5fee",
    ),
    "register_realization": (
        "generalized_card/generalized_card/register_realization.py",
        "bfcf5e4d4b76486d5c9016d5b54e2f99f6e3facb2f322d4e5a63f9d24c4e8dce",
    ),
    "source_provenance": (
        "generalized_card/generalized_card/source_provenance.py",
        "7e4df9acd04652a874930e689d8394025b6e02ad1e6dfb75720499825106726a",
    ),
    "surface_contract": (
        "generalized_card/generalized_card/surface_contract.py",
        "2250c7013a8719969b16f5bf91c18b79a535b9c4d305ac2dd25a1e7fa0dd349e",
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
        "de61458b30395d716a280eac41d5f7f571b4424f548266224389c236051e267f",
    ),
    "legacy_reviser_prompts": (
        "generalized_card/generalized_card/legacy_reviser_prompts.py",
        "6e84767ff10ebf8125d916097c755bcfb8107fa92c4092cbd9056eaa6b31be28",
    ),
    "domain_prompt_adapter": (
        "generalized_card/generalized_card/prompts.py",
        "3173cf5744b854d5d014796f54ae9cb2453047556543b44a3ded37815eb3fb62",
    ),
    "reply_planning": (
        "generalized_card/generalized_card/reply_planning.py",
        "8e2b4382ad6d146f10a8d6665555b23fa68c8bfb9e55eda6f5fa986f28b09719",
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
        "c3dc47398bfa0aa352ba816bed648863da033a301e28f9ae1a5c0247d8c84336",
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
        "244f60adc063e16eec308edbb7fc296c315cef28a9a9396118caa39ac8c2862c",
    ),
    "evaluation_runner": (
        "generalized_card/scripts/run_evaluate.py",
        "40f17c008679e68565559004d8a9e8706b07a8b02eea91cb5f55caf7c38b490f",
    ),
    "output_audit_runner": (
        "generalized_card/scripts/audit_output.py",
        "1955cbf8e4fd057c6f6ada96f4e456a773eaf928cc3051fd374546302ab51335",
    ),
    "content_profile": (
        "generalized_card/generalized_card/content_profile.py",
        "ab13b1b60ebbb7f32736cdf0b6042d7cdeaab5b95b556382d3c806c4b6832df7",
    ),
    "content_profile_analysis": (
        "generalized_card/generalized_card/content_profile_analysis.py",
        "64bf133cb449c3eb012fc6a57a1374ff4cb07ce96db904a79e61144c2d5ca5ef",
    ),
    "content_profile_data": (
        "generalized_card/generalized_card/content_profile_data.py",
        "f44b50627e276ec3ee688cfd9bba1f67c94998fa982b031072169382ec3556c3",
    ),
    "distribution_stats": (
        "generalized_card/generalized_card/distribution_stats.py",
        "e71f5843538cd6be757f44104814fbbf1e3666d34e20ac0aaf11a8b1627dc47a",
    ),
    "thread_metric_suite": (
        "generalized_card/generalized_card/thread_metric_suite.py",
        "0032e510a64cd0082cd643483028908100730d5237d34667124e1c9092a08897",
    ),
    "content_profile_runner": (
        "generalized_card/scripts/compare_content_profile.py",
        "560140c1dfdf68efb658045325bdf62bd4a67b55e39fed4443d12ae338daf709",
    ),
    "parity_auditor": (
        "generalized_card/scripts/audit_core_parity.py",
        "e768e96ec8dc2487018d02ee247fd2a81ee00ccefa708319377e4301674768fc",
    ),
    "score_runner": (
        "scripts/evaluation/score_sampled_generated_runs.py",
        "710625298bbd556afcfc3109f86ef2eba49025a11fc188212b3640bf3503cfe7",
    ),
    "score_disagreement": (
        "scripts/evaluation/score_thread_disagreement.py",
        "769c16db84fbf8a71e11a8ae80d5539fc136e527c06aad11f69fa54f5f5e70bf",
    ),
    "score_self_bleu": (
        "scripts/evaluation/score_thread_self_bleu.py",
        "537399a160ae841ec5ab203465b9e41f698a166523f8313515e159199ba1c1a0",
    ),
    "score_self_bertscore": (
        "scripts/evaluation/score_thread_self_bertscore.py",
        "fd8d6c3d53a1de1577216aacccd7e316f85932a5e1e2457943cc0b05a4f393f8",
    ),
    "score_semantic_uniformity": (
        "scripts/evaluation/score_thread_semantic_uniformity.py",
        "2c2edd07d259adbbdaeebef60e598faa0f089b92329d4783d38a397797b25923",
    ),
    "score_storyseeker": (
        "scripts/evaluation/score_thread_storyseeker.py",
        "76b014cd9d1199555a0cc6e6c44964b46aba02176857a23516c09e84cfa0a5fe",
    ),
    "score_go_emotions": (
        "scripts/evaluation/score_thread_go_emotions.py",
        "3c38d050d58e0fbf1d8b9d84a4c9c490fbbd0b02f759915c22c2e579656191da",
    ),
    "score_politeness": (
        "scripts/evaluation/score_thread_politeness.py",
        "d527a42abdcee750396b60dd1f45025ff80f5e397021af64e3d81ea05f3eadd2",
    ),
    "score_structure": (
        "scripts/evaluation/score_thread_structure.py",
        "8ad909d9168c3966e9dc67ae6abdf7da010687d2f7e63e978bc6afa0e37386d2",
    ),
    "score_detoxify": (
        "scripts/evaluation/score_thread_detoxify.py",
        "6d51c938579ff76cc4440257aa58f98e6519c64197bcf73933661734f9642fc3",
    ),
    "score_summarizer": (
        "scripts/evaluation/summarize_thread_metrics.py",
        "4fa192d72718f4cdafa1f8d499eff2eb390946673cc9f54531b3177f3d290490",
    ),
    "matched_evaluator": (
        "scripts/evaluate_matched_seed_group.py",
        "325de84fc2c37f8f5d7156bd6b00ea13be4b9deb38c12fb8f707d672d53c4122",
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

GENERATION_ADAPTER_CORE_NAMES = (
    "generator_adapter",
    "data_boundary",
    "domain_config",
    "planner_schema",
    "generation_runner",
    "domain_profile",
    "viewpoint_bank",
    "planning_quality",
    "planner_contract",
    "plan_repair",
    "generation_distribution",
    "planner_distribution",
    "branch_routing",
    "task_distribution",
    "first_pass_policy",
    "lexical_quality",
    "reference_metric_calibration",
    "generation_diversity",
    "writer_quality",
    "semantic_realization",
    "length_policy",
    "long_form_planning",
    "surface_contract",
    "surface_typography",
    "sentence_rhythm",
    "story_scope",
    "length_calibration",
    "register_realization",
    "closing_move",
    "opening_move",
    "evaluative_register",
    "source_provenance",
    "comment_structure",
    "tone_length_fit",
    "domain_claim",
    "opener_profile",
    "entity_inventory",
    "entity_spread",
    "reference_link",
    "tone_realization",
    "length_fidelity",
    "domain_prompt_adapter",
    "writer_grounding",
    "speaker_roster",
    "reply_planning",
    "persona_bridge",
    "actor_conditioning",
)

CURRENT_GENERATION_CORE_NAMES = (
    *GENERALIZED_V2_ENGINE_FILES,
    *GENERATION_ADAPTER_CORE_NAMES,
    "generation_backend_runner",
    "token_usage_tracker",
    "token_usage_summarizer",
)

CURRENT_EVALUATION_CORE_NAMES = (
    "data_boundary",
    "domain_config",
    "output_audit",
    "output_audit_runner",
    "evaluation_runner",
    "content_profile",
    "content_profile_analysis",
    "content_profile_data",
    "distribution_stats",
    "content_profile_runner",
    "thread_metric_suite",
    "score_runner",
    "score_disagreement",
    "score_self_bleu",
    "score_self_bertscore",
    "score_semantic_uniformity",
    "score_storyseeker",
    "score_go_emotions",
    "score_politeness",
    "score_structure",
    "score_detoxify",
    "score_summarizer",
    "matched_evaluator",
    "token_usage_tracker",
    "token_usage_summarizer",
)

CURRENT_ACTIVE_CORE_NAMES = tuple(
    dict.fromkeys((*CURRENT_GENERATION_CORE_NAMES, *CURRENT_EVALUATION_CORE_NAMES))
)


CONTRACT_RELATIVE_PATH = "generalized_card/generalized_card/core_contract.py"


def version_source_paths(names: Iterable[str]) -> list[str]:
    """Return every file whose content defines this generator version.

    The pinned sources for `names`, plus this contract. `verify_core_contract`
    cannot check the contract -- a file cannot carry its own hash -- but it names
    the policy version and holds every other pin, so a version whose contract is
    uncommitted is not recoverable even when all of its modules are.
    """

    return [*(CORE_FILES[name][0] for name in names), CONTRACT_RELATIVE_PATH]


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
    if allow_historical and actual in HISTORICAL_GENERATION_POLICY_VERSIONS.get(
        profile, set()
    ):
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
