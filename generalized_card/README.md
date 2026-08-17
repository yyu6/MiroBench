# Generalized CARD

This directory is a domain-configured Planner-Writer-Reviser implementation of CARD. The default `generalized-v2` generator is the domain-neutral Planner-Writer used by the stronger camera run. The historical July 4 CARD snapshot remains available as the explicit `card-snapshot` profile for audits; it is not the default because that older source contains credit-card-specific generation assumptions. Both profiles are isolated from the original credit-card outputs.

## Design

1. **Matched seed construction.** A deterministic, stratified sampler selects real Reddit posts from a product domain. Each seed retains its real post ID and product directory for exact metric matching.
2. **Planner.** The common thread planner and comment-move planner produce structural, behavioral, and local discourse controls. To preserve CARD's semantic-abstraction path without exposing evaluation answers, a frozen bank of real comments is built only from threads outside the full seed pool. For each seed, a domain-neutral retriever selects relevant, source-diverse reference comments. The private Planner abstracts their viewpoint and discourse patterns into slot-level `semantic_move`, `local_topic`, `claim_key`, stance, evidence, reply controls, and a concrete one-use sentence route. A fixed bank of 12 cross-domain decision lenses keeps a perspective distinct from a topic or entity. Matched seed comments still provide structure and surface labels only.
3. **Plan-quality audit.** The normalized Planner output retains its non-test `reference_id`, and output IDs are restricted to the global slots requested by the current batch. Every batch is checked against all prior batches for repeated reference use, claim reuse, invalid decision lenses, perspective concentration, token-level semantic overlap, paraphrased plan overlap measured with a local normalized sentence-embedding model, and payloads that would collapse an anonymous ordinary or long slot into a short whole-comment mode. Enum-only mistakes such as returning `B3` as a decision lens are deterministically changed to the schema fallback `seed_local` and recorded. The control surface plans 24 slots together and permits three bounded, slot-local repairs before any Writer call. Unresolved soft diagnostics are retained in the audit. If the Planner omits a structural slot, that slot is recorded and omitted; the system never fabricates a generic semantic plan to force the original count. Plan diagnostics are recorded in `logs/planning_quality.jsonl`.
4. **Optional domain-derived actor conditioning.** When explicitly enabled, the comment Planner composes an actor state in the same response as each semantic move. It derives that state from the current visible seed/parent context and an evaluation-excluded same-domain reference pattern. The state describes a local knowledge boundary, participation goal, evidence access, attention focus, interaction tendency, context aperture, and realization route. It is not selected from a fixed participant catalog. The default V12-style path disables this extra schema so actor fields cannot increase Planner truncation or omission risk.
5. **Writer.** The common Writer receives a seed or generated parent, one local task, a bounded set of task-relevant factual anchors, and CARD's structured thread blackboard. It never receives matched-real or non-test reference-comment text. Long prior generated comments are represented by semantic controls rather than replayed verbatim; exact short utterances remain visible only as exclusions. The Writer realizes the semantic plan once. Self-BLEU, semantic-cosine, opener, and phrase-family diagnostics are recorded against the current-thread ledger, but they do not trigger candidate resampling or best-of-N selection. Only empty text, exact duplication, parent copying, placeholders, or exposed Planner controls invoke bounded same-slot recovery. The provider output ceiling expands for anonymous long-tail slots, but length remains a soft cue rather than an acceptance gate. Valid comments are persisted even when another slot is omitted.
6. **Health gate.** Empty threads, low writer acceptance, placeholders, refusals, prompt leakage, exact duplicates, semantic-plan collisions, perspective concentration, substantive-slot surface conflicts, long-tail length compression, suspicious matched-real overlap, and suspicious non-test reference-comment overlap are audited before evaluation.
7. **Reviser.** LLMs propose lexical, tone, or story candidates. Deterministic metric and preservation gates choose candidates. Reply-tree structure is revised without changing text. Full-collection controllers accept an improved round or roll back to the most recent accepted collection.
8. **Controller memory.** Every round keeps an immutable history row and a deterministic structured memory snapshot. The snapshot records the distributional direction, strategy and parameters, candidate/thread outcomes, rejection reasons, exact target-metric changes, protected-metric regressions, and accept/rollback decision. The next round avoids strategies already rejected for the same accepted input and receives a bounded prompt summary of prior failures and successful candidate styles. Memory never bypasses local gates or full-collection acceptance.

The structural sampler, per-comment semantic abstraction, task balancing, context apertures and transformations, claim/opening budgets, low-information writer path, writer guards, candidate metrics, full rescoring, accepted-round chaining, rollback, and resume behavior are shared across domains. Context dropout and jitter are fixed CARD controls at `0.42` and `0.32`; they are stored in the frozen domain profile for reproducibility but are not estimated from domain text. Changing either value creates a different configured experiment. Domain-specific data paths, entity dictionaries, community labels, topic facets, and factual guards are configuration boundaries rather than algorithm branches; the decision-lens bank and plan-quality algorithm are shared unchanged.

The default `generalized-v2` control surface keeps CARD's structural and
distribution controls across domains: comment-planner batch size `24`, zero
LLM post-generation metric repair rounds, three bounded pre-Writer plan repairs, Planner API retries `2`, comment-planner token cap `18000`, and a Writer provider ceiling of at least `260` tokens that expands only for anonymous slots above 100 words, and
context dropout/jitter `0.42/0.32`. Domain-derived actor conditioning is enabled
only when requested; the default is `none`. Writer distribution retries, local
repair rounds, and slot retries default to zero; hard same-slot recovery is
bounded to two attempts and whole-post regeneration is disabled. Transport and
provider retries remain separate from distribution optimization. Every value is
recorded in `run_config.json`.

## Domain-Derived Actor Conditioning

`--actor-conditioning domain-derived` is an optional ablation. The
actor is composed by the comment Planner from two leakage-safe inputs: the
visible seed or generated parent context, and behavioral abstractions retrieved
from same-domain real threads excluded from the complete evaluation seed pool.
The resulting actor is local to a discussion turn. It specifies how that turn
knows, attends, reacts, and speaks without asserting a persistent human identity.

This design intentionally has no camera participant list, finance participant
list, or universal demographic persona bank. New domains use their own
evaluation-excluded real discussions through the same extraction and planning
path. CARD's domain-neutral role, tone, payload, stance, and decision-lens
vocabularies remain structured controls; they are not participant identities.

When enabled, actor state and provenance are persisted with every generated comment and
in the run configuration. Distribution diagnostics remain available for audit
and final matched evaluation, but no p-value or metric pass is guaranteed by
generation-time heuristics.

## MatrAIx Persona Conditioning

MatrAIx-Persona-8B is persona data and agent infrastructure, not an 8B-parameter
text-generation model. Generalized CARD uses the official MatrAIx YAML loader,
dimension catalog, Jinja renderer, and `persona_system.md.j2` identity channel.
GPT-5.4-mini remains the Planner and Writer model. CARD still owns the thread
plan, reply tree, per-comment role/tone/payload controls, visible context,
blackboard, distribution pressure, writer guards, and factual constraints.

Clone and pin the audited upstream source once:

```bash
git clone https://github.com/MatrAIx-ai/MatrAIx-Persona-8B \
  third_party/MatrAIx-Persona-8B
git -C third_party/MatrAIx-Persona-8B checkout \
  e85c8772fc8a769ff70662c5368066024b6e15b8
```

`--persona-conditioning matraix-projected` is an optional, separate experimental mode. It
selects communication, decision, affect, skill, and optional configured domain
expertise dimensions, then gives that projection to the official MatrAIx system
renderer. Names, demographics, locations, and life histories are excluded so
persona changes realization rather than injecting facts. The complete official
profile is available as `matraix-full` for short diagnostic runs. In the audited
200-persona development sample, full system prompts can exceed 20,000
characters, so `matraix-full` is not the primary 70-thread condition.

Persona assignment is deterministic from persona seed, seed-post index,
comment slot, and CARD's planned social role. The Planner never sees persona.
The Writer receives persona identity in a dedicated system message and the CARD
task in the user message, matching MatrAIx's documented identity/task boundary.
Every generated comment records persona ID, upstream commit, system hash,
selected dimensions, and system length. The run-level assignment manifest is
`persona_assignment_manifest.json`; run configuration, token cost, elapsed time,
interrupt, and resume behavior use the normal CARD files.

MatrAIx conditioning cannot be combined with `domain-derived` actor
conditioning in one run. Use `--actor-conditioning none` for a MatrAIx
ablation. The default V12-style generalized method uses neither additional
conditioning layer.

The experimental protocol and interpretation limits are documented in
[`MATRAIX_PERSONA_EXPERIMENT.md`](MATRAIX_PERSONA_EXPERIMENT.md).

`generalized_card/core_contract.py` separately pins the generator sources, CARD controllers and revisers, generalized controller entry points, prompt adapter, backend wrappers, and extension controllers. At runtime, the adapter checks every top-level generator and reviser function: only functions named in the domain-boundary allowlist may be replaced. Generation or revision stops before an API call if either the shared source or its adapter drifts. Generator and revision policies are recorded separately so an older generation can be revised without being relabeled as a newer generator.
The algorithm/domain boundary is listed in [`CARD_CORE_PARITY.md`](CARD_CORE_PARITY.md).

Audit the complete boundary without making an API call:

```bash
PYTHONPATH=generalized_card \
python3 generalized_card/scripts/audit_core_parity.py \
  --domain camera \
  --output artifacts/generalized_card/parity_camera.json
```

## Included Domains

- `camera`: 574 unique real discussions; 441 have at least five usable comments.
- `cell_phone`: 258 unique real discussions; 201 have at least five usable comments.
- `headphone`: 240 unique real discussions; 177 have at least five usable comments.
- `laptop`: 276 unique real discussions; 185 have at least five usable comments.

Domain files are under `configs/domains/`. A new domain requires raw post/comment paths, a precomputed real metric CSV, topic facets, technical terms, and protected entities.

## First GPT-5.4-mini Test

Run five camera discussions first:

```bash
OPENAI_API_KEY="$OPENAI_API_KEY" \
python3 generalized_card/scripts/run_generate.py \
  --domain camera \
  --model gpt-5.4-mini \
  --base-url https://api.openai.com/v1 \
  --tag generalized_card_camera_gpt54_smoke5_v1 \
  --pool-size 150 \
  --max-posts 5 \
  --posts-per-run 1 \
  --actor-conditioning none \
  --persona-conditioning none \
  --writer-retries 0 \
  --post-retry-limit 1 \
  --resume
```

`--generator-profile generalized-v2` is the default. Use
`--generator-profile card-snapshot` only for an intentional historical-source
audit; it creates a distinct generator lineage.

The same command is safe after `Ctrl-C`. Completed post slots are skipped,
token records are appended, and elapsed time is accumulated in the same run
directory. API transport and malformed structured Planner responses have finite
retry budgets. `--post-retry-limit` counts total whole-post attempts; its
default and minimum are `1`, which disables automatic whole-post regeneration.
For matched generation, the corresponding real comment's rough amount of detail
and pacing is a soft surface reference. `micro`, `short`, `medium`, and related
labels remain sampling and audit metadata; they do not define word-count ranges,
acceptance gates, or retry conditions. Length deviations are logged for later
distribution evaluation. `--writer-max-tokens` is only a bucket-independent API
safety ceiling against abnormal output. Other hard Writer failures use bounded
slot-local handling before the post is rejected. Soft lexical, semantic, and
length diagnostics never trigger Writer or post regeneration. Permanent
configuration, authentication, and billing errors stop immediately. A resume
attempt with changed generation parameters is rejected; use a new tag for a
different configuration.

When domain-derived actor mode is explicitly enabled, surface-style diagnostics such as an
unexpected first-person frame, question form, generic phrasing, missing optional
anchor, or Markdown quote shape are also retained and logged after the single
Writer realization. They do not invoke a hidden fallback call. Empty output,
exact copying, placeholder literals, and exposed Planner controls remain hard
failures.

Inspect content before spending on 150 threads:

```bash
python3 generalized_card/scripts/audit_output.py \
  artifacts/generalized_card/runs/generalized_card_camera_gpt54_smoke5_v1/generated \
  --domain camera \
  --seed-pool artifacts/generalized_card/seed_pools/camera_product_150_seed42.json \
  --output artifacts/generalized_card/runs/generalized_card_camera_gpt54_smoke5_v1/output_audit.json \
  --require-healthy
```

For 150 threads, use a new tag and change `--max-posts 150 --posts-per-run 5`.

To append a previously completed, contiguous prefix without regenerating it,
reuse the same tag and add `--extend-existing`. The runner requires every
setting other than `--max-posts` to match, records a seed-range policy lineage,
preserves token cost and elapsed time, and invalidates the old evaluation
pointer until the expanded run is evaluated again:

```bash
OPENAI_API_KEY="$OPENAI_API_KEY" \
python3 generalized_card/scripts/run_generate.py \
  --domain camera \
  --model gpt-5.4-mini \
  --base-url https://api.openai.com/v1 \
  --tag generalized_card_camera_gpt54_pw70_v6_20260807_v1 \
  --pool-size 150 \
  --max-posts 150 \
  --posts-per-run 5 \
  --reasoning-effort low \
  --extend-existing \
  --resume
```

## Evaluation

Run the integrity audit, all thread metrics, and exact matched-seed evaluation:

```bash
python3 generalized_card/scripts/run_evaluate.py \
  --tag generalized_card_camera_gpt54_smoke5_v1 \
  --metric-parallel 5 \
  --resume
```

The evaluation audit blocks incomplete, contaminated, copied, or leaked output.
It then stages the generated tree byte-for-byte; it does not clean, rewrite,
delete, or normalize Writer text before scoring.
Planner-distribution findings such as semantic collisions or perspective
concentration are written to `output_audit.json` and reported as warnings, then
evaluation continues so those quality failures remain measurable. After the 12
formal metrics and matched-seed statistics, the same command writes
`content_profile_audit.json` and `.md`. That report decomposes every metric as
matched real → selected evaluation-excluded Planner template → generated output,
then reports Planner→Writer tone/affect/story realization, repetition, and
matched content examples. Lexical customer-service/profanity probes remain weak
diagnostics. With one thread, both the evaluator and report label every test
`DESCRIPTIVE`; only N>1 can receive PASS/PARTIAL/FAIL.

The current research workflow ends at first-pass Planner→Writer evaluation.
Historical reviser entry points remain only for reproducing old artifacts and
are excluded from the default active parity audit. They are not a recommended
way to make a failed generation look matched.

## Output Layout

```text
artifacts/generalized_card/runs/<tag>/
  run_config.json
  run_state.json
  generated/
  output_audit.json
  cleaned/                  byte-identical scoring snapshot of generated/
  evaluation/
  matched_evaluation/
  current_artifact.json
  logs/token_usage.jsonl
  logs/token_usage_summary.json
```

No API key is written to configuration or logs.
