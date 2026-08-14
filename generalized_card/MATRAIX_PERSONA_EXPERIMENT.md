# MatrAIx Persona Experiment

## Research Question

Test whether independent commenter identities improve the human-like
distribution of generated Reddit discussions while CARD continues to control
content, structure, discourse function, tone targets, and factual grounding.
The intervention targets surface and behavioral heterogeneity. It is not
expected to repair every metric and it does not guarantee statistical
non-significance.

## Upstream System Reuse

The integration is pinned to MatrAIx-Persona-8B commit
`e85c8772fc8a769ff70662c5368066024b6e15b8`. It directly imports the official:

- persona YAML loader and `Persona` object;
- dimension catalog and narrative construction;
- Jinja rendering runtime;
- `persona_system.md.j2` identity template.

The Harbor user-simulation loop is intentionally not copied into CARD. That
loop models one user interacting with an application over multiple turns. A
Reddit thread requires many independent authors attached to a preplanned reply
tree. Reusing the Harbor conversation loop would change the CARD algorithm and
confound persona identity with thread planning.

## Conditions

Use the same 70 camera seed indices, sampling seed, domain profile, Planner,
Writer model, and generation parameters in both conditions.

1. **CARD control:** `--persona-conditioning none`.
2. **MatrAIx projected:** `--persona-conditioning matraix-projected`.

The projected condition retains at most ten causal behavior dimensions per
persona. It excludes names, demographics, locations, and biographical facts.
The resulting projection is still rendered with MatrAIx's official system
template. `matraix-full` should first be limited to 5-10 diagnostic threads
because full identities are much longer and therefore test both identity and
context length.

## Assignment And Isolation

The comment-move Planner runs before persona conditioning. After CARD fixes a
comment's role, tone, payload, stance, story mode, local semantic move, parent,
and length bucket, a deterministic role-compatible selector assigns one eligible
persona. The identity is added to the Writer's system channel. The seed post,
visible parent, CARD task, and blackboard remain the only factual sources.

This ordering prevents persona from changing target content or thread
structure. The system boundary also tells the Writer to express identity only
through wording, confidence, attention, and interaction style and forbids new
biography, expertise claims, experiences, or facts.

## Evaluation

Run the same health audit and exact matched-seed metric suite for each
condition. Report MWU and KS p-values together with absolute Cliff's delta and
Wasserstein distance. For diversity, inspect Self-BLEU, Self-BERTScore, and
semantic cosine jointly; lower repetition is not sufficient if semantic
coherence or factual preservation degrades.

Do not select the final method only because one 70-thread run obtains
`p > 0.05`. At this sample size, p-values vary with the draw and do not measure
effect magnitude. Use the paired 70-thread experiment for design selection,
then freeze the method and confirm it on the preregistered 150-thread set.
Self-loop revision must be evaluated separately from the persona generation
ablation.

## Required Provenance

Each run stores:

- generator and revision policy versions;
- MatrAIx repository commit and dataset path;
- persona mode, assignment seed, and eligible population size;
- per-comment persona ID, selected dimensions, system hash, and system length;
- aggregate assignment counts;
- cumulative API usage, estimated cost, and elapsed time across resumes.

These fields make the treatment auditable and prevent a resumed run from
silently changing persona mode, data, assignment, or upstream implementation.
