#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
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
from generalized_card.planning_quality import (  # noqa: E402
    set_isolation_quota,
    set_thread_isolation_share,
)
from generalized_card.viewpoint_bank import set_reference_window  # noqa: E402
from generalized_card.prompts import set_matched_text  # noqa: E402
from generalized_card.branch_routing import set_branch_dictation  # noqa: E402
from generalized_card.plan_vocabulary import set_plan_vocabulary  # noqa: E402
from generalized_card.writer_plan_fields import set_writer_plan_fields  # noqa: E402
from generalized_card.persona_bridge import (  # noqa: E402
    set_persona_draw,
    set_persona_projection,
)
from generalized_card.planner_distribution import set_slot_grid  # noqa: E402
from generalized_card.generation_distribution import (  # noqa: E402
    set_planner_distribution,
)
from generalized_card.domain_profile import (  # noqa: E402
    CARD_CONTEXT_DROPOUT_RATE,
    CARD_CONTEXT_JITTER_RATE,
    build_domain_profile,
    load_domain_profile,
    set_reference_min_comments,
)
from generalized_card.core_contract import (  # noqa: E402
    CORE_POLICY_VERSION,
    CURRENT_GENERATION_CORE_NAMES,
    GENERATION_ADAPTER_CORE_NAMES,
    GENERALIZED_V2_GENERATION_POLICY_VERSION,
    REVISION_CORE_POLICY_VERSION,
    verify_core_contract,
    verify_run_policy,
    version_source_paths,
)
from generalized_card.domain import REPO_ROOT, load_domain_config  # noqa: E402
from generalized_card.source_provenance import (  # noqa: E402
    verify_source_provenance,
)
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
    parser.add_argument(
        "--writer-model",
        default="",
        help=(
            "Writer-only model override. Empty reproduces every release to date, "
            "which sends --model to both the Planner and the Writer. G123 places "
            "the remaining defect in realization, not in the plan, so this is the "
            "axis worth varying."
        ),
    )
    parser.add_argument(
        "--writer-base-url",
        default="",
        help="Endpoint for --writer-model. Empty reuses --base-url.",
    )
    parser.add_argument(
        "--writer-api-key-env",
        default="",
        help="Env var holding the key for --writer-model. Empty reuses --api-key-env.",
    )
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
    parser.add_argument(
        "--reply-novelty-scope",
        choices=("parent_only", "chain"),
        default="parent_only",
        help=(
            "How far back a direct reply's reply_novelty_anchor is checked "
            "against prior plans in its branch. 'parent_only' reproduces "
            "v103 and earlier byte-for-byte: the anchor phrase alone against "
            "only the immediate parent's plan -- a probe asymmetry (a short "
            "phrase vs. a longer compound description) that made the check "
            "nearly never fire (measured: 0 trips on the v103 N=10 "
            "artifact). 'chain' compares the reply's own full plan "
            "(semantic_move, decision_boundary, detail_focus) against every "
            "ancestor already in the thread's plan ledger instead, at the "
            "same similarity threshold -- 60 trips on the same artifact, "
            "including the reply chain measured as a self_bertscore_mean_f1 "
            "excess that grows from +0.0004 at depth 1-2 to +0.0432 at depth "
            "7+ (`generalized_card/analysis/bertscore_pair_diagnosis.py depth`, "
            "`generalized_card/analysis/reply_novelty_chain_diagnosis.py`)."
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
        "--turn-frame",
        choices=("adjudicative_only", "universal"),
        default="adjudicative_only",
        help=(
            "Whether the Writer is told which question its turn settles. "
            "'universal' reproduces v96 and earlier, which rendered that line on "
            "every slot; the 'that's the part that actually matters' frame it "
            "produces was in 18.4%% of v96 comments, worst on personal-datapoint "
            "(29.1%%) and reaction (19.0%%) slots. 'adjudicative_only' renders it "
            "for correction, verdict, and advice turns and never for a story."
        ),
    )
    parser.add_argument(
        "--tone-length-fit",
        choices=("conditional", "median"),
        default="conditional",
        help=(
            "Where the held-out template's tone counts are placed. 'median' "
            "reproduces v96 and earlier: slots ranked by distance from each tone "
            "class's median length, which left the longest slots for the label "
            "assigned last and put 'impolite' on 100%% of v96 slots over 250 "
            "words. 'conditional' fits the measured P(tone | comment size band) "
            "from evaluation-excluded threads, where comments over 250 words are "
            "72%% polite."
        ),
    )
    parser.add_argument(
        "--long-form-layout",
        choices=("measured", "beats_only"),
        default="measured",
        help=(
            "How a long slot is asked to reach its matched scale. 'beats_only' "
            "reproduces v96 and earlier: one thesis developed through up to 40 "
            "planned beats, no layout cue, and one paragraph at every size. "
            "'measured' adds the paragraph count the domain's excluded threads "
            "show at that size and caps the beat request where the Planner still "
            "delivers. v96 long slots realized 0.60x their matched length."
        ),
    )
    parser.add_argument(
        "--reddit-typography",
        choices=("off", "on"),
        default="on",
        help=(
            "Whether accepted Writer text is rewritten with the punctuation a "
            "keyboard produces, drawn once per speaker at the share measured on "
            "evaluation-excluded threads. 'off' reproduces v96 and earlier, "
            "which emitted typographic apostrophes, quotes, em dashes, and "
            "ellipsis on every comment. The self-BLEU tokenizer reads \"it's\" as "
            "one token and the typographic form as three."
        ),
    )
    parser.add_argument(
        "--isolation-quota",
        choices=["measured", "off"],
        default="off",
        help=(
            "Ask the Planner for a measured share of slots unrelated to every "
            "sibling slot. Distinct from --outsider-quota, which asks for distance "
            "from the POST: that gap measures 1.1x on the current configuration "
            "while intra-thread nearest-neighbour cosine is 0.526 against a real "
            "0.487. `off` reproduces every release to date."
        ),
    )
    parser.add_argument(
        "--planner-distribution",
        choices=["off", "full"],
        default="full",
        help=(
            "`off` stops handing the Planner exact whole-thread tone_class, "
            "affect_role and story counts. Those counts are taken from a "
            "different, same-size evaluation-excluded thread, and imposing one "
            "unrelated thread's affect distribution is the likely mechanism "
            "behind three affect metrics low and neutral_rate high."
        ),
    )
    parser.add_argument(
        "--slot-grid",
        choices=["free", "full"],
        default="full",
        help=(
            "`free` withholds the per-slot grid of pre-assigned tone, affect "
            "label, story mode, opener route, length bucket and surface form, "
            "and adds a block telling the Planner what it is, what real threads "
            "of this kind look like, and which statistics the thread is judged "
            "on. The schedule is still computed and recorded, so a free run can "
            "be diffed field by field against one that received it."
        ),
    )
    parser.add_argument(
        "--writer-plan-fields",
        choices=["full", "angle_detail"],
        default="full",
        help=(
            "`angle_detail` withholds `semantic_move`, `decision_boundary` and "
            "`domain_intent` from the Writer, leaving `content_angle` and "
            "`detail_focus`. Measured on v156's own stored Writer prompts, the "
            "first three are finished sentences the Planner wrote and the "
            "Planner writes similar ones for every slot of a thread: any subset "
            "containing `semantic_move` or `domain_intent` prices at plan "
            "cosine ~0.31, while `content_angle`+`detail_focus` reaches 0.2173. "
            "With the run's own realization function (text = 0.384 x plan + "
            "0.0665) that predicts 0.1500 against real's 0.1552, where the full "
            "six predict 0.2070. The prediction is an EXTRAPOLATION: it was "
            "fitted while the Writer could see everything, and a Writer with "
            "less information invents the point from its own priors, which is "
            "what the intercept measures. The intercept may rise and eat the "
            "gain -- that is what the paid run tests. Guardrail: `self_bleu_4` "
            "currently passes at d -0.02 and G181 measured it moving the other "
            "way when comments are pushed apart."
        ),
    )
    parser.add_argument(
        "--plan-vocabulary",
        choices=["open", "closed"],
        default="closed",
        help=(
            "`open` stops handing the Planner a closed taxonomy for the three "
            "fields that decide what a comment is about. The frozen twelve "
            "decision lenses and the eight content angles are product-shopping "
            "categories -- `universal_decision_lens` is their own recorded "
            "source -- so on a domain none of them fit the Planner takes the "
            "escape hatch: 45%% `seed_local` and 62%% `unclear_mixed` on "
            "celebrity top-level slots, and 100%% of both on replies, whose "
            "schema did not carry the fields at all. Under `open` the Planner "
            "abstracts its own lens set from the reference bank it already "
            "sees, both schemas carry all three fields, `seed_local` is "
            "withdrawn as an option, and the canonicalizer stops folding an "
            "unlisted lens back to it. Unlike G191, which removed the grid and "
            "left the escape hatch as the only answer, this removes the hatch."
        ),
    )
    parser.add_argument(
        "--branch-dictation",
        choices=["structural", "full"],
        default="full",
        help=(
            "`structural` strips the branch routes down to shape -- which "
            "branch, which parent, which siblings -- and drops the branch goal, "
            "required perspective, exclusion and owned subject that normally "
            "decide a slot's direction before the Planner sees it. Pair with "
            "--matched-text measured, which puts the slot's own real comment "
            "there instead. LEAK ARM for the same reason --matched-text is."
        ),
    )
    parser.add_argument(
        "--matched-text",
        choices=["measured", "off"],
        default="off",
        help=(
            "Show the Planner the matched real thread's own comment text, not "
            "just its shape. LEAK ARM: ORIENTATION.md s7 forbids matched "
            "evaluation comment text reaching generation, so a run with this on "
            "cannot be pooled with or compared against a held-out release and can "
            "never ship. It measures the ceiling -- with the real thread in front "
            "of it, how much of the gap can the Planner close."
        ),
    )
    parser.add_argument(
        "--reference-window",
        choices=["measured", "unranked", "off"],
        default="off",
        help=(
            "Fill the Planner's reference-example window at the reference bank's "
            "own length distribution instead of taking the lexical top-N. `off` "
            "reproduces every release to date, where BM25 ranking handed the "
            "Planner a window 3x wordier and less than half as off-topic as the "
            "bank it is drawn from, then asked it to produce scatter it had "
            "never been shown."
        ),
    )
    parser.add_argument(
        "--reference-floor",
        choices=["measured", "off"],
        default="off",
        help=(
            "Hold the reference corpus to the seed pool's own comment floor when "
            "measuring the domain profile. `off` reproduces every release to date, "
            "where the profile measured behaviour over threads too small to carry a "
            "discussion: on celebrity 61%% of the reference corpus sat under five "
            "comments against a seed median of 34, moving every affect target by "
            "30-100%%. No effect where the corpus has few small threads."
        ),
    )
    parser.add_argument(
        "--closing-move",
        choices=("measured", "off"),
        default="measured",
        help=(
            "Draw each slot's closing move at its band's measured rate. `off` "
            "reproduces v99, where output ended on an abstract verdict 0.265 of "
            "the time against a real 0.014."
        ),
    )
    parser.add_argument(
        "--verdict-close-guard",
        choices=("off", "on"),
        default="off",
        help=(
            "'off' reproduces `abstract_verdict_close`'s suppression wording "
            "unmodified. 'on' widens it to also name a \"check\"/\"test\" "
            "closing as the same move -- \"that's the check\", \"a solid "
            "check\" -- which the existing wording never named. Measured on "
            "the v103 N=10 artifact and the v106 gate: even where the "
            "existing cue reaches the Writer, the move it targets still "
            "lands at 10-13x real's rate, and the check/test variant it "
            "never named adds another 13-37x on top -- at nearly 3x the "
            "v103 rate on the v106 gate thread specifically, plausibly "
            "because forcing a different novelty angle per reply (v105) "
            "pushes the Writer toward this as a generic fallback when it "
            "runs out of new specific content to name. No domain-profile "
            "change -- this widens the Writer-facing cue only, not the "
            "measurement pattern, so no profile rebuild is needed."
        ),
    )
    parser.add_argument(
        "--semantic-coverage-nonrepeat",
        choices=("off", "on"),
        default="off",
        help=(
            "'off' reproduces the Writer prompt's \"already covered\" block "
            "with no instruction attached to it, unlike its two sibling blocks "
            "(short utterances, sentence routes), which both already tell the "
            "Writer not to reuse what they list. Read against a real chain "
            "(v103 N=10, seed002 comments 40-45): the later comment's own "
            "coverage block already listed all five earlier near-paraphrases "
            "of the same point verbatim, and it restated the point a sixth "
            "time anyway -- the information was present, nothing told the "
            "Writer what to do with it. 'on' appends the same style of "
            "instruction its two sibling blocks already carry. No domain "
            "vocabulary, no domain-profile change."
        ),
    )
    parser.add_argument(
        "--opening-move",
        choices=("measured", "off"),
        default="measured",
        help=(
            "Name each slot's opening word, drawn at its register's measured "
            "distribution, instead of describing the entry category. `off` "
            "reproduces v101, where `polarity_token` openers ran 0.1274 against "
            "a measured 0.0526 and are the highest-disagreement entry there is."
        ),
    )
    parser.add_argument(
        "--evaluation-tier",
        choices=("measured", "off"),
        default="measured",
        help=(
            "Let a slot's positive evaluation land at full strength, drawn at "
            "its register's measured hot-tier share. `off` reproduces v103, "
            "where hot-tier words ran 0.31x real and swapping a warm word for "
            "another warm word moved `polite_rate` by 0.0000."
        ),
    )
    parser.add_argument(
        "--downtoner-tag",
        choices=("suppress", "off"),
        default="suppress",
        help=(
            "Stop closing a sentence on a tag that takes it back (\", sure\", "
            "\", honestly\"). `off` reproduces v103, where the tag ran 40.7x "
            "its real rate."
        ),
    )
    parser.add_argument(
        "--partitive-reference",
        choices=("suppress", "off"),
        default="suppress",
        help=(
            "Stop evaluating a slice of a thing instead of the thing (\"that "
            "part\", \"the useful bit\"). `off` reproduces v103, where the "
            "construction ran 17.3x its real rate."
        ),
    )
    parser.add_argument(
        "--register-realization",
        choices=("measured", "off"),
        default="measured",
        help=(
            "Ask a slot the plan assigned `polite` for the surface moves real "
            "polite comments of its size carry. `off` reproduces v98, where that "
            "register realized 19.3%% of the time."
        ),
    )
    parser.add_argument(
        "--sentence-rhythm",
        choices=("measured", "off"),
        default="measured",
        help=(
            "Whether each slot is given a typing rhythm drawn at the habit "
            "frequencies measured per size band on evaluation-excluded threads. "
            "'off' reproduces v97 and earlier, where every surface cue was a "
            "function of the slot's size, so two same-size slots were asked for "
            "the same shape: their function-word cosine came out 0.502 against a "
            "real 0.368, and 532 v97 comments contained no exclamation mark "
            "against a real 0.079."
        ),
    )
    parser.add_argument(
        "--digit-cue-guard",
        choices=("off", "on"),
        default="off",
        help=(
            "'off' reproduces the digit cue's pre-v106 wording exactly: 'write "
            "it as a figure' with no exclusion for an ordinary quantifier or "
            "negation word, which the Writer sometimes numeralizes too -- '1 "
            "thing I'd check', 'that 1 folder' -- where a person writes the "
            "word. Measured on the v103 artifact "
            "(generalized_card/analysis/digit_cue_diagnosis.py): bare 0/1 in "
            "0.092 of generated comments against 0.020 of evaluation-excluded "
            "real ones (4.6x); real writers do numeralize a plain quantifier "
            "too (55%% of real's own bare-1 occurrences), but generated does "
            "it at 96%% of its own and 8.2x real's per-comment rate for that "
            "pattern specifically, against 1.7x for enumerated/fractional/ "
            "price uses -- the excess concentrates in one sub-pattern, not "
            "the raw digit rate. 'on' adds one sentence naming the failure "
            "mode by example; the underlying instruction is unchanged, so a "
            "genuine count, price, or spec is unaffected."
        ),
    )
    parser.add_argument(
        "--length-calibration",
        choices=("measured", "off"),
        default="measured",
        help=(
            "Whether the length cue asks for the matched slot's own word count "
            "or for the count that realizes it. 'off' reproduces v97 and "
            "earlier, where realized/target ran 1.42x at the shortest slots and "
            "0.71x at 251-400 words, so a thread's mean length survived and its "
            "spread collapsed: length_cv 0.857 against a real 0.947. 'measured' "
            "inverts the fitted transfer function log(realized) = 0.3835 + "
            "0.8925*log(asked), R2 0.894 over the 532 v97 slots."
        ),
    )
    parser.add_argument(
        "--final-punctuation",
        choices=("measured", "off"),
        default="measured",
        help=(
            "Whether a declarative ending is left with no final punctuation at "
            "the rate measured for its size band. 'off' reproduces v97 and "
            "earlier, which ended 95.9%% of comments in a period against a real "
            "80.8%%. Only a trailing period is dropped, so a question mark or an "
            "exclamation the Writer chose is never touched."
        ),
    )
    parser.add_argument(
        "--route-ledger",
        choices=("on", "off"),
        default="on",
        help=(
            "Whether the focused Writer prompt lists the sentence routes this "
            "thread has already reused mid-comment. 'off' reproduces v97 and "
            "earlier, where the ledger reached the 'full' Writer arm only while "
            "'focused' has been the active arm since v82; the adjudication frame "
            "persisted in 14.4%% of v97 comments that never saw the boundary line."
        ),
    )
    parser.add_argument(
        "--reference-link",
        choices=("off", "measured"),
        default="off",
        help=(
            "Offer one real evaluation-excluded URL to a slot whose matched "
            "comment carried a reference link. 'off' reproduces every release "
            "through v112, which wrote zero URLs across 1,974 slots."
        ),
    )
    parser.add_argument(
        "--reference-link-count",
        choices=("off", "measured"),
        default="off",
        help=(
            "Draw HOW MANY reference URLs a routed slot is offered from the "
            "inventory's measured per-carrier distribution. 'off' reproduces "
            "v113-v116, which drew exactly one for every routed slot: 18.0 URL "
            "tokens per carrying comment against a real 34.3 over 15,559 "
            "evaluation-excluded comments. Routing is unchanged; only the count "
            "moves."
        ),
    )
    parser.add_argument(
        "--reference-link-host",
        choices=("off", "measured"),
        default="off",
        help=(
            "When a slot is offered 2+ reference URLs, draw them all from ONE "
            "host at the corpus's measured rate. 'off' reproduces v117, which "
            "drew each independently across 249 folded hosts and stacked four "
            "unrelated links in one 46-word comment (G61). Real puts every URL "
            "on one host 0.771 of the time at k=2, 0.640 at k=3, 0.417 at k=4. "
            "Single-link slots, routing and the count are all unchanged."
        ),
    )
    parser.add_argument(
        "--tone-donor",
        choices=("off", "measured"),
        default="off",
        help=(
            "Hand a slot the Planner assigned POLITE one real short appreciative "
            "sentence, drawn from the evaluation-excluded corpus, to open with. "
            "'off' reproduces every release through v119. G53 measured that "
            "inserting such a sentence flips a non-polite generated comment's "
            "label 0.29-0.50 of the time; at the low end that moves the polite "
            "row of the realization matrix 0.384 -> 0.563 and polite_rate's "
            "P(pass) at N=150 from 0.17 to 0.90. Six cue-based mechanisms are "
            "dead because composing the sentence is the step that fails; this "
            "hands the finished sentence over."
        ),
    )
    parser.add_argument(
        "--rhythm-count",
        choices=("off", "measured"),
        default="off",
        help=(
            "Draw how MANY parenthetical asides a slot is asked for from the "
            "band's measured distribution instead of cueing the literal word "
            "'one'. 'off' reproduces every release through v115, whose "
            "per-carrier count was {1: 48} on the v113 gate -- exactly one, "
            "every time -- against a real 1.47 at long and 3.58 at essay."
        ),
    )
    parser.add_argument(
        "--interaction-scope",
        choices=("off", "conversation", "full"),
        default="off",
        help=(
            "Show the Planner real exchanges instead of isolated opening "
            "statements, and let the Writer take its parent's point as material. "
            "'off' reproduces v125b byte-for-byte. The Planner's only window "
            "into real discourse is the reference block, and its two-rows-per-"
            "thread cap makes a 36-row window come from ~22 threads at 67-83%% "
            "depth-0, so it has seen thousands of opening statements and almost "
            "no reply. 'conversation' appends whole excluded threads in reply "
            "order, chosen for structural richness and topical DISTANCE from the "
            "seed, so only their shape can transfer. 'full' also drops the "
            "Writer rule that tells 55.1%% of prompts to treat the parent's own "
            "point as an exclusion rather than writing material -- that is "
            "redundant with the mechanical parent_copy guard and is why our "
            "replies talk past their parents."
        ),
    )
    parser.add_argument(
        "--recurring-phrase-ledger",
        default="off",
        help=(
            "List the two-word grammatical sequences this thread has already "
            "reused, so the Writer says it another way. 'off' reproduces v133 "
            "byte-for-byte; a number is the minimum reuse count that puts a "
            "pair on the list (3 is the measured operating point: ~12 items "
            "covering 2.1%% of a comment's bigrams, against real's 1.6%%). "
            "G134: our pairwise 2-gram overlap is a flat 2.1x real's at every "
            "comment length, and a controlled ablation of exactly this band "
            "removes 80.5%% of the excess and beats a mass-matched random "
            "deletion in 33 of 33 threads -- the only candidate this session "
            "to survive that test. Restricted to function-word pairs on "
            "purpose: real threads DO reuse their topic nouns (`the ricoh`, "
            "`the sony`, `the price`), so suppressing those would move away "
            "from real; what we over-reuse is grammar (`is the`, `and the`, "
            "`kind of`, `if the`)."
        ),
    )
    parser.add_argument(
        "--writer-temperature",
        default="legacy",
        help=(
            "Writer sampling temperature. 'legacy' reproduces every release "
            "through v128 byte-for-byte; 'schedule' honours the per-slot value "
            "`writer_temperature(task)` already computes (0.82-1.08); a float "
            "sends that fixed temperature. G131: the writer call site has "
            "always passed a computed temperature and `_completion_kwargs` "
            "discards it for any gpt-5* model, on a comment claiming those "
            "endpoints reject it -- probed 2026-08-28, gpt-5.4-mini accepts "
            "temperature, top_p, frequency_penalty and presence_penalty. So "
            "the gpt-5.x line has run at the API default 1.0 and the schedule "
            "(mean ~0.85) has never applied; 'schedule' LOWERS temperature and "
            "is expected to worsen self-similarity, so it is offered for "
            "completeness, not as the candidate. Planner sampling is never "
            "touched: the arm gates on response_format_json, which is True for "
            "both planner calls and False for both writer calls."
        ),
    )
    parser.add_argument(
        "--sentence-pacing",
        choices=("off", "measured"),
        default="off",
        help=(
            "State each slot's own words-per-sentence in the length cue instead "
            "of the word 'pacing'. 'off' reproduces v125b byte-for-byte. G113: "
            "inside the long_turn bucket our coefficient of variation on mean "
            "sentence length is 0.37x real's against 1.10x on word count, so "
            "the realization layer is narrow specifically here. It is not "
            "non-compliance -- the Writer honours the stated sentence count as "
            "well as the word count (median relative error +0.00 against "
            "-0.07). The matched real comments carry words-per-sentence at CV "
            "0.53 and our text realizes 0.39, so the Writer hits both marginals "
            "while pulling their ratio toward its own preferred ~17 words per "
            "sentence. The target is the matched comment's own ratio, which is "
            "why this carries no constant and follows the domain."
        ),
    )
    parser.add_argument(
        "--outsider-quota",
        choices=("off", "measured"),
        default="off",
        help=(
            "Ask the Planner for the share of comments that do not answer the "
            "post at all. 'off' reproduces v124 byte-for-byte. G97: the gap is "
            "entirely in the low tail -- p90 is +0.008 but p1 is +0.038, and "
            "pairs below cosine 0 are 8.09%% of real against 3.30%% of ours. "
            "Word counts already match, but the off-topic rate collapses with "
            "length: at 1-10 words we match real (37.2%% vs 36.7%%) and at 61+ "
            "words real is 3.4%% against our 0.8%%. Of real's low-affinity "
            "comments 6.1%% are >=40 words; of ours, zero. The channels already "
            "exist -- `offtopic_noise` was chosen 0 times in 532 v122 slots -- "
            "and `--social-noise-min-share` cannot reach them because "
            "`rebalance_card_surfaces` discards every share argument by design. "
            "The quota is per-slot, not positional: real threads do not drift "
            "with ordinal position and our depth curve already matches theirs."
        ),
    )
    parser.add_argument(
        "--plan-move-ledger",
        choices=("off", "spent_moves"),
        default="off",
        help=(
            "Name the semantic moves a thread has already spent when asking the "
            "Planner to repair a semantic collision. 'off' reproduces v122 "
            "byte-for-byte. The detector is already correct and already tuned "
            "(0.70 flags 33.6%% of v122 slots on 139 pairs), but the repair "
            "surrendered on 111 slot instances across 22 warnings, with "
            "collision_rate at surrender reaching 0.667 (docs/DECISIONS.md G96). "
            "The instruction is why: it names a category -- 'change the decision "
            "lens, stance, evidence role' -- which E4 prices at 0.23 compliance "
            "against ~1.0 for a concrete token, and it never says which lenses "
            "the thread has already used, so the Planner re-rolls from the same "
            "small vocabulary (G94: greedy dedup at cosine 0.45 would reject 72%% "
            "of slots). This renders the spent-move list and requires a named, "
            "unused move. Raising the repair budget instead is NOT indicated -- "
            "G88 tested that shape one stage later and both priority metrics got "
            "worse."
        ),
    )
    parser.add_argument(
        "--tone-quota",
        choices=("off", "inverted", "calibrate"),
        default="off",
        help=(
            "Render the Planner's tone quota as the assignment whose REALIZED mix "
            "matches the reference template, by inverting the measured realization "
            "matrix. 'off' reproduces every release through v114, where the quota "
            "was the template's own rates and the Writer's 0.854 impolite / 0.384 "
            "polite realization pushed the output to 0.607 impolite against 0.464. "
            "'calibrate' renders a flat quota and is a MEASUREMENT value only: it "
            "populates the (stance, assigned tone) cells the polite cap exists "
            "because nobody has measured. Never use it for a candidate artifact."
        ),
    )
    parser.add_argument(
        "--development-scope",
        choices=("long_only", "measured"),
        default="long_only",
        help=(
            "How far down the enumerated per-slot development beat plan reaches. "
            "'long_only' withholds it below 101 assigned words and reproduces "
            "v110 byte-for-byte. 'measured' extends it to 35 words, the point "
            "where compression starts. This is the length instrument: realized/"
            "assigned words jump 0.816 -> 0.953 across the boundary the legacy "
            "value creates, in all four comparable N=10 runs, and the Writer "
            "delivers 21.3 realized words per delivered beat -- while the asked "
            "word count's own elasticity is -0.02 to 0.11 (docs/DECISIONS.md "
            "G48, G50). The bands the plan never reaches carry 88%% of the word "
            "deficit; extending it is priced at 8-26%% of the self_bleu_4 gap."
        ),
    )
    parser.add_argument(
        "--length-transfer",
        choices=("v97", "refit"),
        default="v97",
        help=(
            "Which fitted word-length transfer function the length calibration "
            "inverts. 'v97' keeps the constants fitted on the v97 run and "
            "reproduces v109 byte-for-byte. 'refit' uses the line refitted on "
            "realized-vs-calibrated-ask over 1,436 slots from four runs "
            "(intercept 0.5580, slope 0.8276, R2 0.879); the v97 constants "
            "regressed on the *uncalibrated* ask and have been under-correcting "
            "ever since, leaving realized/asked at 1.64x below 10 words and "
            "0.68-0.80x above 80 -- the measured cause of the length "
            "compression worth 31-35%% of the self_bleu_4 gap and 14-18%% of "
            "the self_bertscore gap (docs/DECISIONS.md G43, G46)."
        ),
    )
    parser.add_argument(
        "--length-fidelity",
        choices=("off", "measured"),
        default="off",
        help=(
            "Require a slot's realized word count to stay in the same measured "
            "length band as the `real_word_count` it was assigned. 'off' "
            "reproduces v109 byte-for-byte. Realized/assigned words run 1.44x "
            "on slots assigned under 10 words and 0.82x at 50-100, compressing "
            "the thread's length spread; exact pair reweighting puts length "
            "composition at 33-37%% of the self_bleu_4 gap and 17-26%% of the "
            "self_bertscore gap (docs/DECISIONS.md G43). Registered as a soft "
            "problem, so it drives the Writer retry loop and never makes a "
            "matched slot blocking -- it therefore does nothing unless "
            "--writer-retries is above 0."
        ),
    )
    parser.add_argument(
        "--seed-pool-exclude",
        type=Path,
        nargs="*",
        default=(),
        help=(
            "Existing seed-pool JSON files whose threads must not appear in this "
            "run's pool. The held-out set is hashed into the pool's own filename, "
            "so a pool built with exclusions can never be silently replaced by "
            "one built without them. Use it for a calibration run that must share "
            "no thread with any evaluation pool (docs/CALIBRATION_RUNBOOK.md)."
        ),
    )
    parser.add_argument(
        "--length-ceiling",
        choices=("off", "measured"),
        default="off",
        help=(
            "Refuse a realized comment longer than the domain's own measured "
            "p99 comment length (300 words for camera, from the same "
            "evaluation-excluded corpus the band cuts come from). 'off' "
            "reproduces the previous release byte-for-byte. This is NOT a "
            "variant of --length-fidelity and neither implies the other: the "
            "top decile band is open above 108 words, so a slot assigned 150 "
            "words that realizes 523 sits in its own assigned band and the "
            "band check reports nothing. Simulated on the 50 matched DeepSeek "
            "threads, band matching moves length_cv d from +0.23 to +0.27 by "
            "quantising 40%% of slots onto band edges, while the ceiling alone "
            "takes it to +0.04 by touching 1.8%% -- length_cv is a coefficient "
            "of variation and only the tail can move it (docs/DECISIONS.md "
            "G157, G162). Registered soft, so it drives the Writer retry loop "
            "and never makes a matched slot blocking; it therefore does "
            "nothing unless --writer-retries is above 0. Costs the ~1%% of "
            "comments real writes above its own p99."
        ),
    )
    parser.add_argument(
        "--length-ceiling-rounds",
        type=int,
        default=2,
        help=(
            "How many bounded re-draws a comment past the length ceiling gets. "
            "Independent of --writer-retries on purpose: that switch retries on "
            "ANY soft problem, and the v109 gate had 65 of 186 slots raising "
            "one, so reaching the ceiling's 1.8%% through it would rewrite a "
            "third of the corpus for unrelated reasons. Inert unless "
            "--length-ceiling is 'measured'. Exhausting the rounds never skips "
            "the slot -- the last text is stored (ORIENTATION.md s4)."
        ),
    )
    parser.add_argument(
        "--entity-spread",
        choices=("off", "measured"),
        default="off",
        help=(
            "Offer each slot a rotating held-out referent it may name in "
            "passing, drawn at the thread band's measured distinct-designator "
            "rate. 'off' reproduces v108 byte-for-byte. Real matched threads "
            "name 40.8 distinct designators against a generated 7.4, with the "
            "top one taking 0.152 of mentions against 0.485 "
            "(docs/DECISIONS.md G35). Priced by exact ablation at 5.4%% of the "
            "self_bleu_4 gap; shipped primarily as a criterion-2 fix."
        ),
    )
    parser.add_argument(
        "--no-story-scope",
        choices=("sequence", "tense"),
        default="sequence",
        help=(
            "What a no_story slot is barred from. 'tense' reproduces v96 and "
            "v97, which barred any past action or event on 453 of 532 slots: "
            "past-tense verbs appeared in 0.181 of those comments against a "
            "real 0.543, future in 0.031 against 0.226, and the thread lexicon "
            "fell to 2,670 distinct types against a real 3,645, which is the "
            "whole self_bertscore_mean_f1 gap. 'sequence' bars the second "
            "event and the then/after pacing StorySeeker actually scores."
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
        choices=("selective", "planned", "off"),
        default="selective",
        help=(
            "How the Planner assigns separate domain claims. 'selective' is "
            "the active policy: only capacity-compatible slots backed by an "
            "evaluation-excluded factual reference row receive one. 'planned' "
            "preserves the historical near-ubiquitous arm (508/522 comments), "
            "and 'off' disables both planning and delivery."
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
    parser.add_argument(
        "--persona-projection",
        choices=["register", "default"],
        default="default",
        help=(
            "Which persona dimensions reach the Writer. The shipped budget is "
            "ten dimensions spent on a list that never named the register axes, "
            "so `english_proficiency`, `multilingualism`, `urbanicity`, "
            "`age_bracket`, `neurotype` and `political_lean` render on 0%% of "
            "personas while `lstyle_commute_mode` renders on 19 of 123. "
            "`register` puts the writing axes first, spends `lstyle_*` last, "
            "and widens the budget to 16. Selecting a register-diverse persona "
            "set does nothing without this."
        ),
    )
    parser.add_argument(
        "--persona-draw",
        choices=["exhaust", "replace"],
        default="replace",
        help=(
            "`replace` lets two speakers in one thread draw the same persona, "
            "which is the dominant loss of identity variety and is arithmetic: "
            "~30 speakers drawing independently from a band of ~55 yield 21.5 "
            "distinct personas, measured 21.7, against a real corpus at ~29. "
            "`exhaust` takes the highest-ranked candidate the thread has not "
            "used, giving min(speakers, band)."
        ),
    )
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
    # BEFORE build_runtime, not after. The runtime captures both modes at
    # construction and `public_config()` renders every eligible persona to
    # report length statistics, so a setter called later leaves `run_config`
    # and the persona manifest claiming the shipped defaults for a run that
    # used the arms -- which is exactly what the v152 probe recorded.
    set_persona_projection(args.persona_projection)
    set_persona_draw(args.persona_draw)
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
    # The hash contract proves these files have not drifted. This proves git can
    # still hand them back, which is a different question -- see
    # `source_provenance`. Checked before the seed pool and the domain profile so
    # a non-reproducible run stops at the first second of work, not the first
    # dollar.
    source_record = verify_source_provenance(
        version_source_paths(generation_core_names)
    )
    generator_policy_version = (
        GENERALIZED_V2_GENERATION_POLICY_VERSION
        if args.generator_profile == "generalized-v2"
        else CORE_POLICY_VERSION
    )
    run_root = REPO_ROOT / "artifacts" / "generalized_card" / "runs" / args.tag
    generated_root = run_root / "generated"
    if run_root.exists() and not args.resume and not args.prepare_only:
        raise SystemExit(f"Run exists; pass --resume or choose a new --tag: {run_root}")
    exclude_keys: set[tuple[str, str]] = set()
    for pool_path in args.seed_pool_exclude or ():
        payload = json.loads(Path(pool_path).expanduser().resolve().read_text(encoding="utf-8"))
        for row in payload.get("seed_posts") or ():
            exclude_keys.add(
                (str(row.get("source_product_dir")), str(row.get("source_raw_post_id")))
            )
    # The held-out set goes in the FILENAME, not only inside the file. A pool is
    # rebuilt from its name whenever it is missing, so an exclusion that lived
    # only in the contents would be silently dropped by that rebuild.
    exclude_tag = ""
    if exclude_keys:
        digest = hashlib.sha256(
            "\n".join(sorted(f"{a}\t{b}" for a, b in exclude_keys)).encode("utf-8")
        ).hexdigest()[:8]
        exclude_tag = f"_excl{len(exclude_keys)}x{digest}"
    seed_pool = (
        REPO_ROOT
        / "artifacts"
        / "generalized_card"
        / "seed_pools"
        / f"{config.domain_id}_{args.pool_size}_seed{args.sampling_seed}{exclude_tag}.json"
    )
    if not seed_pool.exists():
        build_seed_pool(
            config,
            seed_pool,
            count=args.pool_size,
            seed=args.sampling_seed,
            exclude_keys=exclude_keys or None,
        )

    # Must be set before build_domain_profile runs; it selects the reference corpus.
    set_reference_min_comments(args.reference_floor)
    set_reference_window(args.reference_window)
    set_matched_text(args.matched_text)
    set_branch_dictation(args.branch_dictation)
    set_plan_vocabulary(args.plan_vocabulary)
    set_writer_plan_fields(args.writer_plan_fields)
    set_slot_grid(args.slot_grid)
    set_planner_distribution(args.planner_distribution)
    set_isolation_quota(args.isolation_quota)

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
    # A per-thread isolation share, when the profile carries one, overrides the
    # domain default. matched_profile.py measures it from that seed's own real
    # comments, so a thread whose real discussion genuinely pulls together is
    # generated that way instead of having a fixed quota forced into it.
    set_thread_isolation_share(domain_profile.get("thread_isolation_share"))

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
        "writer_model": args.writer_model or args.model,
        "writer_base_url": args.writer_base_url or args.base_url,
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
            "reply_novelty_scope": args.reply_novelty_scope,
        },
        "domain_claim": args.domain_claim,
        "writer_prompt": args.writer_prompt,
        "writer_route_lock": args.writer_route_lock,
        "social_contract_coherence": args.social_contract_coherence,
        "reply_sibling_visibility": args.reply_sibling_visibility,
        "reddit_typography": args.reddit_typography,
        "long_form_layout": args.long_form_layout,
        "tone_length_fit": args.tone_length_fit,
        "turn_frame": args.turn_frame,
        "sentence_rhythm": args.sentence_rhythm,
        "digit_cue_guard": args.digit_cue_guard,
        "register_realization": args.register_realization,
        "closing_move": args.closing_move,
        "reference_floor": args.reference_floor,
        "reference_window": args.reference_window,
        "matched_text": args.matched_text,
        "branch_dictation": args.branch_dictation,
        "plan_vocabulary": args.plan_vocabulary,
        "writer_plan_fields": args.writer_plan_fields,
        "persona_projection": args.persona_projection,
        "persona_draw": args.persona_draw,
        "slot_grid": args.slot_grid,
        "planner_distribution": args.planner_distribution,
        "isolation_quota": args.isolation_quota,
        "verdict_close_guard": args.verdict_close_guard,
        "semantic_coverage_nonrepeat": args.semantic_coverage_nonrepeat,
        "opening_move": args.opening_move,
        "evaluation_tier": args.evaluation_tier,
        "downtoner_tag": args.downtoner_tag,
        "partitive_reference": args.partitive_reference,
        "length_calibration": args.length_calibration,
        "final_punctuation": args.final_punctuation,
        "route_ledger": args.route_ledger,
        "entity_spread": args.entity_spread,
        "length_fidelity": args.length_fidelity,
        "length_ceiling": args.length_ceiling,
        "length_ceiling_rounds": args.length_ceiling_rounds,
        "length_transfer": args.length_transfer,
        "development_scope": args.development_scope,
        "reference_link": args.reference_link,
        "tone_quota": args.tone_quota,
        "plan_move_ledger": args.plan_move_ledger,
        "outsider_quota": args.outsider_quota,
        "recurring_phrase_ledger": args.recurring_phrase_ledger,
        "writer_temperature": args.writer_temperature,
        "sentence_pacing": args.sentence_pacing,
        "interaction_scope": args.interaction_scope,
        "rhythm_count": args.rhythm_count,
        "reference_link_count": args.reference_link_count,
        "reference_link_host": args.reference_link_host,
        "tone_donor": args.tone_donor,
        "writer_retries": args.writer_retries,
        "no_story_scope": args.no_story_scope,
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
        # Not a resume-immutable field. `generator_core_provenance` already is,
        # and it carries the actual hashes -- so a resume that reaches this line
        # has provably identical source content and any commit recorded here
        # reproduces it. Overwriting with the resume's commit is therefore
        # lossless and needs no history list.
        "source_provenance": source_record,
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
        f"strict={int(args.strict_plan_quality)} "
        f"reply_novelty_scope={args.reply_novelty_scope}",
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
    api_key = os.environ.get(args.api_key_env, "").strip()
    # `--prepare-only` runs the preflight self-test and stops, so it needs no
    # credential. It used to return HERE, before the self-test, which made it a
    # config printer rather than a preflight: the v117 calibration command was
    # "verified" with it and then failed its self-test on the first paid attempt.
    if not api_key and not args.prepare_only:
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
    env["GENERALIZED_CARD_REDDIT_TYPOGRAPHY"] = args.reddit_typography
    env["GENERALIZED_CARD_LONG_FORM_LAYOUT"] = args.long_form_layout
    env["GENERALIZED_CARD_TONE_LENGTH_FIT"] = args.tone_length_fit
    env["GENERALIZED_CARD_TURN_FRAME"] = args.turn_frame
    env["GENERALIZED_CARD_SENTENCE_RHYTHM"] = args.sentence_rhythm
    env["GENERALIZED_CARD_DIGIT_CUE_GUARD"] = args.digit_cue_guard
    env["GENERALIZED_CARD_REGISTER_REALIZATION"] = args.register_realization
    env["GENERALIZED_CARD_CLOSING_MOVE"] = args.closing_move
    env["GENERALIZED_CARD_VERDICT_CLOSE_GUARD"] = args.verdict_close_guard
    env["GENERALIZED_CARD_SEMANTIC_COVERAGE_NONREPEAT"] = args.semantic_coverage_nonrepeat
    env["GENERALIZED_CARD_OPENING_MOVE"] = args.opening_move
    env["GENERALIZED_CARD_EVALUATION_TIER"] = args.evaluation_tier
    env["GENERALIZED_CARD_DOWNTONER_TAG"] = args.downtoner_tag
    env["GENERALIZED_CARD_PARTITIVE_REFERENCE"] = args.partitive_reference
    env["GENERALIZED_CARD_LENGTH_CALIBRATION"] = args.length_calibration
    env["GENERALIZED_CARD_FINAL_PUNCTUATION"] = args.final_punctuation
    env["GENERALIZED_CARD_ROUTE_LEDGER"] = args.route_ledger
    env["GENERALIZED_CARD_ENTITY_SPREAD"] = args.entity_spread
    env["GENERALIZED_CARD_LENGTH_FIDELITY"] = args.length_fidelity
    env["GENERALIZED_CARD_LENGTH_CEILING"] = args.length_ceiling
    env["GENERALIZED_CARD_LENGTH_CEILING_ROUNDS"] = str(args.length_ceiling_rounds)
    env["GENERALIZED_CARD_LENGTH_TRANSFER"] = args.length_transfer
    env["GENERALIZED_CARD_DEVELOPMENT_SCOPE"] = args.development_scope
    env["GENERALIZED_CARD_REFERENCE_LINK"] = args.reference_link
    env["GENERALIZED_CARD_TONE_QUOTA"] = args.tone_quota
    env["GENERALIZED_CARD_PLAN_MOVE_LEDGER"] = args.plan_move_ledger
    env["GENERALIZED_CARD_OUTSIDER_QUOTA"] = args.outsider_quota
    # Generation happens in run_generator_backend.py, a subprocess. The setters
    # called in main() configure THIS process, which builds the domain profile;
    # the Planner prompt is assembled on the other side of this boundary, so a
    # flag that does not appear here does not reach the prompt at all.
    env["GENERALIZED_CARD_ISOLATION_QUOTA"] = args.isolation_quota
    env["GENERALIZED_CARD_MATCHED_TEXT"] = args.matched_text
    env["GENERALIZED_CARD_BRANCH_DICTATION"] = args.branch_dictation
    env["GENERALIZED_CARD_PLAN_VOCABULARY"] = args.plan_vocabulary
    env["GENERALIZED_CARD_WRITER_PLAN_FIELDS"] = args.writer_plan_fields
    env["GENERALIZED_CARD_SLOT_GRID"] = args.slot_grid
    env["GENERALIZED_CARD_PLANNER_DISTRIBUTION"] = args.planner_distribution
    env["GENERALIZED_CARD_REFERENCE_WINDOW"] = args.reference_window
    env["GENERALIZED_CARD_RECURRING_PHRASE_LEDGER"] = str(
        args.recurring_phrase_ledger
    )
    env["GENERALIZED_CARD_WRITER_TEMPERATURE"] = str(args.writer_temperature)
    env["GENERALIZED_CARD_SENTENCE_PACING"] = args.sentence_pacing
    env["GENERALIZED_CARD_INTERACTION_SCOPE"] = args.interaction_scope
    env["GENERALIZED_CARD_RHYTHM_COUNT"] = args.rhythm_count
    env["GENERALIZED_CARD_REFERENCE_LINK_COUNT"] = args.reference_link_count
    env["GENERALIZED_CARD_REFERENCE_LINK_HOST"] = args.reference_link_host
    env["GENERALIZED_CARD_TONE_DONOR"] = args.tone_donor
    env["GENERALIZED_CARD_NO_STORY_SCOPE"] = args.no_story_scope
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
    env["GENERALIZED_CARD_PERSONA_PROJECTION"] = args.persona_projection
    env["GENERALIZED_CARD_PERSONA_DRAW"] = args.persona_draw
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
    env["GENERALIZED_CARD_REPLY_NOVELTY_SCOPE"] = args.reply_novelty_scope
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
    # A writer-only model override usually lives behind a different provider,
    # so it needs its own key; empty reuses the run's key unchanged.
    writer_key = os.environ.get(args.writer_api_key_env, "") if args.writer_api_key_env else ""
    env["WRITER_API_KEY"] = writer_key or api_key
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

    if args.prepare_only:
        print("[prepare-only] preflight self-test passed; no API calls were made",
              flush=True)
        return

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
        args.writer_model or args.model,
        "--writer-base-url",
        args.writer_base_url or args.base_url,
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
    "reddit_typography",
    "long_form_layout",
    "tone_length_fit",
    "turn_frame",
    "sentence_rhythm",
    "digit_cue_guard",
    "register_realization",
    "closing_move",
    "reference_floor",
    "reference_window",
    "matched_text",
    "branch_dictation",
    "plan_vocabulary",
    "writer_plan_fields",
    "persona_projection",
    "persona_draw",
    "slot_grid",
    "planner_distribution",
    "isolation_quota",
    "verdict_close_guard",
    "semantic_coverage_nonrepeat",
    "opening_move",
    "evaluation_tier",
    "downtoner_tag",
    "partitive_reference",
    "length_calibration",
    "final_punctuation",
    "route_ledger",
    "entity_spread",
    "length_fidelity",
    "length_ceiling",
    "length_ceiling_rounds",
    "length_transfer",
    "development_scope",
    "reference_link",
    "tone_quota",
    "plan_move_ledger",
    "outsider_quota",
    "recurring_phrase_ledger",
    "writer_temperature",
    "sentence_pacing",
    "interaction_scope",
    "rhythm_count",
    "reference_link_count",
    "reference_link_host",
    "tone_donor",
    "writer_retries",
    "no_story_scope",
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
