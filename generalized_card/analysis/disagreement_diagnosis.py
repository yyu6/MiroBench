#!/usr/bin/env python3
"""Reproduce the `hard_disagree_rate` diagnosis. No API calls.

Every number in `tasks/v102-worklog.md` comes from one of the subcommands below.

    python3 generalized_card/analysis/disagreement_diagnosis.py structure
    python3 generalized_card/analysis/disagreement_diagnosis.py openers
    python3 generalized_card/analysis/disagreement_diagnosis.py echo
    python3 generalized_card/analysis/disagreement_diagnosis.py surrogate   # sklearn
    python3 generalized_card/analysis/disagreement_diagnosis.py ablate      # torch
    python3 generalized_card/analysis/disagreement_diagnosis.py all

Discipline built into the shared loader:

  * the 150-thread evaluation seed pool is excluded from anything fitted, and
    the ten matched threads are kept separate from the rest of the corpus
  * the real per-pair stance labels come from the evaluation scorer's own
    `stance_disagreement_results.json` tables, so no metric is approximated
  * **pairs are deduplicated by (thread_id, reply_id).** One Reddit post can sit
    under two product folders, so a naive read double-counts 24-32% of the pairs
  * `ablate` re-scores text with the evaluator's own scorer classes and
    reproduces the shipped artifact byte-for-byte before it changes anything

The metric, read from the scorer: `hard_disagree_rate` is the share of
parent -> reply pairs whose argmax over {disagree, neutral, agree} is `disagree`.
Root comments are pairs too -- their parent is the post's title plus selftext.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable

REPO = Path(__file__).resolve().parents[2]
REAL_DIR = REPO / "data/raw/discussions/camera_product"
SEED_POOL = REPO / "artifacts/generalized_card/seed_pools/camera_product_150_seed42.json"
SCORER = REPO / "scripts/evaluation/score_thread_disagreement.py"
STANCE_MODEL = REPO / "Stance_Rel/RoBERT_rel_1.5e-05"
DEFAULT_RUN = (
    REPO
    / "artifacts/generalized_card/runs"
    / "generalized_card_camera_gpt54_v101_register_n10_20260820_v1"
)

TOKEN = re.compile(r"[a-z']+")
STOP = frozenset(
    """the a an and or of to in is are was were be been it its this that i you
    he she they we my your for on at with as but if so not no do does did have
    has had will would can could me him her them from about""".split()
)

# The Writer-facing opener instructions, verbatim from opener_profile.py. The
# rendered prompt is the only place the assignment survives -- `opener_type` is
# not persisted into `discussion.json`, so the plan is recovered from the prompt.
OPENER_INSTRUCTION_MARKERS = {
    "content_phrase": "Open on the substance itself",
    "first_person": "Open with your own experience or position",
    "noun_phrase": "Open with the thing being discussed",
    "discourse_marker": "Open with a short conversational connective",
    "polarity_token": "Open with a bare agreement or disagreement token",
    "question": "Open with the question itself",
    "quote": "Open with a brief markdown quote",
    "conditional": "Open with the condition or circumstance",
    "address": "Open by addressing the person",
    "imperative": "Open with the action you are recommending",
    "link": "Open with the reference itself",
}

# Leading agreement/polarity tokens, the surface form the excess is made of.
POLARITY_OPEN = re.compile(
    r"^\W*(yeah|yea|yep|yup|yes|no|nope|nah|true|exactly|agreed|same|fair|right"
    r"|sure)\b[\s,.:;!—–-]*",
    re.I,
)


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def content_words(text: str) -> set[str]:
    return set(TOKEN.findall(str(text).lower())) - STOP


def parent_echo(row: dict[str, Any]) -> float:
    """Share of the reply's content words that also appear in its parent."""

    parent = content_words(row["parent_text"])
    reply = content_words(row["text"])
    return len(parent & reply) / len(reply) if parent and reply else 0.0


def rate(rows: Iterable[dict[str, Any]]) -> float:
    rows = list(rows)
    if not rows:
        return float("nan")
    return sum(1 for row in rows if row["pred"] == "disagree") / len(rows)


def margin(row: dict[str, Any]) -> float:
    return row["p_dis"] - max(row["p_neu"], row["p_agr"])


class Corpus:
    """Generated, matched-real and excluded-real stance pairs, loaded once."""

    def __init__(self, run: Path) -> None:
        self.run = run
        pool = _load_json(SEED_POOL) or {}
        seeds = pool.get("seed_posts") or []
        self.seed_ids = {str(row.get("source_raw_post_id") or "") for row in seeds}
        self.generated = self._generated_pairs(run)
        matched = {row["real_post_id"] for row in self.generated}
        real = self._real_pairs()
        self.matched = [row for row in real if row["thread_id"] in matched]
        self.excluded = [row for row in real if row["thread_id"] not in self.seed_ids]

    # -- loaders ---------------------------------------------------------- #

    def _real_pairs(self) -> list[dict[str, Any]]:
        seen: set[tuple[str, str]] = set()
        out: list[dict[str, Any]] = []
        for product in sorted(REAL_DIR.iterdir()):
            payload = _load_json(product / "stance_disagreement_results.json")
            for pair in (payload or {}).get("pairs") or []:
                key = (str(pair["thread_id"]), str(pair["reply_id"]))
                if key in seen:
                    continue
                seen.add(key)
                out.append(
                    {
                        "thread_id": str(pair["thread_id"]),
                        "reply_id": str(pair["reply_id"]),
                        "is_root": not str(pair["parent_id"]).startswith("t1_"),
                        "pred": pair["pred_label"],
                        "p_dis": pair["stance_probs"]["disagree"],
                        "p_neu": pair["stance_probs"]["neutral"],
                        "p_agr": pair["stance_probs"]["agree"],
                        "text": pair["reply_text"],
                        "parent_text": pair["parent_text"],
                        "words": len(pair["reply_text"].split()),
                        "parent_words": len(pair["parent_text"].split()),
                    }
                )
        return out

    def _generated_pairs(self, run: Path) -> list[dict[str, Any]]:
        pool = _load_json(SEED_POOL) or {}
        real_by_seed = {
            int(row["seed_index"]): str(row["source_raw_post_id"])
            for row in pool.get("seed_posts") or []
        }
        out: list[dict[str, Any]] = []
        for sim_dir in sorted(run.glob("cleaned/run_*_sampled_reddit")):
            discussion = _load_json(sim_dir / "discussion.json") or {}
            stance = _load_json(sim_dir / "stance_disagreement_results.json") or {}
            plan: dict[tuple[str, str], str] = {}
            for post in discussion.get("posts") or []:
                for record in post.get("generation_records") or []:
                    comment = record.get("comment") or {}
                    key = (str(post["post_id"]), str(comment.get("comment_id")))
                    plan[key] = _planned_opener(record.get("prompt") or "")
            seed_by_post = {
                str(post["post_id"]): int(post.get("seed_index") or -1)
                for post in discussion.get("posts") or []
            }
            for pair in stance.get("pairs") or []:
                key = (str(pair["thread_id"]), str(pair["reply_id"]))
                out.append(
                    {
                        "thread_id": key[0],
                        "reply_id": key[1],
                        "real_post_id": real_by_seed.get(seed_by_post.get(key[0], -1), ""),
                        "is_root": int(pair["depth"] or 0) == 0,
                        "pred": pair["pred_label"],
                        "p_dis": pair["stance_probs"]["disagree"],
                        "p_neu": pair["stance_probs"]["neutral"],
                        "p_agr": pair["stance_probs"]["agree"],
                        "text": pair["reply_text"],
                        "parent_text": pair["parent_text"],
                        "words": len(pair["reply_text"].split()),
                        "parent_words": len(pair["parent_text"].split()),
                        "planned_opener": plan.get(key, "?"),
                    }
                )
        return out

    # -- views ------------------------------------------------------------ #

    def replies(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [row for row in rows if not row["is_root"]]

    def roots(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [row for row in rows if row["is_root"]]


def _planned_opener(prompt: str) -> str:
    hits = [
        name
        for name, marker in OPENER_INSTRUCTION_MARKERS.items()
        if marker in prompt
    ]
    return hits[0] if len(hits) == 1 else "?"


def classify_opener(text: str) -> str:
    """The evaluation-side grammatical opener class, from `opener_profile`."""

    sys.path.insert(0, str(REPO / "generalized_card"))
    from generalized_card.opener_profile import classify_opener as _classify

    return _classify(text)


# --------------------------------------------------------------------------- #
# structure: where in the thread the gap lives
# --------------------------------------------------------------------------- #


def cmd_structure(corpus: Corpus) -> None:
    """The whole gap is the reply-pair conditional; root pairs already match."""

    print("== hard_disagree_rate, split by pair kind ==\n")
    header = f"{'corpus':16s} {'n':>6s} {'root share':>10s} {'P(d|root)':>10s} {'P(d|reply)':>11s} {'overall':>8s}"
    print(header)
    for name, rows in (
        ("generated", corpus.generated),
        ("matched-real", corpus.matched),
        ("excluded-real", corpus.excluded),
    ):
        roots, replies = corpus.roots(rows), corpus.replies(rows)
        print(
            f"{name:16s} {len(rows):6d} {len(roots)/len(rows):10.3f} "
            f"{rate(roots):10.4f} {rate(replies):11.4f} {rate(rows):8.4f}"
        )

    gen, real = corpus.generated, corpus.matched
    g_root, g_reply = corpus.roots(gen), corpus.replies(gen)
    counterfactual = (
        len(g_root) / len(gen) * rate(g_root)
        + len(g_reply) / len(gen) * rate(corpus.replies(real))
    )
    print(
        f"\ngenerated with the real REPLY conditional -> {counterfactual:.4f} "
        f"(shipped {rate(gen):.4f}, matched real {rate(real):.4f})"
    )

    print("\n== the head is nearly degenerate: the decision margin is knife-edge ==\n")
    for name, rows in (
        ("generated", corpus.replies(corpus.generated)),
        ("matched-real", corpus.replies(corpus.matched)),
        ("excluded-real", corpus.replies(corpus.excluded)),
    ):
        margins = sorted(margin(row) for row in rows)


        def pick(q: float) -> float:
            return margins[min(len(margins) - 1, int(round(q * (len(margins) - 1))))]

        print(
            f"{name:16s} n={len(rows):5d} mean p_disagree={statistics.mean(r['p_dis'] for r in rows):.4f} "
            f"margin p10={pick(0.10):+.4f} med={pick(0.50):+.4f} p90={pick(0.90):+.4f} rate={rate(rows):.4f}"
        )
    real_margins = sorted(margin(row) for row in corpus.replies(corpus.matched))
    for shift in (0.005, 0.010, 0.015, 0.020):
        shifted = sum(1 for value in real_margins if value + shift > 0) / len(real_margins)
        print(f"  matched-real margins shifted by +{shift:.3f} -> rate {shifted:.4f}")
    print(
        "  A uniform ~+0.017 shift of the decision margin reproduces the whole gap.\n"
        "  No subset of pairs carries it; the whole distribution is translated."
    )


# --------------------------------------------------------------------------- #
# openers: the plan is right, the opener is not realized
# --------------------------------------------------------------------------- #


def cmd_openers(corpus: Corpus) -> None:
    """`polarity_token` openers are 2.5x their scheduled share and 3x as
    likely to be labelled `disagree`."""

    profile = _load_json(corpus.run / "domain_profile.json") or {}
    shares = ((profile.get("opener_profile") or {}).get("shares")) or {}

    print("== the schedule reaches the Writer prompt, and is not obeyed ==\n")
    matrix: dict[str, Counter] = defaultdict(Counter)
    for row in corpus.generated:
        matrix[row["planned_opener"]][classify_opener(row["text"])] += 1
    print(f"{'planned':18s} {'n':>4s} {'obeyed':>7s} | top realized")
    for name in sorted(matrix, key=lambda key: -sum(matrix[key].values())):
        total = sum(matrix[name].values())
        top = ", ".join(f"{k}:{v}" for k, v in matrix[name].most_common(4))
        print(f"{name:18s} {total:4d} {matrix[name].get(name, 0)/total:7.3f} | {top}")

    realized = Counter(classify_opener(row["text"]) for row in corpus.generated)
    total = len(corpus.generated)
    print(f"\n{'opener':18s} {'measured share':>14s} {'realized':>9s} {'ratio':>6s}")
    for name in sorted(set(shares) | set(realized), key=lambda k: -shares.get(k, 0)):
        share = shares.get(name, 0.0)
        got = realized.get(name, 0) / total
        ratio = got / share if share else float("inf")
        print(f"{name:18s} {share:14.4f} {got:9.4f} {ratio:6.2f}")

    print("\n== P(disagree | realized opener), reply pairs only ==\n")
    print(
        f"{'opener':18s} | {'Xprev':>6s} {'XP(d)':>6s} | {'Mprev':>6s} {'MP(d)':>6s} | "
        f"{'Gprev':>6s} {'GP(d)':>6s} | {'contrib':>8s}"
    )
    groups = {
        key: _by_opener(corpus.replies(rows))
        for key, rows in (
            ("X", corpus.excluded),
            ("M", corpus.matched),
            ("G", corpus.generated),
        )
    }
    sizes = {key: sum(len(v) for v in group.values()) for key, group in groups.items()}
    base = rate(corpus.replies(corpus.excluded))
    total_contrib = 0.0
    order = sorted(groups["X"], key=lambda k: -len(groups["X"][k]))
    for name in order:
        cells = {key: groups[key].get(name, []) for key in groups}
        prev = {key: len(cells[key]) / sizes[key] for key in cells}
        x_rate = rate(cells["X"])
        contrib = (prev["G"] - prev["M"]) * (x_rate - base if x_rate == x_rate else 0.0)
        total_contrib += contrib
        print(
            f"{name:18s} | {prev['X']:6.3f} {x_rate:6.3f} | {prev['M']:6.3f} {rate(cells['M']):6.3f} | "
            f"{prev['G']:6.3f} {rate(cells['G']):6.3f} | {contrib:+8.4f}"
        )
    gen_replies = corpus.replies(corpus.generated)
    print(
        f"\nprevalence contribution total {total_contrib:+.4f} against a reply-pair gap of "
        f"{rate(gen_replies) - rate(corpus.replies(corpus.matched)):+.4f}"
    )
    excess = [
        row
        for row in gen_replies
        if classify_opener(row["text"]) == "polarity_token"
        and row["planned_opener"] not in ("polarity_token", "?")
    ]
    print(
        f"reply slots that prepended a polarity token they were not assigned: {len(excess)} "
        f"of {len(gen_replies)} ({len(excess)/len(gen_replies):.3f})"
    )
    print("  by planned class:", Counter(r["planned_opener"] for r in excess).most_common())


def _by_opener(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        out[classify_opener(row["text"])].append(row)
    return out


# --------------------------------------------------------------------------- #
# echo: replies re-use the parent's words 1.4-1.6x as often as real replies
# --------------------------------------------------------------------------- #


def cmd_echo(corpus: Corpus) -> None:
    """Parent echo is real, monotone, and survives controlling for length."""

    print("== share of the reply's content words that appear in its parent ==\n")
    for name, rows in (
        ("generated", corpus.replies(corpus.generated)),
        ("matched-real", corpus.replies(corpus.matched)),
        ("excluded-real", corpus.replies(corpus.excluded)),
    ):
        print(f"{name:16s} n={len(rows):5d} mean={statistics.mean(parent_echo(r) for r in rows):.4f}")

    print("\n== P(disagree) rises monotonically with parent echo (excluded real) ==\n")
    bins = ((0, 0.05), (0.05, 0.10), (0.10, 0.15), (0.15, 0.20), (0.20, 0.30), (0.30, 1.01))
    print(f"{'echo bin':12s} | {'Xn':>6s} {'Xrate':>6s} | {'Gn':>5s} {'Grate':>6s}")
    x_replies = corpus.replies(corpus.excluded)
    g_replies = corpus.replies(corpus.generated)
    for low, high in bins:
        x_cell = [r for r in x_replies if low <= parent_echo(r) < high]
        g_cell = [r for r in g_replies if low <= parent_echo(r) < high]
        print(
            f"[{low:.2f},{high:.2f}) | {len(x_cell):6d} {rate(x_cell):6.3f} | "
            f"{len(g_cell):5d} {rate(g_cell):6.3f}"
        )
    counterfactual = sum(
        (len([r for r in x_replies if low <= parent_echo(r) < high]) / len(x_replies))
        * (rate([r for r in g_replies if low <= parent_echo(r) < high]) or 0.0)
        for low, high in bins
    )
    print(
        f"\ngenerated conditionals at the real echo distribution -> {counterfactual:.4f} "
        f"(shipped {rate(g_replies):.4f}, excluded real {rate(x_replies):.4f})"
    )

    print("\n== not a length artifact: echo is higher in every length cell ==\n")
    parent_bins = ((0, 25), (25, 80), (80, 200), (200, 10**6))
    reply_bins = ((0, 15), (15, 35), (35, 70), (70, 10**6))
    print(f"{'parent w':>12s} {'reply w':>12s} | {'realN':>6s} {'realEcho':>8s} | {'genN':>5s} {'genEcho':>8s} {'ratio':>6s}")
    for p_low, p_high in parent_bins:
        for r_low, r_high in reply_bins:
            def cell(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
                return [
                    r
                    for r in rows
                    if p_low <= r["parent_words"] < p_high
                    and r_low <= r["words"] < r_high
                ]

            x_cell, g_cell = cell(x_replies), cell(g_replies)
            if len(x_cell) < 40 or len(g_cell) < 8:
                continue
            x_echo = statistics.mean(parent_echo(r) for r in x_cell)
            g_echo = statistics.mean(parent_echo(r) for r in g_cell)
            print(
                f"{p_low:5d}-{min(p_high, 999):5d} {r_low:5d}-{min(r_high, 999):5d} | "
                f"{len(x_cell):6d} {x_echo:8.4f} | {len(g_cell):5d} {g_echo:8.4f} {g_echo/x_echo:6.2f}"
            )


# --------------------------------------------------------------------------- #
# surrogate: what the head reads
# --------------------------------------------------------------------------- #


def cmd_surrogate(corpus: Corpus) -> None:
    """The head is a reply-text classifier; the parent adds little."""

    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold

    rows = corpus.replies(corpus.excluded)
    y = np.array([1 if r["pred"] == "disagree" else 0 for r in rows])
    m = np.array([margin(r) for r in rows])
    groups = np.array([r["thread_id"] for r in rows])
    print("== held-out surrogate for the stance head, excluded real reply pairs ==\n")
    for label, texts in (
        ("reply only", [r["text"] for r in rows]),
        ("parent only", [r["parent_text"] for r in rows]),
        ("parent + reply", [r["parent_text"] + " || " + r["text"] for r in rows]),
    ):
        vec = TfidfVectorizer(min_df=5, ngram_range=(1, 2), sublinear_tf=True)
        features = vec.fit_transform(texts)
        aucs, r2s = [], []
        for train, test in GroupKFold(n_splits=5).split(features, y, groups):
            clf = LogisticRegression(max_iter=2000).fit(features[train], y[train])
            aucs.append(roc_auc_score(y[test], clf.predict_proba(features[test])[:, 1]))
            reg = Ridge(alpha=1.0).fit(features[train], m[train])
            pred = reg.predict(features[test])
            r2s.append(1 - ((m[test] - pred) ** 2).sum() / ((m[test] - m[test].mean()) ** 2).sum())
        print(f"{label:16s} vocab={features.shape[1]:6d}  AUC={np.mean(aucs):.4f}  margin R^2={np.mean(r2s):.4f}")

    vec = TfidfVectorizer(min_df=5, ngram_range=(1, 2), sublinear_tf=True)
    features = vec.fit_transform([r["text"] for r in rows])
    reg = Ridge(alpha=1.0).fit(features, m)
    names = np.array(vec.get_feature_names_out())
    order = np.argsort(reg.coef_)
    print("\nthe `disagree` class is keyed by explicit stance tokens, agreement included:")
    print("  +:", ", ".join(f"{names[i]}({reg.coef_[i]:+.3f})" for i in order[::-1][:16]))


# --------------------------------------------------------------------------- #
# ablate: causal edits, scored with the evaluator's own scorer
# --------------------------------------------------------------------------- #


def _scorer() -> Any:
    import importlib.util

    spec = importlib.util.spec_from_file_location("stance_scorer", SCORER)
    module = importlib.util.module_from_spec(spec)
    sys.modules["stance_scorer"] = module
    spec.loader.exec_module(module)
    return module


def _score(module: Any, rows: list[dict[str, Any]], transform: Callable[[dict], str]) -> list[dict]:
    pairs = [
        module.StancePair(
            thread_id=row["thread_id"],
            thread_title="",
            parent_id="p",
            reply_id=row["reply_id"],
            parent_author="",
            reply_author="",
            parent_text=row["parent_text"],
            reply_text=transform(row),
            depth=1 if not row["is_root"] else 0,
        )
        for row in rows
    ]
    scorer = module.StanceRelScorer(
        model_dir=STANCE_MODEL,
        label_order=("disagree", "neutral", "agree"),
        device="cpu",
        max_length=256,
        graph_author="none",
    )
    scored = scorer.score_pairs(pairs, batch_size=32)
    return [{"pred": row["pred_label"]} for row in scored]


def _strip_polarity(text: str) -> str:
    out = POLARITY_OPEN.sub("", text, count=1).strip()
    if not out:
        return text
    return out[0].upper() + out[1:] if out[0].islower() else out


def cmd_ablate(corpus: Corpus) -> None:
    """Fidelity check, then the one causal edit that moves the metric."""

    module = _scorer()
    shipped = rate(corpus.generated)
    reproduced = _score(module, corpus.generated, lambda row: row["text"])
    print("== fidelity: the harness must reproduce the artifact before it edits it ==\n")
    agreement = sum(
        1 for a, b in zip(reproduced, corpus.generated) if a["pred"] == b["pred"]
    ) / len(corpus.generated)
    print(f"shipped {shipped:.4f}  re-scored {rate(reproduced):.4f}  label agreement {agreement:.4f}")
    if agreement < 1.0:
        print("  FIDELITY FAILED -- do not read anything below this line.")
        return

    replies = corpus.replies(corpus.generated)
    excess = {
        (row["thread_id"], row["reply_id"])
        for row in replies
        if classify_opener(row["text"]) == "polarity_token"
        and row["planned_opener"] not in ("polarity_token", "?")
    }
    every = {
        (row["thread_id"], row["reply_id"])
        for row in replies
        if classify_opener(row["text"]) == "polarity_token"
    }
    print("\n== causal edit, reply pairs ==\n")
    for label, keys in (
        ("baseline (v101 as shipped)", set()),
        ("obey the plan: strip the UNPLANNED polarity openers", excess),
        ("strip EVERY polarity opener (over-correction)", every),
    ):

        def edit(row: dict[str, Any], keys: set[tuple[str, str]] = keys) -> str:
            if (row["thread_id"], row["reply_id"]) in keys:
                return _strip_polarity(row["text"])
            return row["text"]

        scored = _score(module, replies, edit)
        touched = sum(1 for row in replies if edit(row) != row["text"])
        print(f"{label:54s} rate={rate(scored):.4f}  edited={touched}")
    print(f"{'matched real reply pairs':54s} rate={rate(corpus.replies(corpus.matched)):.4f}")


COMMANDS: dict[str, Callable[[Corpus], None]] = {
    "structure": cmd_structure,
    "openers": cmd_openers,
    "echo": cmd_echo,
    "surrogate": cmd_surrogate,
    "ablate": cmd_ablate,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=[*COMMANDS, "all"])
    parser.add_argument("--run", default=str(DEFAULT_RUN))
    args = parser.parse_args()
    corpus = Corpus(Path(args.run))
    names = list(COMMANDS) if args.command == "all" else [args.command]
    for name in names:
        print(f"\n{'#' * 76}\n# {name}\n{'#' * 76}\n")
        COMMANDS[name](corpus)


if __name__ == "__main__":
    main()
