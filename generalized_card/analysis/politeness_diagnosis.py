#!/usr/bin/env python3
"""Reproduce the v99 politeness diagnosis. No API calls, no model loading.

Every number in `tasks/v99-worklog.md` comes from one of the subcommands below.
They exist so that evidence is reproducible rather than described -- the same
reason a generator version has to be committed before it runs.

    python3 generalized_card/analysis/politeness_diagnosis.py markers
    python3 generalized_card/analysis/politeness_diagnosis.py experience
    python3 generalized_card/analysis/politeness_diagnosis.py dismissive
    python3 generalized_card/analysis/politeness_diagnosis.py lexical
    python3 generalized_card/analysis/politeness_diagnosis.py bands
    python3 generalized_card/analysis/politeness_diagnosis.py realization
    python3 generalized_card/analysis/politeness_diagnosis.py moves
    python3 generalized_card/analysis/politeness_diagnosis.py all

Discipline built into the shared loaders:

  * the 150-thread evaluation seed pool is excluded from anything fitted, and
    any derivation is fitted on half the excluded threads and scored on the
    other half, split by thread so no thread straddles the split
  * the real per-comment labels come from the evaluation classifier's own
    `politeness_results.json` tables, so no metric is approximated and no model
    is re-run
  * `lexical` needs scikit-learn; every other subcommand is stdlib only
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable

REPO = Path(__file__).resolve().parents[2]
REAL_DIR = REPO / "data/raw/discussions/camera_product"
SEED_POOL = REPO / "artifacts/generalized_card/seed_pools/camera_product_150_seed42.json"
DEFAULT_RUN = (
    REPO
    / "artifacts/generalized_card/runs"
    / "generalized_card_camera_gpt54_v98_rhythm_n10_20260820_v1"
)

TOKEN = re.compile(r"[a-z']+")
BANDS = ((0, 15), (15, 30), (30, 60), (60, 120), (120, 10**6))

# Hand-listed only for the "does the derived set agree with an obvious one" check
# in `markers`; nothing downstream depends on it.
WARMTH = frozenset(
    """thank thanks appreciate love loved nice great awesome amazing excellent
    fantastic perfect beautiful glad happy enjoy enjoying excited incredible
    favorite truly helpful""".split()
)

# Surface families. Form, never topic, so a finding transfers to another domain.
FAMILIES: dict[str, re.Pattern[str]] = {
    "negate_premise": re.compile(
        r"\b(?:i don'?t (?:see|get|buy|think)|you can'?t|that'?s not|it'?s not|"
        r"isn'?t (?:the|a|clearly|really)|doesn'?t (?:change|matter|move|tell)|"
        r"nobody|no one|hardly)\b"
    ),
    "dismiss_noun": re.compile(
        r"\b(?:junk|noise|useless|pointless|fluff|dead weight|hand-?waving|"
        r"nonsense|meaningless|worthless|marketing speak|spec[- ]sheet|a wash|"
        r"moot|irrelevant|overkill|gimmick)\b"
    ),
    "adjudge": re.compile(
        r"\b(?:the only (?:thing|part|question|one) that (?:matters|counts)|"
        r"that'?s the (?:part|bit|thing|only) that|what actually matters|"
        r"the real question|what exactly is the point|the whole point is|"
        r"that'?s where it (?:matters|counts)|the deciding factor|"
        r"whether it actually matters)\b"
    ),
    "contrastive": re.compile(
        r"\b(?:but|though|although|however|still|yet|except|unless|whereas|"
        r"that said|even so|not that|other than)\b"
    ),
}

_IRREGULAR = (
    "got|took|bought|went|had|saw|made|found|came|gave|kept|sold|brought|ran|"
    "shot|put|sent|left|felt|thought|said|told|knew|held|won|lost|paid"
)
EXPERIENCE: dict[str, re.Pattern[str]] = {
    "i_past": re.compile(rf"\bi (?:\w+ed|{_IRREGULAR})\b"),
    "i_own": re.compile(r"\bi (?:have|own|use|shoot|run|carry|keep|prefer|love|like)\b"),
    "my_thing": re.compile(r"\bmy \w+"),
    "i_have_done": re.compile(r"\bi'?(?:ve| have) \w+"),
    "any_past": re.compile(rf"\b(?:\w+ed|{_IRREGULAR})\b"),
    "future": re.compile(r"\b(?:will|'ll|gonna|going to)\b"),
    "time_ref": re.compile(
        r"\b(?:yesterday|last (?:night|week|month|year|summer|winter)|"
        r"this morning|a (?:week|month|year)s? ago|recently|since)\b"
    ),
    "showing": re.compile(r"\b(?:here'?s|heres|attached|photo|pic|pics|shot)\b"),
}

# The four moves `register_realization` ships, plus the three it excluded, so the
# exclusion reasons stay checkable.
MOVES: dict[str, re.Pattern[str]] = {
    "any_intensifier": re.compile(
        r"\b(?:very|really|super|pretty|so|incredibly|absolutely|definitely)\b"
    ),
    "plain_verdict": re.compile(
        r"\b(?:great|good|excellent|fantastic|awesome|amazing|perfect|lovely|"
        r"beautiful|incredible|superb|brilliant)\b"
    ),
    "own_thing": re.compile(r"\bmy \w+"),
    "love_like": re.compile(r"\b(?:love|loved|loving|adore|enjoy|enjoyed|enjoying)\b"),
    "gratitude": re.compile(r"\b(?:thank|thanks|thx|appreciate|appreciated)\b"),
    "reassure_you": re.compile(
        r"\b(?:you'?ll be fine|you can'?t go wrong|no problem|"
        r"you'?ll (?:love|enjoy|like)|good luck|hope (?:it|you|this|that))\b"
    ),
    "link": re.compile(r"https?://"),
}


# --------------------------------------------------------------------------- #
# shared loaders
# --------------------------------------------------------------------------- #


def toks(text: str) -> list[str]:
    return TOKEN.findall(str(text).lower())


def band_of(words: int) -> str:
    for low, high in BANDS:
        if low <= words < high:
            return f"{low}-{high if high < 10**6 else 'inf'}"
    return "?"


def _rows(path: Path, keep: set[str] | None = None) -> list[dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    out = []
    for thread in payload.get("threads") or []:
        if not isinstance(thread, dict):
            continue
        thread_id = str(thread.get("thread_id") or "")
        if keep is not None and thread_id not in keep:
            continue
        for comment in thread.get("comments") or []:
            text = str((comment or {}).get("text") or "")
            if not text.strip():
                continue
            out.append(
                {
                    "thread_id": thread_id,
                    "label": str(comment.get("pred_label") or ""),
                    "text": text,
                    "words": len(text.split()),
                }
            )
    return out


class Corpus:
    """The three comment sets every subcommand needs, loaded once."""

    def __init__(self, run: Path) -> None:
        self.run = run
        self.seeds = {
            str(row.get("source_raw_post_id") or "").strip()
            for row in json.loads(SEED_POOL.read_text()).get("seed_posts") or []
            if isinstance(row, dict)
        }
        real: list[dict[str, Any]] = []
        for product in sorted(REAL_DIR.iterdir()):
            real.extend(_rows(product / "politeness_results.json"))
        self.excluded = [row for row in real if row["thread_id"] not in self.seeds]
        self.generated = [
            row
            for path in sorted(run.glob("cleaned/*/politeness_results.json"))
            for row in _rows(path)
        ]
        matched_ids = self._matched_ids()
        self.matched = [row for row in real if row["thread_id"] in matched_ids]
        # Fitted on half the excluded threads, scored on the other half. Split by
        # thread so no thread contributes to both sides.
        ids = sorted({row["thread_id"] for row in self.excluded})
        fit_ids = set(ids[::2])
        self.fit = [row for row in self.excluded if row["thread_id"] in fit_ids]
        self.held = [row for row in self.excluded if row["thread_id"] not in fit_ids]

    def _matched_ids(self) -> set[str]:
        """The real threads this run was matched against, from its own artifact."""

        path = self.run / "matched_evaluation/matched_real_thread_scores.csv"
        if not path.is_file():
            return set()
        import csv

        with path.open() as handle:
            return {
                str(row.get("thread_id") or "").strip()
                for row in csv.DictReader(handle)
                if str(row.get("thread_id") or "").strip()
            }

    def describe(self) -> None:
        print(
            f"excluded real {len(self.excluded)} (fit {len(self.fit)} / held "
            f"{len(self.held)})   matched real {len(self.matched)}   "
            f"generated {len(self.generated)}"
        )


def share(rows: Iterable[dict[str, Any]], label: str) -> float:
    rows = list(rows)
    return sum(1 for row in rows if row["label"] == label) / max(1, len(rows))


def lift(
    rows: list[dict[str, Any]], predicate: Callable[[dict[str, Any]], bool], label: str
) -> tuple[float, float, float, float]:
    """Return P(label|hit), P(label|miss), the ratio, and the hit prevalence."""

    hit = [row for row in rows if predicate(row)]
    miss = [row for row in rows if not predicate(row)]
    p, q = share(hit, label), share(miss, label)
    return p, q, p / max(1e-9, q), len(hit) / max(1, len(rows))


def matches(pattern: re.Pattern[str]) -> Callable[[dict[str, Any]], bool]:
    return lambda row: bool(pattern.search(row["text"].lower()))


# --------------------------------------------------------------------------- #
# subcommands
# --------------------------------------------------------------------------- #


def cmd_markers(corpus: Corpus) -> None:
    """REJECTED: marker frequency. The gap is in the conditional, not the count."""

    print("=== polite register derived by token log-odds on the fit half ===")
    in_hit: Counter[str] = Counter()
    in_miss: Counter[str] = Counter()
    n_hit = n_miss = 0
    for row in corpus.fit:
        types = set(toks(row["text"]))
        if row["label"] == "polite":
            n_hit += 1
            in_hit.update(types)
        else:
            n_miss += 1
            in_miss.update(types)
    ranked = []
    for token in set(in_hit) | set(in_miss):
        docs = in_hit[token] + in_miss[token]
        if docs < 40:
            continue
        p = (in_hit[token] + 0.5) / (n_hit + 1)
        q = (in_miss[token] + 0.5) / (n_miss + 1)
        ranked.append((token, math.log(p / q), docs))
    ranked.sort(key=lambda item: -item[1])
    print("  top 15:", " ".join(token for token, _, _ in ranked[:15]))
    print()
    for top_n in (10, 20, 30, 50):
        markers = {token for token, _, _ in ranked[:top_n]}
        p, q, ratio, prev = lift(
            corpus.held, lambda row: bool(set(toks(row["text"])) & markers), "polite"
        )
        print(
            f"  top-{top_n:<3} held-out: prevalence {prev:.3f}  P(polite|marker) "
            f"{p:.3f}  P(polite|none) {q:.3f}  lift {ratio:.2f}x"
        )
    print()
    markers = {token for token, _, _ in ranked[:30]}
    print("=== the rejection: same markers, real against generated ===")
    print(f"{'corpus':<18} {'presence':>9} {'P(pol|mk)':>10} {'P(pol|none)':>12}")
    for name, rows in (
        ("excluded real", corpus.excluded),
        ("matched real", corpus.matched),
        ("generated", corpus.generated),
    ):
        p, q, _, prev = lift(
            rows, lambda row: bool(set(toks(row["text"])) & markers), "polite"
        )
        print(f"{name:<18} {prev:>9.3f} {p:>10.3f} {q:>12.3f}")
    gen_p, gen_q, _, _ = lift(
        corpus.generated,
        lambda row: bool(set(toks(row["text"])) & markers),
        "polite",
    )
    _, _, _, real_prev = lift(
        corpus.matched, lambda row: bool(set(toks(row["text"])) & markers), "polite"
    )
    print()
    print(
        f"  counterfactual -- generated at real prevalence, generated "
        f"conditionals: {real_prev * gen_p + (1 - real_prev) * gen_q:.3f}"
    )
    print(f"  generated actual {share(corpus.generated, 'polite'):.3f}   "
          f"matched real {share(corpus.matched, 'polite'):.3f}")


def cmd_experience(corpus: Corpus) -> None:
    """REJECTED: first-person lived experience. Warmth outlifts every feature."""

    print("=== P(polite | feature), fitted then scored out of sample ===")
    print(f"{'feature':<14} {'fit':>7} {'held':>7} {'P(pol|f)':>9} {'prev':>7}")
    for name, pattern in EXPERIENCE.items():
        _, _, l_fit, _ = lift(corpus.fit, matches(pattern), "polite")
        p, _, l_held, prev = lift(corpus.held, matches(pattern), "polite")
        print(f"{name:<14} {l_fit:>7.2f} {l_held:>7.2f} {p:>9.3f} {prev:>7.3f}")
    warm = lambda row: bool(set(toks(row["text"])) & WARMTH)  # noqa: E731
    p, _, l_held, prev = lift(corpus.held, warm, "polite")
    print(f"{'warmth':<14} {'':>7} {l_held:>7.2f} {p:>9.3f} {prev:>7.3f}")
    print()
    print("=== the rejection: a flat gap in every cell of warmth x experience ===")
    cells = (
        ("warmth + i_past", lambda r: warm(r) and matches(EXPERIENCE["i_past"])(r)),
        ("warmth only", lambda r: warm(r) and not matches(EXPERIENCE["i_past"])(r)),
        ("i_past only", lambda r: not warm(r) and matches(EXPERIENCE["i_past"])(r)),
        ("neither", lambda r: not warm(r) and not matches(EXPERIENCE["i_past"])(r)),
    )
    print(f"{'cell':<18} {'real P(pol)':>12} {'gen P(pol)':>11} {'real n':>7} {'gen n':>6}")
    for tag, predicate in cells:
        real = [row for row in corpus.matched if predicate(row)]
        gen = [row for row in corpus.generated if predicate(row)]
        print(
            f"{tag:<18} {share(real,'polite'):>12.3f} {share(gen,'polite'):>11.3f} "
            f"{len(real):>7} {len(gen):>6}"
        )


def cmd_dismissive(corpus: Corpus) -> None:
    """REJECTED as a cause, but two prevalence findings survive as tells."""

    print("=== out-of-sample lift on P(impolite), and the generated prevalence ===")
    print(f"{'family':<16} {'held lift':>10} {'excl real':>10} {'matched':>9} "
          f"{'generated':>10} {'gen/real':>9}")
    for name, pattern in FAMILIES.items():
        _, _, l_held, _ = lift(corpus.held, matches(pattern), "impolite")
        rates = [
            sum(1 for row in rows if matches(pattern)(row)) / max(1, len(rows))
            for rows in (corpus.excluded, corpus.matched, corpus.generated)
        ]
        print(
            f"{name:<16} {l_held:>10.2f} {rates[0]:>10.4f} {rates[1]:>9.4f} "
            f"{rates[2]:>10.4f} {rates[2]/max(1e-9,rates[1]):>8.2f}x"
        )
    print()
    print("=== the rejection: the counterfactual on excluded real ===")
    dismissive = ("negate_premise", "dismiss_noun", "adjudge")
    any_hit = lambda row: any(  # noqa: E731
        matches(FAMILIES[name])(row) for name in dismissive
    )
    p, q, _, prev_excl = lift(corpus.excluded, any_hit, "polite")
    print(f"  excluded real P(polite | dismissive)    = {p:.3f}")
    print(f"  excluded real P(polite | no dismissive) = {q:.3f}   <- no effect")
    _, _, _, prev_real = lift(corpus.matched, any_hit, "polite")
    gen_p, gen_q, _, prev_gen = lift(corpus.generated, any_hit, "polite")
    print(f"  prevalence: matched real {prev_real:.3f}  generated {prev_gen:.3f}")
    print(
        "  generated polite_rate if prevalence alone moved to real: "
        f"{prev_real * gen_p + (1 - prev_real) * gen_q:.3f}"
    )
    print()
    print("  and with no dismissive family present at all:")
    for name, rows in (("matched real", corpus.matched), ("generated", corpus.generated)):
        bare = [row for row in rows if not any_hit(row)]
        print(f"    {name:<14} n={len(bare):<5} P(polite)={share(bare,'polite'):.3f}")


def cmd_lexical(corpus: Corpus) -> None:
    """CONFIRMED: the gap is lexical, distributed, and a deficit not an excess."""

    try:
        import numpy as np
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
    except ImportError:
        print("`lexical` needs numpy and scikit-learn; every other subcommand does not.")
        return

    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_features=40000, sublinear_tf=True)
    x_fit = vec.fit_transform([row["text"] for row in corpus.fit])
    y_fit = np.array([1 if row["label"] == "polite" else 0 for row in corpus.fit])
    model = LogisticRegression(max_iter=2000, class_weight="balanced").fit(x_fit, y_fit)
    threshold = float(
        np.quantile(model.predict_proba(x_fit)[:, 1], 1 - y_fit.mean())
    )

    print("=== does a bag of words reproduce the classifier? ===")
    print(f"{'corpus':<20} {'actual':>8} {'model':>7} {'AUC':>6}")
    for name, rows in (
        ("held-out excluded", corpus.held),
        ("matched real", corpus.matched),
        ("generated", corpus.generated),
    ):
        x = vec.transform([row["text"] for row in rows])
        prob = model.predict_proba(x)[:, 1]
        y = np.array([1 if row["label"] == "polite" else 0 for row in rows])
        auc = roc_auc_score(y, prob) if 0 < y.sum() < len(y) else float("nan")
        print(f"{name:<20} {y.mean():>8.3f} {(prob>=threshold).mean():>7.3f} {auc:>6.3f}")

    coef = model.coef_[0]
    x_gen = vec.transform([row["text"] for row in corpus.generated])
    x_mat = vec.transform([row["text"] for row in corpus.matched])
    df_gen = np.asarray((x_gen > 0).mean(axis=0)).ravel()
    df_mat = np.asarray((x_mat > 0).mean(axis=0)).ravel()
    pos, neg = coef > 0, coef < 0
    print()
    print(f"  polite-feature deficit in generated : "
          f"{float((coef[pos]*(df_mat[pos]-df_gen[pos])).sum()):+.3f}")
    print(f"  impolite-feature excess in generated: "
          f"{float((-coef[neg]*(df_gen[neg]-df_mat[neg])).sum()):+.3f}")
    print("  (a negative excess means generated uses LESS of the impolite vocabulary)")
    print()
    print("=== per-1,000-token rate for the strongest polite unigrams ===")
    real_counts, gen_counts = Counter(), Counter()
    for row in corpus.matched:
        real_counts.update(toks(row["text"]))
    for row in corpus.generated:
        gen_counts.update(toks(row["text"]))
    real_n = sum(real_counts.values()) / 1000
    gen_n = sum(gen_counts.values()) / 1000
    names = vec.get_feature_names_out()
    print(f"{'token':<16} {'real/1k':>9} {'gen/1k':>8} {'ratio':>7}")
    shown = 0
    for index in np.argsort(-coef):
        token = names[index]
        if " " in token or shown >= 16:
            continue
        r, g = real_counts[token] / real_n, gen_counts[token] / gen_n
        print(f"{token:<16} {r:>9.3f} {g:>8.3f} {g/max(1e-9,r):>6.2f}x")
        shown += 1


def cmd_bands(corpus: Corpus) -> None:
    """CONFIRMED: the gap grows monotonically with comment length."""

    print("=== polite_rate by realized length ===")
    print(f"{'band':<10} {'real n':>7} {'gen n':>6} {'real P(pol)':>12} {'gen P(pol)':>11}")
    for low, high in BANDS:
        key = f"{low}-{high if high < 10**6 else 'inf'}"
        real = [r for r in corpus.matched if low <= r["words"] < high]
        gen = [r for r in corpus.generated if low <= r["words"] < high]
        if not real or not gen:
            continue
        print(
            f"{key:<10} {len(real):>7} {len(gen):>6} {share(real,'polite'):>12.3f} "
            f"{share(gen,'polite'):>11.3f}"
        )
    print()
    print(f"mean words: matched real "
          f"{statistics.mean(r['words'] for r in corpus.matched):.1f}  "
          f"generated {statistics.mean(r['words'] for r in corpus.generated):.1f}")


def cmd_realization(corpus: Corpus) -> None:
    """CONFIRMED: the plan is right, realization is the whole failure."""

    plans: dict[str, dict[str, Any]] = {}

    def walk(comments, out):
        for comment in comments or []:
            out.append(comment)
            walk(comment.get("replies"), out)

    for path in sorted(corpus.run.glob("generated/*/discussion.json")):
        payload = json.loads(path.read_text())
        for post in payload.get("posts") or []:
            flat: list[dict] = []
            walk(post.get("comments"), flat)
            for comment in flat:
                text = " ".join(str(comment.get("content") or "").split())
                if text:
                    plans[text] = comment
    labels = {" ".join(row["text"].split()): row["label"] for row in corpus.generated}
    joined = [
        {
            "label": labels[text],
            "planned": str(plan.get("tone_target") or "").strip().lower(),
            "words": len(text.split()),
        }
        for text, plan in plans.items()
        if text in labels
    ]
    if not joined:
        print("no plan/label join available for this run")
        return
    print(f"joined {len(joined)} comments")
    print()
    print("=== planned marginal against realized marginal ===")
    planned, realized = Counter(r["planned"] for r in joined), Counter(r["label"] for r in joined)
    for key in ("polite", "somewhat_polite", "neutral", "impolite"):
        print(f"  {key:<16} planned {planned[key]/len(joined):.3f}   "
              f"realized {realized[key]/len(joined):.3f}   "
              f"matched real {share(corpus.matched, key):.3f}")
    print()
    print("=== confusion matrix, planned -> realized ===")
    keys = ("polite", "somewhat_polite", "neutral", "impolite")
    print(f"{'planned':<16} {'n':>4}  " + "  ".join(f"{k:>15}" for k in keys))
    for tone in keys:
        rows = [r for r in joined if r["planned"] == tone]
        if not rows:
            continue
        counts = Counter(r["label"] for r in rows)
        cells = "  ".join(f"{counts[k]/len(rows):>15.3f}" for k in keys)
        print(f"{tone:<16} {len(rows):>4}  {cells}")
    print()
    print("=== planned polite by band, and whether it lands ===")
    by_band: dict[str, list[dict]] = defaultdict(list)
    for row in joined:
        by_band[band_of(row["words"])].append(row)
    print(f"{'band':<10} {'n':>4} {'planned polite':>15} {'landed':>8}")
    for low, high in BANDS:
        key = f"{low}-{high if high < 10**6 else 'inf'}"
        rows = by_band.get(key) or []
        if not rows:
            continue
        pp = [r for r in rows if r["planned"] == "polite"]
        print(f"{key:<10} {len(rows):>4} {len(pp)/len(rows):>15.3f} "
              f"{share(pp,'polite'):>8.3f}")


def cmd_moves(corpus: Corpus) -> None:
    """The profile `register_realization` ships, and the exclusion reasons."""

    print("=== each move: out-of-sample lift, real prevalence, generated rate ===")
    print(f"{'move':<18} {'held lift':>10} {'real prev':>10} {'gen prev':>9} "
          f"{'gen/real':>9}  verdict")
    verdicts = {
        "any_intensifier": "shipped",
        "plain_verdict": "shipped",
        "own_thing": "shipped",
        "love_like": "shipped",
        "gratitude": "excluded: generated already above real",
        "reassure_you": "excluded: real prevalence too low",
        "link": "excluded: needs a real URL",
    }
    for name, pattern in MOVES.items():
        _, _, l_held, _ = lift(corpus.held, matches(pattern), "polite")
        pr = sum(1 for r in corpus.matched if matches(pattern)(r)) / len(corpus.matched)
        pg = sum(1 for r in corpus.generated if matches(pattern)(r)) / len(corpus.generated)
        print(
            f"{name:<18} {l_held:>10.2f} {pr:>10.3f} {pg:>9.3f} "
            f"{pg/max(1e-9,pr):>8.2f}x  {verdicts.get(name,'')}"
        )
    print()
    print("=== the shipped profile: share among excluded real `polite` comments ===")
    polite = [row for row in corpus.excluded if row["label"] == "polite"]
    shipped = ("any_intensifier", "plain_verdict", "own_thing", "love_like", "gratitude")
    print(f"{'band':<10} {'n':>5}  " + "  ".join(f"{m[:15]:>15}" for m in shipped))
    for low, high in BANDS:
        rows = [r for r in polite if low <= r["words"] < high]
        if len(rows) < 40:
            continue
        cells = "  ".join(
            f"{sum(1 for r in rows if matches(MOVES[m])(r))/len(rows):>15.3f}"
            for m in shipped
        )
        print(f"{band_of(low):<10} {len(rows):>5}  {cells}")
    print()
    print("  `gratitude` runs backwards to every other move, which is why one")
    print("  flat warmth cue would be wrong at both ends of the length range.")


COMMANDS: dict[str, Callable[[Corpus], None]] = {
    "markers": cmd_markers,
    "experience": cmd_experience,
    "dismissive": cmd_dismissive,
    "lexical": cmd_lexical,
    "bands": cmd_bands,
    "realization": cmd_realization,
    "moves": cmd_moves,
}


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("command", choices=[*COMMANDS, "all"])
    parser.add_argument(
        "--run",
        type=Path,
        default=DEFAULT_RUN,
        help="run directory to analyse (default: the v98 N=10 run)",
    )
    args = parser.parse_args()
    if not args.run.is_dir():
        print(f"run directory not found: {args.run}", file=sys.stderr)
        return 2
    corpus = Corpus(args.run.resolve())
    corpus.describe()
    print()
    names = list(COMMANDS) if args.command == "all" else [args.command]
    for name in names:
        print("#" * 78)
        print(f"# {name}  --  {COMMANDS[name].__doc__.splitlines()[0]}")
        print("#" * 78)
        COMMANDS[name](corpus)
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
