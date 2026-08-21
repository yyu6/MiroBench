#!/usr/bin/env python3
"""Why `polite_rate` and `impolite_rate` have failed since v96.

Eight versions moved sub-sentence vocabulary and none moved the metric. This
script establishes why: Polite Guard decides on whether the comment contains one
*whole sentence* that is unambiguously appreciative, and generated text almost
never writes one.

Everything here is offline. It reads the shipped per-comment classifier tables
under `data/raw/discussions/` and a run artifact, and it re-runs the evaluation's
own `Intel/polite-guard` checkpoint locally for the sentence-level and
counterfactual work. No API call is made.

Discipline, same as the other two diagnosis scripts:

  * every profile is fitted on evaluation-excluded threads only (the 150-thread
    seed pool is held out of anything derived)
  * `sentences` and `counterfactual` refuse to print a derived number until the
    local harness has reproduced the shipped artifact's labels exactly
  * donor sentences come from excluded real threads and are reported by surface
    form, never by topic, so a finding transfers to another domain

  python3 generalized_card/analysis/polite_sentence_diagnosis.py bands
  python3 generalized_card/analysis/polite_sentence_diagnosis.py sentences
  python3 generalized_card/analysis/polite_sentence_diagnosis.py carriers
  python3 generalized_card/analysis/polite_sentence_diagnosis.py profile
  python3 generalized_card/analysis/polite_sentence_diagnosis.py counterfactual
  python3 generalized_card/analysis/polite_sentence_diagnosis.py all
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

REPO = Path(__file__).resolve().parents[2]
REAL_DIR = REPO / "data/raw/discussions/camera_product"
SEED_POOL = REPO / "artifacts/generalized_card/seed_pools/camera_product_150_seed42.json"
DEFAULT_RUN = (
    REPO
    / "artifacts/generalized_card/runs"
    / "generalized_card_camera_gpt54_v103_stance_opening_n10_20260821_v1"
)

# The threshold that defines an "unambiguously appreciative" sentence. 0.80 is
# well clear of the decision boundary; the finding is flat between 0.70 and 0.90.
POLITE_SENTENCE = 0.80
BANDS = ((0, 15), (15, 30), (30, 60), (60, 120), (120, 10**6))
SENTENCE = re.compile(r"(?<=[.!?])\s+|\n+")

# Surface families, form and never topic, so the taxonomy transfers to another
# domain. Coverage of the harvested set is reported by `sentences` -- it is
# currently partial, which is why a mechanism must ship the families it can
# actually name rather than "a sentence Polite Guard likes".
FORMS: dict[str, re.Pattern[str]] = {
    "gratitude": re.compile(r"\b(?:thank|thanks|thx|appreciate)\b", re.I),
    "praise_object": re.compile(
        r"\b(?:nice|great|good|awesome|lovely|beautiful|solid|amazing|excellent|perfect)\b"
        r".{0,40}\b(?:shot|work|pic|photo|job|choice|call|find|set|kit|one|camera|lens)\b"
        r"|\b(?:congrat|well done)\b",
        re.I,
    ),
    "well_wish": re.compile(
        r"\b(?:good luck|have fun|enjoy|hope (?:you|it|that|this)|"
        r"you'?ll (?:love|enjoy|be fine|do great))\b",
        re.I,
    ),
    "self_positive": re.compile(
        r"\b(?:i (?:love|loved|really like|adore|enjoy|enjoyed))\b"
        r"|\b(?:love|really like) (?:my|the|it|this|that)\b",
        re.I,
    ),
    "affirm_other": re.compile(
        r"^\s*(?:this|that|yes|exactly|agreed|same|seconding|absolutely|definitely|totally)\b",
        re.I,
    ),
    "superlative": re.compile(
        r"\b(?:best|favou?rite|greatest|most fun|never regret)\b", re.I
    ),
    "offer_help": re.compile(
        r"\b(?:happy to|let me know|feel free|dm me|if you (?:want|need|have))\b", re.I
    ),
}


# --------------------------------------------------------------------------- #
# loaders
# --------------------------------------------------------------------------- #


def band_of(words: int) -> str:
    for low, high in BANDS:
        if low <= words < high:
            return f"{low}-{high if high < 10**6 else 'inf'}"
    return "?"


def _rows(path: Path) -> list[dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    out = []
    for thread in payload.get("threads") or []:
        if not isinstance(thread, dict):
            continue
        for comment in thread.get("comments") or []:
            text = str((comment or {}).get("text") or "")
            if not text.strip():
                continue
            out.append(
                {
                    "thread_id": str(thread.get("thread_id") or ""),
                    "text": text,
                    "words": len(text.split()),
                    "label": str(comment.get("pred_label") or ""),
                    "polite": float(comment.get("polite_probability") or 0.0),
                }
            )
    return out


class Corpus:
    """Excluded real, matched real and generated comments, loaded once."""

    def __init__(self, run: Path) -> None:
        self.run = run
        seeds = {
            str(row.get("source_raw_post_id") or "").strip()
            for row in json.loads(SEED_POOL.read_text()).get("seed_posts") or []
            if isinstance(row, dict)
        }
        real = [row for product in sorted(REAL_DIR.iterdir())
                for row in _rows(product / "politeness_results.json")]
        self.excluded = [row for row in real if row["thread_id"] not in seeds]
        matched_ids = self._matched_ids()
        self.matched = [row for row in real if row["thread_id"] in matched_ids]
        self.generated = [
            row
            for path in sorted(run.glob("cleaned/*/politeness_results.json"))
            for row in _rows(path)
        ]

    def _matched_ids(self) -> set[str]:
        path = self.run / "matched_evaluation/matched_real_thread_scores.csv"
        if not path.is_file():
            return set()
        with path.open() as handle:
            return {
                str(row.get("thread_id") or "").strip()
                for row in csv.DictReader(handle)
                if str(row.get("thread_id") or "").strip()
            }

    def describe(self) -> None:
        print(
            f"excluded real {len(self.excluded)}   matched real {len(self.matched)}   "
            f"generated {len(self.generated)}"
        )


def share(rows: list[dict[str, Any]], label: str) -> float:
    return sum(1 for row in rows if row["label"] == label) / max(1, len(rows))


def sentences(text: str) -> list[str]:
    parts = (part.strip() for part in SENTENCE.split(" ".join(text.split())))
    return [part for part in parts if len(part.split()) >= 2]


def load_scorer() -> Callable[[list[str]], list[dict[str, float]]]:
    """The evaluation's own checkpoint, same settings as `score_thread_politeness`."""

    try:
        import torch
        from transformers import (
            AutoConfig,
            AutoModelForSequenceClassification,
            AutoTokenizer,
        )
    except ImportError as exc:  # pragma: no cover - environment guard
        raise SystemExit("needs torch and transformers (system python3 here)") from exc

    name = "Intel/polite-guard"
    config = AutoConfig.from_pretrained(name)
    index = {int(i): str(label).replace(" ", "_") for i, label in config.id2label.items()}
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSequenceClassification.from_pretrained(name).eval()

    def score(texts: list[str], batch: int = 64) -> list[dict[str, float]]:
        out: list[dict[str, float]] = []
        for start in range(0, len(texts), batch):
            encoded = tokenizer(
                texts[start : start + batch],
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt",
            )
            with torch.inference_mode():
                probs = torch.softmax(model(**encoded).logits, dim=-1).tolist()
            out.extend({index[i]: float(row[i]) for i in range(len(row))} for row in probs)
        return out

    return score


def assert_reproduces(score, rows: list[dict[str, Any]], n: int = 120) -> None:
    """Refuse to print a derived number from a harness that is not the scorer."""

    probe = rows[:n]
    got = score([row["text"] for row in probe])
    flips = sum(1 for row, g in zip(probe, got) if max(g, key=g.get) != row["label"])
    drift = max(abs(g["polite"] - row["polite"]) for row, g in zip(probe, got))
    print(f"[fidelity] n={len(probe)}  label flips={flips}  max |dP(polite)|={drift:.6f}")
    if flips or drift > 1e-4:
        sys.exit("harness does not reproduce the artifact -- refusing to print anything derived")


def carrier_index(score, rows: list[dict[str, Any]]) -> list[list[float]]:
    """Per-comment list of sentence-level P(polite)."""

    flat = [(i, s) for i, row in enumerate(rows) for s in sentences(row["text"])]
    scored = score([s for _, s in flat])
    per: list[list[float]] = [[] for _ in rows]
    for (i, _), g in zip(flat, scored):
        per[i].append(g["polite"])
    return per


# --------------------------------------------------------------------------- #
# subcommands
# --------------------------------------------------------------------------- #


def cmd_bands(corpus: Corpus) -> None:
    """REJECTED: the length mix. The gap is the within-band conditional."""

    def mix(rows):
        counts: dict[str, int] = defaultdict(int)
        for row in rows:
            counts[band_of(row["words"])] += 1
        return {k: v / len(rows) for k, v in counts.items()}

    def cond(rows, label):
        out = {}
        for low, high in BANDS:
            key = f"{low}-{high if high < 10**6 else 'inf'}"
            sub = [r for r in rows if band_of(r["words"]) == key]
            out[key] = share(sub, label) if sub else 0.0
        return out

    print("=== P(label | length band) ===")
    print(f"{'band':<10}{'real n':>8}{'gen n':>7}{'real pol':>10}{'gen pol':>9}"
          f"{'real imp':>10}{'gen imp':>9}")
    for low, high in BANDS:
        key = f"{low}-{high if high < 10**6 else 'inf'}"
        real = [r for r in corpus.matched if band_of(r["words"]) == key]
        gen = [r for r in corpus.generated if band_of(r["words"]) == key]
        if not real or not gen:
            continue
        print(f"{key:<10}{len(real):>8}{len(gen):>7}{share(real,'polite'):>10.3f}"
              f"{share(gen,'polite'):>9.3f}{share(real,'impolite'):>10.3f}"
              f"{share(gen,'impolite'):>9.3f}")
    print()
    for label in ("polite", "impolite"):
        mr, mg = mix(corpus.matched), mix(corpus.generated)
        cr, cg = cond(corpus.matched, label), cond(corpus.generated, label)
        act_r, act_g = share(corpus.matched, label), share(corpus.generated, label)
        gap = act_r - act_g
        only_mix = sum(mr[k] * cg.get(k, 0.0) for k in mr)
        only_cond = sum(mg[k] * cr.get(k, 0.0) for k in mg)
        print(f"{label}: real {act_r:.4f}  generated {act_g:.4f}  gap {gap:+.4f}")
        print(f"   fix only the length mix       -> {only_mix:.4f}  "
              f"closes {(only_mix-act_g)/gap*100:5.1f}%")
        print(f"   fix only the within-band rate -> {only_cond:.4f}  "
              f"closes {(only_cond-act_g)/gap*100:5.1f}%")


def cmd_sentences(corpus: Corpus) -> None:
    """CONFIRMED: the deficit is at sentence level and flat across position."""

    score = load_scorer()
    assert_reproduces(score, corpus.generated)
    for name, rows in (("MATCHED REAL", corpus.matched), ("GENERATED", corpus.generated)):
        longs = [r for r in rows if r["words"] >= 40]
        flat = []
        for row in longs:
            parts = sentences(row["text"])
            flat.extend((j, part, len(parts)) for j, part in enumerate(parts))
        scored = score([part for _, part, _ in flat])
        print(f"\n=== {name}  40w+ comments {len(longs)}  sentences {len(flat)} ===")
        print(f"  comment-level mean P(polite) {statistics.mean(r['polite'] for r in longs):.3f}")
        print(f"  {'position':<10}{'n':>6}{'mean P(pol)':>13}")
        for label, keep in (("first", lambda j, n: j == 0),
                            ("middle", lambda j, n: 0 < j < n - 1),
                            ("last", lambda j, n: j == n - 1 and n > 1)):
            vals = [g["polite"] for (j, _, n), g in zip(flat, scored) if keep(j, n)]
            if vals:
                print(f"  {label:<10}{len(vals):>6}{statistics.mean(vals):>13.3f}")
        hits = [g["polite"] for g in scored]
        print(f"  sentences individually P(polite)>{POLITE_SENTENCE}: "
              f"{sum(1 for h in hits if h > POLITE_SENTENCE)/len(hits):.3f}")


def cmd_carriers(corpus: Corpus) -> None:
    """CONFIRMED: one binary feature reconstructs both failing rates."""

    score = load_scorer()
    assert_reproduces(score, corpus.generated)
    random.seed(31)
    for name, rows, cap in (("EXCLUDED REAL", corpus.excluded, 2200),
                            ("GENERATED", corpus.generated, 10**6)):
        sample = rows if len(rows) <= cap else random.sample(rows, cap)
        per = carrier_index(score, sample)
        carries = [i for i, ps in enumerate(per) if any(p > POLITE_SENTENCE for p in ps)]
        rest = [i for i in range(len(sample)) if i not in set(carries)]
        prevalence = len(carries) / len(sample)
        print(f"\n{name}  n={len(sample)}   carries a polite sentence {prevalence:.3f}")
        for label in ("polite", "impolite"):
            a = share([sample[i] for i in carries], label)
            b = share([sample[i] for i in rest], label)
            print(f"   P({label:<8} | carries) {a:.3f}   | does not {b:.3f}"
                  f"   -> reconstructed rate {prevalence*a + (1-prevalence)*b:.3f}"
                  f"   actual {share(sample, label):.3f}")


def cmd_profile(corpus: Corpus) -> None:
    """The per-band profile a mechanism would ship, and where the forms are."""

    score = load_scorer()
    assert_reproduces(score, corpus.generated)
    random.seed(19)
    rates: dict[str, dict[str, float]] = {}
    for name, rows, cap in (("EXCLUDED REAL", corpus.excluded, 3000),
                            ("GENERATED", corpus.generated, 10**6)):
        sample = rows if len(rows) <= cap else random.sample(rows, cap)
        per = carrier_index(score, sample)
        print(f"\n=== {name}: share carrying a P(polite)>{POLITE_SENTENCE} sentence ===")
        print(f"  {'band':<10}{'n':>6}{'carries':>10}{'rel position':>14}")
        rates[name] = {}
        for low, high in BANDS:
            key = f"{low}-{high if high < 10**6 else 'inf'}"
            idx = [i for i, r in enumerate(sample) if band_of(r["words"]) == key]
            if len(idx) < 30:
                continue
            hit = [i for i in idx if any(p > POLITE_SENTENCE for p in per[i])]
            pos = [min(j for j, p in enumerate(per[i]) if p > POLITE_SENTENCE) / (len(per[i]) - 1)
                   for i in hit if len(per[i]) > 1]
            rates[name][key] = len(hit) / len(idx)
            print(f"  {key:<10}{len(idx):>6}{len(hit)/len(idx):>10.3f}"
                  f"{(statistics.mean(pos) if pos else float('nan')):>14.3f}")
    # what those sentences are, by form
    sample = random.sample(corpus.excluded, 2500)
    flat = [s for r in sample for s in sentences(r["text"])]
    scored = score(flat)
    harvest = [s for s, g in zip(flat, scored) if g["polite"] > POLITE_SENTENCE]
    print(f"\n=== the {len(harvest)} harvested sentences, by surface form "
          f"({len(harvest)/len(flat):.3f} of all excluded-real sentences) ===")
    widths = sorted(len(s.split()) for s in harvest)
    print(f"  length: median {widths[len(widths)//2]} words, p90 {widths[int(.9*len(widths))]}")
    covered: set[int] = set()
    for name, pattern in FORMS.items():
        hits = {i for i, s in enumerate(harvest) if pattern.search(s)}
        covered |= hits
        print(f"  {name:<16}{len(hits)/len(harvest):>7.3f}")
    print(f"  {'ANY named form':<16}{len(covered)/len(harvest):>7.3f}"
          f"   <- the rest are not yet nameable; a mechanism must ship forms, not a score")


def cmd_counterfactual(corpus: Corpus) -> None:
    """CAUSAL: insert one real polite sentence, re-score, at the real per-band rate."""

    score = load_scorer()
    assert_reproduces(score, corpus.generated)
    random.seed(23)
    donors_pool = random.sample(corpus.excluded, 2500)
    flat = [s for r in donors_pool for s in sentences(r["text"])]
    scored = score(flat)
    donors = [s for s, g in zip(flat, scored)
              if g["polite"] > 0.90 and 4 <= len(s.split()) <= 14]
    controls = [s for s, g in zip(flat, scored)
                if 0.02 < g["polite"] < 0.10 and 4 <= len(s.split()) <= 14]
    print(f"donor pool {len(donors)}   control pool {len(controls)}")

    targets = [r for r in corpus.generated if r["words"] >= 40 and r["label"] != "polite"]
    base = statistics.mean(r["polite"] for r in targets)
    print(f"\n=== insert ONE sentence into {len(targets)} generated non-polite 40w+ comments ===")
    for where, pool in (("front", donors), ("end", donors), ("after_first", donors),
                        ("end (CONTROL)", controls)):
        edited = []
        for k, row in enumerate(targets):
            donor = pool[k % len(pool)]
            parts = sentences(row["text"])
            if where == "front":
                edited.append(donor + " " + " ".join(parts))
            elif where == "after_first":
                edited.append(" ".join(parts[:1] + [donor] + parts[1:]))
            else:
                edited.append(" ".join(parts) + " " + donor)
        got = score(edited)
        flip = sum(1 for g in got if max(g, key=g.get) == "polite") / len(got)
        print(f"  {where:<16} -> becomes polite {flip:.3f}   "
              f"mean dP(polite) {statistics.mean(g['polite'] for g in got) - base:+.3f}")


COMMANDS: dict[str, Callable[[Corpus], None]] = {
    "bands": cmd_bands,
    "sentences": cmd_sentences,
    "carriers": cmd_carriers,
    "profile": cmd_profile,
    "counterfactual": cmd_counterfactual,
}


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("command", choices=(*COMMANDS, "all"))
    parser.add_argument("--run", default=str(DEFAULT_RUN))
    args = parser.parse_args()
    corpus = Corpus(Path(args.run).expanduser().resolve())
    corpus.describe()
    names = tuple(COMMANDS) if args.command == "all" else (args.command,)
    for name in names:
        print("\n" + "#" * 78)
        print(f"# {name}  --  {COMMANDS[name].__doc__.splitlines()[0]}")
        print("#" * 78)
        COMMANDS[name](corpus)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
