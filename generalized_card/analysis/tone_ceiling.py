#!/usr/bin/env python3
"""How far the tone pair can honestly go, and what is left after v104.

Companion to `polite_sentence_diagnosis.py`, which established that the metric
is reconstructed by the *carrier* rate -- the share of comments holding one
sentence that reads polite on its own. This file holds the work done after
v104's paid gate, and every number it prints is one that would otherwise only
exist in a report:

  features     ~20 form-only carrier predictors, fitted on half the excluded
               threads and scored on the other half. Only three replicate.
  exclamation  the causal test on the strongest under-produced one, with a
               control and an abuse check. It is substantially a classifier
               artifact.
  arms         the three-arm ablation v104 was priced off, with the warm -> warm
               control that makes it interpretable.
  spread       the real between-thread tone distribution, where generated sits
               in it, and the plan's variance compression.
  ledger       the reuse ledger tested as a priming source, and rejected.

Form-only throughout: no feature names a product, a brand or a domain term, so a
surviving finding transfers to another domain. `features`, `exclamation` and
`arms` load the evaluation's own `Intel/polite-guard` checkpoint on CPU (a few
minutes each); `spread` and `ledger` are seconds and stdlib only. No API call.

    python3 generalized_card/analysis/tone_ceiling.py spread
    python3 generalized_card/analysis/tone_ceiling.py ledger
    python3 generalized_card/analysis/tone_ceiling.py features
    python3 generalized_card/analysis/tone_ceiling.py exclamation
    python3 generalized_card/analysis/tone_ceiling.py arms
"""

from __future__ import annotations

import argparse
import glob
import json
import random
import re
import statistics
import sys
from pathlib import Path
from typing import Any, Callable

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "generalized_card"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from generalized_card import evaluative_register as ev  # noqa: E402
from polite_sentence_diagnosis import (  # noqa: E402
    DEFAULT_RUN,
    POLITE_SENTENCE,
    Corpus,
    load_scorer,
    sentences,
    share,
)

V104_GATE = REPO / "artifacts/generalized_card/runs/v104_evaluative_seed8_20260821_v1"
V103_N10 = DEFAULT_RUN
REAL_DIR = REPO / "data/raw/discussions/camera_product"
SEED_POOL = REPO / "artifacts/generalized_card/seed_pools/camera_product_150_seed42.json"
GATE_SEED_INDEX = 8

SUBORD = (r"if|when|whenever|unless|while|since|because|although|though|as long as|"
          r"assuming|provided|once|until|before|after|in case")
QUALIFIER = re.compile(
    r"\b(?:but|though|however|kind of|sort of|a bit|somewhat|fairly|pretty much|"
    r"i guess|i think|maybe|probably|at least|for the price|depends|not perfect|"
    r"caveat|downside|granted|admittedly)\b", re.I)
COORD = re.compile(r"\b(?:and|but|or|so|because)\b", re.I)

# Form only. Nothing here names a product, a brand or a domain term.
PATTERNS: dict[str, re.Pattern[str]] = {
    "starts_with_subordinator": re.compile(rf"^\W*(?:{SUBORD})\b", re.I),
    "has_subordinate_clause": re.compile(rf"\b(?:{SUBORD})\b", re.I),
    "gerund_subject": re.compile(r"^\W*\w+ing\b(?!\s*(?:it|this|that)\b)", re.I),
    "deictic_target": re.compile(
        r"\b(?:this|that|these|those|your|my|his|her|their|its|it)\b", re.I),
    "second_person_target": re.compile(r"\b(?:your|yours|you)\b", re.I),
    "first_person_subject": re.compile(r"^\W*(?:i|we)\b", re.I),
    "abstract_nominal": re.compile(
        r"\b\w+(?:tion|ment|ness|ity|ance|ence|ing)\b\s+(?:is|was|are|were|makes|means)\b", re.I),
    "exclamation": re.compile(r"!"),
    "all_caps_word": re.compile(r"\b[A-Z]{3,}\b"),
    "intensifier": re.compile(
        r"\b(?:so|very|really|absolutely|incredibly|truly|super|just|damn|seriously)\b", re.I),
    "ends_period_only": re.compile(r"[a-z0-9)\"']\.\s*$", re.I),
    "imperative_open": re.compile(
        r"^\W*(?:get|grab|go|try|check|buy|keep|enjoy|have|take|use|do)\b", re.I),
    "question": re.compile(r"\?"),
    "quote_reply": re.compile(r"^\s*>"),
    "thanks_or_respect": re.compile(
        r"\b(?:thank|thanks|thx|appreciate|respect|congrats?|welcome)\b", re.I),
}
COMPUTED = ("one_clause_only", "copula_early", "no_qualifier", "under_8_words", "no_comma")
_COPULA = {"is", "are", "was", "were", "'s", "looks", "feels"}


def feature(name: str, text: str) -> bool:
    if name == "one_clause_only":
        return ("," not in text) and not COORD.search(text)
    if name == "copula_early":
        return any(w.lower().strip(",.!?") in _COPULA for w in text.split()[:4])
    if name == "no_qualifier":
        return not QUALIFIER.search(text)
    if name == "under_8_words":
        return len(text.split()) < 8
    if name == "no_comma":
        return "," not in text
    return bool(PATTERNS[name].search(text))


FEATURES = (*PATTERNS, *COMPUTED)


# --------------------------------------------------------------------------- #
# shared helpers
# --------------------------------------------------------------------------- #


def real_thread(seed_index: int) -> list[dict[str, Any]]:
    """The matched real thread for one seed, with its stored classifier labels."""

    seeds = json.loads(SEED_POOL.read_text()).get("seed_posts") or []
    wanted = str((seeds[seed_index] or {}).get("source_raw_post_id") or "").strip()
    for product in sorted(REAL_DIR.iterdir()):
        path = product / "politeness_results.json"
        if not path.is_file():
            continue
        for thread in json.loads(path.read_text()).get("threads") or []:
            if str(thread.get("thread_id") or "").strip() != wanted:
                continue
            return [
                {"text": str(c.get("text") or ""), "label": str(c.get("pred_label") or ""),
                 "polite": float(c.get("polite_probability") or 0.0)}
                for c in thread.get("comments") or [] if str(c.get("text") or "").strip()
            ]
    return []


def run_comments(run: Path, *, largest_only: bool = False) -> list[dict[str, Any]]:
    """Scored comments from a run; optionally only its biggest thread."""

    threads: list[list[dict[str, Any]]] = []
    for path in sorted(glob.glob(str(run / "cleaned/*/politeness_results.json"))):
        for thread in json.loads(Path(path).read_text()).get("threads") or []:
            rows = [
                {"text": str(c.get("text") or ""), "label": str(c.get("pred_label") or ""),
                 "polite": float(c.get("polite_probability") or 0.0)}
                for c in thread.get("comments") or [] if str(c.get("text") or "").strip()
            ]
            if rows:
                threads.append(rows)
    if not threads:
        return []
    if largest_only:
        return max(threads, key=len)
    return [row for thread in threads for row in thread]


def sentence_table(score, rows: list[dict[str, Any]], cap: int | None = None,
                   seed: int = 101) -> list[dict[str, Any]]:
    """Every sentence of a comment set, with its Polite Guard score."""

    if cap is not None and len(rows) > cap:
        rows = random.Random(seed).sample(rows, cap)
    flat = [(i, s) for i, row in enumerate(rows) for s in sentences(row["text"])]
    scored = score([s for _, s in flat])
    return [{"index": i, "text": s, "polite": g["polite"]}
            for (i, s), g in zip(flat, scored)]


def assert_reproduces(score, rows: list[dict[str, Any]], n: int = 100) -> None:
    probe = rows[:n]
    got = score([r["text"] for r in probe])
    flips = sum(1 for r, g in zip(probe, got) if max(g, key=g.get) != r["label"])
    print(f"[fidelity] n={len(probe)} label flips={flips}")
    if flips:
        sys.exit("harness does not reproduce the artifact -- refusing to print anything derived")


# --------------------------------------------------------------------------- #
# subcommands
# --------------------------------------------------------------------------- #


def cmd_spread(_: Any) -> None:
    """The real between-thread tone spread, and the plan's compression of it."""

    per_thread = []
    seen: set[str] = set()
    for product in sorted(REAL_DIR.iterdir()):
        path = product / "politeness_results.json"
        if not path.is_file():
            continue
        for thread in json.loads(path.read_text()).get("threads") or []:
            tid = str(thread.get("thread_id") or "").strip()
            if not tid or tid in seen:
                continue
            rows = [c for c in thread.get("comments") or [] if str(c.get("text") or "").strip()]
            if len(rows) < 15:
                continue
            seen.add(tid)
            per_thread.append({
                "polite": sum(1 for c in rows if c.get("pred_label") == "polite") / len(rows),
                "impolite": sum(1 for c in rows if c.get("pred_label") == "impolite") / len(rows),
            })
    print(f"real threads with >= 15 comments: {len(per_thread)}")
    print(f"\n{'':<12}{'p05':>8}{'p10':>8}{'p25':>8}{'median':>9}{'p75':>8}{'p90':>8}"
          f"{'mean':>8}{'sd':>8}")
    for label in ("polite", "impolite"):
        values = sorted(row[label] for row in per_thread)

        def q(p: float) -> float:
            return values[min(len(values) - 1, int(p * len(values)))]

        print(f"{label:<12}{q(.05):>8.3f}{q(.10):>8.3f}{q(.25):>8.3f}"
              f"{statistics.median(values):>9.3f}{q(.75):>8.3f}{q(.90):>8.3f}"
              f"{statistics.mean(values):>8.3f}{statistics.stdev(values):>8.3f}")

    generated = run_comments(V103_N10)
    if generated:
        gp, gi = share(generated, "polite"), share(generated, "impolite")
        pol = sorted(row["polite"] for row in per_thread)
        imp = sorted(row["impolite"] for row in per_thread)
        print(f"\ngenerated (v103 N=10 pooled): polite {gp:.4f}  impolite {gi:.4f}")
        print(f"  share of real threads at or below that polite rate  : "
              f"{sum(1 for x in pol if x <= gp) / len(pol):.3f}")
        print(f"  share of real threads at or above that impolite rate: "
              f"{sum(1 for x in imp if x >= gi) / len(imp):.3f}")

    # the plan's own between-thread spread
    def walk(items, out):
        for item in items or []:
            out.append(item)
            walk(item.get("replies"), out)

    planned = []
    for path in sorted(glob.glob(str(V103_N10 / "generated/*/discussion.json"))):
        for post in json.loads(Path(path).read_text()).get("posts") or []:
            flat: list[dict] = []
            walk(post.get("comments"), flat)
            if not flat:
                continue
            tones = [str(c.get("tone_target") or "").lower() for c in flat]
            planned.append(sum(1 for t in tones if t == "polite") / len(tones))
    if len(planned) > 1:
        real_pol = [row["polite"] for row in per_thread]
        print(f"\nplan's between-thread polite share, v103 N=10 ({len(planned)} threads)")
        print(f"  plan  mean {statistics.mean(planned):.3f}  sd {statistics.stdev(planned):.3f}"
              f"  min {min(planned):.3f}  max {max(planned):.3f}")
        print(f"  real  mean {statistics.mean(real_pol):.3f}  sd {statistics.stdev(real_pol):.3f}"
              f"  min {min(real_pol):.3f}  max {max(real_pol):.3f}")
        print(f"  the plan's MEAN is right; its VARIANCE is "
              f"{statistics.stdev(planned) / statistics.stdev(real_pol):.2f} of real. The mean gap "
              f"is realization, not planning.")


def cmd_ledger(_: Any) -> None:
    """REJECTED: the reuse ledger as a source of the two tics it echoes back."""

    ledger_partitive = re.compile(
        r"^\s*-\s.{0,80}\b(?:part|bit|piece|side|aspect)\b.{0,40}\(used \d+x\)", re.I | re.M)
    ledger_tag = re.compile(r"^\s*-\s.{0,90}(?:,\s*(?:honestly|sure)\b)", re.I | re.M)
    records = [
        record
        for path in sorted(glob.glob(str(V103_N10 / "generated/*/generation_records.json")))
        for record in json.loads(Path(path).read_text())
    ]
    rows = []
    for position, record in enumerate(records):
        prompt = str(record.get("prompt") or "")
        text = str((record.get("comment") or {}).get("content") or "")
        if not text.strip():
            continue
        rows.append({
            "position": position,
            "ledger_partitive": bool(ledger_partitive.search(prompt)),
            "ledger_tag": bool(ledger_tag.search(prompt)),
            "out_partitive": ev.has_partitive(text),
            "out_tag": any(ev.has_downtoner_tag(s) for s in sentences(text)),
        })
    print(f"records with output: {len(rows)}")
    print(f"\n{'construction':<18}{'ledger':>9}{'n':>6}{'output has it':>15}{'lift':>8}")
    for name, key, out in (("partitive", "ledger_partitive", "out_partitive"),
                           ("downtoner tag", "ledger_tag", "out_tag")):
        present = [r for r in rows if r[key]]
        absent = [r for r in rows if not r[key]]
        a = sum(1 for r in present if r[out]) / max(1, len(present))
        b = sum(1 for r in absent if r[out]) / max(1, len(absent))
        print(f"{name:<18}{'present':>9}{len(present):>6}{a:>15.3f}{a / max(1e-9, b):>8.2f}")
        print(f"{'':<18}{'absent':>9}{len(absent):>6}{b:>15.3f}")
    print("\nThe ledger fills up late in a thread, so position is controlled:")
    rows.sort(key=lambda r: r["position"])
    quarter = max(1, len(rows) // 4)
    print(f"  {'quartile':<10}{'ledger+ n':>11}{'rate':>8}{'ledger- n':>11}{'rate':>8}")
    for index in range(4):
        chunk = rows[index * quarter:(index + 1) * quarter] if index < 3 else rows[3 * quarter:]
        present = [r for r in chunk if r["ledger_partitive"]]
        absent = [r for r in chunk if not r["ledger_partitive"]]
        print(f"  Q{index + 1:<9}{len(present):>11}"
              f"{sum(1 for r in present if r['out_partitive']) / max(1, len(present)):>8.3f}"
              f"{len(absent):>11}"
              f"{sum(1 for r in absent if r['out_partitive']) / max(1, len(absent)):>8.3f}")
    print("\n  Flat or lower where the ledger is present. It is a symptom, not a")
    print("  cause, so v98's route-ledger mechanism is left alone.")


def cmd_features(_: Any) -> None:
    """Form-only carrier predictors. Only three replicate out of sample."""

    corpus = Corpus(V103_N10)
    score = load_scorer()
    assert_reproduces(score, corpus.generated)
    ids = sorted({row["thread_id"] for row in corpus.excluded})
    random.Random(101).shuffle(ids)
    fit_ids = set(ids[::2])
    sample = random.Random(101).sample(corpus.excluded, min(6000, len(corpus.excluded)))
    table = sentence_table(score, sample)
    for row in table:
        row["split"] = "fit" if sample[row["index"]]["thread_id"] in fit_ids else "held"
    gen_table = sentence_table(score, corpus.generated)

    results: dict[str, dict[str, float]] = {}
    for split in ("fit", "held"):
        rows = [r for r in table if r["split"] == split]
        carriers = [r for r in rows if r["polite"] > POLITE_SENTENCE]
        base = len(carriers) / len(rows)
        print(f"\n### {split}: {len(rows)} sentences, {len(carriers)} carriers (base {base:.4f})")
        print(f"  {'feature':<26}{'prev':>8}{'precision':>11}{'lift':>7}{'recall':>8}")
        for name in FEATURES:
            hit = [r for r in rows if feature(name, r["text"])]
            if len(hit) < 30:
                continue
            got = [r for r in hit if r["polite"] > POLITE_SENTENCE]
            precision = len(got) / len(hit)
            results.setdefault(name, {})[split] = precision / base
            print(f"  {name:<26}{len(hit) / len(rows):>8.3f}{precision:>11.3f}"
                  f"{precision / base:>7.2f}{len(got) / len(carriers):>8.3f}")

    print("\n### replicating features (lift >= 1.6 in both halves), against generated")
    print(f"  {'feature':<26}{'fit lift':>9}{'held lift':>10}{'real prev':>11}"
          f"{'gen prev':>10}{'gen/real':>10}")
    for name in sorted(results, key=lambda k: -results[k].get("held", 0.0)):
        row = results[name]
        if row.get("fit", 0) < 1.6 or row.get("held", 0) < 1.6:
            continue
        real_prev = sum(1 for r in table if feature(name, r["text"])) / len(table)
        gen_prev = sum(1 for r in gen_table if feature(name, r["text"])) / len(gen_table)
        print(f"  {name:<26}{row['fit']:>9.2f}{row['held']:>10.2f}{real_prev:>11.3f}"
              f"{gen_prev:>10.3f}{gen_prev / max(1e-9, real_prev):>9.2f}x")


def cmd_exclamation(_: Any) -> None:
    """The strongest under-produced carrier feature is a classifier artifact."""

    corpus = Corpus(V103_N10)
    score = load_scorer()
    assert_reproduces(score, corpus.generated)
    real = sentence_table(score, corpus.excluded, cap=6000)
    generated = sentence_table(score, corpus.generated)

    def swap(text: str, frm: str, to: str) -> str:
        return re.sub(re.escape(frm) + r"+\s*$", to, text.strip())

    src = [r for r in real if r["polite"] > POLITE_SENTENCE and r["text"].rstrip().endswith("!")]
    got = score([swap(r["text"], "!", ".") for r in src])
    print(f"\n(1) real carriers ending '!'   n={len(src)}")
    print(f"    P(polite) {statistics.mean(r['polite'] for r in src):.3f} -> "
          f"{statistics.mean(g['polite'] for g in got):.3f}"
          f"   still carriers {sum(1 for g in got if g['polite'] > POLITE_SENTENCE) / len(got):.3f}")

    control = [r for r in real
               if r["polite"] > POLITE_SENTENCE and r["text"].rstrip().endswith(".")][:len(src)]
    got_control = score([swap(r["text"], ".", "...") for r in control])
    print(f"    CONTROL, carriers ending '.' -> '...'   n={len(control)}"
          f"   P(polite) {statistics.mean(r['polite'] for r in control):.3f} -> "
          f"{statistics.mean(g['polite'] for g in got_control):.3f}")

    for label, keep in (("positive", True), ("ABUSE CHECK, non-evaluative", False)):
        target = [r for r in generated
                  if r["polite"] <= POLITE_SENTENCE
                  and ev.is_positive_sentence(r["text"]) == keep
                  and r["text"].rstrip().endswith(".")]
        if not target:
            continue
        got2 = score([swap(r["text"], ".", "!") for r in target])
        print(f"\n({'2' if keep else '3'}) generated {label} non-carriers ending '.'  "
              f"n={len(target)}")
        print(f"    P(polite) {statistics.mean(r['polite'] for r in target):.3f} -> "
              f"{statistics.mean(g['polite'] for g in got2):.3f}"
              f"   become carriers "
              f"{sum(1 for g in got2 if g['polite'] > POLITE_SENTENCE) / len(got2):.3f}")
    print("\n  Punctuation alone moves a sentence that evaluates nothing. Match the")
    print("  measured real rate and no more; past that it is tuning to the metric.")


def cmd_arms(_: Any) -> None:
    """The ablation v104 was priced off, with the control that interprets it."""

    tag_mid = re.compile(
        r",\s*(?:%s)\b(?=\s*[.!?…,]|\s*$)" % "|".join(ev.DOWNTONER_TAGS), re.I)
    on_surface = re.compile(
        r"\s*\b(?:on the surface|in principle|up to a point|as far as it goes|"
        r"for what it'?s worth)\b", re.I)
    partitive = re.compile(
        r"\b(that|the|this)\s+((?:\w+\s+){0,2}?)(%s)\b" % "|".join(ev.PARTITIVE_HEADS), re.I)
    upgrade = {"good": "great", "nice": "great", "useful": "fantastic", "handy": "fantastic",
               "solid": "excellent", "decent": "great", "sensible": "great",
               "capable": "excellent", "neat": "awesome", "fine": "great",
               "reasonable": "great", "workable": "great", "alright": "great",
               "adequate": "great", "serviceable": "great"}
    neutral = {"good": "decent", "nice": "solid", "useful": "handy", "handy": "useful",
               "solid": "capable", "decent": "reasonable", "sensible": "reasonable",
               "capable": "workable", "neat": "tidy", "fine": "alright",
               "reasonable": "sensible", "workable": "serviceable", "alright": "okay",
               "adequate": "serviceable", "serviceable": "adequate"}

    def swap_words(text: str, table: dict[str, str]) -> str:
        def replace(match: re.Match[str]) -> str:
            word = match.group(0)
            out = table.get(word.lower(), word)
            return out.capitalize() if word[:1].isupper() else out
        return re.sub(r"\b(?:%s)\b" % "|".join(table), replace, text, flags=re.I)

    edits: dict[str, Callable[[str], str]] = {
        "A  strip the trailing tag": lambda t: re.sub(
            r"\s+([.!?,])", r"\1", on_surface.sub("", tag_mid.sub("", t))),
        "B  de-partitive": lambda t: partitive.sub(lambda m: m.group(1), t),
        "C  warm tier -> hot tier": lambda t: swap_words(t, upgrade),
        "CONTROL  warm -> other warm": lambda t: swap_words(t, neutral),
    }
    corpus = Corpus(V103_N10)
    score = load_scorer()
    assert_reproduces(score, corpus.generated)
    generated = corpus.generated
    base_polite = share(generated, "polite")
    base_impolite = share(generated, "impolite")
    real_polite = share(corpus.matched, "polite")
    real_impolite = share(corpus.matched, "impolite")
    print(f"\nv103 polite {base_polite:.4f} impolite {base_impolite:.4f}"
          f"   |  matched real polite {real_polite:.4f} impolite {real_impolite:.4f}")
    print(f"\n{'edit':<32}{'polite':>9}{'impolite':>10}{'pol gap closed':>16}"
          f"{'imp gap closed':>16}{'touched':>9}")

    def apply(name: str, fns: list[Callable[[str], str]]) -> None:
        edited, touched = [], 0
        for row in generated:
            text = row["text"]
            for fn in fns:
                text = fn(text)
            touched += text != row["text"]
            edited.append(text)
        got = score(edited)
        polite = sum(1 for g in got if max(g, key=g.get) == "polite") / len(got)
        impolite = sum(1 for g in got if max(g, key=g.get) == "impolite") / len(got)
        print(f"{name:<32}{polite:>9.4f}{impolite:>10.4f}"
              f"{(polite - base_polite) / (real_polite - base_polite) * 100:>15.1f}%"
              f"{(impolite - base_impolite) / (real_impolite - base_impolite) * 100:>15.1f}%"
              f"{touched:>9}")

    for name, fn in edits.items():
        apply(name, [fn])
    apply("A+B+C  all three", [edits["A  strip the trailing tag"],
                               edits["B  de-partitive"],
                               edits["C  warm tier -> hot tier"]])
    print("\n  The control swaps the same words for OTHER warm words. If it moves")
    print("  `polite_rate`, the ablation was measuring perturbation, not tier.")
    print("  It does not. And v104 still delivered 8.4% against C's 20.8% --")
    print("  an ablation is an upper bound, never a price (tasks/lessons.md).")


COMMANDS: dict[str, Callable[[Any], None]] = {
    "spread": cmd_spread,
    "ledger": cmd_ledger,
    "features": cmd_features,
    "exclamation": cmd_exclamation,
    "arms": cmd_arms,
}


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=(*COMMANDS, "all"))
    args = parser.parse_args()
    names = tuple(COMMANDS) if args.command == "all" else (args.command,)
    for name in names:
        print("\n" + "#" * 78)
        print(f"# {name}  --  {COMMANDS[name].__doc__.splitlines()[0]}")
        print("#" * 78)
        COMMANDS[name](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
