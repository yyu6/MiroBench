#!/usr/bin/env python3
"""Which channel carries `polite_rate`'s gap -- prevalence, or conversion?

`docs/DECISIONS.md` G6 established that a *carrier* -- a comment holding one
sentence at `P(polite) > 0.80` -- reconstructs `polite_rate` and `impolite_rate`
to three decimals on real and generated alike. Eight releases then tried to make
the generator write more appreciative sentences. This script measures whether
that was ever the channel.

It is not (G53). At the sentence level the deficit is 3.4x -- real 0.0817 of
sentences are carriers, generated 0.0239 -- and giving generated real's mix of
appreciative surface forms while keeping its own conversion closes **-0.7%**,
while giving it real's conversion closes **101.8%**. The generator writes
`gratitude` at 1.58x real's rate and `affirm_other` at 3.46x. They do not land.

Subcommands, each a hypothesis about what a carrier is, and each fidelity-checked
against the shipped labels before printing (rule E6):

    forms     prevalence vs conversion per named appreciative form, and the
              Kitagawa split of the sentence-level carrier gap.       -> the finding
    subject   REJECTED: that a carrier predicates on the thing rather than
              narrating the speaker.  P(carrier | starts with "I") 0.0819 vs
              0.0816, ratio 1.00.
    tail      REJECTED: that a carrier is the appreciation occupying the whole
              sentence rather than prefixing a continuation.  Real's tail
              distribution closes 2.3%; real's within-bin conversion closes 94.2%.
    examples  the same-form sentences from both sides with their scores, for the
              read-the-text step that must precede any story.

Everything is offline and no API call is made. Real sentences come from
evaluation-excluded threads only.
"""
from __future__ import annotations

import argparse
import random
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from polite_sentence_diagnosis import (  # noqa: E402
    FORMS,
    POLITE_SENTENCE,
    REPO,
    Corpus,
    assert_reproduces,
    load_scorer,
    sentences,
)

DEFAULT_TAG = "v110_length_transfer_n10_20260824_v1"
REAL_SAMPLE = 900
SEED = 11

I_INITIAL = re.compile(r"^\s*(?:i|i'm|i've|i'd|i'll|im|ive)\b", re.I)
APPRECIATIVE = re.compile(
    r"\b(?:thanks?|thank you|thx|appreciate|great|awesome|amazing|wonderful|excellent|"
    r"perfect|lovely|beautiful|brilliant|fantastic|solid|nice|good|love|loved|best|"
    r"congrats|congratulations|useful|helpful)\b",
    re.I,
)
TAIL_BINS = [(0, 0), (1, 3), (4, 8), (9, 15), (16, 30), (31, 10 ** 6)]


def scored_sentences(score, rows, cap):
    random.seed(SEED)
    sample = rows if len(rows) <= cap else random.sample(rows, cap)
    flat = [s for row in sample for s in sentences(row["text"])]
    got = score(flat)
    return sample, [(s, g["polite"]) for s, g in zip(flat, got)]


def kitagawa(real_cells, gen_cells, label_mix, label_conv):
    """Print the two counterfactuals for a set of {cell: (share, conversion)}."""
    keys = [k for k in real_cells
            if real_cells[k][1] is not None and gen_cells.get(k, (0, None))[1] is not None]
    actual_r = sum(real_cells[k][0] * real_cells[k][1] for k in keys)
    actual_g = sum(gen_cells[k][0] * gen_cells[k][1] for k in keys)
    mix = sum(real_cells[k][0] * gen_cells[k][1] for k in keys)
    conv = sum(gen_cells[k][0] * real_cells[k][1] for k in keys)
    gap = actual_r - actual_g
    print(f"\n  carrier rate: real {actual_r:.4f}  generated {actual_g:.4f}  gap {gap:+.4f}")
    print(f"    {label_mix:<52} {mix:.4f}  closes {(mix - actual_g) / gap:6.1%}")
    print(f"    {label_conv:<52} {conv:.4f}  closes {(conv - actual_g) / gap:6.1%}")


def cmd_forms(score, corpus) -> None:
    def measure(rows, cap):
        _, car = scored_sentences(score, rows, cap)
        flat = [s for s, _ in car]
        hit_any = [any(p.search(s) for p in FORMS.values()) for s in flat]
        out = {}
        for name, pattern in FORMS.items():
            idx = [i for i, s in enumerate(flat) if pattern.search(s)]
            out[name] = ((len(idx) / len(flat),
                          sum(car[i][1] > POLITE_SENTENCE for i in idx) / len(idx))
                         if idx else (0.0, None))
        idx = [i for i, hit in enumerate(hit_any) if not hit]
        out["(no named form)"] = (len(idx) / len(flat),
                                  sum(car[i][1] > POLITE_SENTENCE for i in idx) / len(idx))
        out["ALL"] = (1.0, sum(p > POLITE_SENTENCE for _, p in car) / len(car))
        return out

    real, gen = measure(corpus.excluded, REAL_SAMPLE), measure(corpus.generated, 10 ** 6)
    print(f"\n  {'form':<18}{'prevalence R':>13}{'G':>9}{'ratio':>8}   "
          f"{'conversion R':>13}{'G':>9}{'ratio':>8}")
    for name in list(FORMS) + ["(no named form)", "ALL"]:
        pr, cr = real[name]
        pg, cg = gen[name]
        pr_ratio = pg / pr if pr else float("nan")
        cv_ratio = (cg / cr) if (cr and cg is not None) else float("nan")
        print(f"  {name:<18}{pr:>13.4f}{pg:>9.4f}{pr_ratio:>8.2f}   "
              f"{(cr if cr is not None else float('nan')):>13.3f}"
              f"{(cg if cg is not None else float('nan')):>9.3f}{cv_ratio:>8.2f}")
    cells_r = {k: real[k] for k in list(FORMS) + ["(no named form)"]}
    cells_g = {k: gen[k] for k in list(FORMS) + ["(no named form)"]}
    kitagawa(cells_r, cells_g,
             "generated at REAL'S FORM MIX, own conversion :",
             "generated at REAL'S CONVERSION, own form mix :")


def cmd_subject(score, corpus) -> None:
    print("\n  REJECTED hypothesis: a carrier predicates on the thing, not the speaker.")
    for rows, label, cap in ((corpus.excluded, "EXCLUDED REAL", REAL_SAMPLE),
                             (corpus.generated, "GENERATED", 10 ** 6)):
        _, car = scored_sentences(score, rows, cap)
        n_i = sum(1 for s, _ in car if I_INITIAL.match(s))
        carriers = [(s, p) for s, p in car if p > POLITE_SENTENCE]
        ci = sum(1 for s, _ in carriers if I_INITIAL.match(s))
        print(f"\n  {label}: {len(car)} sentences")
        print(f"    carrier rate                            {len(carriers) / len(car):.4f}")
        print(f"    share of carriers starting 'I'          {ci / len(carriers):.4f}")
        print(f"    P(carrier | starts with 'I')            {ci / n_i:.4f}   (n={n_i})")
        print(f"    P(carrier | anything else)              "
              f"{(len(carriers) - ci) / (len(car) - n_i):.4f}   (n={len(car) - n_i})")


def tail_words(sentence: str):
    match = None
    for match in APPRECIATIVE.finditer(sentence):
        pass
    return None if match is None else len(sentence[match.end():].split())


def cmd_tail(score, corpus) -> None:
    print("\n  REJECTED hypothesis: a carrier is the appreciation occupying the whole sentence.")
    cells = {}
    for rows, label, cap in ((corpus.excluded, "EXCLUDED REAL", REAL_SAMPLE),
                             (corpus.generated, "GENERATED", 10 ** 6)):
        _, car = scored_sentences(score, rows, cap)
        hits = [(s, p, tail_words(s)) for s, p in car]
        hits = [h for h in hits if h[2] is not None]
        print(f"\n  {label}: {len(hits)} sentences carry an appreciative token "
              f"({len(hits) / len(car):.3f} of all)")
        print(f"    {'words after the token':>22}{'n':>7}{'P(carrier)':>12}")
        table = {}
        for low, high in TAIL_BINS:
            sub = [h for h in hits if low <= h[2] <= high]
            share = len(sub) / len(hits)
            conv = (sum(1 for h in sub if h[1] > POLITE_SENTENCE) / len(sub)) if len(sub) >= 8 else None
            table[(low, high)] = (share, conv)
            if conv is not None:
                name = f"{low}" if low == high else (f"{low}-{high}" if high < 10 ** 6 else f"{low}+")
                print(f"    {name:>22}{len(sub):>7}{conv:>12.3f}")
        cells[label] = table
    kitagawa(cells["EXCLUDED REAL"], cells["GENERATED"],
             "generated at REAL'S TAIL DISTRIBUTION, own conversion :",
             "generated at REAL'S WITHIN-BIN CONVERSION, own tails  :")


def cmd_examples(score, corpus) -> None:
    _, real = scored_sentences(score, corpus.excluded, REAL_SAMPLE)
    _, gen = scored_sentences(score, corpus.generated, 10 ** 6)
    for form in ("gratitude", "praise_object", "affirm_other"):
        pattern = FORMS[form]
        rr = [(s, p) for s, p in real if pattern.search(s)]
        gg = [(s, p) for s, p in gen if pattern.search(s)]
        if len(gg) < 5:
            continue
        print(f"\n  == {form}: real n={len(rr)} mean P={statistics.fmean(p for _, p in rr):.3f}"
              f"   generated n={len(gg)} mean P={statistics.fmean(p for _, p in gg):.3f}")
        print("     real, highest:")
        for s, p in sorted(rr, key=lambda x: -x[1])[:4]:
            print(f"       {p:.2f}  {s[:96]}")
        print("     generated, LOWEST (same form, does not land):")
        for s, p in sorted(gg, key=lambda x: x[1])[:4]:
            print(f"       {p:.2f}  {s[:96]}")


COMMANDS = {"forms": cmd_forms, "subject": cmd_subject, "tail": cmd_tail, "examples": cmd_examples}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=[*COMMANDS, "all"])
    ap.add_argument("--tag", default=DEFAULT_TAG)
    args = ap.parse_args()
    corpus = Corpus(REPO / "artifacts/generalized_card/runs" / args.tag)
    score = load_scorer()
    assert_reproduces(score, corpus.generated)
    print("fidelity: local harness reproduces the shipped labels exactly")
    for name in (COMMANDS if args.command == "all" else [args.command]):
        COMMANDS[name](score, corpus)


if __name__ == "__main__":
    main()
