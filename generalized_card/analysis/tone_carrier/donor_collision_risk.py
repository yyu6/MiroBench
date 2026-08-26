#!/usr/bin/env python3
"""What v120's donor sentences cost `self_bleu_4` and `self_bertscore` (G37).

The arm is priced to take `polite_rate` at N=150 from P(pass) 0.17 to 0.90. It
would be worthless if it did that by breaking the two metrics next to it, and G37
is a measured reason to expect it might: in v109 slots given the same prescribed
speech act scored **+0.0255** on `self_bertscore` against slots given none.

The donor is prescribed text drawn from a shared pool, so this is the same shape
of risk. It is mitigated by pool size and per-slot hash keying -- 825 sentences
against the link arm's 802, which runs at 0.95 compliance with zero repeats -- but
mitigation is not measurement.

Method: take the shipped artifact, prefix each POLITE-ASSIGNED comment with the
sentence the arm would actually draw for that slot, and rescore both metrics with
the project's own scorers. Fidelity first: the untouched recomputation must
reproduce the shipped values.

This OVERSTATES the cost in one direction and understates it in another, and both
are stated rather than buried: it applies the donor at 100% compliance (the Writer
will not reach that), and it prefixes rather than integrating (the Writer will
blend it, which is more similar to real, not less).

Usage:  python3 generalized_card/analysis/tone_carrier/donor_collision_risk.py
"""
from __future__ import annotations

import collections
import json
import statistics as st
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "scripts" / "evaluation"))
sys.path.insert(0, str(REPO / "generalized_card"))

from generalized_card.tone_donor import (  # noqa: E402
    draw_donor_sentence,
    load_donor_inventory,
    set_tone_donor_mode,
    tone_donor_slot,
)
from score_thread_self_bertscore import load_bert_scorer  # noqa: E402
from score_thread_self_bleu import pairwise_self_bleu_for_order, tokenize  # noqa: E402

TAG = "v117_calibration_20260826_v1"
RUN = REPO / "artifacts/generalized_card/runs" / TAG


class Task:
    def __init__(self, payload: dict) -> None:
        self.__dict__.update(payload or {})


def self_bleu(texts: list[str]) -> float:
    usable = [t for t in texts if len(str(t or "").split()) >= 2]
    if len(usable) < 2:
        return float("nan")
    return pairwise_self_bleu_for_order([tokenize(t) for t in usable], 4)


def main() -> None:
    inventory = load_donor_inventory("camera_product")
    print(f"donor inventory: {inventory.get('sentence_count')} sentences")
    if not inventory.get("available"):
        raise SystemExit("no donor inventory; run harvest_donor_sentences.py first")

    threads: dict[str, list[str]] = collections.defaultdict(list)
    donored: dict[str, list[str]] = collections.defaultdict(list)
    drawn: list[str] = []
    routed = total = 0
    set_tone_donor_mode("measured")
    try:
        for folder in sorted((RUN / "generated").glob("run_*_sampled_reddit")):
            for record in json.loads((folder / "generation_records.json").read_text()):
                comment = record.get("comment") or {}
                text = str(comment.get("content") or "").strip()
                if not text:
                    continue
                key = f"{record.get('seed_index')}"
                total += 1
                threads[key].append(text)
                task = Task(record.get("task") or {})
                sentence = draw_donor_sentence(task, inventory)
                if tone_donor_slot(task):
                    routed += 1
                if sentence:
                    drawn.append(sentence)
                    donored[key].append(f"{sentence} {text}")
                else:
                    donored[key].append(text)
    finally:
        set_tone_donor_mode("off")

    counts = collections.Counter(drawn)
    print(f"comments {total}   polite-assigned (routed) {routed} = {routed/total:.3f}")
    print(f"distinct donors drawn {len(counts)} of {len(drawn)} draws"
          f"   most reused {counts.most_common(1)[0][1] if counts else 0}x")
    per_thread = collections.defaultdict(list)
    set_tone_donor_mode("measured")
    try:
        for folder in sorted((RUN / "generated").glob("run_*_sampled_reddit")):
            for record in json.loads((folder / "generation_records.json").read_text()):
                task = Task(record.get("task") or {})
                sentence = draw_donor_sentence(task, inventory)
                if sentence:
                    per_thread[f"{record.get('seed_index')}"].append(sentence)
    finally:
        set_tone_donor_mode("off")
    dup_pairs = 0
    for used in per_thread.values():
        dup_pairs += sum(c * (c - 1) // 2 for c in collections.Counter(used).values())
    print(f"within-thread exact collisions: {dup_pairs} pairs across "
          f"{len(per_thread)} threads   (v113's link arm: 0)")
    print(f"mean donor words {st.mean(len(s.split()) for s in drawn):.2f}"
          if drawn else "no donors drawn")

    keys = sorted(threads)
    base_bleu = st.mean(self_bleu(threads[k]) for k in keys)
    donor_bleu = st.mean(self_bleu(donored[k]) for k in keys)
    print(f"\nself_bleu_4   shipped {base_bleu:.6f}  ->  with donors {donor_bleu:.6f}"
          f"   {donor_bleu - base_bleu:+.6f}  ({100*(donor_bleu-base_bleu)/base_bleu:+.2f}%)")

    scorer, _, _, _, _, _ = load_bert_scorer(
        bert_score_path=REPO / "bert_score-master",
        model_type="microsoft/deberta-xlarge-mnli", num_layers=None, batch_size=8,
        device="auto", idf=False, idf_sents=[], rescale_with_baseline=False,
        local_files_only=True,
    )

    def sbert(texts: list[str]) -> float:
        usable = [t for t in texts if len(str(t or "").split()) >= 2]
        cand, ref = [], []
        for i in range(len(usable)):
            for j in range(i + 1, len(usable)):
                cand.append(usable[i]); ref.append(usable[j])
        if not cand:
            return float("nan")
        _, _, f1 = scorer.score(cand, ref, batch_size=8)
        return float(sum(float(v) for v in f1) / len(f1))

    base_sb = st.mean(sbert(threads[k]) for k in keys)
    donor_sb = st.mean(sbert(donored[k]) for k in keys)
    print(f"self_bertscore shipped {base_sb:.6f}  ->  with donors {donor_sb:.6f}"
          f"   {donor_sb - base_sb:+.6f}  ({100*(donor_sb-base_sb)/base_sb:+.2f}%)")
    print("\nReference points: generated already sits +8.67% on self_bleu_4 and "
          "+1.59% on self_bertscore,\nand G37's prescribed-speech-act effect was "
          "+0.0255 on self_bertscore. Read the moves against those.")
    print("Caveats, both directions: 100% compliance is assumed (the Writer will "
          "not reach it),\nand the donor is PREFIXED rather than blended, which is "
          "the least integrated form.")


if __name__ == "__main__":
    main()
