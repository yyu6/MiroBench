#!/usr/bin/env python3
"""Post-gate audit of the v109 `--entity-spread measured` arm (seed 8).

The v109 gate is a **randomised within-run experiment**, which is unusual for
this project and worth exploiting. `entity_spread.slot_offers_referent` draws
per slot from `sha256("entity_spread:{seed_key}:{local_task_id}")`, and
`local_task_id` is assigned by the Planner's traversal order, not by anything
about the slot's plan, tone, depth or claim. So "this slot was offered extra
referents" is independent of slot content, and a fired-vs-not-fired contrast
inside one thread is a legitimate treatment-effect estimate -- not the
between-thread correlational slope that `docs/DECISIONS.md` G35 retracted.

Four blocks:

1. `rate`   -- E3 realized-rate audit. How many slots the draw selected, how
              many actually rendered the block, and where the difference goes.
2. `naming` -- the proximal effect: designator mentions per comment, fired vs
              not-fired, plus the same numbers for the previous release and for
              the matched real thread. This is what the mechanism was built to
              move.
3. `cosine` -- the distal effect on the two metrics that regressed
              (`semantic_mean_cosine`, and BERTScore as its close relative):
              pairwise similarity split into both-fired / one-fired /
              neither-fired. Uses the scorer's own embedding model.
4. `shape`  -- the naming *shape* comparison that reframes the target: real
              names many things rarely, generated names few things often.

    python3 generalized_card/analysis/entity_spread_gate_audit.py rate
    python3 generalized_card/analysis/entity_spread_gate_audit.py all
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = REPO / "generalized_card"
for extra in (str(PACKAGE_ROOT), str(REPO / "scripts" / "evaluation")):
    if extra not in sys.path:
        sys.path.insert(0, extra)

from generalized_card.content_profile_analysis import DESIGNATOR  # noqa: E402
from generalized_card.entity_inventory import slot_equipment_options  # noqa: E402

RUNS = REPO / "artifacts/generalized_card/runs"
TREATED = "v109_entity_spread_seed8_20260824_v1"
CONTROL = "v108_semantic_coverage_nonrepeat_seed8_20260823_v2"
BLOCK = "Other things in this space you may name"


def mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else float("nan")


def records(tag: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted((RUNS / tag).glob("generated/run_*/generation_records.json")):
        rows.extend(json.loads(path.read_text(encoding="utf-8")))
    if not rows:
        raise SystemExit(f"no generation records under {tag}")
    return rows


def slots(tag: str) -> list[dict[str, Any]]:
    """One row per accepted comment, tagged with whether the arm fired on it."""

    out = []
    for rec in records(tag):
        comment = rec.get("comment") or {}
        text = (comment.get("content") or "").strip()
        if not text:
            continue
        task = rec.get("task") or {}
        out.append(
            {
                "comment_id": str(comment.get("comment_id")),
                "text": text,
                "depth": int(comment.get("depth", task.get("depth", 0)) or 0),
                "fired": BLOCK in (rec.get("prompt") or ""),
                "local_task_id": int(task.get("local_task_id") or 0),
                "anchors": [
                    str(v) for v in (task.get("concrete_anchors") or ()) if str(v).strip()
                ],
            }
        )
    return out


def seed_key(tag: str) -> str:
    for path in sorted((RUNS / tag).glob("logs/*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict) and row.get("seed_key"):
                return str(row["seed_key"])
    raise SystemExit(f"no seed_key logged under {tag}")


def spread_profile(tag: str) -> dict[str, Any]:
    profile = json.loads((RUNS / tag / "domain_profile.json").read_text(encoding="utf-8"))
    return profile.get("entity_spread_profile") or {}


def inventory(tag: str) -> dict[str, Any]:
    profile = json.loads((RUNS / tag / "domain_profile.json").read_text(encoding="utf-8"))
    return profile.get("entity_inventory") or {}


def draw(key: str) -> float:
    digest = hashlib.sha256(f"entity_spread:{key}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False) / float(1 << 64)


def cmd_rate(_: Any) -> None:
    """E3: did the arm fire at the rate it was designed to fire at?"""

    rows = slots(TREATED)
    key = seed_key(TREATED)
    bands = (spread_profile(TREATED).get("bands") or {})
    share = float((bands.get("large") or {}).get("distinct_per_comment") or 0.0)
    inv = inventory(TREATED)

    drawn = [r for r in rows if draw(f"{key}:{r['local_task_id']}") < share]
    fired = [r for r in rows if r["fired"]]
    lost = [r for r in drawn if not r["fired"]]

    print("\n== realized-rate audit ==\n")
    print(f"  seed_key                {key}")
    print(f"  thread band / rate      large / {share:.4f}")
    print(f"  slots                   {len(rows)}")
    print(f"  draw selected           {len(drawn)}  ({len(drawn) / len(rows):.1%})")
    print(f"  block rendered          {len(fired)}  ({len(fired) / len(rows):.1%})")
    print(f"  selected but not shown  {len(lost)}  ({len(lost) / max(1, len(drawn)):.1%} of selected)")
    print(f"  shown but not selected  {sum(1 for r in fired if r not in drawn)}")

    if lost:
        empty = 0
        for row in lost:
            options = slot_equipment_options(
                inv, slot_index=row["local_task_id"], limit=3, excluded=row["anchors"]
            )
            empty += not options
        print(f"\n  of the {len(lost)} lost slots, {empty} got an empty option list from")
        print("  slot_equipment_options -- the only other early return in the render path.")


def _designators(text: str) -> list[str]:
    return [m.group().casefold() for m in DESIGNATOR.finditer(text)]


def cmd_naming(_: Any) -> None:
    """The proximal effect the mechanism was built to produce."""

    print("\n== proximal effect: designator naming ==\n")
    rows = slots(TREATED)
    print("  Within v109, treatment is assigned by a content-independent hash draw,")
    print("  so this is a randomised contrast, not a correlational one.\n")
    print(f"  {'group':>14s} {'comments':>9s} {'mentions/comment':>17s} {'>=1 mention':>12s} {'distinct/comment':>17s}")
    for label, want in (("fired", True), ("not fired", False)):
        group = [r for r in rows if r["fired"] is want]
        hits = [_designators(r["text"]) for r in group]
        distinct = len({d for h in hits for d in h})
        print(
            f"  {label:>14s} {len(group):9d} {sum(len(h) for h in hits) / len(group):17.3f} "
            f"{sum(1 for h in hits if h) / len(group):12.3f} {distinct / len(group):17.3f}"
        )

    print("\n  Release comparison on the same seed-8 thread:\n")
    print(f"  {'run':>14s} {'mentions':>9s} {'distinct':>9s} {'top name':>10s} {'top count':>10s} {'top share':>10s}")
    for tag, label in ((CONTROL, "v108"), (TREATED, "v109")):
        counts: Counter[str] = Counter()
        for row in slots(tag):
            counts.update(_designators(row["text"]))
        total = sum(counts.values())
        name, top = counts.most_common(1)[0]
        print(
            f"  {label:>14s} {total:9d} {len(counts):9d} {name:>10s} {top:10d} {top / total:10.4f}"
        )


def matched_real_texts(tag: str) -> list[str]:
    from generalized_card.content_profile_data import matched_threads
    from generalized_card.domain import load_domain_config

    recs = records(tag)
    matches, _ = matched_threads(RUNS / tag, load_domain_config("camera"), recs)
    return [text for row in matches.values() for text in row["texts"]]


def cmd_shape(_: Any) -> None:
    """Real names many things rarely; generated names few things often."""

    print("\n== naming shape: real vs generated ==\n")
    print(f"  {'side':>14s} {'comments':>9s} {'mentions':>9s} {'distinct':>9s} "
          f"{'mentions/name':>14s} {'>=1 mention':>12s} {'top share':>10s}")
    real = matched_real_texts(TREATED)
    sides: list[tuple[str, list[str]]] = [("matched real", real)]
    for tag, label in ((CONTROL, "v108"), (TREATED, "v109")):
        sides.append((label, [r["text"] for r in slots(tag)]))
    for label, texts in sides:
        counts: Counter[str] = Counter()
        with_hit = 0
        for text in texts:
            hits = _designators(text)
            if hits:
                with_hit += 1
            counts.update(hits)
        total = sum(counts.values()) or 1
        top = counts.most_common(1)[0][1] if counts else 0
        print(
            f"  {label:>14s} {len(texts):9d} {sum(counts.values()):9d} {len(counts):9d} "
            f"{total / max(1, len(counts)):14.3f} {with_hit / len(texts):12.3f} {top / total:10.4f}"
        )


def scorer_json(tag: str, name: str) -> dict[str, Any]:
    path = RUNS / tag / "cleaned/run_00_sampled_reddit" / name
    return json.loads(path.read_text(encoding="utf-8"))


def shipped_embeddings(tag: str) -> tuple[list[str], Any]:
    """The scorer's own saved comment embeddings, so no model is re-run.

    `score_thread_semantic_uniformity` persists one embedding per scored
    comment, which is what `semantic_mean_cosine` is computed from. Using them
    rather than re-embedding keeps this analysis on the shipped artifact
    (project rule E6) and makes the fidelity check exact.
    """

    import numpy as np

    thread = scorer_json(tag, "semantic_uniformity_results.json")["threads"][0]
    ids = [str(row["comment_id"]) for row in thread["comments"]]
    vectors = np.asarray([row["embedding"] for row in thread["comments"]], dtype=float)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return ids, vectors / np.clip(norms, 1e-12, None)


def cmd_cosine(_: Any) -> None:
    """Did the fired slots cause the semantic-uniformity regression?"""

    import numpy as np

    ids, unit = shipped_embeddings(TREATED)
    thread = scorer_json(TREATED, "semantic_uniformity_results.json")["threads"][0]
    sim = unit @ unit.T
    iu = np.triu_indices(len(ids), k=1)
    pair_sim = sim[iu]
    shipped = float(thread["mean_cosine_similarity"])
    print("\n== fidelity: pairwise cosine from saved embeddings must match the metric ==\n")
    print(f"  shipped={shipped:.12f} recomputed={float(pair_sim.mean()):.12f} "
          f"delta={abs(shipped - float(pair_sim.mean())):.2e}")
    if abs(shipped - float(pair_sim.mean())) > 1e-6:
        raise SystemExit("saved embeddings do not reproduce the metric; refusing to report")

    by_id = {row["comment_id"]: row for row in slots(TREATED)}
    fired = np.array([bool(by_id[i]["fired"]) for i in ids], dtype=bool)
    left, right = fired[iu[0]], fired[iu[1]]
    groups = {
        "both fired": left & right,
        "one fired": left ^ right,
        "neither fired": ~left & ~right,
    }
    print("\n== distal effect: pairwise cosine, by treatment of each side ==\n")
    print(f"  {'pair group':>16s} {'pairs':>8s} {'mean cosine':>12s}")
    for label, mask in groups.items():
        print(f"  {label:>16s} {int(mask.sum()):8d} {float(pair_sim[mask].mean()):12.4f}")
    print(f"  {'all pairs':>16s} {len(pair_sim):8d} {float(pair_sim.mean()):12.4f}")

    bf = float(pair_sim[groups["both fired"]].mean())
    nf = float(pair_sim[groups["neither fired"]].mean())
    print(f"\n  both-fired minus neither-fired = {bf - nf:+.4f}")

    cthread = scorer_json(CONTROL, "semantic_uniformity_results.json")["threads"][0]
    print(f"\n  v108 (spread arm off)   mean cosine = {float(cthread['mean_cosine_similarity']):.4f}")
    print(f"  v109 untreated subgroup mean cosine = {nf:.4f}")
    print(f"  v109 thread             mean cosine = {shipped:.4f}")


def cmd_labels(_: Any) -> None:
    """Within-run randomised contrast on every saved per-comment scorer label.

    The same content-independent draw, applied to the model outputs the twelve
    formal metrics are built from. This is the only way to attribute a tone,
    story or emotion movement to this arm on this run, because the v109 gate
    command also turned `--semantic-coverage-nonrepeat` off, so the v108->v109
    thread-level difference is not a one-arm contrast.
    """

    rows = {r["comment_id"]: r for r in slots(TREATED)}
    story = scorer_json(TREATED, "storyseeker_results.json")["threads"][0]["comments"]
    polite = scorer_json(TREATED, "politeness_results.json")["threads"][0]["comments"]
    emotion = scorer_json(TREATED, "go_emotions_results.json")["threads"][0]["comments"]

    def split(comments: list[dict[str, Any]]) -> dict[bool, list[dict[str, Any]]]:
        out: dict[bool, list[dict[str, Any]]] = {True: [], False: []}
        for row in comments:
            slot = rows.get(str(row["comment_id"]))
            if slot is not None:
                out[bool(slot["fired"])].append(row)
        return out

    print("\n== within-run randomised contrast on saved per-comment labels ==\n")
    s, p, e = split(story), split(polite), split(emotion)
    print(f"  {'group':>12s} {'n':>5s} {'P(story)':>9s} {'story>0.5':>10s} "
          f"{'polite':>8s} {'impolite':>9s} {'neutral':>8s} {'distinct emo':>13s} {'words':>7s}")
    for label, want in (("fired", True), ("not fired", False)):
        sg, pg, eg = s[want], p[want], e[want]
        polite_share = sum(1 for r in pg if r["pred_label"] in {"polite", "somewhat_polite"}) / len(pg)
        impolite = sum(1 for r in pg if r["pred_label"] == "impolite") / len(pg)
        neutral = sum(1 for r in pg if r["pred_label"] == "neutral") / len(pg)
        words = mean([float(len(r["text"].split())) for r in sg])
        print(
            f"  {label:>12s} {len(sg):5d} {mean([r['story_probability'] for r in sg]):9.4f} "
            f"{sum(1 for r in sg if r['story_probability'] > 0.5) / len(sg):10.4f} "
            f"{polite_share:8.4f} {impolite:9.4f} {neutral:8.4f} "
            f"{len({r['dominant_emotion'] for r in eg}):13d} {words:7.1f}"
        )
    print("\n  `polite` here pools polite + somewhat_polite; the formal metric counts")
    print("  only the strict `polite` class, so read this column as a direction.")


def cmd_bertscore(args: Any) -> None:
    """The same randomised contrast on the shipped `self_bertscore` scorer.

    Needs a pair-level rerun of the project's own scorer, which the metric does
    not save by default:

        python3 scripts/evaluation/score_thread_self_bertscore.py \\
          artifacts/generalized_card/runs/<tag>/cleaned/run_00_sampled_reddit \\
          --target-kind generated --device cpu --batch-size 32 --include-pairs \\
          --output-file <pairs.json>

    The rerun must reproduce the shipped thread mean exactly (project rule E6);
    this command checks that before reporting anything.
    """

    path = Path(args.pairs)
    data = json.loads(path.read_text(encoding="utf-8"))
    thread = data["threads"][0]
    shipped = json.loads(
        (RUNS / TREATED / "cleaned/run_00_sampled_reddit/self_bertscore_results.json").read_text(
            encoding="utf-8"
        )
    )["threads"][0]["mean_bert_f1"]
    delta = abs(thread["mean_bert_f1"] - shipped)
    print("\n== fidelity: the pair rerun must reproduce the shipped thread mean ==\n")
    print(f"  shipped={shipped:.12f} recomputed={thread['mean_bert_f1']:.12f} delta={delta:.2e}")
    if delta > 1e-9:
        raise SystemExit("pair rerun does not reproduce the shipped metric; refusing to report")

    rows = {r["comment_id"]: r for r in slots(TREATED)}
    print("\n== distal effect: pairwise BERTScore F1, by treatment of each side ==\n")
    groups: dict[str, list[float]] = {"both fired": [], "one fired": [], "neither fired": []}
    shared: dict[str, list[float]] = {"share a name": [], "no shared name": []}
    unmapped = 0
    for pair in thread["pairs"]:
        left = rows.get(str(pair["left_comment_id"]))
        right = rows.get(str(pair["right_comment_id"]))
        if left is None or right is None:
            unmapped += 1
            continue
        count = int(left["fired"]) + int(right["fired"])
        label = ("neither fired", "one fired", "both fired")[count]
        groups[label].append(float(pair["bert_f1"]))
        if count == 2:
            ln = {m.group().casefold() for m in DESIGNATOR.finditer(left["text"])}
            rn = {m.group().casefold() for m in DESIGNATOR.finditer(right["text"])}
            shared["share a name" if (ln & rn) else "no shared name"].append(float(pair["bert_f1"]))
    print(f"  unmapped pairs: {unmapped}\n")
    print(f"  {'pair group':>16s} {'pairs':>8s} {'mean F1':>10s}")
    for label, vals in groups.items():
        print(f"  {label:>16s} {len(vals):8d} {mean(vals):10.4f}")
    print(f"\n  both-fired minus neither-fired = {mean(groups['both fired']) - mean(groups['neither fired']):+.4f}")
    print("\n  Within both-fired pairs only, split by whether the two comments")
    print("  actually name a designator in common:\n")
    print(f"  {'subgroup':>16s} {'pairs':>8s} {'mean F1':>10s}")
    for label, vals in shared.items():
        print(f"  {label:>16s} {len(vals):8d} {mean(vals):10.4f}")


def cmd_mediation(_: Any) -> None:
    """Is the similarity effect the entity names, or just the extra length?

    `labels` shows treated comments run 48.5 words against 33.6 untreated -- the
    cue adds about 15 words. Length alone raises pairwise similarity (more
    tokens, more chance of overlap), so the treatment effect has to be checked
    inside length strata before it can be called a semantic effect. If it
    survives, the cue changes what the comment says; if it collapses, the fix is
    to make the referent *replace* text rather than extend it.
    """

    import numpy as np

    ids, unit = shipped_embeddings(TREATED)
    by_id = {row["comment_id"]: row for row in slots(TREATED)}
    fired = np.array([bool(by_id[i]["fired"]) for i in ids], dtype=bool)
    words = np.array([len(by_id[i]["text"].split()) for i in ids], dtype=float)
    sim = unit @ unit.T
    iu = np.triu_indices(len(ids), k=1)
    pair_sim, left, right = sim[iu], fired[iu[0]], fired[iu[1]]
    pair_words = words[iu[0]] + words[iu[1]]
    edges = np.quantile(pair_words, [0.0, 0.25, 0.5, 0.75, 1.0])

    print("\n== treatment effect inside pair-length strata ==\n")
    print(f"  {'combined words':>16s} {'both n':>7s} {'both':>8s} {'neither n':>10s} "
          f"{'neither':>8s} {'effect':>8s}")
    for low, high in zip(edges[:-1], edges[1:]):
        band = (pair_words >= low) & (pair_words <= high if high == edges[-1] else pair_words < high)
        both = band & left & right
        neither = band & ~left & ~right
        if both.sum() < 30 or neither.sum() < 30:
            continue
        b, n = float(pair_sim[both].mean()), float(pair_sim[neither].mean())
        print(
            f"  {f'[{low:.0f},{high:.0f}]':>16s} {int(both.sum()):7d} {b:8.4f} "
            f"{int(neither.sum()):10d} {n:8.4f} {b - n:+8.4f}"
        )
    both_all = left & right
    neither_all = ~left & ~right
    print(f"\n  unstratified effect = "
          f"{float(pair_sim[both_all].mean()) - float(pair_sim[neither_all].mean()):+.4f}")
    print("\n  Length is a real confounder here, so the stratified column is the")
    print("  estimate that matters; the unstratified one is an upper bound.")


COMMANDS = {
    "rate": cmd_rate,
    "mediation": cmd_mediation,
    "naming": cmd_naming,
    "shape": cmd_shape,
    "cosine": cmd_cosine,
    "bertscore": cmd_bertscore,
    "labels": cmd_labels,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("command", choices=[*COMMANDS, "all"])
    parser.add_argument(
        "--pairs",
        default="",
        help="Pair-level self_bertscore rerun JSON, for the `bertscore` block.",
    )
    args = parser.parse_args()
    names = list(COMMANDS) if args.command == "all" else [args.command]
    for name in names:
        COMMANDS[name](args)


if __name__ == "__main__":
    main()
