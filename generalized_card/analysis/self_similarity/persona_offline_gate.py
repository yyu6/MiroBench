#!/usr/bin/env python3
"""Offline gate for the persona arm: does it fire, per speaker, before any spend?

E9/E12/E15: an arm can record itself ON and reach zero prompts. This replays a
paid run's own thread structure through the persona runtime and reports what
each slot would actually receive -- no API calls.
"""
from __future__ import annotations
import argparse, json, glob, sys
from collections import Counter, defaultdict
from pathlib import Path
REPO = Path("/Users/yaoningyu/Desktop/UIUC/GEO")
sys.path.insert(0, str(REPO / "generalized_card"))
from generalized_card.persona_bridge import (  # noqa: E402
    MODE_FULL, MODE_PROJECTED, build_runtime, _speaker_id_from_author,
)

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="v128_interaction_n10_20260828_v1")
    ap.add_argument("--mode", default=MODE_PROJECTED, choices=[MODE_PROJECTED, MODE_FULL])
    ap.add_argument("--dimensions", default="fam_photography,ind_consumer_electronics")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    root = REPO / "third_party" / "MatrAIx-Persona-8B"
    rt = build_runtime(
        mode=a.mode, matraix_root=root,
        dataset_dir=root / "persona" / "datasets" / "matraix-persona-dev-sample",
        assignment_seed=a.seed,
        expertise_dimensions=tuple(x for x in a.dimensions.split(",") if x),
    )
    cfg = rt.public_config()
    print(f"mode={a.mode}  eligible_personas={cfg['eligible_personas']}  "
          f"system_chars mean={cfg['system_chars_mean']} max={cfg['system_chars_max']}")

    per_thread = defaultdict(list)
    for p in sorted(glob.glob(str(REPO/"artifacts/generalized_card/runs"/a.run/"cleaned/run_*_sampled_reddit/discussion.json"))):
        d = json.loads(Path(p).read_text())
        for post in d.get("posts") or []:
            seed = int(post.get("seed_index") or 0)
            def walk(cs):
                for c in cs:
                    yield c
                    yield from walk(c.get("replies") or [])
            for c in walk(post.get("comments") or []):
                per_thread[seed].append(c)

    print(f"\n{'thread':>7} {'slots':>6} {'speakers':>9} {'personas':>9} {'per-slot would be':>18} {'consistent':>11}")
    tot_slots = tot_sp = tot_pe = 0
    incons = 0
    all_personas = Counter()
    for seed, comments in sorted(per_thread.items()):
        by_speaker = defaultdict(set)
        slot_personas = set()
        for c in comments:
            sid = _speaker_id_from_author(c.get("author"))
            proxy = dict(c)
            proxy["local_task_id"] = int(c.get("comment_id") or 0) % 10000
            asn = rt.assign(seed_index=seed, task=proxy, speaker_id=sid)
            by_speaker[sid].add(asn.persona_id)
            all_personas[asn.persona_id] += 1
            slot = rt.assign(seed_index=seed, task=proxy, speaker_id="")
            slot_personas.add(slot.persona_id)
        personas = {p for v in by_speaker.values() for p in v}
        bad = sum(1 for v in by_speaker.values() if len(v) > 1)
        incons += bad
        tot_slots += len(comments); tot_sp += len(by_speaker); tot_pe += len(personas)
        print(f"{seed:>7} {len(comments):>6} {len(by_speaker):>9} {len(personas):>9} "
              f"{len(slot_personas):>18} {'OK' if bad == 0 else f'{bad} BROKEN':>11}")
    print(f"\ntotals: {tot_slots} slots, {tot_sp} speakers, {tot_pe} persona-assignments")
    print(f"distinct personas used across the corpus: {len(all_personas)} of {cfg['eligible_personas']}")
    print(f"speakers with an inconsistent persona: {incons}   (must be 0)")
    top = all_personas.most_common(3)
    print(f"most-used personas: {top}  (max share {100*top[0][1]/tot_slots:.1f}%)")
    print(f"\nGATE: {'PASS' if incons == 0 and len(all_personas) >= 20 else 'FAIL'}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
