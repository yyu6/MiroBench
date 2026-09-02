#!/usr/bin/env python3
"""Does the Planner reuse the lenses it named, or invent a new set per batch?

  python3 experiments/geo_v137ds/lens_growth.py v153_20260903

Under `--plan-vocabulary open` the Planner names its own lens set, and it sees
only its own batch of eight slots plus a ledger summary of what earlier batches
used. That ledger's instruction reads "Shift independent new rows away from
dominant combinations", which under a closed taxonomy correctly stops P01 from
filling a thread -- and under an open one pushes toward naming something new
every batch. A 97-comment thread is 13 batches; if each names five fresh lenses
the thread ends with ~65, which no real thread has.

This reports the cumulative distinct-lens count per batch. Flat growth after the
first few batches means reuse; linear growth to the batch count times five means
the ledger is pushing the wrong way and the instruction needs a mode-aware form.
"""
import collections
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "generalized_card"))
from generalized_card.plan_vocabulary import canonical_lens  # noqa: E402


def _real_positions(run_name: str) -> int:
    """The matched real thread's own position count, measured.

    This line used to print "真人 thread 一般 5-12 种" -- a range I invented,
    and the same invented range that was in the Planner prompt and that led me
    to call a correct result an explosion. A tool that prints a made-up
    baseline will mislead the next reading of it too.
    """
    import re

    m = re.match(r"(.+)_p(\d+)$", run_name)
    if not m:
        return 0
    seed = int(m.group(2))
    try:
        sys.path.insert(0, str(REPO / "experiments/geo_v137ds"))
        from surface_vs_content import real_by_seed

        from generalized_card.planning_quality import PlanSemanticIndex
        from generalized_card.plan_vocabulary import real_position_count

        bodies = (real_by_seed().get(seed) or [])
        if len(bodies) < 4:
            return 0
        index = PlanSemanticIndex(
            model_name="sentence-transformers/all-mpnet-base-v2", device="cpu"
        )
        return real_position_count(bodies, index.encode_texts)
    except Exception:  # noqa: BLE001
        return 0


def main() -> None:
    prefix = sys.argv[1] if len(sys.argv) > 1 else "v153_20260903"
    for d in sorted((REPO / "artifacts/generalized_card/runs").glob(f"{prefix}_p*")):
        log = d / "logs/planning_quality.jsonl"
        if not log.exists():
            continue
        print(f"\n{'='*66}\n{d.name}\n{'='*66}")
        seen, seen_canon, rows = set(), set(), []
        for i, line in enumerate(log.read_text().splitlines(), 1):
            rec = json.loads(line)
            plans = [p for p in (rec.get("selected_plans") or {}).values() if isinstance(p, dict)]
            if not plans:
                continue
            batch = [str(p.get("perspective_id") or "") for p in plans]
            new = [x for x in batch if x not in seen]
            seen.update(batch)
            seen_canon.update(canonical_lens(x) for x in batch)
            rows.append((i, len(plans), len(set(batch)), len(new), len(seen), len(seen_canon)))
        if not rows:
            continue
        print(f"{'批':>4}{'slot':>6}{'本批不同':>9}{'本批新增':>9}{'累计不同':>9}{'归一后':>8}")
        for r in rows:
            print(f"{r[0]:>4}{r[1]:>6}{r[2]:>9}{r[3]:>9}{r[4]:>9}{r[5]:>8}")
        slots = sum(r[1] for r in rows)
        print(f"\n  {slots} slot -> {rows[-1][4]} 个不同 lens（归一后 {rows[-1][5]}）")
        real = _real_positions(d.name)
        target = f"真实 thread 实测 {real} 个位置" if real else "真实位置数未知"
        print(f"  每 slot 新增 lens: {rows[-1][4]/slots:.2f}   ({target})")
        counts = collections.Counter()
        for line in log.read_text().splitlines():
            for p in (json.loads(line).get("selected_plans") or {}).values():
                if isinstance(p, dict):
                    counts[canonical_lens(p.get("perspective_id"))] += 1
        top = counts.most_common(8)
        print(f"  最高频占 {top[0][1]/slots:.0%}（目标 <= 33%）")
        for k, v in top:
            print(f"      {v:>4}  {k[:58]}")


if __name__ == "__main__":
    main()
