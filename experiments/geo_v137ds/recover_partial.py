#!/usr/bin/env python3
"""Rebuild a killed run's artifacts from its incremental record log.

  python3 experiments/geo_v137ds/recover_partial.py v156_20260903_p1

`records.partial.jsonl` holds every comment the moment it was written. If the
process dies before `write_discussion_bundle`, that log is the only copy -- this
turns it back into the `generation_records.json` shape the analysis tools read,
so a partial run is still worth something.

It does NOT reconstruct `discussion.json`: the reply tree is assembled by the
generator and a hand-rebuilt one would be a guess presented as an artifact. The
scoring pipeline is therefore still out of reach for a killed run; what this
recovers is the plan-and-text pairing every diagnostic in this directory uses.
"""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def main() -> int:
    if len(sys.argv) < 2:
        sys.exit("usage: recover_partial.py <run_tag> [more tags...]")
    for tag in sys.argv[1:]:
        run = REPO / "artifacts/generalized_card/runs" / tag
        partial = run / "generated/run_00_sampled_reddit/records.partial.jsonl"
        final = run / "generated/run_00_sampled_reddit/generation_records.json"
        if final.exists():
            print(f"{tag}: 已有完整产物，不需要恢复")
            continue
        if not partial.exists():
            print(f"{tag}: 没有 {partial.name}（这次运行早于增量落盘）")
            continue
        rows, bad = [], 0
        for line in partial.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                # A kill mid-write leaves one truncated line; the rest are good.
                bad += 1
        out = run / "generated/run_00_sampled_reddit/generation_records.recovered.json"
        out.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
        by_seed = {}
        for r in rows:
            by_seed[r.get("seed_index")] = by_seed.get(r.get("seed_index"), 0) + 1
        print(f"{tag}: 恢复 {len(rows)} 条"
              + (f"（{bad} 行损坏，已跳过）" if bad else "")
              + f"  seed 分布 {by_seed}")
        print(f"   -> {out.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
