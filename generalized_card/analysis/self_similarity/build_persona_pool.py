#!/usr/bin/env python3
"""Select a larger, register-diverse persona pool from the 1M public release.

The bundled `matraix-persona-dev-sample` is 200 records; after the English/adult
filter and the usability filters it yields 123 personas and, more to the point,
only **112 distinct rendered system prompts** and **24 distinct
(register, tone_expected, skill_writing) combinations**. Since the mechanism is
"one register per speaker", the binding constraint is distinct prompts, not
record count -- adding thin records from the same sample collapses onto prompts
already in use.

This writes a new dataset directory in the same layout the loader expects, so
`--matraix-dataset` picks it up with no code change.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path("/Users/yaoningyu/Desktop/UIUC/GEO")
sys.path.insert(0, str(REPO / "generalized_card"))
from generalized_card.persona_bridge import (  # noqa: E402
    _NULLISH, _eligible_for_english_reddit,
)

VOICE = ("register", "tone_expected", "skill_writing")


def filled(dimensions: dict, keys) -> int:
    return sum(
        1
        for k in keys
        if str(dimensions.get(k) or "").strip().lower() not in _NULLISH
    )


def voice_key(dimensions: dict) -> tuple:
    return tuple(str(dimensions.get(k) or "").strip() for k in VOICE)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="dataset directory to write")
    ap.add_argument("--target", type=int, default=600, help="personas to select")
    ap.add_argument("--min-voice", type=int, default=1)
    ap.add_argument("--per-voice-cap", type=int, default=40,
                    help="max personas sharing one (register,tone,skill_writing) key")
    ap.add_argument("--scan", type=int, default=200000, help="rows to scan")
    a = ap.parse_args()

    shards = sorted(glob.glob(str(
        Path.home() / ".cache/huggingface/hub/datasets--MatrAIx2026--MatrAIx_Persona_1M_Public_Release"
        / "snapshots/*/data/*.parquet")))
    if not shards:
        raise SystemExit("no shard downloaded yet")
    import pyarrow.parquet as pq

    # The public release packs 1,290 attributes as 4-bit codes plus a null
    # bitmap; the repository ships the official decoder, so use it rather than
    # reimplementing the layout.
    sys.path.insert(0, str(REPO / "third_party" / "MatrAIx-Persona-8B"
                           / "persona" / "validation" / "scripts"))
    from decode_persona_1m import decode_row, load_schema

    cols = load_schema()

    seen_voice: Counter = Counter()
    chosen: list[dict] = []
    scanned = 0
    for shard in shards:
        pf = pq.ParquetFile(shard)
        for batch in pf.iter_batches(batch_size=4096):
            for row in batch.to_pylist():
                scanned += 1
                if scanned > a.scan or len(chosen) >= a.target:
                    break
                dims = decode_row(row, cols)
                if not dims:
                    continue
                if not _eligible_for_english_reddit(dims):
                    continue
                if filled(dims, VOICE) < a.min_voice:
                    continue
                key = voice_key(dims)
                # Spread across register combinations: the dev sample's whole
                # failure is 24 combinations for 326 speakers, and an uncapped
                # scan would just refill the most common one.
                if seen_voice[key] >= a.per_voice_cap:
                    continue
                seen_voice[key] += 1
                chosen.append({"persona_id": f"{scanned:07d}",
                               "dimensions": dims})
            if scanned > a.scan or len(chosen) >= a.target:
                break
        if scanned > a.scan or len(chosen) >= a.target:
            break

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    import yaml
    for i, rec in enumerate(chosen, 1):
        (out / f"persona_{i:04d}.yaml").write_text(
            yaml.safe_dump({"persona_id": rec["persona_id"],
                            "schema_version": "1.0",
                            "dimensions": rec["dimensions"]},
                           sort_keys=True, allow_unicode=True),
            encoding="utf-8")
    (out / "manifest.json").write_text(json.dumps({
        "kind": "card-register-diverse",
        "count": len(chosen),
        "source_dataset": "MatrAIx2026/MatrAIx_Persona_1M_Public_Release",
        "scanned_rows": scanned,
        "min_voice_dimensions": a.min_voice,
        "per_voice_cap": a.per_voice_cap,
        "distinct_voice_keys": len(seen_voice),
    }, indent=2) + "\n", encoding="utf-8")
    print(f"scanned {scanned} rows -> selected {len(chosen)} personas")
    print(f"distinct (register,tone_expected,skill_writing) keys: {len(seen_voice)}")
    print(f"top keys: {seen_voice.most_common(5)}")
    print(f"written to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
