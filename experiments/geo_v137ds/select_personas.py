#!/usr/bin/env python3
"""Select a register-diverse persona set from the public 1M coreset.

  python3 experiments/geo_v137ds/select_personas.py celebrity_geo --count 400

Every persona run so far used `matraix-persona-dev-sample` -- 200 rows whose own
manifest calls itself a dev sample, and whose `parent_pool` field points at the
1M coreset this reads instead. In that sample the dimensions that decide how a
person WRITES are nearly empty: `register` on 26% of rows across 3 values,
`english_proficiency` 37%/3, `primary_language` 31%/4, `skill_writing` 12%,
`skill_storytelling` 5%, and a median of 68 populated dimensions. The projection
that renders a persona keeps ten dimensions, and with `register` present on only
30 of 123 eligible rows those ten filled up with `tech_savviness`,
`lstyle_work_schedule` and `lstyle_commute_mode`. The resulting 123 system
prompts were 0.809 alike with 17 exact duplicates, which is why G201 read the
persona layer as having no room -- a property of the sample, not of the method.

In the coreset the same dimensions are 64%/6, 72%/5, 72%/45, 53% and 51%, with a
median of 510 populated dimensions.

Selection is not "the first N eligible". The failing metric is register
uniformity, so the set is chosen to SPREAD on the register axes: rows are
bucketed by the cross-product of the axes below and taken round-robin across
buckets, so a rare register value is represented before a common one repeats.
Within a bucket, human-grounded sources come before synthetic and a row with
more populated dimensions comes before a sparser one. Deterministic given the
same shards and count.
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MATRAIX = REPO / "third_party" / "MatrAIx-Persona-8B"
BANK = MATRAIX / "persona" / "datasets" / "matraix-persona-1m"

# The axes register variation actually lives on. A row must carry at least
# --min-register-axes of these to be eligible, and the strata are their product.
REGISTER_AXES = (
    "register",
    "english_proficiency",
    "primary_language",
    "multilingualism",
    "age_bracket",
    "region",
    "urbanicity",
    "socioeconomic_band",
)
# Kept when present because the projection renders them, but not required.
VOICE_AXES = (
    "skill_writing",
    "skill_storytelling",
    "tone_expected",
    "dominant_trait",
    "emotional_state",
    "neurotype",
    "cultural_background",
    "political_lean",
    "highest_education",
    "media_diet",
    "tech_savviness",
    "trust_level",
    "expertise_gap",
    "intent",
    "query_complexity",
    "decision_style",
    "risk_tolerance",
    "time_pressure",
)
# Strata are built from a subset -- the full product of eight axes would give
# more buckets than rows and degrade to "one row per bucket", which is just a
# shuffle. These four carry most of the writing variance.
# `english_proficiency` is deliberately NOT a stratum axis: the runtime keeps
# only three of its values, so stratifying on it buys three buckets and spends
# the budget that `urbanicity` and `socioeconomic_band` -- both free to vary --
# would otherwise get.
STRATUM_AXES = ("register", "age_bracket", "urbanicity", "socioeconomic_band")

# The runtime discards personas before they can ever be drawn:
# `persona_bridge._eligible_for_english_reddit` drops minors, and drops anyone
# whose `english_proficiency` is outside {Native, Fluent, Intermediate} unless
# that field is missing AND no conflicting primary language is recorded. A first
# pass selected for spread ACROSS proficiency and language and then watched the
# runtime throw 196 of 400 rows away -- selecting for diversity that cannot
# survive to generation. These mirror the runtime's rule; if it changes, this
# must change with it, and `test_selector_matches_runtime_eligibility` fails if
# it does not.
MINOR_AGES = {"Under 5", "5-12", "13-17"}
ENGLISH_OK = {"Native", "Fluent (C1-C2)", "Intermediate (B1-B2)"}


def runtime_eligible(vals: dict) -> bool:
    if vals.get("age_bracket") in MINOR_AGES:
        return False
    english = vals.get("english_proficiency", "__missing__")
    primary = vals.get("primary_language", "__missing__")
    if english in ENGLISH_OK:
        return True
    return english == "__missing__" and primary in {"__missing__", "English"}


SOURCE_RANK = {
    "real_human_survey": 0, "gss": 1, "prism": 2,
    "stackoverflow": 3, "amazon": 4, "wiki": 5, "synthetic": 9,
}

DOMAIN_ELIGIBILITY = {
    # one Expertise:Domains plus one Professional:Industry, mirroring camera's
    # fam_photography + ind_consumer_electronics
    "celebrity_geo": ("fam_film_studies", "ind_entertainment"),
    "camera": ("fam_photography", "ind_consumer_electronics"),
}
PRESENT = {"Aware", "Familiar", "Proficient", "Expert",
           "Some exposure", "Experienced", "Veteran"}
# The codec carries "null" and "None" as real string labels, not as absent
# values. Counting them as present is how a first pass selected 7 personas whose
# rendered register line read `register: null`.
ABSENT = {"", "null", "none", "n/a", "na", "unknown", "unspecified"}


def has(vals: dict, field: str) -> bool:
    return str(vals.get(field) or "").strip().lower() not in ABSENT


def load_codec():
    sys.path.insert(0, str(MATRAIX))
    from persona.post_process.unified_dataset.schema import AttributeCodec

    return AttributeCodec.from_codes_schema(BANK / "persona_codes.schema.json")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("domain")
    ap.add_argument("--count", type=int, default=400)
    ap.add_argument("--min-register-axes", type=int, default=5)
    ap.add_argument("--min-populated", type=int, default=150)
    ap.add_argument("--require-domain", action="store_true",
                    help="keep only rows showing exposure on the domain's own "
                         "expertise dimensions; off by default because the "
                         "register axes are what this set is for and the domain "
                         "filter is what cut the dev sample to 123 rows")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    import duckdb
    import numpy as np

    codec = load_codec()
    index_of = {f: i for i, f in enumerate(codec.field_ids)}
    # decode_row scans every value map for every field; we need ~26 fields, so
    # invert the maps once and read only those indices.
    wanted = [f for f in (*REGISTER_AXES, *VOICE_AXES) if f in index_of]
    dom_axes = [f for f in DOMAIN_ELIGIBILITY.get(a.domain, ()) if f in index_of]
    read_fields = wanted + [f for f in dom_axes if f not in wanted]
    rev = {
        f: {i: v for v, i in codec.value_codes[index_of[f]].items()}
        for f in read_fields
    }
    idx = {f: index_of[f] for f in read_fields}

    shards = sorted(BANK.glob("data/persona-1m-*.parquet"))
    if not shards:
        sys.exit(f"没有分片: {BANK/'data'}")
    print(f"分片 {len(shards)} 个 (完整库是 10 个)")

    buckets: dict[tuple, list] = collections.defaultdict(list)
    seen = kept = 0
    src_counts = collections.Counter()
    # pyarrow 19 raises "Repetition level histogram size mismatch" on some of
    # these shards -- a known reader bug against the writer that produced them,
    # not corruption; duckdb reads the same bytes. Reading through duckdb also
    # pushes the column projection down so only the seven needed columns are
    # materialised out of 12.
    con = duckdb.connect()
    for shard in shards:
        reader = con.execute(
            "SELECT source, source_record_id, attributes, null_bitmap, "
            "attribute_overrides, populated_attribute_count "
            f"FROM read_parquet('{shard}')"
        ).fetch_record_batch(4096)
        for batch in reader:
            d = batch.to_pydict()
            for k in range(batch.num_rows):
                seen += 1
                if int(d["populated_attribute_count"][k] or 0) < a.min_populated:
                    continue
                packed = np.frombuffer(d["attributes"][k], dtype=np.uint8)
                codes = np.empty(len(codec.field_ids), dtype=np.uint8)
                codes[0::2] = packed[: len(codes) // 2] & 0x0F
                codes[1::2] = (packed[: len(codes) // 2] >> 4) & 0x0F
                nulls = np.zeros(len(codec.field_ids), dtype=np.uint8)
                nb = d["null_bitmap"][k]
                if nb:
                    u = np.unpackbits(np.frombuffer(nb, dtype=np.uint8), bitorder="little")
                    nulls[: min(len(nulls), u.size)] = u[: len(nulls)]
                vals = {}
                for f in read_fields:
                    i = idx[f]
                    if nulls[i]:
                        continue
                    lab = rev[f].get(int(codes[i]))
                    if lab:
                        vals[f] = lab
                for ov in (d["attribute_overrides"][k] or ()):
                    fi = int(ov["field_index"])
                    if 0 <= fi < len(codec.field_ids):
                        fid = codec.field_ids[fi]
                        if fid in rev and ov.get("value"):
                            vals[fid] = str(ov["value"])
                if not runtime_eligible(vals):
                    continue
                if not has(vals, "register"):
                    # `register` is the axis the failing metric is about; a row
                    # without it cannot contribute to the spread this set exists
                    # to create.
                    continue
                if sum(1 for f in REGISTER_AXES if has(vals, f)) < a.min_register_axes:
                    continue
                if a.require_domain and dom_axes and not any(
                    vals.get(f) in PRESENT for f in dom_axes
                ):
                    continue
                kept += 1
                src = str(d["source"][k])
                src_counts[src] += 1
                key = tuple(
                    vals[f] if has(vals, f) else "" for f in STRATUM_AXES
                )
                buckets[key].append((
                    SOURCE_RANK.get(src, 8),
                    -int(d["populated_attribute_count"][k] or 0),
                    str(d["source_record_id"][k] or ""),
                    src, vals,
                ))
    print(f"扫过 {seen:,} 行，通过筛选 {kept:,} 行，落在 {len(buckets):,} 个分层里")
    print("来源构成: " + ", ".join(f"{s}={c:,}" for s, c in src_counts.most_common()))
    if not kept:
        sys.exit("没有一行通过筛选；放宽 --min-register-axes / --min-populated")

    for key in buckets:
        buckets[key].sort(key=lambda r: (r[0], r[1], r[2]))

    # Two levels, because one flat round-robin over strata sorted by size takes
    # from the biggest bucket first and the biggest buckets ARE the common
    # register values: a first pass returned 40 of 60 as "Formal / standard",
    # which is the concentration this set exists to avoid. `register` gets an
    # equal quota per value, and the sub-strata inside a value are visited
    # rarest-first so an unusual (register, proficiency, age, region) corner is
    # represented before a common one repeats.
    by_register: dict[str, list[tuple]] = collections.defaultdict(list)
    for key in buckets:
        by_register[key[0]].append(key)
    for reg in by_register:
        by_register[reg].sort(key=lambda k: (len(buckets[k]), k))
    reg_values = sorted(by_register, key=lambda r: (-len(by_register[r]), r))
    cursor = {r: 0 for r in reg_values}
    depth = {k: 0 for k in buckets}
    chosen: list[tuple] = []
    stalled = 0
    while len(chosen) < a.count and stalled < len(reg_values):
        stalled = 0
        for reg in reg_values:
            if len(chosen) >= a.count:
                break
            keys = by_register[reg]
            took = False
            for _ in range(len(keys)):
                key = keys[cursor[reg] % len(keys)]
                cursor[reg] += 1
                if depth[key] < len(buckets[key]):
                    chosen.append(buckets[key][depth[key]])
                    depth[key] += 1
                    took = True
                    break
            if not took:
                stalled += 1
    strata = len({tuple(c[4].get(f, "") for f in STRATUM_AXES) for c in chosen})
    print(f"选出 {len(chosen)} 个，覆盖 {strata} 个分层，{len(reg_values)} 种 register 等额配比")

    out = Path(a.out) if a.out else (
        MATRAIX / "persona" / "datasets" / f"matraix-persona-{a.domain}-{len(chosen)}"
    )
    out.mkdir(parents=True, exist_ok=True)
    manifest = []
    for n, (_, _, rid, src, vals) in enumerate(chosen, 1):
        pid = f"{n:04d}"
        body = ["persona_id: '%s'" % pid, "version: '1.0'", f"source: {src}",
                "dimensions:"]
        # Drop the codec's placeholder labels. `english_proficiency: None` is a
        # real string in the value list, and writing it renders a persona line
        # that says the dimension is unset -- noise in the system prompt, and it
        # counts against the ten the projection keeps.
        written = {f: v for f, v in sorted(vals.items()) if has(vals, f)}
        for f, v in written.items():
            body.append(f"  {f}: {json.dumps(v, ensure_ascii=False)}")
        (out / f"persona_{pid}.yaml").write_text("\n".join(body) + "\n")
        manifest.append({
            "persona_id": pid,
            # The loader resolves a manifest path relative to matraix_root when
            # the set lives under it, and absolutely otherwise, so a set written
            # to a scratch directory still loads.
            "path": (
                f"{out.relative_to(MATRAIX)}/persona_{pid}.yaml"
                if MATRAIX in out.parents or out == MATRAIX
                else str(out / f"persona_{pid}.yaml")
            ),
            "source": src, "source_record_id": rid,
            "display_name": None,
            "dimensions": {f: vals[f] for f in STRATUM_AXES if has(vals, f)},
        })
    (out / "manifest.json").write_text(json.dumps({
        "kind": out.name,
        "count": len(manifest),
        "schema_version": "1.0",
        "dimension_count": len(codec.field_ids),
        "parent_pool": "persona/datasets/matraix-persona-1m",
        "hf_repo": "MatrAIx2026/MatrAIx_Persona_1M_Public_Release",
        "selected_for": "register spread",
        "stratum_axes": list(STRATUM_AXES),
        "register_axes": list(REGISTER_AXES),
        "min_register_axes": a.min_register_axes,
        "min_populated_attributes": a.min_populated,
        "require_domain": bool(a.require_domain),
        "shards_read": [s.name for s in shards],
        "source_counts": dict(src_counts),
        "personas": manifest,
    }, indent=1))
    print(f"-> {out}")
    for f in ("register", "english_proficiency", "primary_language", "skill_writing"):
        c = collections.Counter(v[4].get(f, "(缺)") for v in chosen)
        print(f"  {f}: " + ", ".join(f"{k}={n}" for k, n in c.most_common(7)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
