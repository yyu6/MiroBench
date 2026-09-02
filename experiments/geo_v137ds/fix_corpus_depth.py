#!/usr/bin/env python3
"""Materialise a comments.jsonl the scorers can actually read.

The multidomain scrape and the product-thread scrape disagree on two things, and
both make a corpus score as garbage rather than fail loudly:

  1. FIELD NAMES. The scorers read `comment_fullname`/`comment_id`; the
     multidomain scrape writes `fullname`/`id`. Every comment therefore loads
     with comment_id "", the parent links never resolve, and anything keyed on a
     comment identity collapses -- politeness came back 13183/13183 `polite`,
     go_emotions 13183/13183 `neutral`, and semantic cosine all None.
  2. DEPTH. parent_id is recorded but `depth` is not, and load_real_comments
     reads `int(row.get("depth") or 0)`, so a 99-comment thread scores
     max_depth 1.0.

Both are aliased/derived here rather than in the scorers, which are pinned core
sources shared with the product-thread domains that already have these fields.


The multidomain scrape records parent_id (t1_/t3_ prefixed) but leaves `depth`
absent, while the product-thread scrape fills it. load_real_comments reads
`int(row.get("depth") or 0)`, so every comment in a scraped-without-depth corpus
lands at depth 0 and the thread scores as flat: max_depth and avg_depth come out
1.0 for a 99-comment thread, and structural_virality with them.

Derives depth by walking parent_id, and writes a real file in place of the
symlink so the adapter directory holds data in the same shape the scorers were
written against.
"""
import argparse, json, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ap = argparse.ArgumentParser()
ap.add_argument("domain", help="<name> in data/raw/discussions/<name>_geo")
a = ap.parse_args()

d = a.domain if a.domain.endswith("_geo") else f"{a.domain}_geo"
adapter = REPO / "data/raw/discussions" / d / d.replace("_geo", "")
src = next((p for p in adapter.glob("*.comments.jsonl")), None)
if src is None:
    sys.exit(f"no comments.jsonl under {adapter}")

rows = [json.loads(line) for line in open(src.resolve()) if line.strip()]
by_id = {str(r.get("id")): r for r in rows}
fullname_to_id = {str(r.get("fullname")): str(r.get("id")) for r in rows}

def depth_of(row, seen=None):
    seen = seen or set()
    p = str(row.get("parent_id") or "")
    if not p or p.startswith("t3_"):
        return 0
    pid = fullname_to_id.get(p) or p[3:] if p.startswith("t1_") else None
    parent = by_id.get(str(pid))
    if parent is None or str(parent.get("id")) in seen:
        return 0
    seen.add(str(row.get("id")))
    return 1 + depth_of(parent, seen)

sys.setrecursionlimit(10000)
changed = 0
aliased = 0
for r in rows:
    if r.get("depth") in (None, "", "None"):
        r["depth"] = depth_of(r); changed += 1
    if not str(r.get("comment_id") or "").strip():
        r["comment_id"] = str(r.get("id") or ""); aliased += 1
    if not str(r.get("comment_fullname") or "").strip():
        r["comment_fullname"] = str(r.get("fullname") or "")

from collections import Counter
dist = Counter(r["depth"] for r in rows)
if src.is_symlink():
    src.unlink()
with open(src, "w", encoding="utf-8") as fh:
    for r in rows:
        fh.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"{changed} of {len(rows)} comments given a derived depth, "
      f"{aliased} given comment_id/comment_fullname -> {src.relative_to(REPO)}")
print("  depth distribution:", dict(sorted(dist.items())[:10]))
