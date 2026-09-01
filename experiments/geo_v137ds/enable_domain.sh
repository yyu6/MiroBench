#!/usr/bin/env bash
# Make one reddit_multidomain_baselines domain runnable by GEO.
#
#   ./experiments/geo_v137ds/enable_domain.sh celebrity
#   ./experiments/geo_v137ds/enable_domain.sh celebrity --score   # also score real threads
#
# GEO needs two things a multidomain domain does not have yet:
#
#   1. generalized_card/configs/domains/<domain>.json -- five required fields
#      plus the vocabulary lists that shape Planner prompts.  The raw corpus is
#      already in the layout GEO's loaders expect (*.jsonl + *.comments.jsonl),
#      one directory level down, so this links rather than copies.
#   2. a real thread_scores.csv -- the matched evaluator pairs each generated
#      thread with its own source thread by `thread_id`, so every eligible real
#      thread must be scored once.  Local models only, no API, but slow; it is
#      behind --score so the config step stays instant.
#
# Generation does NOT read real_scores_csv; only evaluation does.  So a domain
# can generate as soon as step 1 is done, and be scored later.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${GEO_PYTHON:-/Users/yaoningyu/.pyenv/versions/3.11.8/bin/python3}"
SRC_ROOT="${GEO_MULTIDOMAIN_DATA:-$ROOT/data/reddit_domain_posts 2}"

domain="${1:-}"; shift || true
do_score=0; device="auto"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --score)  do_score=1; shift ;;
    --device) device="$2"; shift 2 ;;
    -h|--help) grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) shift ;;
  esac
done
[[ -n "$domain" ]] || { echo "usage: enable_domain.sh <domain> [--score] [--device auto|cpu|mps]" >&2
  echo "available:" >&2; ls "$SRC_ROOT" 2>/dev/null | sed 's/^/  /' >&2; exit 2; }

src="$SRC_ROOT/$domain"
[[ -d "$src" ]] || { echo "ERROR: no raw corpus at $src" >&2
  echo "available:" >&2; ls "$SRC_ROOT" 2>/dev/null | sed 's/^/  /' >&2; exit 2; }
posts="$(ls "$src"/*.jsonl 2>/dev/null | grep -v '\.comments\.jsonl$' | head -1)"
comments="$(ls "$src"/*.comments.jsonl 2>/dev/null | head -1)"
[[ -n "$posts" && -n "$comments" ]] || { echo "ERROR: $src needs both <d>.jsonl and <d>.comments.jsonl" >&2; exit 2; }

# GEO's loaders walk category/<product dir>/*.comments.jsonl, one level deeper
# than the multidomain layout. Link rather than copy: 12 domains of raw JSONL is
# large, and a copy would silently diverge from the corpus the baselines use.
adapter="$ROOT/data/raw/discussions/${domain}_geo/$domain"
mkdir -p "$adapter"
ln -sf "$posts" "$adapter/"; ln -sf "$comments" "$adapter/"
echo "corpus linked -> data/raw/discussions/${domain}_geo/$domain/"

scores="$ROOT/artifacts/baselines/${domain}_geo/real/thread_scores.csv"
GEO_SRC_ROOT="$SRC_ROOT" GEO_DOMAIN="$domain" GEO_ADAPTER="$adapter" GEO_SCORES="$scores" GEO_ROOT="$ROOT" "$PY" - <<'PYX'
"""Write the domain config, deriving vocabulary from the corpus itself."""
import json, os, re, sys, pathlib
from collections import Counter
sys.path.insert(0, str(pathlib.Path(os.environ["GEO_ROOT"]) / "scripts" / "evaluation"))
from score_thread_semantic_uniformity import load_real_comments

domain = os.environ["GEO_DOMAIN"]; root = pathlib.Path(os.environ["GEO_ROOT"])
adapter = pathlib.Path(os.environ["GEO_ADAPTER"])
by_thread, _ = load_real_comments(adapter)
eligible = {k: v for k, v in by_thread.items() if len(v) >= 5}

posts_file = next(p for p in adapter.iterdir() if p.name.endswith(".jsonl")
                  and not p.name.endswith(".comments.jsonl"))
subs, titles = Counter(), []
for line in open(posts_file):
    try: r = json.loads(line)
    except Exception: continue
    subs[str(r.get("subreddit") or "")] += 1
    titles.append(str(r.get("title") or ""))

STOP = set("""the a an and or but of to in for on at by with from as is are was were be been
this that these those it its his her their they them you your i we our not no do does did so
if then than there here what when where which who whom why how all any some most more much
just also very can could would should will shall may might must about into over under after
before out up down off again once only own same too s t don now new get got go going one two
he she him his my me us who's it's i'm don't didn't doesn't isn't that's there's what's has
have had having said says say like really actually still even way lot make made makes think
know time good people want need thing things going right yes yeah well never always something
someone anything everything because they're you're we're i've they've it'd he's she's http
https www com reddit imgur gt amp nbsp deleted removed edit
""".split())

def tokens(texts):
    for t in texts:
        for w in re.findall(r"[a-z][a-z'-]{3,}", t.lower()):
            if w not in STOP and not w.startswith("http"):
                yield w

def distinctive(bodies, background, n, banned=()):
    """Terms this domain uses far more than the other domains do.

    Raw frequency returns function words no matter how long the stop list is --
    the first attempt produced "people, think, because, https". Scoring each
    term against its rate in the other eleven corpora surfaces what the domain
    is actually about, with no hand-written vocabulary per domain.
    """
    fg = Counter(tokens(bodies)); bg = Counter(tokens(background))
    nf = sum(fg.values()) or 1; nb = sum(bg.values()) or 1
    scored = []
    for w, k in fg.items():
        if k < 12: continue
        # Subreddit names and site furniture score as maximally distinctive --
        # they appear in one corpus and nowhere else -- but they describe the
        # forum, not the subject.
        if w in banned or "reddit" in w or any(b and b in w for b in banned): continue
        rate = k / nf
        base = (bg.get(w, 0) + 1) / nb
        scored.append((rate / base, k, w))
    scored.sort(reverse=True)
    return [w for _, _, w in scored[:n]]

def proper(texts, n):
    """Capitalised spans that recur -- names the generator must not invent around."""
    c = Counter()
    for t in texts:
        for m in re.findall(r"\b[A-Z][a-zA-Z'&.-]+(?:\s+[A-Z][a-zA-Z'&.-]+){0,2}", t):
            m = m.strip()
            if len(m) < 4: continue
            if m.lower() in STOP: continue
            if len(m.split()) == 1 and m.lower() in {w for w in STOP}: continue
            c[m] += 1
    # prefer multi-word names, then frequent single tokens
    multi = [(k, v) for k, v in c.items() if len(k.split()) > 1 and v >= 3]
    multi.sort(key=lambda kv: -kv[1])
    out = [k for k, _ in multi[:n]]
    if len(out) < n:
        single = [(k, v) for k, v in c.items() if len(k.split()) == 1 and v >= 8]
        single.sort(key=lambda kv: -kv[1])
        out += [k for k, _ in single[: n - len(out)]]
    return out[:n]

bodies = [c.text for v in eligible.values() for c in v][:20000]

# background = the other domains' corpora, for the distinctiveness score
background = []
src_root = pathlib.Path(os.environ.get("GEO_SRC_ROOT", root / "data/reddit_domain_posts 2"))
for other in sorted(p for p in src_root.iterdir() if p.is_dir() and p.name != domain):
    f = next((x for x in other.glob("*.comments.jsonl")), None)
    if not f: continue
    with open(f) as fh:
        for i, line in enumerate(fh):
            if i >= 1200: break
            try: background.append(str(json.loads(line).get("body") or ""))
            except Exception: pass
cfg = {
    "domain_id": f"{domain}_geo",
    "display_name": domain.replace("_", " "),
    "community_context": "Reddit " + ", ".join(f"r/{s}" for s, _ in subs.most_common(4) if s),
    "raw_discussions_dir": f"data/raw/discussions/{domain}_geo",
    "real_scores_csv": f"artifacts/baselines/{domain}_geo/real/thread_scores.csv",
    "min_comments": 5,
    "topic_facets": [f"discussion in r/{s}" for s, _ in subs.most_common(6) if s],
    "technical_terms": distinctive(
        bodies, background, 16,
        banned={s.lower() for s in subs if s} | {"subreddit", "mods", "mod", "automod", "sub"},
    ),
    "protected_entity_terms": proper(titles, 14),
    "persona_expertise_dimensions": [],
    "_generated_by": "experiments/geo_v137ds/enable_domain.sh",
    "_corpus": {"threads": len(by_thread), "eligible_threads": len(eligible),
                "comments": sum(len(v) for v in by_thread.values())},
}
out = root / "generalized_card/configs/domains" / f"{domain}_geo.json"
out.write_text(json.dumps(cfg, indent=2) + "\n")
print(f"config written -> {out.relative_to(root)}")
print(f"  corpus     : {len(by_thread)} threads, {len(eligible)} with >=5 comments")
print(f"  community  : {cfg['community_context']}")
print(f"  terms      : {', '.join(cfg['technical_terms'][:8])} ...")
print(f"  entities   : {', '.join(cfg['protected_entity_terms'][:6])} ...")
if len(eligible) < 150:
    print(f"  ⚠ only {len(eligible)} eligible threads -- a 150-seed pool is not possible; "
          f"use --pool-size {len(eligible)}")
PYX

if [[ "$do_score" == "1" ]]; then
  echo
  echo "scoring real threads (local models, no API; this is the slow part)"
  mkdir -p "$(dirname "$scores")"
  # NOT run_baseline_evaluation.py: it has no --real-only and its second phase
  # runs simulations, which spends API money. score_real_threads.py calls that
  # script's phase-1 function directly, so this is local compute only.
  HF_HUB_OFFLINE=1 "$PY" "$ROOT/experiments/geo_v137ds/score_real_threads.py" \
    "$domain" --device "$device" 2>&1 | tail -20
else
  echo
  echo "real thread scores NOT built. Generation works without them; evaluation does not."
  echo "  ./experiments/geo_v137ds/enable_domain.sh $domain --score --device mps"
fi

cat <<EOF

generate with:
  HF_HUB_OFFLINE=1 ./experiments/geo_v137ds/run_geo_domain.sh ${domain}_geo \\
    --planner gpt-5.4-mini --writer deepseek-v4-flash \\
    --pool-size 150 --sampling-seed 907 --shard-size 3 --max-parallel 50
EOF
