#!/usr/bin/env python3
"""Build a real reviser prompt for every configured domain and show it adapts.

The requirement is that the Planner, Writer and reviser work unchanged on
celebrity, news, movies, laptop -- not only on the domain they were built for.
For the reviser that means the prompt has to carry the domain WITHOUT naming a
product category anywhere in the code, so this renders one against every config
in `generalized_card/configs/domains/` and reports what changed.
"""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "selfloop"))
import reviser as R
import strategies as S

# Deliberately mentions an entity that only ONE domain's corpus knows, so the
# protected-terms channel is exercised rather than merely present.
COMMENT = ('Honestly the $800 number keeps getting repeated and nobody checks it. '
           'Olivia Rodrigo said the opposite on the Sony a7 IV thread yesterday.')
NEIGHBOURS = ["Sure, but the follow-up covered it.", "Where did that figure come from?"]
BANNED = ("card", "bank", "apr", "camera", "lens", "megapixel", "laptop",
          "headphone", "shutter", "warranty", "product")

rows = []
for config_path in sorted((REPO / "generalized_card/configs/domains").glob("*.json")):
    config = json.loads(config_path.read_text())
    target = R.Target(thread_id="t", index=0, comment_id="1", text=COMMENT,
                      parent_text="The report said the tradeoff was 800.",
                      neighbours=NEIGHBOURS)
    prompt = R.build_prompt(
        target, metric="semantic_mean_cosine", measured=0.28, thread_target=0.17,
        community=str(config.get("community_context") or "Reddit"),
        protected=list(config.get("protected_entity_terms") or []),
        candidates=5,
    )
    leaked = [w for w in BANNED if f" {w} " in f" {prompt.lower()} "
              and w not in str(config).lower()]
    rows.append((config["domain_id"], config.get("community_context", "")[:52],
                 len(config.get("protected_entity_terms") or []), len(prompt), leaked))

print(f"{'domain':<20}{'community (from its own corpus)':<54}{'实体名':>6}{'提示词字符':>10}{'领域词泄露':>10}")
print("-" * 104)
bad = 0
for domain, community, entities, size, leaked in rows:
    bad += len(leaked)
    print(f"{domain:<20}{community:<54}{entities:>6}{size:>10}{','.join(leaked) or '无':>10}")
print(f"\n{len(rows)} 个 domain 配置全部渲染成功，代码里写死的领域词泄露: {bad}")

print("\n--- 每个 domain 从自己语料里认出的实体（锚点通道）---")
for config_path in sorted((REPO / "generalized_card/configs/domains").glob("*.json")):
    config = json.loads(config_path.read_text())
    protected = list(config.get("protected_entity_terms") or [])
    hit = [t for t in protected if t.lower() in COMMENT.lower()]
    print(f"  {config['domain_id']:<20}命中本域实体表: {hit or '（无，靠通用抽取兜底）'}")
print(f"  不依赖任何领域表的锚点: {S.anchors_in(COMMENT, [])}")

print("\n--- celebrity 与 camera 的提示词差异（只应差在领域内容上）---")
import difflib

def render(name):
    config = json.loads((REPO / f"generalized_card/configs/domains/{name}.json").read_text())
    return R.build_prompt(
        R.Target(thread_id="t", index=0, comment_id="1", text=COMMENT,
                 parent_text="The report said the tradeoff was 800.", neighbours=NEIGHBOURS),
        metric="semantic_mean_cosine", measured=0.28, thread_target=0.17,
        community=str(config.get("community_context") or "Reddit"),
        protected=list(config.get("protected_entity_terms") or []), candidates=5,
    ).splitlines()

diff = [l for l in difflib.unified_diff(render("camera"), render("celebrity_geo"),
                                        "camera", "celebrity", lineterm="", n=0)
        if l[:1] in "+-" and l[:3] not in ("---", "+++")]
for line in diff:
    print("  " + line[:150])
sys.exit(1 if bad else 0)
