# GEO 版本记录

每个版本 = 一组 CLI arm + 一批 run tag。回溯时用 tag 前缀取回全部产物，
用 `git log` 取回当时的代码，用 `freeze.sh --verify` 确认代码没漂移。

| 版本 | tag 前缀 | domain | N | 结果 | 与上一版的唯一差异 |
|---|---|---|---|---|---|
| v137ds | `v137ds_*` `v137ext_*` `v137app_*` | camera | 150 | **8/12** | 基线：Planner gpt-5.4-mini + Writer deepseek-v4-flash |
| v137ds | `v137gpt_*` | camera | 150 | 6/12 | writer 换 gpt-5.4-mini |
| v137ds | `v1374o_*` `geo4ofill_*` | camera | 125 | 4/12 | writer 换 gpt-4o-mini |
| v137ds | `v137pro_*` | camera | 95 | 7/12 | writer 换 deepseek-v4-pro |
| v137ds | `geo137_celebrity_*` `celebfill_*` | celebrity | 113 | 4/12 | 换 domain，参考语料未过滤 |
| **v138ref** | `celebv2_20260902_*` | celebrity | 48 | **8/12** | `--reference-floor measured`：参考语料加上和种子池一样的 ≥5 条评论门槛 |
| **v139mp** | `mprof_20260902_s*` | celebrity | 50 | 10/12 (N=15) | 每个 seed 用它自己匹配的真实 thread 的 polite/impolite/story 作为目标。**泄露 arm**，不可与上面各版并列比较 |
| **v140out** | `outq_20260902_*` | celebrity | 50 | 待测 | v139mp + `--outsider-quota measured` |

## 复现任意一版

```bash
./experiments/geo_v137ds/freeze.sh --verify          # 确认代码未漂移
./experiments/geo_v137ds/eval_geo_domain.sh <domain> --tags <该版全部 tag>
```

## 已知的坑

- **参考语料污染**（v138ref 修复）：种子池要求 ≥5 条评论，参考语料没有同样门槛。
  celebrity 61% 的参考帖少于 5 条，学到的目标平均偏 51%。四个 domain 都中招
  （camera 29% / celebrity 61% / game 76% / news 61%）。
- **新 domain 的语料字段缺失**：多域抓取写 `id`/`fullname`，scorer 读
  `comment_id`/`comment_fullname`，且缺 `depth`。不修的话打分静默产出退化值
  （polite 100%、emotion 只有 1 类）。用 `fix_corpus_depth.py` 修。
- **outsider quota 的历史**：G99 记录它是第一个同时改善两个目标指标的 arm，
  但 Planner 只按 1.9% 执行（目标 12%）；G102 调整批次大小后仍只有 0.66%，被否决；
  G104 测出即使完美执行，天花板也只有差距的 ~36%。v140out 是在换了 writer、
  修了参考语料之后重试。
