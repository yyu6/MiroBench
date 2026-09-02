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
| **v139mp** | `mprof_20260902_s*` | celebrity | **46** | **7/12** | 同上，跑满 N=50 后复测。tone 组全过，`self_bertscore` d +0.39 / `semantic_mean_cosine` d +0.55 纹丝不动 |
| **v141iso** | `iso2_20260902_p*` | celebrity | 50 | 跑中 | v139mp + `--isolation-quota measured --fixed-isolation`，全域固定 12% |
| **v142isopt** | `isopt_20260902_p*` | celebrity | 50 | 跑中 | v141iso 去掉 `--fixed-isolation`：每条 thread 用它自己真实 thread 量出的孤立比例。已核对两支的 profile 除该字段外逐字节一致 |

### 为什么要有 isolation quota

真实语料内部自己的规律（80 条 celebrity thread，孤立=最近邻余弦 < 0.35）：

| 指标 | 与该 thread 孤立比例的 Spearman | p |
|---|---|---|
| `semantic_mean_cosine` | **-0.570** | 0.0000 |
| `self_bertscore` | -0.274 | 0.0140 |
| `self_bleu_4` | **+0.300** | 0.0068 |

所以话题散开确实压低前两个指标 —— 这是真实语料自己的性质，不是我们假设的。
但 `self_bleu_4` 是反号，它现在 d -0.01 完美通过，**是这个 arm 要盯的副作用**。

孤立比例在真实 thread 之间差异极大（中位 0.255，范围 0.04~1.00），
所以全域固定 12% 对一半 thread 是错的：有的 thread 真人几乎全在各说各的，
有的几乎完全聚焦。v141iso 和 v142isopt 就是在测这个差别值不值。

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
- **arm 之间必须只差一个 flag**：v141iso 第一次启动时漏了 `--reference-floor
  measured`，而那是把 celebrity 从 4/12 拉到 8/12 的修复，等于和 v139mp 差了两个
  flag，归因作废。启动后用 preflight 日志里的 `--tone-polite-min-share` 核对：
  过滤后是 0.219，未过滤是 0.406。
- **调用加了但 import 没加**，两者都是运行时解析，语法检查通不过不了这一关。
  第一次 isolation 跑的 50 个 shard 全部在 preflight 崩于
  `NameError: set_isolation_quota is not defined`，`gen.done` 照样写了 DONE。
  改完 `run_generate.py` 后 grep 一次 import 再启动。
- **outsider quota 的历史**：G99 记录它是第一个同时改善两个目标指标的 arm，
  但 Planner 只按 1.9% 执行（目标 12%）；G102 调整批次大小后仍只有 0.66%，被否决；
  G104 测出即使完美执行，天花板也只有差距的 ~36%。v140out 是在换了 writer、
  修了参考语料之后重试。
