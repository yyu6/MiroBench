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
| **v143obs** | `obs_20260902_p*` `obsb_*` | celebrity | 50 | 跑中 | v139mp + `--adopt-observations`：采纳 profile 自己已经测出来、却被硬编码常数顶掉的 8 个行为率（short 0.18→0.48、calm 0.78→0.29 等） |
| **v144win2** | `win2_20260902_p*` | celebrity | 50 | 跑中 | v138ref + `--reference-window measured`：Planner 的参考例句按参考库自身长度分布取，档内仍按 BM25 排 |
| **v146raw2** | `raw2_20260902_p*` | celebrity | 50 | 跑中 | v138ref + `--reference-window unranked`：完全不按词汇相关度排，等价于从参考库随机抽 |
| **v145** | `iso3_20260902_p*` | celebrity | 50 | 跑中 | v139mp + 逐 thread 孤立配额，且配额不再自带词数范围 |

### 什么时候可以提前砍掉一支

**N≈13 时 `semantic_mean_cosine` 仍判 FAIL，就不必跑到 N=50。**

小样本意味着统计功效低，也就是*很难*拒绝原假设。在这种条件下仍被拒，
说明差距大到低功效都挡不住，加样本只会更显著，不可能翻盘。
v141iso 在 N=13 时 mwu p=0.024 / ks p=0.044，据此停在 23/50。

反向不成立：小 N 的 PASS 不能当数。同一张表里 `self_bertscore` 是
p=0.073 擦过去的 PASS，而 d=+0.42，放到 N=48 必然 FAIL。
**小 N 只能用来否定，不能用来肯定。**

### v141iso 读数（N=13，`iso2_*` + `iso2b_*`，16 个 shard）

11/12 PASS，但**这是 N 撑出来的**：N=13 时 Cliff d 的标准差是 0.23，几乎测不出显著。
唯一能比的是效应量：

| 指标 | v138ref (N=48) | v141iso (N=13) |
|---|---|---|
| `self_bertscore` | d +0.37 | d **+0.42** |
| `semantic_mean_cosine` | d +0.54 | d **+0.53** |

**配额执行了但没转化**：thread 内孤立比例 16.5% → 20.6%（真实 24.9%），
两个目标指标的效应量原地不动。参照检验其实已经预告了一半——
`self_bertscore` 与孤立度的相关只有 -0.274，本来就弱。

担心的副作用没有出现：`self_bleu_4` d -0.16，`length_cv`、`emotion_entropy` 都在。
所以这一支是无害无效，不是有害。

### Planner 被堵死在哪里

配额执行率实测 **69%**（`check_isolation_compliance.py`，要求 0.349 / 实到 0.239），
对比 outsider quota 当年的 1.9%，说明 Planner 是照做的。挡住它的是三个结构限制：

1. **12 个 perspective 是购物决策框架**（`UNIVERSAL_VIEWPOINTS`，写死的 tuple，
   celebrity 和 camera 共用）。"性价比""兼容性""故障排查"——每条评论必须落在
   其中一个轴上，跑题在结构上不存在。**这一条还没动**，因为它是 Planner 的核心
   语义空间，改动面比另外两条大得多。
2. **参考例句被 BM25 筛过**。profile 里有 1217 条真实评论，Planner 每批只看到
   与帖子词汇相关度最高的 36 条：

   | | 相关度 | 跑题(<0.10) | 词数中位 |
   |---|---|---|---|
   | 完整参考库 | 0.083 | 62.7% | 12 |
   | Planner 看到 | 0.190 | 27.8% | 34 |
   | v144win 之后 | 0.152 | 47.2% | 16 |

   BM25 按词汇重叠打分，短句 token 少排不上去——跑题的和短的是同一批评论
   （真实孤立评论 70% 不到 10 词）。v144win2 / v146raw2 修这一条。

   **两种挑法实际发出去的内容**（celebrity seed 7，帖子是 Mahershala Ali 谈 Marvel
   没拍 Blade）。两边都是 36 条散装评论，不是完整 thread；差别只在挑法：

   BM25 挑的（36 条来自 24 个 thread）——全在往帖子话题上贴，而且全是在论证：
   ```
   [ 3词] VERY disappointed!!!! SMH
   [12词] Marvel: Ok but where do I fit Thor, Cap and Iron Man?
   [41词] .....no CGI? You're kidding right? There isn't a shot in DD BA...
   [45词] Apparently S2 of Born Again dropped a lot of viewers...
   ```

   不排序挑的（36 条来自 32 个 thread）——就是真人随口说的话：
   ```
   [ 6词] He was fantastic in that movie.
   [13词] I'm extremely here for the renaissance of the 90s comic hotties (Lillard, Fraser).
   [ 8词] Gonna need a pic to prove that cheif
   [ 9词] Glad someone remains an icon in these trying times.
   ```

   提示词要 Planner 做的是「从例句里抽取真人做了什么对话动作，套到当前帖子上」。
   喂一堆论证进去，它就规划出一堆论证，于是整个 thread 围着同一个论点转。
   不排序还顺带把来源从 24 个 thread 提到 32 个——BM25 会把话题相近的帖子
   里的评论整批顶上来。
3. **`short_max_share` 卡 0.18，实测 0.48**。结构上禁止了那种散落短句。
   v143obs 修这一条。

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
- **有 run 在飞时不要改 core contract 文件**。三支跑到一半时改了
  `viewpoint_bank.py` 和 `run_generate.py`，之后启动的每个 shard 都撞
  `RuntimeError: CARD core contract mismatch`，每支只活下十几个。而且**死掉的
  shard 会留下日志、不留 run 目录**，按目录数看进度会高估。用 `watch_arms.sh`。
- **补跑必须换 tag 前缀**。复用原前缀会走 resume 校验，只要 run_generate 之后
  加过任何字段（哪怕默认值就是旧行为）都会被拒。`refill.sh` 接受新前缀，
  `chain.sh` 的 prefix 可以传 `(a|b)_日期` 这样的 ERE 交替式覆盖两者。
- **outsider quota 的历史**：G99 记录它是第一个同时改善两个目标指标的 arm，
  但 Planner 只按 1.9% 执行（目标 12%）；G102 调整批次大小后仍只有 0.66%，被否决；
  G104 测出即使完美执行，天花板也只有差距的 ~36%。v140out 是在换了 writer、
  修了参考语料之后重试。
