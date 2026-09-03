# 跑题槽位 (off-topic slots) — 已否决，未上线

> **2026-09-03 结论：前提被自己的对照实验推翻，代码已撤回。**
> 下面第 1、3 节仍然成立，第 2 节（记账）是错的。

## 结论摘要

真实 thread 里 24.5% 的评论和全场都不搭界；我们只有 10.4%。这解释了
`semantic_mean_cosine` 缺口的 63%。根本原因是**逐槽规划器的输入里没有任何
能表达"话题距离"的字段**。

---

## 1. 确认确实是这个问题

- [x] 数出来：真人 89/364 (24.5%)，v156 38/364 (10.4%)，差 51 条
- [x] 排除机器人内容：364 条真人评论里只有 1 条是 AutoModerator (0.3%)
- [x] 排除"只是短"：按词数分段做直接标准化 —— 用我们自己的长度构成套真人
      各段比例应有 81 条，实际 38 条。长度只解释 8/51 (15%)，43 条是话题层面
- [x] 特征签名：1-5 词档我们没问题 (49.1% vs 53.1%)；11-20 词档
      真人 22.7%、我们 3.9%。**我们只要写实质性评论，就一定在回应主帖**

## 2. 确认它解释得了指标差距 —— **失败，本节作废**

- [x] 正向：真人原样 0.1647 → 去掉不搭界的 0.2255；我们是 0.2356。
      真人去掉跑题评论之后基本就是我们
- [x] 反向：把缺的 k 条不相干评论移植进我们的 thread，0.2356 → 0.1909。
      填掉缺口的 63% (0.0447 / 0.0709)
- [x] 两个方向一致

## 3. 根本原因（代码级）

- [x] `comment_planner_prompt` (prompts.py:1149) 第一行就是 `del matched_real_thread`
- [x] 它调用 `_render_matched_slots` (prompts.py:3197)，只渲染
      `depth / parent / words / surface / development_plan`
- [x] `surface_only_label` (surface_contract.py:61) 只有 6 个取值，
      全部由词数和问号决定
- [x] 唯一响应 `--matched-text measured` 的是 `_render_matched_structure`
      (prompts.py:3157)，只被 `planner_prompt` (1006) 调用 —— 那是 thread 级
      规划器，不是逐槽规划器。**我们每一跑都带着这个开关，它从没进过
      决定"每条评论说什么"的提示词**
- [x] `--social-noise-min-share 0.18` 是 accepted-and-ignored：
      `rebalance_card_surfaces` (task_distribution.py:150) 明确 `del kwargs`，
      文档写"share arguments are accepted so the caller's contract is unchanged
      and ignored"，规划器是唯一所有者
- [x] 规划器看到的格子是混的：micro@depth2 跑题率 75%，
      ordinary_turn@depth0 是 4.3%。看到 (surface, depth) 之后仍然只能
      在 25% 的先验上瞎猜

**一句话**：语料里有这个事实，管道从不提取，规划器从没被告知。

---

## 设计

新 arm `--offtopic-slots measured`（默认 `off`，默认逐字节复现现有行为）。

1. 预处理阶段，对每个真实 thread 算出每条评论"对全场其他评论的平均相似度"，
   低于 0.10 的标为 detached。这个数**从真实 thread 数出来，不估计**
   （沿用 `--exact-matched-thread-size` 精确复现评论数、说话人数的同一套路）
2. 把 detached 标记按槽位传进 `_render_matched_slots`，渲染成
   `S17: depth=1; parent=S3; words=14; surface=short_turn; detached=yes`
3. 逐槽规划器的 schema 里加一句：detached 槽不回应主帖的论点，
   写一个自成一体的插话 —— 玩笑、跑题、冲另一个评论者的一句、话题外的联想
4. 写手侧：detached 槽不注入主帖论点上下文

## 验收标准（先写死，事后不改）

主判据（N=10，配对检验）：
- [ ] "不搭界"评论占比 10.4% → 达到 20% 以上（真人 24.5%）
- [ ] `semantic_mean_cosine` 从 0.2428 降到 0.19 以下（真人 0.1702）
- [ ] 上述两条都要配对 Wilcoxon p<0.05，并报升/降 thread 数

护栏（任何一条破了就算失败，不许事后放宽）：
- [ ] 其余 10 个指标不能从 PASS 掉到 FAIL
- [ ] `avg_depth` / `structural_virality` / 评论数 / 说话人数 保持精确匹配
- [ ] `length_cv` 不能被跑题槽拖坏
- [ ] `mean_story_probability` 不能升（跑题容易写成小故事）

失败判定：如果跑题占比上去了但 `semantic_mean_cosine` 没动，说明记账错了，
这条路和 v153/v156 一样封盘，如实报告，不追加尝试。

## 已封盘（不要再试）

- `--writer-temperature`：280 次调用探针，5 档无剂量反应，
  最好一档 p=0.096 只填 14%，且完全不动低尾
- `--plan-vocabulary open`：lens 涨 3.7 倍，计划语义散布只动 3%，文本不动


---

# 为什么否决（2026-09-03）

## 我漏掉的对照

我报告过「真人去掉不搭界的评论 0.2255 ≈ 我们 0.2356」，并读成"这些评论就是差距"。
但**去掉任何分布的底部都会抬高均值**。按同样条数、同样规则去掉我们自己最低的：

    真人抬升 +0.0554     我们抬升 +0.0665
    去掉前差距 +0.0628   去掉后差距 +0.0740   <- 反而变大

真人的跑题评论没有做我们的跑题评论没做的事。

## 分位表说的是相反的事

    p1  +0.013   p25 +0.036   p50 +0.055   p75 +0.070   p90 +0.086   p99 +0.039

缺口**最小**在 p1，**最大**在 p90。"低于 0.05 的对更少"（15.0% vs 24.0%）
只是整条分布右移的后果，不是独立缺陷。这在 mpnet 上复现了 G121 在 BERTScore
上的结论 —— 低尾读法两把尺子上都死了。

## 仓库早就做过这件事

写代码之前没查 DECISIONS，这是我的错。已有：
- G97 `--outsider-quota measured` 提出同一机制
- G99 / G102 两次付费跑：12% 的指标只落地 1.9%、0.66%
- G104 直接注入测天花板：即使 100% 合规也只有约 36%
- G121 撤回低尾前提
- G183 「下发的配额不等于被执行的配额」

G102 还把根本原因说得比我准：**per-slot 的结构性绑定压过全局段落；
`story_mode` 之所以有效，是因为 `apply_slot_distribution_schedule` 直接覆盖
规划器 151 次 —— "story is scheduled; outsider is merely requested."**

## 同一人重复自己 —— 也否决

G97 说我们的同一作者对比不同作者高 5 倍。v156 上不成立：
真人 +0.0368，我们 +0.0236，我们**比真人还低**。（G97 量的是 v122 的 BERTScore。）

## 唯一还活着的线索

我们有 **18.1%** 的评论对超过真人自己的 p90（真人按定义 10%）。读那些对，
全是**两个槽位在争同一个细分论点**：两条都在说"$800 那个取舍到底在不在原片段里"，
两条都在说"对比该不该只留在国防预算的例子里"。

这就是 G97 归给 "~2/3 是我们的切题评论互相太近" 的那部分，也是 G94 指认过、
`--plan-vocabulary open` 试过没打动的那个硬问题。下一个设计必须针对它，
并且必须先说明它凭什么不重蹈 G98/G102/v156 的覆辙。

## 已封盘清单（不要再试）

- `--writer-temperature`：280 次调用探针，5 档无剂量反应
- `--plan-vocabulary open`：lens 涨 3.7 倍，计划语义散布只动 3%
- `--offtopic-slots` / `--outsider-quota` 家族：本节
- 同一作者重复自己：v156 上我们优于真人

---

# v159 (--writer-retries 3) — 机制成功，指标未过

`semantic_overlap_high` 的检测器一直在工作但从不被执行：`--writer-retries`
默认 0，而重试循环是它唯一的出口（帮助文本原话："it therefore does nothing
unless --writer-retries is above 0"）。

## 机制层面：确实修好了

                        v157      v159
最后仍不合格却留下      50.2%     18.7%
重写救回来的             0.4%     25.8%
semantic_overlap_high      50  →     29
lexical_overlap_high        ?  →     13   <- 新出现

## 指标层面：全部不显著

配对 9 个 thread，没有一个指标 p<0.05：

    semantic_mean_cosine   6/9 更好  p=0.164   <- 最接近
    emotion_entropy        2/9       p=0.195   <- 最接近变差
    其余 8 个              p>=0.25

N=9 的 12 指标：11/12（semantic_mean_cosine FAIL，KS p=0.034）。

## 结论

**重写 4 次把内容推开了，但腔调叠加了 4 层。** 换了说什么，没换怎么说，
所以 `emotion_entropy` 反而更集中（-19.5%）。

不上线，但记录：**重试机制本身是通的且有力（救回 26%）**，问题在重写指令
只针对语义，完全没管说话方式。若要再用，应配合逐指标的重写指令，
而这正是 selfloop reviser 做的事。
