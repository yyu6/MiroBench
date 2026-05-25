"""Patch the vanilla MiroFish Reddit runner with GEO-specific runtime logic.

This module exists so `third_party/MiroFish` can stay close to upstream
vanilla OASIS while GEO still preserves its previous patched behavior in a
separate, explicit layer.
"""
from __future__ import annotations

import os
import random
import sqlite3
from datetime import datetime
from typing import Any

from .oasis_reddit import (
    _overlay_has_explicit_reply_depth_guidance,
    apply_oasis_reddit_realism_patch,
    configure_reddit_runtime,
    create_reddit_platform,
    generate_reddit_agent_graph,
)

_COMMENT_JOB_TARGETS = (
    "blunt verdict: answer in one clear stance without writing a balanced review",
    "narrow caveat: add one exception, edge case, or condition the thread is missing",
    "personal datapoint: mention one concrete use, fee, redemption, denial, or habit",
    "skeptical challenge: push back on one visible claim or assumption",
    "pointed question: ask one specific follow-up that would change the recommendation",
    "correction: fix one factual/detail-level claim without summarizing the whole topic",
    "annoyance or vent: name one friction point in a human, slightly biased way",
    "tiny pile-on or joke: react like a Redditor, not like a product advisor",
)

_EVIDENCE_TARGETS = (
    "firsthand story, if plausible for the persona",
    "numbers or threshold: fee, spend, APR, credit limit, category cap, or redemption value",
    "household/workflow detail: who uses it, how often, what gets forgotten",
    "negative experience: denial, clawback, support issue, confusing rule, missed credit",
    "secondhand/forum memory: something you saw others complain about",
    "no evidence, just a blunt preference or vibe check",
    "local correction of the parent comment's exact wording",
    "tradeoff from a niche use case rather than generic rewards math",
)

_LENGTH_TARGETS = {
    "micro": "3-10 words. A blunt answer, tiny correction, joke, or short question. Do not explain it.",
    "short": "11-25 words. One narrow point only; no setup, no wrap-up.",
    "medium": "26-55 words. One concrete datapoint, caveat, or disagreement.",
    "long": "80-140 words. Use only for a real story, timeline, numbers, or messy context; not a balanced mini-review.",
}

_LENGTH_TARGET_SEQUENCE = (
    "micro",
    "short",
    "medium",
    "short",
    "micro",
    "medium",
    "long",
    "short",
    "micro",
    "medium",
    "short",
)


def _build_turn_diversity_plan(
    agent_id: int,
    sequence_index: int,
    total_comments: int,
) -> dict[str, str]:
    """Rotate per-turn comment targets so active agents do not converge."""

    offset = int(agent_id) + int(sequence_index) + int(total_comments)
    length_target = _LENGTH_TARGET_SEQUENCE[offset % len(_LENGTH_TARGET_SEQUENCE)]
    return {
        "length_target": length_target,
        "length_instruction": _LENGTH_TARGETS[length_target],
        "comment_job_target": _COMMENT_JOB_TARGETS[offset % len(_COMMENT_JOB_TARGETS)],
        "evidence_target": _EVIDENCE_TARGETS[
            ((int(agent_id) * 3) + int(sequence_index) + int(total_comments))
            % len(_EVIDENCE_TARGETS)
        ],
    }


def apply_geo_runner_patch(vanilla_module: Any) -> None:
    """Monkeypatch the vanilla MiroFish runner module in place."""

    runner_cls = vanilla_module.RedditSimulationRunner
    if getattr(runner_cls, "_geo_patch_applied", False):
        return

    apply_oasis_reddit_realism_patch()

    original_init = runner_cls.__init__

    def __init__(self, config_path: str, wait_for_commands: bool = True):
        original_init(self, config_path, wait_for_commands)
        self.agent_config_by_id = {
            int(cfg.get("agent_id", 0)): dict(cfg)
            for cfg in self.config.get("agent_configs", [])
            if cfg.get("agent_id") is not None
        }
        self._next_available_minute = {
            agent_id: 0 for agent_id in self.agent_config_by_id
        }
        self._round_action_delays = {}

    def _get_time_multiplier(self, current_hour: int) -> float:
        time_config = self.config.get("time_config", {})
        peak_hours = time_config.get("peak_hours", [9, 10, 11, 14, 15, 20, 21, 22])
        off_peak_hours = time_config.get("off_peak_hours", [0, 1, 2, 3, 4, 5])
        morning_hours = time_config.get("morning_hours", [7, 8, 9])
        work_hours = time_config.get("work_hours", list(range(10, 18)))

        if current_hour in peak_hours:
            return float(time_config.get("peak_activity_multiplier", 1.5))
        if current_hour in off_peak_hours:
            return float(time_config.get("off_peak_activity_multiplier", 0.3))
        if current_hour in morning_hours:
            return float(time_config.get("morning_activity_multiplier", 0.5))
        if current_hour in work_hours:
            return float(time_config.get("work_activity_multiplier", 0.7))
        return 1.0

    def _compute_activation_weight(self, cfg: dict[str, Any], time_multiplier: float) -> float:
        activity_level = float(cfg.get("activity_level", 0.5) or 0.5)
        posts_per_hour = float(cfg.get("posts_per_hour", 0.3) or 0.3)
        comments_per_hour = float(cfg.get("comments_per_hour", 0.8) or 0.8)
        influence_weight = float(cfg.get("influence_weight", 1.0) or 1.0)
        stance = str(cfg.get("stance", "neutral") or "neutral").lower()

        engagement_factor = 0.4 + (0.25 * posts_per_hour) + (0.35 * comments_per_hour)
        influence_factor = 0.9 + (0.15 * max(0.0, influence_weight))
        stance_factor = 0.75 if stance == "observer" else 1.0

        return max(
            0.01,
            activity_level * time_multiplier * engagement_factor * influence_factor * stance_factor,
        )

    def _weighted_sample_agent_ids(
        self,
        weighted_candidates: list[tuple[int, float]],
        target_count: int,
    ) -> list[int]:
        if target_count <= 0 or not weighted_candidates:
            return []

        pool = [(agent_id, max(weight, 0.0)) for agent_id, weight in weighted_candidates]
        selected: list[int] = []

        while pool and len(selected) < target_count:
            total_weight = sum(weight for _, weight in pool)
            if total_weight <= 0:
                break

            pick = random.uniform(0, total_weight)
            running = 0.0
            for idx, (agent_id, weight) in enumerate(pool):
                running += weight
                if running >= pick:
                    selected.append(agent_id)
                    pool.pop(idx)
                    break

        return selected

    def _draw_agent_delay(self, agent_id: int) -> int:
        cfg = self.agent_config_by_id.get(agent_id, {})
        min_delay = int(cfg.get("response_delay_min", 5) or 5)
        max_delay = int(cfg.get("response_delay_max", 60) or 60)
        if max_delay < min_delay:
            min_delay, max_delay = max_delay, min_delay
        return random.randint(min_delay, max_delay)

    def _get_runtime_float(self, name: str, default: float) -> float:
        runtime_config = dict(self.config.get("runtime_config", {}) or {})
        value = runtime_config.get(name, default)
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _get_runtime_int(self, name: str, default: int) -> int:
        runtime_config = dict(self.config.get("runtime_config", {}) or {})
        value = runtime_config.get(name, default)
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    def _schedule_agent_cooldown(self, agent_id: int, current_minute: int) -> int:
        delay = self._draw_agent_delay(agent_id)
        self._next_available_minute[agent_id] = current_minute + max(1, delay)
        return delay

    def _schedule_batch_cooldowns(self, agent_ids: list[int] | set[int], current_minute: int) -> None:
        for agent_id in set(agent_ids):
            self._schedule_agent_cooldown(int(agent_id), current_minute)

    def _build_round_batches(
        self,
        active_agents: list[tuple[int, Any]],
        minutes_per_round: int,
    ) -> list[list[tuple[int, Any]]]:
        if not active_agents:
            return []

        bucket_span = max(1, minutes_per_round // 3)
        buckets: dict[int, list[tuple[int, Any]]] = {}

        for agent_id, agent in active_agents:
            delay = int(self._round_action_delays.get(agent_id, 0))
            capped_delay = min(delay, max(0, minutes_per_round - 1))
            bucket_id = capped_delay // bucket_span
            buckets.setdefault(bucket_id, []).append((agent_id, agent))

        return [buckets[bucket_id] for bucket_id in sorted(buckets)]

    async def _run_batch_sequentially(
        self,
        batch: list[tuple[int, Any]],
        vanilla_module: Any,
    ) -> None:
        """Run one batch agent-by-agent so later agents see earlier comments.

        The old behavior sent one whole batch through a single ``env.step``.
        That made sibling top-level comments effectively simultaneous, so they
        all reacted to the same stale feed snapshot. For Reddit discussion
        realism we want the next agent to see what the previous one just wrote.
        """

        ordered_batch = sorted(
            batch,
            key=lambda item: (
                int(self._round_action_delays.get(item[0], 0)),
                int(item[0]),
            ),
        )
        while ordered_batch:
            # Recompute the discussion-plan hints against the live DB after each
            # action so later agents adapt to the newly visible thread state.
            self._assign_batch_discussion_plans(ordered_batch)
            agent_id, agent = ordered_batch.pop(0)
            await self.env.step({agent: vanilla_module.LLMAction()})

    def _get_sparse_thread_stats(self) -> dict[str, int]:
        db_path = self._get_db_path()
        if not os.path.exists(db_path):
            return {"total_posts": 0, "sparse_posts": 0, "total_comments": 0}

        try:
            conn = sqlite3.connect(db_path)
            try:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT
                        COUNT(*) AS total_posts,
                        SUM(CASE WHEN COALESCE(c.comment_count, 0) <= 1 THEN 1 ELSE 0 END) AS sparse_posts,
                        SUM(COALESCE(c.comment_count, 0)) AS total_comments
                    FROM post AS p
                    LEFT JOIN (
                        SELECT post_id, COUNT(*) AS comment_count
                        FROM comment
                        GROUP BY post_id
                    ) AS c
                    ON c.post_id = p.post_id
                    """
                )
                row = cursor.fetchone()
                if not row:
                    return {"total_posts": 0, "sparse_posts": 0, "total_comments": 0}
                total_posts, sparse_posts, total_comments = row
                return {
                    "total_posts": int(total_posts or 0),
                    "sparse_posts": int(sparse_posts or 0),
                    "total_comments": int(total_comments or 0),
                }
            finally:
                conn.close()
        except sqlite3.Error:
            return {"total_posts": 0, "sparse_posts": 0, "total_comments": 0}

    def _get_comment_structure_stats(self) -> dict[str, int]:
        db_path = self._get_db_path()
        if not os.path.exists(db_path):
            return {
                "top_level_comments": 0,
                "nested_comments": 0,
                "deep_comments": 0,
                "max_depth": 0,
                "flat_threads": 0,
                "buried_threads": 0,
            }

        try:
            conn = sqlite3.connect(db_path)
            try:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    WITH thread_structure AS (
                        SELECT
                            p.post_id,
                            SUM(CASE WHEN c.comment_id IS NOT NULL
                                AND (c.parent_comment_id IS NULL OR c.parent_comment_id = 0)
                                THEN 1 ELSE 0 END) AS top_level_comments,
                            SUM(CASE WHEN c.comment_id IS NOT NULL
                                AND c.parent_comment_id IS NOT NULL
                                AND c.parent_comment_id != 0
                                THEN 1 ELSE 0 END) AS nested_comments,
                            MAX(COALESCE(c.depth, 0)) AS max_depth
                        FROM post AS p
                        LEFT JOIN comment AS c
                        ON c.post_id = p.post_id
                        GROUP BY p.post_id
                    )
                    SELECT
                        SUM(COALESCE(top_level_comments, 0)) AS top_level_comments,
                        SUM(COALESCE(nested_comments, 0)) AS nested_comments,
                        SUM(CASE WHEN COALESCE(max_depth, 0) >= 2 THEN 1 ELSE 0 END) AS buried_threads,
                        SUM(CASE WHEN COALESCE(top_level_comments, 0) >= 2
                            AND COALESCE(nested_comments, 0) = 0
                            THEN 1 ELSE 0 END) AS flat_threads,
                        MAX(COALESCE(max_depth, 0)) AS max_depth
                    FROM thread_structure
                    """
                )
                row = cursor.fetchone()
                if not row:
                    return {
                        "top_level_comments": 0,
                        "nested_comments": 0,
                        "deep_comments": 0,
                        "max_depth": 0,
                        "flat_threads": 0,
                        "buried_threads": 0,
                    }
                top_level_comments, nested_comments, buried_threads, flat_threads, max_depth = row
                return {
                    "top_level_comments": int(top_level_comments or 0),
                    "nested_comments": int(nested_comments or 0),
                    "deep_comments": int(nested_comments or 0),
                    "max_depth": int(max_depth or 0),
                    "flat_threads": int(flat_threads or 0),
                    "buried_threads": int(buried_threads or 0),
                }
            finally:
                conn.close()
        except sqlite3.Error:
            return {
                "top_level_comments": 0,
                "nested_comments": 0,
                "deep_comments": 0,
                "max_depth": 0,
                "flat_threads": 0,
                "buried_threads": 0,
            }

    def _build_structure_guidance(
        self,
        sparse_stats: dict[str, int],
        structure_stats: dict[str, int],
    ) -> dict[str, Any]:
        overlay = dict(self.config.get("overlay") or {})
        if _overlay_has_explicit_reply_depth_guidance(overlay):
            return {
                "structure_preference": "balanced",
                "structure_reason": (
                    "Overlay prompt explicitly controls top-level vs reply vs subsub depth; planner structure hints are weak context only."
                ),
            }

        total_posts = int(sparse_stats.get("total_posts", 0))
        top_level_comments = int(structure_stats.get("top_level_comments", 0))
        nested_comments = int(structure_stats.get("nested_comments", 0))
        flat_threads = int(structure_stats.get("flat_threads", 0))
        buried_threads = int(structure_stats.get("buried_threads", 0))
        max_depth = int(structure_stats.get("max_depth", 0))

        if total_posts <= 0:
            return {"structure_preference": "balanced"}

        if top_level_comments <= total_posts or nested_comments > max(1, top_level_comments):
            return {
                "structure_preference": "top_level",
                "structure_reason": "Too much of the activity is concentrated in reply chains instead of independent answers.",
            }
        if flat_threads >= 1 and nested_comments < max(1, flat_threads):
            return {
                "structure_preference": "reply",
                "structure_reason": (
                    "Several threads have standalone answers but not enough back-and-forth yet."
                ),
            }
        if buried_threads >= max(1, total_posts // 2):
            return {
                "structure_preference": "top_level",
                "structure_reason": "Many threads already have reply chains. Branch out unless a visible parent comment gives a concrete reason to continue that sub-argument.",
            }
        return {
            "structure_preference": "balanced",
            "structure_reason": "Keep a mix of standalone answers and replies.",
        }

    def _score_discussion_candidate(self, agent_id: int) -> float:
        cfg = self.agent_config_by_id.get(agent_id, {})
        comments_per_hour = float(cfg.get("comments_per_hour", 0.8) or 0.8)
        activity_level = float(cfg.get("activity_level", 0.5) or 0.5)
        influence_weight = float(cfg.get("influence_weight", 1.0) or 1.0)
        stance = str(cfg.get("stance", "neutral") or "neutral").lower()

        score = (1.25 * comments_per_hour) + (0.5 * activity_level) + (0.15 * influence_weight)
        if stance == "observer":
            score -= 0.8
        elif stance == "opposing":
            score += 0.2
        elif stance == "supportive":
            score += 0.15
        return score

    def _get_batch_discussion_budget(
        self,
        batch: list[tuple[int, Any]],
        stats: dict[str, int],
    ) -> int:
        if len(batch) < 2:
            return 0

        total_posts = int(stats.get("total_posts", 0))
        sparse_posts = int(stats.get("sparse_posts", 0))
        total_comments = int(stats.get("total_comments", 0))
        if total_posts == 0 or sparse_posts == 0:
            return 0

        avg_comments_per_post = total_comments / max(1, total_posts)
        avg_comments_per_hour = sum(
            float(
                self.agent_config_by_id.get(agent_id, {}).get("comments_per_hour", 0.8) or 0.8
            )
            for agent_id, _ in batch
        ) / len(batch)

        budget = 1
        if (
            sparse_posts >= 3
            and avg_comments_per_post < 1.2
            and len(batch) >= 6
            and avg_comments_per_hour >= 0.9
        ):
            budget = 2

        budget_multiplier = max(
            0.5,
            min(3.0, self._get_runtime_float("required_comment_budget_multiplier", 1.0)),
        )
        budget = max(1, int(round(budget * budget_multiplier)))
        return min(len(batch), budget)

    def _assign_batch_discussion_plans(
        self,
        batch: list[tuple[int, Any]],
    ) -> None:
        for _, agent in batch:
            if hasattr(agent, "_geo_discussion_plan"):
                delattr(agent, "_geo_discussion_plan")

        stats = self._get_sparse_thread_stats()
        structure_stats = self._get_comment_structure_stats()
        structure_guidance = self._build_structure_guidance(stats, structure_stats)
        budget = self._get_batch_discussion_budget(batch, stats)

        ranked_candidates = sorted(
            batch,
            key=lambda item: self._score_discussion_candidate(item[0]),
            reverse=True,
        )
        required_agent_ids = {
            int(agent_id)
            for agent_id, _ in ranked_candidates[: max(0, budget)]
        }

        reason = (
            f"{stats['sparse_posts']} of {stats['total_posts']} threads still have at most one comment"
        )
        structure_reason = str(structure_guidance.get("structure_reason", "") or "").strip()
        if structure_reason:
            reason = f"{reason}. {structure_reason}"
        for sequence_index, (agent_id, agent) in enumerate(batch):
            turn_plan = _build_turn_diversity_plan(
                int(agent_id),
                sequence_index,
                int(stats.get("total_comments", 0)),
            )
            turn_plan.update(
                {
                    "required": int(agent_id) in required_agent_ids,
                    "structure_preference": structure_guidance.get("structure_preference", "balanced"),
                }
            )
            if int(agent_id) in required_agent_ids:
                turn_plan["reason"] = reason
            elif structure_reason:
                turn_plan["reason"] = structure_reason
            setattr(
                agent,
                "_geo_discussion_plan",
                turn_plan,
            )

    def _get_active_agents_for_round(self, env, current_hour: int, round_num: int) -> list:
        time_config = self.config.get("time_config", {})
        agent_configs = self.config.get("agent_configs", [])
        minutes_per_round = int(time_config.get("minutes_per_round", 60) or 60)
        current_minute = round_num * minutes_per_round

        base_min = time_config.get("agents_per_hour_min", 5)
        base_max = time_config.get("agents_per_hour_max", 20)

        multiplier = self._get_time_multiplier(current_hour)
        target_count = max(0, int(round(random.uniform(base_min, base_max) * multiplier)))

        self._round_action_delays = {}
        weighted_candidates: list[tuple[int, float]] = []
        for cfg in agent_configs:
            agent_id = int(cfg.get("agent_id", 0))
            active_hours = cfg.get("active_hours", list(range(8, 23)))

            if current_hour not in active_hours:
                continue

            if current_minute < self._next_available_minute.get(agent_id, 0):
                continue

            weighted_candidates.append(
                (agent_id, self._compute_activation_weight(cfg, multiplier))
            )

        selected_ids = self._weighted_sample_agent_ids(weighted_candidates, target_count)

        active_agents = []
        for agent_id in selected_ids:
            try:
                agent = env.agent_graph.get_agent(agent_id)
                active_agents.append((agent_id, agent))
                self._round_action_delays[agent_id] = self._schedule_agent_cooldown(
                    agent_id,
                    current_minute,
                )
            except Exception:
                pass

        return active_agents

    async def run(self, max_rounds: int = None):
        print("=" * 60)
        print("OASIS Reddit模拟")
        print(f"配置文件: {self.config_path}")
        print(f"模拟ID: {self.config.get('simulation_id', 'unknown')}")
        print(f"等待命令模式: {'启用' if self.wait_for_commands else '禁用'}")
        print("=" * 60)

        time_config = self.config.get("time_config", {})
        total_hours = time_config.get("total_simulation_hours", 72)
        minutes_per_round = time_config.get("minutes_per_round", 30)
        total_rounds = (total_hours * 60) // minutes_per_round

        if max_rounds is not None and max_rounds > 0:
            original_rounds = total_rounds
            total_rounds = min(total_rounds, max_rounds)
            if total_rounds < original_rounds:
                print(f"\n轮数已截断: {original_rounds} -> {total_rounds} (max_rounds={max_rounds})")

        print(f"\n模拟参数:")
        print(f"  - 总模拟时长: {total_hours}小时")
        print(f"  - 每轮时间: {minutes_per_round}分钟")
        print(f"  - 总轮数: {total_rounds}")
        if max_rounds:
            print(f"  - 最大轮数限制: {max_rounds}")
        print(f"  - Agent数量: {len(self.config.get('agent_configs', []))}")

        print("\n初始化LLM模型...")
        model = self._create_model()

        db_path = self._get_db_path()
        os.environ["OASIS_DB_PATH"] = db_path
        configure_reddit_runtime(self.config, db_path=db_path)

        print("加载Agent Profile...")
        profile_path = self._get_profile_path()
        if not os.path.exists(profile_path):
            print(f"错误: Profile文件不存在: {profile_path}")
            return

        self.agent_graph = await generate_reddit_agent_graph(
            profile_path=profile_path,
            model=model,
            available_actions=self.AVAILABLE_ACTIONS,
        )

        if os.path.exists(db_path):
            os.remove(db_path)
            print(f"已删除旧数据库: {db_path}")

        print("创建OASIS环境...")
        reddit_platform = create_reddit_platform(db_path)
        self.env = vanilla_module.oasis.make(
            agent_graph=self.agent_graph,
            platform=reddit_platform,
            semaphore=30,
        )

        await self.env.reset()
        print("环境初始化完成\n")

        self.ipc_handler = vanilla_module.IPCHandler(self.simulation_dir, self.env, self.agent_graph)
        self.ipc_handler.update_status("running")

        event_config = self.config.get("event_config", {})
        initial_posts = event_config.get("initial_posts", [])

        if initial_posts:
            print(f"执行初始事件 ({len(initial_posts)}条初始帖子)...")
            initial_actions = {}
            for post in initial_posts:
                agent_id = post.get("poster_agent_id", 0)
                content = post.get("content", "")
                try:
                    agent = self.env.agent_graph.get_agent(agent_id)
                    if agent in initial_actions:
                        if not isinstance(initial_actions[agent], list):
                            initial_actions[agent] = [initial_actions[agent]]
                        initial_actions[agent].append(
                            vanilla_module.ManualAction(
                                action_type=vanilla_module.ActionType.CREATE_POST,
                                action_args={"content": content},
                            )
                        )
                    else:
                        initial_actions[agent] = vanilla_module.ManualAction(
                            action_type=vanilla_module.ActionType.CREATE_POST,
                            action_args={"content": content},
                        )
                except Exception as exc:
                    print(f"  警告: 无法为Agent {agent_id}创建初始帖子: {exc}")

            if initial_actions:
                await self.env.step(initial_actions)
                self._schedule_batch_cooldowns(
                    [post.get("poster_agent_id", 0) for post in initial_posts],
                    current_minute=0,
                )
                print(f"  已发布 {len(initial_actions)} 条初始帖子")

        print("\n开始模拟循环...")
        start_time = datetime.now()

        for round_num in range(total_rounds):
            simulated_minutes = round_num * minutes_per_round
            simulated_hour = (simulated_minutes // 60) % 24
            simulated_day = simulated_minutes // (60 * 24) + 1

            active_agents = self._get_active_agents_for_round(
                self.env, simulated_hour, round_num
            )

            if not active_agents:
                continue

            round_batches = self._build_round_batches(
                active_agents,
                minutes_per_round=minutes_per_round,
            )
            for batch in round_batches:
                await self._run_batch_sequentially(batch, vanilla_module)

            if (round_num + 1) % 10 == 0 or round_num == 0:
                elapsed = (datetime.now() - start_time).total_seconds()
                progress = (round_num + 1) / total_rounds * 100
                print(
                    f"  [Day {simulated_day}, {simulated_hour:02d}:00] "
                    f"Round {round_num + 1}/{total_rounds} ({progress:.1f}%) "
                    f"- {len(active_agents)} agents active "
                    f"- elapsed: {elapsed:.1f}s"
                )

        total_elapsed = (datetime.now() - start_time).total_seconds()
        print(f"\n模拟循环完成!")
        print(f"  - 总耗时: {total_elapsed:.1f}秒")
        print(f"  - 数据库: {db_path}")

        if self.wait_for_commands:
            print("\n" + "=" * 60)
            print("进入等待命令模式 - 环境保持运行")
            print("支持的命令: interview, batch_interview, close_env")
            print("=" * 60)

            self.ipc_handler.update_status("alive")

            try:
                while not vanilla_module._shutdown_event.is_set():
                    should_continue = await self.ipc_handler.process_commands()
                    if not should_continue:
                        break
                    try:
                        await vanilla_module.asyncio.wait_for(
                            vanilla_module._shutdown_event.wait(),
                            timeout=0.5,
                        )
                        break
                    except vanilla_module.asyncio.TimeoutError:
                        pass
            except KeyboardInterrupt:
                print("\n收到中断信号")
            except vanilla_module.asyncio.CancelledError:
                print("\n任务被取消")
            except Exception as exc:
                print(f"\n命令处理出错: {exc}")

            print("\n关闭环境...")

        self.ipc_handler.update_status("stopped")
        await self.env.close()

        print("环境已关闭")
        print("=" * 60)

    runner_cls.__init__ = __init__
    runner_cls._get_time_multiplier = _get_time_multiplier
    runner_cls._compute_activation_weight = _compute_activation_weight
    runner_cls._weighted_sample_agent_ids = _weighted_sample_agent_ids
    runner_cls._draw_agent_delay = _draw_agent_delay
    runner_cls._get_runtime_float = _get_runtime_float
    runner_cls._get_runtime_int = _get_runtime_int
    runner_cls._schedule_agent_cooldown = _schedule_agent_cooldown
    runner_cls._schedule_batch_cooldowns = _schedule_batch_cooldowns
    runner_cls._build_round_batches = _build_round_batches
    runner_cls._run_batch_sequentially = _run_batch_sequentially
    runner_cls._get_sparse_thread_stats = _get_sparse_thread_stats
    runner_cls._get_comment_structure_stats = _get_comment_structure_stats
    runner_cls._build_structure_guidance = _build_structure_guidance
    runner_cls._score_discussion_candidate = _score_discussion_candidate
    runner_cls._get_batch_discussion_budget = _get_batch_discussion_budget
    runner_cls._assign_batch_discussion_plans = _assign_batch_discussion_plans
    runner_cls._get_active_agents_for_round = _get_active_agents_for_round
    runner_cls.run = run
    runner_cls._geo_patch_applied = True
