"""In-process minimal OASIS Reddit simulation runner.

This module replaces the previous out-of-process MiroFish runner with a
focused, dependency-light wrapper around the public ``camel-oasis`` API.
Behavior matches the upstream vanilla baseline: load a simulation config,
build an LLM, generate the Reddit agent graph, run the env loop with the
configured time-of-day activation policy, and persist a SQLite DB that
the exporter consumes downstream.

The class deliberately preserves the attribute and method names that
``oasis_runner_patch.apply_geo_runner_patch`` reaches into, so the GEO
patched backbone can monkeypatch this class identically to how it used
to monkeypatch MiroFish's runner.

The class intentionally omits MiroFish's IPC / interview command loop,
Flask integration, Zep memory, and custom logging — none of which are
needed for non-interactive thread generation.
"""
from __future__ import annotations

import asyncio
import json
import os
import random
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

try:
    from camel.models import ModelFactory
    from camel.types import ModelPlatformType
    import oasis
    from oasis import (
        ActionType,
        LLMAction,
        ManualAction,
        generate_reddit_agent_graph,
    )
except ImportError as exc:  # pragma: no cover - environment guard
    raise SystemExit(
        "Reddit simulation requires the `camel-oasis` and `camel-ai` pip "
        "packages. Install with: pip install camel-oasis camel-ai"
    ) from exc


class RedditSimulationRunner:
    """Minimal vanilla OASIS Reddit runner.

    Compatible with ``mirobench.generation.oasis_runner_patch`` — the
    patch module monkeypatches this class's ``__init__``, ``run``, and
    ``_get_active_agents_for_round`` and adds ~15 more methods to it.

    Public API mirrors the upstream behavior used by GEO's pipeline.
    """

    AVAILABLE_ACTIONS = [
        ActionType.LIKE_POST,
        ActionType.DISLIKE_POST,
        ActionType.CREATE_POST,
        ActionType.CREATE_COMMENT,
        ActionType.LIKE_COMMENT,
        ActionType.DISLIKE_COMMENT,
        ActionType.SEARCH_POSTS,
        ActionType.SEARCH_USER,
        ActionType.TREND,
        ActionType.REFRESH,
        ActionType.DO_NOTHING,
        ActionType.FOLLOW,
        ActionType.MUTE,
    ]

    def __init__(self, config_path: str, wait_for_commands: bool = False) -> None:
        """Open the config and stash paths.

        ``wait_for_commands`` is accepted for compatibility with the patch
        layer but is always treated as ``False`` (non-interactive).
        """

        self.config_path = config_path
        self.simulation_dir = os.path.dirname(os.path.abspath(config_path))
        self.config: Dict[str, Any] = self._load_config()
        self.wait_for_commands = False  # always non-interactive
        self.env = None
        self.agent_graph = None
        self.ipc_handler = None

    def _load_config(self) -> Dict[str, Any]:
        """Load ``simulation_config.json``."""

        with open(self.config_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _get_profile_path(self) -> str:
        """Profile JSON sits next to the simulation config."""

        return os.path.join(self.simulation_dir, "reddit_profiles.json")

    def _get_db_path(self) -> str:
        """SQLite DB sits next to the simulation config."""

        return os.path.join(self.simulation_dir, "reddit_simulation.db")

    def _create_model(self):
        """Build a CAMEL OpenAI model from environment + config."""

        llm_api_key = os.environ.get("LLM_API_KEY", "")
        llm_base_url = os.environ.get("LLM_BASE_URL", "")
        llm_model = os.environ.get("LLM_MODEL_NAME", "") or self.config.get(
            "llm_model", "gpt-4o-mini"
        )

        if llm_api_key:
            os.environ["OPENAI_API_KEY"] = llm_api_key
        if not os.environ.get("OPENAI_API_KEY"):
            raise ValueError(
                "Missing API key. Set LLM_API_KEY (or OPENAI_API_KEY) in your "
                "environment or a .env file at the repository root."
            )
        if llm_base_url:
            os.environ["OPENAI_API_BASE_URL"] = llm_base_url

        return ModelFactory.create(
            model_platform=ModelPlatformType.OPENAI,
            model_type=llm_model,
        )

    def _get_active_agents_for_round(
        self,
        env,
        current_hour: int,
        round_num: int,
    ) -> List[Tuple[int, Any]]:
        """Pick which agents act this round based on time-of-day config.

        The GEO patch overrides this with a more sophisticated weighted
        sampler; this baseline implementation matches the upstream
        vanilla activation policy.
        """

        time_config = self.config.get("time_config", {})
        agent_configs = self.config.get("agent_configs", [])

        base_min = time_config.get("agents_per_hour_min", 5)
        base_max = time_config.get("agents_per_hour_max", 20)
        peak_hours = time_config.get(
            "peak_hours", [9, 10, 11, 14, 15, 20, 21, 22]
        )
        off_peak_hours = time_config.get("off_peak_hours", [0, 1, 2, 3, 4, 5])

        if current_hour in peak_hours:
            multiplier = time_config.get("peak_activity_multiplier", 1.5)
        elif current_hour in off_peak_hours:
            multiplier = time_config.get("off_peak_activity_multiplier", 0.3)
        else:
            multiplier = 1.0

        target_count = int(random.uniform(base_min, base_max) * multiplier)

        candidates: List[int] = []
        for cfg in agent_configs:
            agent_id = cfg.get("agent_id", 0)
            active_hours = cfg.get("active_hours", list(range(8, 23)))
            activity_level = cfg.get("activity_level", 0.5)
            if current_hour not in active_hours:
                continue
            if random.random() < activity_level:
                candidates.append(agent_id)

        selected_ids = (
            random.sample(candidates, min(target_count, len(candidates)))
            if candidates
            else []
        )

        active_agents: List[Tuple[int, Any]] = []
        for agent_id in selected_ids:
            try:
                agent = env.agent_graph.get_agent(agent_id)
                active_agents.append((agent_id, agent))
            except Exception:  # noqa: BLE001
                pass
        return active_agents

    async def run(self, max_rounds: Optional[int] = None) -> None:
        """Run the Reddit simulation loop."""

        time_config = self.config.get("time_config", {})
        total_hours = time_config.get("total_simulation_hours", 72)
        minutes_per_round = time_config.get("minutes_per_round", 30)
        total_rounds = (total_hours * 60) // minutes_per_round
        if max_rounds is not None and max_rounds > 0:
            total_rounds = min(total_rounds, max_rounds)

        print(f"OASIS Reddit simulation — {total_rounds} rounds, "
              f"{len(self.config.get('agent_configs', []))} agents")

        model = self._create_model()

        profile_path = self._get_profile_path()
        if not os.path.exists(profile_path):
            raise FileNotFoundError(
                f"Reddit profile JSON not found at {profile_path}. "
                "Generate it via mirobench.generation.config_builder."
            )

        self.agent_graph = await generate_reddit_agent_graph(
            profile_path=profile_path,
            model=model,
            available_actions=self.AVAILABLE_ACTIONS,
        )

        db_path = self._get_db_path()
        if os.path.exists(db_path):
            os.remove(db_path)

        self.env = oasis.make(
            agent_graph=self.agent_graph,
            platform=oasis.DefaultPlatformType.REDDIT,
            database_path=db_path,
            semaphore=30,
        )
        await self.env.reset()

        # Seed posts
        initial_posts = self.config.get("event_config", {}).get("initial_posts", [])
        if initial_posts:
            initial_actions: Dict[Any, Any] = {}
            for post in initial_posts:
                agent_id = post.get("poster_agent_id", 0)
                content = post.get("content", "")
                try:
                    agent = self.env.agent_graph.get_agent(agent_id)
                except Exception as e:  # noqa: BLE001
                    print(f"  WARN: could not seed post for agent {agent_id}: {e}")
                    continue
                action = ManualAction(
                    action_type=ActionType.CREATE_POST,
                    action_args={"content": content},
                )
                if agent in initial_actions:
                    if not isinstance(initial_actions[agent], list):
                        initial_actions[agent] = [initial_actions[agent]]
                    initial_actions[agent].append(action)
                else:
                    initial_actions[agent] = action
            if initial_actions:
                await self.env.step(initial_actions)

        # Main simulation loop
        start_time = datetime.now()
        for round_num in range(total_rounds):
            simulated_minutes = round_num * minutes_per_round
            simulated_hour = (simulated_minutes // 60) % 24

            active_agents = self._get_active_agents_for_round(
                self.env, simulated_hour, round_num
            )
            if not active_agents:
                continue

            actions = {agent: LLMAction() for _, agent in active_agents}
            await self.env.step(actions)

            if (round_num + 1) % 10 == 0 or round_num == 0:
                elapsed = (datetime.now() - start_time).total_seconds()
                progress = (round_num + 1) / total_rounds * 100
                print(
                    f"  round {round_num + 1}/{total_rounds} "
                    f"({progress:.1f}%) — {len(active_agents)} agents "
                    f"— elapsed {elapsed:.1f}s"
                )

        total_elapsed = (datetime.now() - start_time).total_seconds()
        print(f"Simulation done in {total_elapsed:.1f}s. DB: {db_path}")

        await self.env.close()


async def _amain(config_path: str, max_rounds: Optional[int]) -> None:
    runner = RedditSimulationRunner(config_path=config_path, wait_for_commands=False)
    await runner.run(max_rounds=max_rounds)


def run_simulation_in_process(
    config_path: str,
    max_rounds: Optional[int] = None,
    apply_geo_patch: bool = False,
) -> None:
    """Synchronous entry point that runs one OASIS simulation in-process.

    When ``apply_geo_patch`` is True, GEO's runtime patches are applied to
    the runner class and OASIS Reddit modules before the simulation runs.
    """

    if apply_geo_patch:
        # Import lazily so users who only want vanilla never trigger the
        # patch module's heavy OASIS-internal imports.
        from . import oasis_runner as _self_module
        from .oasis_runner_patch import apply_geo_runner_patch
        apply_geo_runner_patch(_self_module)

    asyncio.run(_amain(config_path, max_rounds))
