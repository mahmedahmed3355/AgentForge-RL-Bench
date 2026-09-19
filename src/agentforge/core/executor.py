"""End-to-end episode execution for AgentForge-RL-Bench."""

from __future__ import annotations

from typing import Any, Callable

from agentforge.environments import BaseEnvironment

from .episode import EpisodeRunner
from .contracts import EpisodeResult


class EpisodeExecutor:
    """Connect an agent, environment, episode runner, and trajectory recorder."""

    def __init__(
        self,
        environment: BaseEnvironment,
        agent: Callable[[Any], Any],
        max_steps: int = 100,
    ) -> None:
        self.environment = environment
        self.agent = agent
        self.max_steps = max_steps

    def run(
        self,
        episode_id: str,
        task_id: str,
    ) -> EpisodeResult:
        """Run one complete agent-environment episode."""

        runner = EpisodeRunner(
            episode_id=episode_id,
            task_id=task_id,
            max_steps=self.max_steps,
        )

        reset_result = self.environment.reset()
        if isinstance(reset_result, tuple) and len(reset_result) == 2:
            observation, _ = reset_result
        else:
            observation = reset_result

        while not runner.done:
            action = self.agent(observation)

            result = self.environment.step(action)

            runner.record_step(
                observation=observation,
                action=action,
                reward=result.reward,
                next_observation=result.observation,
                status="success" if result.done and action == "solve" else (
                    "timeout" if result.done else "running"
                ),
                metadata=result.info,
            )

            observation = result.observation

            if result.done and not runner.done:
                runner.finish(
                    success=False,
                    status="failure",
                )

        return runner.result()
