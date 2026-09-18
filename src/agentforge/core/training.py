"""Training loop primitives for AgentForge-RL-Bench."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agentforge.agents.base import BaseAgent
from agentforge.environments import BaseEnvironment

from .agent_executor import AgentEpisodeExecutor


@dataclass
class TrainingEpisodeMetrics:
    """Metrics collected from one training episode."""

    episode_id: str
    task_id: str
    steps: int
    total_reward: float
    success: bool


@dataclass
class TrainingRunResult:
    """Aggregated result of a training run."""

    run_id: str
    episodes_completed: int
    total_steps: int
    total_reward: float
    successful_episodes: int
    episode_metrics: list[TrainingEpisodeMetrics] = field(
        default_factory=list
    )

    @property
    def success_rate(self) -> float:
        """Return episode success rate."""

        if self.episodes_completed == 0:
            return 0.0

        return self.successful_episodes / self.episodes_completed


class TrainingLoop:
    """Run repeated agent-environment training episodes."""

    def __init__(
        self,
        agent: BaseAgent,
        environment_factory: Any,
    ) -> None:
        self.agent = agent
        self.environment_factory = environment_factory

    def run(
        self,
        run_id: str,
        task_id: str,
        episodes: int,
        max_steps: int,
    ) -> TrainingRunResult:
        """Execute a deterministic training run."""

        if episodes < 0:
            raise ValueError("episodes must be non-negative")

        if max_steps <= 0:
            raise ValueError("max_steps must be positive")

        metrics: list[TrainingEpisodeMetrics] = []

        total_steps = 0
        total_reward = 0.0
        successful_episodes = 0

        for episode_number in range(1, episodes + 1):
            environment = self.environment_factory()

            executor = AgentEpisodeExecutor(
                agent=self.agent,
                environment=environment,
            )

            result = executor.run(
                episode_id=f"{run_id}-episode-{episode_number:04d}",
                task_id=task_id,
                max_steps=max_steps,
            )

            episode_metrics = TrainingEpisodeMetrics(
                episode_id=result.episode_id,
                task_id=result.task_id,
                steps=result.steps,
                total_reward=result.total_reward,
                success=result.done,
            )

            metrics.append(episode_metrics)

            total_steps += result.steps
            total_reward += result.total_reward

            if result.done:
                successful_episodes += 1

            environment.close()

        return TrainingRunResult(
            run_id=run_id,
            episodes_completed=episodes,
            total_steps=total_steps,
            total_reward=total_reward,
            successful_episodes=successful_episodes,
            episode_metrics=metrics,
        )
