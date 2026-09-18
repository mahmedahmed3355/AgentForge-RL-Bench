"""Training environment orchestration for AgentForge-RL-Bench."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

from agentforge.agents.base import BaseAgent
from agentforge.core.agent_executor import AgentEpisodeExecutor
from agentforge.core.agent_executor import AgentEpisodeResult
from agentforge.environments import BaseEnvironment


EnvironmentFactory = Callable[[], BaseEnvironment]


@dataclass(frozen=True)
class TrainingTask:
    """A task available to the RL training loop."""

    task_id: str
    environment_factory: EnvironmentFactory
    metadata: dict[str, Any]


class TrainingEnvironmentRunner:
    """Run agent episodes against a controlled training task set."""

    def __init__(
        self,
        tasks: Sequence[TrainingTask],
    ) -> None:
        if not tasks:
            raise ValueError(
                "At least one training task is required."
            )

        task_ids = [task.task_id for task in tasks]

        if len(task_ids) != len(set(task_ids)):
            raise ValueError(
                "Training task IDs must be unique."
            )

        self._tasks = tuple(tasks)
        self._task_by_id = {
            task.task_id: task
            for task in self._tasks
        }

    @property
    def tasks(self) -> tuple[TrainingTask, ...]:
        """Return the immutable training task collection."""

        return self._tasks

    @property
    def task_ids(self) -> tuple[str, ...]:
        """Return training task IDs in deterministic order."""

        return tuple(
            task.task_id
            for task in self._tasks
        )

    def get_task(
        self,
        task_id: str,
    ) -> TrainingTask:
        """Return a registered training task."""

        try:
            return self._task_by_id[task_id]
        except KeyError as exc:
            raise KeyError(
                f"Unknown training task: {task_id}"
            ) from exc

    def create_environment(
        self,
        task_id: str,
    ) -> BaseEnvironment:
        """Create a fresh environment for one task."""

        return self.get_task(
            task_id
        ).environment_factory()

    def run_episode(
        self,
        *,
        agent: BaseAgent,
        task_id: str,
        episode_id: str,
        max_steps: int,
    ) -> AgentEpisodeResult:
        """Run one isolated training episode."""

        if max_steps < 1:
            raise ValueError(
                "max_steps must be >= 1."
            )

        task = self.get_task(task_id)
        environment = task.environment_factory()

        try:
            executor = AgentEpisodeExecutor(
                agent=agent,
                environment=environment,
            )

            return executor.run(
                episode_id=episode_id,
                task_id=task.task_id,
                max_steps=max_steps,
            )
        finally:
            environment.close()
