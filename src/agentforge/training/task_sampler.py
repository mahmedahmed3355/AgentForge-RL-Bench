"""Task sampling and curriculum utilities for AgentForge-RL-Bench."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .environment_runner import TrainingTask


@dataclass(frozen=True)
class TaskSample:
    """One deterministic task-selection decision."""

    task_id: str
    episode_index: int
    selection_index: int


class TaskSampler:
    """Select training tasks while preserving reproducibility."""

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

    @property
    def tasks(self) -> tuple[TrainingTask, ...]:
        """Return the registered tasks."""

        return self._tasks

    def sample(
        self,
        episode_index: int,
    ) -> TaskSample:
        """Select a task deterministically using round-robin sampling."""

        if episode_index < 1:
            raise ValueError(
                "episode_index must be >= 1."
            )

        selection_index = (
            (episode_index - 1)
            % len(self._tasks)
        )

        return TaskSample(
            task_id=self._tasks[
                selection_index
            ].task_id,
            episode_index=episode_index,
            selection_index=selection_index,
        )

    def sequence(
        self,
        start_episode: int,
        count: int,
    ) -> tuple[TaskSample, ...]:
        """Return a deterministic sequence of task selections."""

        if start_episode < 1:
            raise ValueError(
                "start_episode must be >= 1."
            )

        if count < 0:
            raise ValueError(
                "count must be >= 0."
            )

        return tuple(
            self.sample(
                start_episode + offset
            )
            for offset in range(count)
        )
