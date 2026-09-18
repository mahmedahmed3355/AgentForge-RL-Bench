"""Training and held-out evaluation task split utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .environment_runner import TrainingTask


@dataclass(frozen=True)
class TaskSplit:
    """Immutable train/evaluation task partition."""

    training_tasks: tuple[TrainingTask, ...]
    evaluation_tasks: tuple[TrainingTask, ...]

    def __post_init__(self) -> None:
        if not self.training_tasks:
            raise ValueError(
                "Training task pool cannot be empty."
            )

        if not self.evaluation_tasks:
            raise ValueError(
                "Evaluation task pool cannot be empty."
            )

        training_ids = {
            task.task_id
            for task in self.training_tasks
        }

        evaluation_ids = {
            task.task_id
            for task in self.evaluation_tasks
        }

        overlap = training_ids & evaluation_ids

        if overlap:
            raise ValueError(
                "Training and evaluation task IDs must not overlap: "
                + ", ".join(sorted(overlap))
            )

    @property
    def training_task_ids(self) -> tuple[str, ...]:
        """Return training task IDs in declared order."""

        return tuple(
            task.task_id
            for task in self.training_tasks
        )

    @property
    def evaluation_task_ids(self) -> tuple[str, ...]:
        """Return held-out task IDs in declared order."""

        return tuple(
            task.task_id
            for task in self.evaluation_tasks
        )

    def get_training_task(
        self,
        task_id: str,
    ) -> TrainingTask:
        """Return a training task only."""

        for task in self.training_tasks:
            if task.task_id == task_id:
                return task

        raise KeyError(
            f"Unknown training task: {task_id}"
        )

    def get_evaluation_task(
        self,
        task_id: str,
    ) -> TrainingTask:
        """Return an evaluation task only."""

        for task in self.evaluation_tasks:
            if task.task_id == task_id:
                return task

        raise KeyError(
            f"Unknown evaluation task: {task_id}"
        )


class HeldOutEvaluationPool:
    """Read-only access to tasks reserved for evaluation."""

    def __init__(
        self,
        tasks: Sequence[TrainingTask],
    ) -> None:
        if not tasks:
            raise ValueError(
                "Held-out evaluation pool cannot be empty."
            )

        task_ids = [
            task.task_id
            for task in tasks
        ]

        if len(task_ids) != len(set(task_ids)):
            raise ValueError(
                "Evaluation task IDs must be unique."
            )

        self._tasks = tuple(tasks)

    @property
    def tasks(self) -> tuple[TrainingTask, ...]:
        """Return the immutable held-out task collection."""

        return self._tasks

    @property
    def task_ids(self) -> tuple[str, ...]:
        """Return evaluation task IDs in deterministic order."""

        return tuple(
            task.task_id
            for task in self._tasks
        )

    def get(
        self,
        task_id: str,
    ) -> TrainingTask:
        """Retrieve one held-out evaluation task."""

        for task in self._tasks:
            if task.task_id == task_id:
                return task

        raise KeyError(
            f"Unknown held-out evaluation task: {task_id}"
        )

    def as_tuple(self) -> tuple[TrainingTask, ...]:
        """Expose the pool without allowing mutation."""

        return self._tasks


def build_task_split(
    training_tasks: Sequence[TrainingTask],
    evaluation_tasks: Sequence[TrainingTask],
) -> TaskSplit:
    """Build and validate a strict train/evaluation split."""

    return TaskSplit(
        training_tasks=tuple(training_tasks),
        evaluation_tasks=tuple(evaluation_tasks),
    )
