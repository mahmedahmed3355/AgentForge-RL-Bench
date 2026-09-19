"""Canonical benchmark task registry."""

from __future__ import annotations

from dataclasses import dataclass

from .contracts import Split, TaskSpec


@dataclass(frozen=True)
class TaskRegistry:
    """Validated registry of versioned benchmark task specifications."""

    _tasks: dict[str, TaskSpec]

    def __init__(
        self,
        tasks: tuple[TaskSpec, ...] = (),
    ) -> None:
        registry: dict[str, TaskSpec] = {}

        for task in tasks:
            self._validate_task(task)

            if task.identity in registry:
                raise ValueError(
                    f"Duplicate task identity: {task.identity}"
                )

            registry[task.identity] = task

        object.__setattr__(
            self,
            "_tasks",
            registry,
        )

    @staticmethod
    def _validate_task(task: TaskSpec) -> None:
        if not isinstance(task, TaskSpec):
            raise TypeError(
                "TaskRegistry accepts only TaskSpec instances."
            )

    def register(self, task: TaskSpec) -> "TaskRegistry":
        """Return a new registry containing ``task``."""

        self._validate_task(task)

        if task.identity in self._tasks:
            raise ValueError(
                f"Task already registered: {task.identity}"
            )

        return TaskRegistry(
            tasks=tuple(self._tasks.values())
            + (task,)
        )

    def get(
        self,
        task_id: str,
        version: str | None = None,
    ) -> TaskSpec:
        """Retrieve a task by stable ID and optional version."""

        if version is not None:
            identity = f"{task_id}@{version}"

            try:
                return self._tasks[identity]
            except KeyError as exc:
                raise KeyError(
                    f"Unknown task: {identity}"
                ) from exc

        matches = [
            task
            for task in self._tasks.values()
            if task.task_id == task_id
        ]

        if not matches:
            raise KeyError(
                f"Unknown task: {task_id}"
            )

        if len(matches) > 1:
            raise ValueError(
                f"Multiple versions registered for {task_id}; "
                "specify version explicitly."
            )

        return matches[0]

    def all(self) -> tuple[TaskSpec, ...]:
        """Return registered tasks in deterministic identity order."""

        return tuple(
            sorted(
                self._tasks.values(),
                key=lambda task: task.identity,
            )
        )

    def for_split(
        self,
        split: Split,
    ) -> tuple[TaskSpec, ...]:
        """Return tasks belonging to one split."""

        if split not in {"train", "eval"}:
            raise ValueError(
                f"Unsupported split: {split!r}"
            )

        return tuple(
            task
            for task in self.all()
            if task.split == split
        )

    def __len__(self) -> int:
        return len(self._tasks)
