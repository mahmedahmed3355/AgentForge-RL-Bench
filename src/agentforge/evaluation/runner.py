"""Before/after evaluation runner for AgentForge-RL-Bench."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from agentforge.agents.base import BaseAgent
from agentforge.core.agent_executor import AgentEpisodeExecutor
from agentforge.training.task_split import HeldOutEvaluationPool


@dataclass(frozen=True)
class EvaluationTaskResult:
    """Result of evaluating one held-out task."""

    task_id: str
    episode_id: str
    success: bool
    steps: int
    total_reward: float
    final_status: str
    failure_mode: str | None = None
    metadata: dict[str, Any] = field(
        default_factory=dict
    )


@dataclass(frozen=True)
class EvaluationRunResult:
    """Aggregate result for one evaluation phase."""

    evaluation_id: str
    phase: str
    task_results: tuple[EvaluationTaskResult, ...]

    @property
    def task_count(self) -> int:
        """Return the number of evaluated tasks."""

        return len(self.task_results)

    @property
    def successful_tasks(self) -> int:
        """Return the number of successful tasks."""

        return sum(
            1
            for result in self.task_results
            if result.success
        )

    @property
    def failed_tasks(self) -> int:
        """Return the number of failed tasks."""

        return (
            self.task_count
            - self.successful_tasks
        )

    @property
    def pass_rate(self) -> float:
        """Return the task pass rate."""

        if self.task_count == 0:
            return 0.0

        return (
            self.successful_tasks
            / self.task_count
        )

    @property
    def average_steps(self) -> float:
        """Return average steps per evaluated task."""

        if not self.task_results:
            return 0.0

        return sum(
            result.steps
            for result in self.task_results
        ) / self.task_count

    @property
    def total_reward(self) -> float:
        """Return aggregate reward."""

        return sum(
            result.total_reward
            for result in self.task_results
        )

    @property
    def failure_modes(self) -> dict[str, int]:
        """Count observed failure modes."""

        counts: dict[str, int] = {}

        for result in self.task_results:
            if result.failure_mode is None:
                continue

            counts[result.failure_mode] = (
                counts.get(result.failure_mode, 0)
                + 1
            )

        return counts

    def to_dict(self) -> dict[str, Any]:
        """Serialize the complete evaluation result."""

        return {
            "evaluation_id": self.evaluation_id,
            "phase": self.phase,
            "task_count": self.task_count,
            "successful_tasks": self.successful_tasks,
            "failed_tasks": self.failed_tasks,
            "pass_rate": self.pass_rate,
            "average_steps": self.average_steps,
            "total_reward": self.total_reward,
            "failure_modes": dict(
                self.failure_modes
            ),
            "tasks": [
                {
                    "task_id": result.task_id,
                    "episode_id": result.episode_id,
                    "success": result.success,
                    "steps": result.steps,
                    "total_reward": result.total_reward,
                    "final_status": result.final_status,
                    "failure_mode": result.failure_mode,
                    "metadata": dict(
                        result.metadata
                    ),
                }
                for result in self.task_results
            ],
        }


class EvaluationRunner:
    """Evaluate an agent against held-out tasks."""

    VALID_PHASES = {
        "baseline",
        "after_training",
    }

    def __init__(
        self,
        evaluation_pool: HeldOutEvaluationPool,
    ) -> None:
        self.evaluation_pool = evaluation_pool

    def run(
        self,
        *,
        agent: BaseAgent,
        evaluation_id: str,
        phase: str,
        max_steps: int,
    ) -> EvaluationRunResult:
        """Run evaluation across every held-out task."""

        if phase not in self.VALID_PHASES:
            raise ValueError(
                f"Unsupported evaluation phase: {phase}"
            )

        if max_steps < 1:
            raise ValueError(
                "max_steps must be >= 1."
            )

        results: list[EvaluationTaskResult] = []

        for index, task in enumerate(
            self.evaluation_pool.tasks,
            start=1,
        ):
            episode_id = (
                f"{evaluation_id}"
                f"-{phase}"
                f"-task-{index:04d}"
            )

            environment = task.environment_factory()

            try:
                executor = AgentEpisodeExecutor(
                    agent=agent,
                    environment=environment,
                )

                result = executor.run(
                    episode_id=episode_id,
                    task_id=task.task_id,
                    max_steps=max_steps,
                )

                failure_mode = (
                    None
                    if result.done
                    else "incomplete"
                )

                results.append(
                    EvaluationTaskResult(
                        task_id=task.task_id,
                        episode_id=episode_id,
                        success=result.done,
                        steps=result.steps,
                        total_reward=result.total_reward,
                        final_status=(
                            "success"
                            if result.done
                            else "incomplete"
                        ),
                        failure_mode=failure_mode,
                        metadata={
                            "phase": phase,
                            "evaluation_id": evaluation_id,
                            **task.metadata,
                        },
                    )
                )
            finally:
                environment.close()

        return EvaluationRunResult(
            evaluation_id=evaluation_id,
            phase=phase,
            task_results=tuple(results),
        )
