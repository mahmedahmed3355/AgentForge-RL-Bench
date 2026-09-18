"""End-to-end experiment orchestration for AgentForge-RL-Bench."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from agentforge.agents.base import BaseAgent
from agentforge.evaluation import (
    EvaluationRunResult,
    EvaluationRunner,
)
from agentforge.core.evaluation import (
    EvaluationComparator,
    EvaluationMetrics,
    LearningDelta,
)


@dataclass(frozen=True)
class ExperimentExecutionResult:
    """Complete result of one training/evaluation experiment."""

    experiment_id: str
    baseline: EvaluationRunResult
    after_training: EvaluationRunResult
    learning_delta: LearningDelta

    def to_dict(self) -> dict[str, Any]:
        """Serialize the complete experiment result."""

        return {
            "experiment_id": self.experiment_id,
            "baseline": self.baseline.to_dict(),
            "after_training": self.after_training.to_dict(),
            "learning_delta": {
                "pass_rate_delta": (
                    self.learning_delta.pass_rate_delta
                ),
                "average_steps_delta": (
                    self.learning_delta.average_steps_delta
                ),
                "total_reward_delta": (
                    self.learning_delta.total_reward_delta
                ),
                "successful_tasks_delta": (
                    self.learning_delta.successful_tasks_delta
                ),
                "failed_tasks_delta": (
                    self.learning_delta.failed_tasks_delta
                ),
            },
        }


class ExperimentOrchestrator:
    """Coordinate baseline, training, and after-training evaluation."""

    def __init__(
        self,
        evaluation_runner: EvaluationRunner,
    ) -> None:
        self.evaluation_runner = evaluation_runner

    def run(
        self,
        *,
        experiment_id: str,
        agent: BaseAgent,
        max_steps: int,
        train: Callable[[BaseAgent], None],
    ) -> ExperimentExecutionResult:
        """Run one complete baseline → training → evaluation experiment."""

        baseline = self.evaluation_runner.run(
            agent=agent,
            evaluation_id=f"{experiment_id}-baseline",
            phase="baseline",
            max_steps=max_steps,
        )

        train(agent)

        after_training = self.evaluation_runner.run(
            agent=agent,
            evaluation_id=f"{experiment_id}-after-training",
            phase="after_training",
            max_steps=max_steps,
        )

        baseline_metrics = EvaluationMetrics(
            pass_rate=baseline.pass_rate,
            average_steps=baseline.average_steps,
            total_reward=baseline.total_reward,
            successful_tasks=baseline.successful_tasks,
            failed_tasks=baseline.failed_tasks,
            failure_modes=baseline.failure_modes,
        )

        after_training_metrics = EvaluationMetrics(
            pass_rate=after_training.pass_rate,
            average_steps=after_training.average_steps,
            total_reward=after_training.total_reward,
            successful_tasks=after_training.successful_tasks,
            failed_tasks=after_training.failed_tasks,
            failure_modes=after_training.failure_modes,
        )

        learning_delta = EvaluationComparator.compare(
            baseline_metrics,
            after_training_metrics,
        )

        return ExperimentExecutionResult(
            experiment_id=experiment_id,
            baseline=baseline,
            after_training=after_training,
            learning_delta=learning_delta,
        )
