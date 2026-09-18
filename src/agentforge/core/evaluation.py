"""Evaluation metrics and before/after comparison."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class EvaluationMetrics:
    """Metrics produced by an evaluation run."""

    pass_rate: float
    average_steps: float
    total_reward: float
    successful_tasks: int
    failed_tasks: int
    failure_modes: dict[str, int] = field(default_factory=dict)

    @property
    def total_tasks(self) -> int:
        """Return total evaluated tasks."""

        return self.successful_tasks + self.failed_tasks

    def to_dict(self) -> dict[str, Any]:
        """Serialize metrics."""

        return {
            "pass_rate": self.pass_rate,
            "average_steps": self.average_steps,
            "total_reward": self.total_reward,
            "successful_tasks": self.successful_tasks,
            "failed_tasks": self.failed_tasks,
            "failure_modes": self.failure_modes,
        }


@dataclass
class LearningDelta:
    """Difference between baseline and post-training evaluation."""

    pass_rate_delta: float
    average_steps_delta: float
    total_reward_delta: float
    successful_tasks_delta: int
    failed_tasks_delta: int

    baseline: EvaluationMetrics
    after_training: EvaluationMetrics

    def to_dict(self) -> dict[str, Any]:
        """Serialize comparison results."""

        return {
            "pass_rate_delta": self.pass_rate_delta,
            "average_steps_delta": self.average_steps_delta,
            "total_reward_delta": self.total_reward_delta,
            "successful_tasks_delta": self.successful_tasks_delta,
            "failed_tasks_delta": self.failed_tasks_delta,
            "baseline": self.baseline.to_dict(),
            "after_training": self.after_training.to_dict(),
        }


class EvaluationComparator:
    """Compare evaluation results before and after training."""

    @staticmethod
    def compare(
        baseline: EvaluationMetrics,
        after_training: EvaluationMetrics,
    ) -> LearningDelta:
        """Calculate measurable learning deltas."""

        return LearningDelta(
            pass_rate_delta=(
                after_training.pass_rate
                - baseline.pass_rate
            ),
            average_steps_delta=(
                after_training.average_steps
                - baseline.average_steps
            ),
            total_reward_delta=(
                after_training.total_reward
                - baseline.total_reward
            ),
            successful_tasks_delta=(
                after_training.successful_tasks
                - baseline.successful_tasks
            ),
            failed_tasks_delta=(
                after_training.failed_tasks
                - baseline.failed_tasks
            ),
            baseline=baseline,
            after_training=after_training,
        )

    @staticmethod
    def save(
        delta: LearningDelta,
        path: str | Path,
    ) -> Path:
        """Save comparison results."""

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        output_path.write_text(
            json.dumps(
                delta.to_dict(),
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        return output_path
