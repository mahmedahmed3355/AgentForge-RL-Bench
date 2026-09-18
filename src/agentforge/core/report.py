"""Unified experiment reporting for AgentForge-RL-Bench."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .evaluation import EvaluationMetrics, LearningDelta
from .failure_analysis import FailureAnalysis


@dataclass
class ExperimentReport:
    """Unified report for one AgentForge experiment."""

    experiment_id: str
    name: str
    domain: str
    status: str

    baseline: EvaluationMetrics
    after_training: EvaluationMetrics
    learning_delta: LearningDelta
    failure_analysis: FailureAnalysis

    training_runs: list[dict[str, Any]]

    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Convert the complete report into a JSON-safe dictionary."""

        return {
            "schema_version": "1.0",
            "experiment": {
                "experiment_id": self.experiment_id,
                "name": self.name,
                "domain": self.domain,
                "status": self.status,
            },
            "baseline": self.baseline.to_dict(),
            "after_training": self.after_training.to_dict(),
            "learning_delta": self.learning_delta.to_dict(),
            "failure_analysis": self.failure_analysis.to_dict(),
            "training_runs": self.training_runs,
            "metadata": self.metadata,
        }

    def save(self, path: str | Path) -> Path:
        """Persist the complete experiment report."""

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        output_path.write_text(
            json.dumps(
                self.to_dict(),
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        return output_path

    @classmethod
    def load(cls, path: str | Path) -> "ExperimentReport":
        """Restore a report from JSON."""

        input_path = Path(path)

        payload = json.loads(
            input_path.read_text(encoding="utf-8")
        )

        baseline_data = payload["baseline"]
        after_data = payload["after_training"]
        delta_data = payload["learning_delta"]
        failure_data = payload["failure_analysis"]

        baseline = EvaluationMetrics(
            pass_rate=baseline_data["pass_rate"],
            average_steps=baseline_data["average_steps"],
            total_reward=baseline_data["total_reward"],
            successful_tasks=baseline_data["successful_tasks"],
            failed_tasks=baseline_data["failed_tasks"],
            failure_modes=baseline_data.get(
                "failure_modes",
                {},
            ),
        )

        after_training = EvaluationMetrics(
            pass_rate=after_data["pass_rate"],
            average_steps=after_data["average_steps"],
            total_reward=after_data["total_reward"],
            successful_tasks=after_data["successful_tasks"],
            failed_tasks=after_data["failed_tasks"],
            failure_modes=after_data.get(
                "failure_modes",
                {},
            ),
        )

        learning_delta = LearningDelta(
            pass_rate_delta=delta_data["pass_rate_delta"],
            average_steps_delta=delta_data["average_steps_delta"],
            total_reward_delta=delta_data["total_reward_delta"],
            successful_tasks_delta=delta_data[
                "successful_tasks_delta"
            ],
            failed_tasks_delta=delta_data[
                "failed_tasks_delta"
            ],
            baseline=baseline,
            after_training=after_training,
        )

        mode_deltas = []

        from .failure_analysis import FailureModeDelta

        for item in failure_data["mode_deltas"]:
            mode_deltas.append(
                FailureModeDelta(
                    mode=item["mode"],
                    baseline_count=item["baseline_count"],
                    after_training_count=item[
                        "after_training_count"
                    ],
                )
            )

        failure_analysis = FailureAnalysis(
            baseline_failures=failure_data[
                "baseline_failures"
            ],
            after_training_failures=failure_data[
                "after_training_failures"
            ],
            mode_deltas=mode_deltas,
        )

        experiment_data = payload["experiment"]

        return cls(
            experiment_id=experiment_data["experiment_id"],
            name=experiment_data["name"],
            domain=experiment_data["domain"],
            status=experiment_data["status"],
            baseline=baseline,
            after_training=after_training,
            learning_delta=learning_delta,
            failure_analysis=failure_analysis,
            training_runs=payload.get(
                "training_runs",
                [],
            ),
            metadata=payload.get(
                "metadata",
                {},
            ),
        )
