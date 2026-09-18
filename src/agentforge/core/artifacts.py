"""Persistent experiment artifacts for AgentForge-RL-Bench."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .orchestrator import ExperimentExecutionResult


class ExperimentArtifactStore:
    """Persist complete experiment results as inspectable JSON artifacts."""

    def __init__(self, root: str | Path = "runs") -> None:
        self.root = Path(root)

    def experiment_dir(self, experiment_id: str) -> Path:
        """Return the artifact directory for one experiment."""

        return self.root / experiment_id

    def save(
        self,
        result: ExperimentExecutionResult,
        config: dict[str, Any] | None = None,
    ) -> Path:
        """Persist an experiment result and its layers."""

        directory = self.experiment_dir(
            result.experiment_id
        )
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        baseline = result.baseline.to_dict()
        after_training = result.after_training.to_dict()

        learning_delta = {
            "pass_rate_delta": (
                result.learning_delta.pass_rate_delta
            ),
            "average_steps_delta": (
                result.learning_delta.average_steps_delta
            ),
            "total_reward_delta": (
                result.learning_delta.total_reward_delta
            ),
            "successful_tasks_delta": (
                result.learning_delta.successful_tasks_delta
            ),
            "failed_tasks_delta": (
                result.learning_delta.failed_tasks_delta
            ),
        }

        experiment = result.to_dict()

        self._write_json(
            directory / "config.json",
            config or {},
        )
        self._write_json(
            directory / "baseline.json",
            baseline,
        )
        self._write_json(
            directory / "after_training.json",
            after_training,
        )
        self._write_json(
            directory / "learning_delta.json",
            learning_delta,
        )
        self._write_json(
            directory / "experiment_result.json",
            experiment,
        )

        return directory

    def load(
        self,
        experiment_id: str,
    ) -> dict[str, Any]:
        """Load the complete persisted experiment artifact."""

        directory = self.experiment_dir(
            experiment_id
        )

        if not directory.exists():
            raise FileNotFoundError(
                f"Experiment artifacts not found: "
                f"{experiment_id}"
            )

        return {
            "config": self._read_json(
                directory / "config.json"
            ),
            "baseline": self._read_json(
                directory / "baseline.json"
            ),
            "after_training": self._read_json(
                directory / "after_training.json"
            ),
            "learning_delta": self._read_json(
                directory / "learning_delta.json"
            ),
            "experiment_result": self._read_json(
                directory / "experiment_result.json"
            ),
        }

    @staticmethod
    def _write_json(
        path: Path,
        payload: dict[str, Any],
    ) -> None:
        path.write_text(
            json.dumps(
                payload,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    @staticmethod
    def _read_json(
        path: Path,
    ) -> dict[str, Any]:
        if not path.exists():
            raise FileNotFoundError(
                f"Artifact file not found: {path}"
            )

        return json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )
