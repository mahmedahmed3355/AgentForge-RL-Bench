"""Experiment management for AgentForge-RL-Bench."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ExperimentManager:
    """Manage related training and evaluation runs."""

    experiment_id: str
    name: str
    domain: str
    description: str = ""

    runs: list[dict[str, Any]] = field(default_factory=list)
    status: str = "created"

    def start(self) -> None:
        """Start the experiment."""

        if self.status == "completed":
            raise RuntimeError("Cannot start a completed experiment.")

        self.status = "running"

    def add_run(
        self,
        run_id: str,
        phase: str,
        metrics: dict[str, Any] | None = None,
    ) -> None:
        """Register a run belonging to this experiment."""

        if phase not in {"baseline", "training", "evaluation"}:
            raise ValueError(
                "phase must be baseline, training, or evaluation"
            )

        self.runs.append(
            {
                "run_id": run_id,
                "phase": phase,
                "metrics": metrics or {},
            }
        )

    def complete(self) -> None:
        """Mark the experiment as completed."""

        self.status = "completed"

    @property
    def run_count(self) -> int:
        """Return the number of registered runs."""

        return len(self.runs)

    def runs_for_phase(self, phase: str) -> list[dict[str, Any]]:
        """Return runs belonging to a specific phase."""

        return [
            run
            for run in self.runs
            if run["phase"] == phase
        ]

    def save(self, path: str | Path) -> Path:
        """Persist the experiment."""

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "experiment_id": self.experiment_id,
            "name": self.name,
            "domain": self.domain,
            "description": self.description,
            "status": self.status,
            "run_count": self.run_count,
            "runs": self.runs,
        }

        output_path.write_text(
            json.dumps(
                payload,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        return output_path

    @classmethod
    def load(cls, path: str | Path) -> "ExperimentManager":
        """Restore an experiment from disk."""

        input_path = Path(path)

        payload = json.loads(
            input_path.read_text(encoding="utf-8")
        )

        experiment = cls(
            experiment_id=payload["experiment_id"],
            name=payload["name"],
            domain=payload["domain"],
            description=payload.get("description", ""),
        )

        experiment.status = payload.get("status", "created")
        experiment.runs = payload.get("runs", [])

        return experiment
