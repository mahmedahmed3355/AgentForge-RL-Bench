"""Checkpoint and resume support for AgentForge training runs."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from .training import TrainingEpisodeMetrics, TrainingRunResult


class TrainingCheckpoint:
    """Persist and restore training-run state."""

    SCHEMA_VERSION = "1.0"

    @classmethod
    def save(
        cls,
        result: TrainingRunResult,
        path: str | Path,
        status: str = "running",
    ) -> Path:
        """Save the current training state."""

        if status not in {"running", "completed"}:
            raise ValueError(
                "status must be 'running' or 'completed'"
            )

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "schema_version": cls.SCHEMA_VERSION,
            "status": status,
            "run_id": result.run_id,
            "episodes_completed": result.episodes_completed,
            "total_steps": result.total_steps,
            "total_reward": result.total_reward,
            "successful_episodes": result.successful_episodes,
            "episode_metrics": [
                asdict(metric)
                for metric in result.episode_metrics
            ],
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
    def load(
        cls,
        path: str | Path,
    ) -> tuple[TrainingRunResult, str]:
        """Restore training state from a checkpoint."""

        input_path = Path(path)

        payload = json.loads(
            input_path.read_text(encoding="utf-8")
        )

        if payload.get("schema_version") != cls.SCHEMA_VERSION:
            raise ValueError(
                "Unsupported training checkpoint schema version"
            )

        status = payload["status"]

        metrics = [
            TrainingEpisodeMetrics(**item)
            for item in payload["episode_metrics"]
        ]

        result = TrainingRunResult(
            run_id=payload["run_id"],
            episodes_completed=payload["episodes_completed"],
            total_steps=payload["total_steps"],
            total_reward=payload["total_reward"],
            successful_episodes=payload["successful_episodes"],
            episode_metrics=metrics,
        )

        return result, status
