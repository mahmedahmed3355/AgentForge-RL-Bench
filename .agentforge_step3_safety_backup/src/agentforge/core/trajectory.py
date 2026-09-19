"""Trajectory recording utilities for AgentForge-RL-Bench."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .contracts import StepRecord, Trajectory


class TrajectoryRecorder:
    """Record and persist agent-environment interactions."""

    def __init__(self, episode_id: str, task_id: str) -> None:
        self.trajectory = Trajectory(
            episode_id=episode_id,
            task_id=task_id,
        )

    def record_step(
        self,
        observation: Any,
        action: Any,
        reward,
        next_observation: Any,
        status: str,
        metadata: dict[str, Any] | None = None,
    ) -> StepRecord:
        """Record one environment step."""

        step = StepRecord(
            step_id=self.trajectory.length + 1,
            observation=observation,
            action=action,
            reward=reward,
            next_observation=next_observation,
            status=status,
            metadata=metadata or {},
        )

        self.trajectory.append(step)
        return step

    def save_json(self, path: str | Path) -> Path:
        """Save the complete trajectory as JSON."""

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "episode_id": self.trajectory.episode_id,
            "task_id": self.trajectory.task_id,
            "length": self.trajectory.length,
            "total_reward": self.trajectory.total_reward,
            "steps": [
                {
                    "step_id": step.step_id,
                    "observation": step.observation,
                    "action": step.action,
                    "reward": {
                        "progress": step.reward.progress,
                        "correctness": step.reward.correctness,
                        "tests": step.reward.tests,
                        "efficiency": step.reward.efficiency,
                        "terminal": step.reward.terminal,
                        "penalties": step.reward.penalties,
                        "total": step.reward.total,
                    },
                    "next_observation": step.next_observation,
                    "status": step.status,
                    "metadata": step.metadata,
                }
                for step in self.trajectory.steps
            ],
        }

        output_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        return output_path
