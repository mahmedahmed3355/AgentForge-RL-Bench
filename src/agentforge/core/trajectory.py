"""Trajectory recording utilities."""

from __future__ import annotations

import json
from dataclasses import asdict
from enum import Enum
from pathlib import Path
from typing import Any

from .contracts import (
    StepRecord,
    Trajectory,
    TrajectoryEventRecord,
)


class TrajectoryRecorder:
    """Record complete step-level benchmark trajectories."""

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
        events: list[TrajectoryEventRecord] | None = None,
    ) -> StepRecord:
        """Record one transition and its typed semantic events."""

        step = StepRecord(
            step_id=self.trajectory.length + 1,
            observation=observation,
            action=action,
            reward=reward,
            next_observation=next_observation,
            status=status,
            events=list(events or []),
            metadata=metadata or {},
        )

        for event in step.events:
            if event.step_id != step.step_id:
                raise ValueError(
                    "Trajectory event step_id must match "
                    "the recorded step."
                )

        self.trajectory.append(step)
        return step

    @staticmethod
    def _event_payload(event: TrajectoryEventRecord) -> dict[str, Any]:
        payload = asdict(event)

        event_type = payload.get("event_type")

        if isinstance(event_type, Enum):
            payload["event_type"] = event_type.value

        return payload

    @staticmethod
    def _reward_payload(reward) -> dict[str, Any]:
        return {
            "progress": reward.progress,
            "correctness": reward.correctness,
            "tests": reward.tests,
            "testing": reward.testing,
            "recovery": reward.recovery,
            "efficiency": reward.efficiency,
            "terminal": reward.terminal,
            "penalties": reward.penalties,
            "total": reward.total,
            "provenance": [
                {
                    "source": item.source,
                    "component": item.component,
                    "value": item.value,
                    "reason": item.reason,
                    "step_id": item.step_id,
                    "event_type": (
                        item.event_type.value
                        if item.event_type is not None
                        else None
                    ),
                }
                for item in reward.provenance
            ],
        }

    def save_json(self, path: str | Path) -> Path:
        """Save the complete trajectory as JSON."""

        output_path = Path(path)
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

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
                    "reward": self._reward_payload(
                        step.reward
                    ),
                    "next_observation": step.next_observation,
                    "status": step.status,
                    "events": [
                        self._event_payload(event)
                        for event in step.events
                    ],
                    "metadata": step.metadata,
                }
                for step in self.trajectory.steps
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
