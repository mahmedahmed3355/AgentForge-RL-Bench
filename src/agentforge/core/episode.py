"""Episode execution state for AgentForge-RL-Bench."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .contracts import EpisodeResult, RewardBreakdown
from .trajectory import TrajectoryRecorder


@dataclass
class EpisodeRunner:
    """Manage one agent attempt on one task."""

    episode_id: str
    task_id: str
    max_steps: int = 100
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.recorder = TrajectoryRecorder(
            episode_id=self.episode_id,
            task_id=self.task_id,
        )
        self._finished = False
        self._success = False
        self._final_status = "running"

    @property
    def step_count(self) -> int:
        """Return the number of recorded steps."""

        return self.recorder.trajectory.length

    @property
    def done(self) -> bool:
        """Return whether the episode has finished."""

        return self._finished

    def record_step(
        self,
        observation: Any,
        action: Any,
        reward: RewardBreakdown,
        next_observation: Any,
        status: str = "running",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record one step and update episode state."""

        if self._finished:
            raise RuntimeError("Cannot record a step after the episode has finished.")

        if self.step_count >= self.max_steps:
            raise RuntimeError("Maximum episode steps have already been reached.")

        self.recorder.record_step(
            observation=observation,
            action=action,
            reward=reward,
            next_observation=next_observation,
            status=status,
            metadata=metadata,
        )

        if status in {"success", "failure", "timeout"}:
            self._finished = True
            self._success = status == "success"
            self._final_status = status
        elif self.step_count >= self.max_steps:
            self._finished = True
            self._success = False
            self._final_status = "timeout"

    def finish(self, success: bool, status: str | None = None) -> EpisodeResult:
        """Explicitly finish an episode."""

        if not self._finished:
            self._finished = True
            self._success = success
            self._final_status = status or ("success" if success else "failure")

        return self.result()

    def result(self) -> EpisodeResult:
        """Build the current episode result."""

        return EpisodeResult(
            episode_id=self.episode_id,
            task_id=self.task_id,
            success=self._success,
            steps=self.step_count,
            total_reward=self.recorder.trajectory.total_reward,
            final_status=self._final_status,
            metadata=self.metadata,
        )
