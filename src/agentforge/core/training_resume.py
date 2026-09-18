"""Training resume orchestration for AgentForge-RL-Bench."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .training import TrainingRunResult
from .training_checkpoint import TrainingCheckpoint


@dataclass
class TrainingResumeState:
    """State required to resume a training run."""

    run_id: str
    next_episode_index: int
    episodes_completed: int
    total_steps: int
    total_reward: float
    successful_episodes: int
    status: str


class TrainingResumeManager:
    """Create and restore resumable training state."""

    def save(
        self,
        result: TrainingRunResult,
        checkpoint_path: str | Path,
        status: str = "running",
    ) -> TrainingResumeState:
        """Save training state and return the resume position."""

        TrainingCheckpoint.save(
            result=result,
            path=checkpoint_path,
            status=status,
        )

        return TrainingResumeState(
            run_id=result.run_id,
            next_episode_index=result.episodes_completed + 1,
            episodes_completed=result.episodes_completed,
            total_steps=result.total_steps,
            total_reward=result.total_reward,
            successful_episodes=result.successful_episodes,
            status=status,
        )

    def load(
        self,
        checkpoint_path: str | Path,
    ) -> tuple[TrainingRunResult, TrainingResumeState]:
        """Restore a training result and calculate the next episode."""

        result, status = TrainingCheckpoint.load(
            checkpoint_path
        )

        state = TrainingResumeState(
            run_id=result.run_id,
            next_episode_index=result.episodes_completed + 1,
            episodes_completed=result.episodes_completed,
            total_steps=result.total_steps,
            total_reward=result.total_reward,
            successful_episodes=result.successful_episodes,
            status=status,
        )

        return result, state

    @staticmethod
    def episode_id(
        run_id: str,
        episode_index: int,
    ) -> str:
        """Create a deterministic episode identifier."""

        if episode_index < 1:
            raise ValueError(
                "episode_index must be >= 1"
            )

        return (
            f"{run_id}-episode-{episode_index:04d}"
        )

    @staticmethod
    def validate_resume(
        state: TrainingResumeState,
        requested_total_episodes: int,
    ) -> None:
        """Validate that a run can continue."""

        if requested_total_episodes < 1:
            raise ValueError(
                "requested_total_episodes must be >= 1"
            )

        if state.episodes_completed > requested_total_episodes:
            raise ValueError(
                "Checkpoint contains more completed episodes "
                "than the requested training run."
            )

        if state.status == "completed":
            raise RuntimeError(
                "Training run is already completed."
            )

    @staticmethod
    def merge_episode(
        result: TrainingRunResult,
        *,
        steps: int,
        reward: float,
        success: bool,
        episode_id: str,
        task_id: str,
    ) -> TrainingRunResult:
        """Append one newly completed episode to a restored result."""

        from .training import TrainingEpisodeMetrics

        metric = TrainingEpisodeMetrics(
            episode_id=episode_id,
            task_id=task_id,
            steps=steps,
            total_reward=reward,
            success=success,
        )

        return TrainingRunResult(
            run_id=result.run_id,
            episodes_completed=result.episodes_completed + 1,
            total_steps=result.total_steps + steps,
            total_reward=result.total_reward + reward,
            successful_episodes=(
                result.successful_episodes
                + (1 if success else 0)
            ),
            episode_metrics=[
                *result.episode_metrics,
                metric,
            ],
        )
