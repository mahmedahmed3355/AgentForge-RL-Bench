"""Checkpoint persistence for AgentForge-RL-Bench."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .episode import EpisodeRunner
from .contracts import RewardBreakdown


class EpisodeCheckpoint:
    """Save and restore an episode's execution state."""

    @staticmethod
    def save(runner: EpisodeRunner, path: str | Path) -> Path:
        """Persist the current episode state to JSON."""

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        trajectory = runner.recorder.trajectory

        payload: dict[str, Any] = {
            "episode_id": runner.episode_id,
            "task_id": runner.task_id,
            "max_steps": runner.max_steps,
            "metadata": runner.metadata,
            "finished": runner.done,
            "success": runner._success,
            "final_status": runner._final_status,
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
                    },
                    "next_observation": step.next_observation,
                    "status": step.status,
                    "metadata": step.metadata,
                }
                for step in trajectory.steps
            ],
        }

        output_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        return output_path

    @staticmethod
    def load(path: str | Path) -> EpisodeRunner:
        """Restore an EpisodeRunner from a checkpoint."""

        checkpoint_path = Path(path)

        payload = json.loads(
            checkpoint_path.read_text(encoding="utf-8")
        )

        runner = EpisodeRunner(
            episode_id=payload["episode_id"],
            task_id=payload["task_id"],
            max_steps=payload["max_steps"],
            metadata=payload.get("metadata", {}),
        )

        for item in payload["steps"]:
            reward_data = item["reward"]

            reward = RewardBreakdown(
                progress=reward_data["progress"],
                correctness=reward_data["correctness"],
                tests=reward_data["tests"],
                efficiency=reward_data["efficiency"],
                terminal=reward_data["terminal"],
                penalties=reward_data["penalties"],
            )

            runner.recorder.record_step(
                observation=item["observation"],
                action=item["action"],
                reward=reward,
                next_observation=item["next_observation"],
                status=item["status"],
                metadata=item.get("metadata", {}),
            )

        runner._finished = payload["finished"]
        runner._success = payload["success"]
        runner._final_status = payload["final_status"]

        return runner
