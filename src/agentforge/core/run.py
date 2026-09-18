"""Training run management for AgentForge-RL-Bench."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class RunManager:
    """Manage metadata and episode results for one training run."""

    run_id: str
    domain: str
    agent_name: str
    config: dict[str, Any] = field(default_factory=dict)

    episodes: list[dict[str, Any]] = field(default_factory=list)
    status: str = "created"

    def start(self) -> None:
        """Mark the run as active."""

        if self.status == "completed":
            raise RuntimeError("Cannot start a completed run.")

        self.status = "running"

    def record_episode(self, result: Any) -> None:
        """Record an episode result."""

        self.episodes.append(
            {
                "episode_id": result.episode_id,
                "task_id": result.task_id,
                "success": result.success,
                "steps": result.steps,
                "total_reward": result.total_reward,
                "final_status": result.final_status,
                "metadata": result.metadata,
            }
        )

    def complete(self) -> None:
        """Mark the run as completed."""

        self.status = "completed"

    @property
    def episode_count(self) -> int:
        """Return the number of recorded episodes."""

        return len(self.episodes)

    @property
    def successful_episodes(self) -> int:
        """Return the number of successful episodes."""

        return sum(
            1
            for episode in self.episodes
            if episode["success"]
        )

    @property
    def success_rate(self) -> float:
        """Return episode success rate."""

        if not self.episodes:
            return 0.0

        return self.successful_episodes / self.episode_count

    @property
    def total_reward(self) -> float:
        """Return accumulated reward across episodes."""

        return sum(
            episode["total_reward"]
            for episode in self.episodes
        )

    def save(self, path: str | Path) -> Path:
        """Persist the complete run metadata."""

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "run_id": self.run_id,
            "domain": self.domain,
            "agent_name": self.agent_name,
            "config": self.config,
            "status": self.status,
            "episode_count": self.episode_count,
            "successful_episodes": self.successful_episodes,
            "success_rate": self.success_rate,
            "total_reward": self.total_reward,
            "episodes": self.episodes,
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
    def load(cls, path: str | Path) -> "RunManager":
        """Restore a training run from disk."""

        input_path = Path(path)

        payload = json.loads(
            input_path.read_text(encoding="utf-8")
        )

        manager = cls(
            run_id=payload["run_id"],
            domain=payload["domain"],
            agent_name=payload["agent_name"],
            config=payload.get("config", {}),
        )

        manager.status = payload.get("status", "created")
        manager.episodes = payload.get("episodes", [])

        return manager
