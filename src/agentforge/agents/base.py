"""Agent interface for AgentForge-RL-Bench."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseAgent(ABC):
    """Framework contract implemented by every Agent."""

    @abstractmethod
    def reset(self) -> None:
        """Reset internal episode state."""

    @abstractmethod
    def act(self, observation: Any) -> Any:
        """Produce an action from an observation."""

    @abstractmethod
    def update(
        self,
        observation: Any,
        action: Any,
        reward: float,
        next_observation: Any,
        done: bool,
    ) -> None:
        """Update the agent from one transition."""

    @abstractmethod
    def save_checkpoint(self, path: str) -> None:
        """Save agent state."""

    @abstractmethod
    def load_checkpoint(self, path: str) -> None:
        """Load agent state."""
