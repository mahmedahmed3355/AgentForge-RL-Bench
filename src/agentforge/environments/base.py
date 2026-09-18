"""Base environment contract for AgentForge-RL-Bench."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from agentforge.core import RewardBreakdown


class EnvironmentStep:
    """Result returned by one environment step."""

    def __init__(
        self,
        observation: Any,
        reward: RewardBreakdown,
        done: bool,
        info: dict[str, Any] | None = None,
    ) -> None:
        self.observation = observation
        self.reward = reward
        self.done = done
        self.info = info or {}


class BaseEnvironment(ABC):
    """Minimal contract every AgentForge environment must implement."""

    @abstractmethod
    def reset(self) -> Any:
        """Reset the environment and return the initial observation."""
        raise NotImplementedError

    @abstractmethod
    def step(self, action: Any) -> EnvironmentStep:
        """Apply an action and return the resulting environment state."""
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        """Release environment resources."""
        raise NotImplementedError
