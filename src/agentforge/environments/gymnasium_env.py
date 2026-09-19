"""Gymnasium-native environment contract for benchmark tasks."""

from __future__ import annotations

from abc import abstractmethod
from typing import Any

import gymnasium as gym
from gymnasium import spaces


class AgentForgeGymEnv(gym.Env):
    """Canonical Gymnasium environment contract.

    Concrete benchmark tasks will subclass this class later.
    No task implementation is included here.
    """

    metadata: dict[str, Any] = {}

    observation_space: spaces.Space
    action_space: spaces.Space

    @abstractmethod
    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        """Reset and return ``(observation, info)``."""
        super().reset(seed=seed)
        raise NotImplementedError

    @abstractmethod
    def step(
        self,
        action: Any,
    ) -> tuple[
        Any,
        float,
        bool,
        bool,
        dict[str, Any],
    ]:
        """Return Gymnasium's five-value step tuple."""
        raise NotImplementedError
