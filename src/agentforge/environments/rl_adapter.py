"""Gymnasium adapter for legacy AgentForge environments."""

from __future__ import annotations

from typing import Any

import gymnasium as gym

from .base import BaseEnvironment, EnvironmentStep


class RLEnvironmentAdapter(gym.Env):
    """Expose an existing BaseEnvironment as a Gymnasium environment.

    This preserves the existing AgentForge environment implementation while
    making the public adapter a genuine Gymnasium ``Env``.
    """

    metadata: dict[str, Any] = {}

    def __init__(
        self,
        environment: BaseEnvironment,
    ) -> None:
        super().__init__()

        self.environment = environment
        self._terminated = False
        self._truncated = False

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        """Reset and return Gymnasium's ``(observation, info)``."""

        super().reset(seed=seed)

        self._terminated = False
        self._truncated = False

        try:
            observation = self.environment.reset(
                seed=seed
            )
        except TypeError:
            # Preserve compatibility with the existing BaseEnvironment
            # contract, whose reset() predates Gymnasium seeding.
            observation = self.environment.reset()

        info: dict[str, Any] = {}

        if options:
            info["reset_options"] = dict(options)

        if seed is not None:
            info["seed"] = seed

        return observation, info

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
        """Execute one action using Gymnasium semantics."""

        if self._terminated or self._truncated:
            raise RuntimeError(
                "Cannot call step() after the episode has ended. "
                "Call reset() first."
            )

        transition: EnvironmentStep = (
            self.environment.step(action)
        )

        terminated = bool(transition.done)
        truncated = False

        self._terminated = terminated
        self._truncated = truncated

        info = dict(transition.info)

        return (
            transition.observation,
            transition.reward.total,
            terminated,
            truncated,
            info,
        )

    def close(self) -> None:
        """Close the wrapped environment."""

        self.environment.close()

    @property
    def terminated(self) -> bool:
        """Return whether the episode terminated naturally."""

        return self._terminated

    @property
    def truncated(self) -> bool:
        """Return whether the episode was truncated."""

        return self._truncated
