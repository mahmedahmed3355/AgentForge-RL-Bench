"""RL adapter for exposing AgentForge environments through a Gymnasium-style API."""

from __future__ import annotations

from typing import Any

from .base import BaseEnvironment, EnvironmentStep


class RLEnvironmentAdapter:
    """Adapt an AgentForge environment to a Gymnasium-style interface.

    The adapter intentionally does not depend on Gymnasium itself.
    This keeps AgentForge's core environment layer lightweight while
    exposing the reset()/step() semantics expected by RL libraries.
    """

    def __init__(self, environment: BaseEnvironment) -> None:
        self.environment = environment
        self._terminated = False
        self._truncated = False

    def reset(self) -> Any:
        """Reset the environment and return the initial observation."""

        self._terminated = False
        self._truncated = False

        return self.environment.reset()

    def step(
        self,
        action: Any,
    ) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        """Execute one action using Gymnasium-style semantics."""

        if self._terminated or self._truncated:
            raise RuntimeError(
                "Cannot call step() after the episode has ended. "
                "Call reset() first."
            )

        transition: EnvironmentStep = self.environment.step(action)

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
