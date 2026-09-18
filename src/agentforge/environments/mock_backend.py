"""Deterministic mock backend environment used for contract testing."""

from __future__ import annotations

from typing import Any

from agentforge.core import RewardBreakdown

from .base import BaseEnvironment, EnvironmentStep


class MockBackendEnvironment(BaseEnvironment):
    """Small deterministic environment for testing the RL interface."""

    def __init__(self, max_steps: int = 5) -> None:
        self.max_steps = max_steps
        self.step_count = 0
        self.state = "initial"

    def reset(self) -> dict[str, Any]:
        self.step_count = 0
        self.state = "initial"

        return {
            "state": self.state,
            "step": self.step_count,
        }

    def step(self, action: Any) -> EnvironmentStep:
        if self.step_count >= self.max_steps:
            raise RuntimeError("Environment episode has already finished.")

        self.step_count += 1

        if action == "solve":
            self.state = "solved"
            reward = RewardBreakdown(
                progress=0.2,
                correctness=0.5,
                terminal=1.0,
            )
            done = True
        else:
            self.state = "working"
            reward = RewardBreakdown(progress=0.1)
            done = self.step_count >= self.max_steps

        return EnvironmentStep(
            observation={
                "state": self.state,
                "step": self.step_count,
            },
            reward=reward,
            done=done,
            info={
                "action": action,
                "step": self.step_count,
            },
        )

    def close(self) -> None:
        self.state = "closed"
