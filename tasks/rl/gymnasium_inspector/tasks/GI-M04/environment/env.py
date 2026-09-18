from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agentforge.environments.base import BaseEnvironment, EnvironmentStep
from agentforge.core import RewardBreakdown


@dataclass
class GiM04Environment(BaseEnvironment):
    """Stateful Gymnasium/Inspector skeleton for GI-M04."""

    seed: int | None = None
    max_steps: int = 100

    def __post_init__(self) -> None:
        self._step_count = 0
        self._state: dict[str, Any] = {}
        self._done = False

    def reset(self) -> dict[str, Any]:
        self._step_count = 0
        self._done = False
        self._state = {
            "task_id": "GI-M04",
            "phase": "initial",
            "progress": 0.0,
        }
        return dict(self._state)

    def step(self, action: Any) -> EnvironmentStep:
        if self._done:
            raise RuntimeError("Cannot step a finished environment.")

        self._step_count += 1

        next_state = dict(self._state)

        reward = RewardBreakdown(
            progress=0.0,
            correctness=0.0,
            tests=0.0,
            efficiency=0.0,
            terminal=0.0,
            penalties=0.0,
        )

        done = self._step_count >= self.max_steps

        if done:
            self._done = True

        self._state = next_state

        return EnvironmentStep(
            observation=dict(next_state),
            reward=reward,
            done=done,
            info={
                "task_id": "GI-M04",
                "step": self._step_count,
            },
        )

    def close(self) -> None:
        self._done = True
