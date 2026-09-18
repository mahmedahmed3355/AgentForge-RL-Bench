from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class GymInspectorEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, max_steps: int = 4):
        super().__init__()

        self.max_steps = max_steps

        self.action_space = spaces.Discrete(3)

        self.observation_space = spaces.Dict(
            {
                "position": spaces.Box(
                    low=0.0,
                    high=4.0,
                    shape=(1,),
                    dtype=np.float32,
                ),
                "remaining": spaces.Box(
                    low=0.0,
                    high=4.0,
                    shape=(1,),
                    dtype=np.float32,
                ),
            }
        )

        self.position = 0
        self.step_count = 0

    def _observation(self) -> dict[str, np.ndarray]:
        return {
            "position": np.asarray(
                [self.position],
                dtype=np.float32,
            ),
            "remaining": np.asarray(
                [self.max_steps - self.step_count],
                dtype=np.float32,
            ),
        }

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ):
        super().reset(seed=seed)

        self.position = 0
        self.step_count = 0

        observation = self._observation()

        info = {
            "step": 0,
            "seeded": seed is not None,
        }

        return observation, info

    def step(self, action: int):
        if not self.action_space.contains(action):
            raise ValueError("action outside action_space")

        self.step_count += 1

        if action == 0:
            self.position = max(0, self.position - 1)
        elif action == 1:
            self.position = min(4, self.position + 1)
        else:
            self.position = self.position

        terminated = self.position == 4
        truncated = self.step_count >= self.max_steps and not terminated

        progress_reward = float(self.position) / 4.0
        efficiency_reward = (
            0.25 if action == 1 else 0.0
        )
        terminal_reward = 1.0 if terminated else 0.0

        reward = float(
            progress_reward
            + efficiency_reward
            + terminal_reward
        )

        observation = self._observation()

        info = {
            "step": self.step_count,
            "position": self.position,
        }

        return (
            observation,
            reward,
            bool(terminated),
            bool(truncated),
            info,
        )

    def close(self) -> None:
        return None
