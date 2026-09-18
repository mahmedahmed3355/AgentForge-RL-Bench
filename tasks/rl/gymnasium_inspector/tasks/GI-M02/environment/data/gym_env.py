from __future__ import annotations

import gymnasium as gym
from gymnasium import spaces
import numpy as np


class GymInspectorEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, max_steps: int = 5):
        super().__init__()

        if max_steps <= 0:
            raise ValueError("max_steps must be positive")

        self.max_steps = int(max_steps)

        self.observation_space = spaces.Dict(
            {
                "position": spaces.Box(
                    low=np.array([0], dtype=np.int32),
                    high=np.array([4], dtype=np.int32),
                    dtype=np.int32,
                ),
                "steps": spaces.Box(
                    low=np.array([0], dtype=np.int32),
                    high=np.array([self.max_steps], dtype=np.int32),
                    dtype=np.int32,
                ),
            }
        )

        self.action_space = spaces.Discrete(2)

        self.position = 0
        self.steps = 0

    def _observation(self):
        return {
            "position": np.array([self.position], dtype=np.int32),
            "steps": np.array([self.steps], dtype=np.int32),
        }

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.position = 0
        self.steps = 0

        observation = self._observation()
        info = {
            "target_position": 4,
            "max_steps": self.max_steps,
        }

        return observation, info

    def step(self, action):
        if not self.action_space.contains(action):
            raise ValueError(f"invalid action: {action!r}")

        self.steps += 1

        if int(action) == 1:
            self.position = min(self.position + 1, 4)

        terminated = self.position == 4
        truncated = self.steps >= self.max_steps and not terminated

        if terminated:
            reward = 2.0
        elif int(action) == 1:
            reward = 0.5
        else:
            reward = -0.1

        observation = self._observation()
        info = {
            "target_position": 4,
            "distance": 4 - self.position,
        }

        return (
            observation,
            float(reward),
            bool(terminated),
            bool(truncated),
            info,
        )

    def close(self):
        return None
