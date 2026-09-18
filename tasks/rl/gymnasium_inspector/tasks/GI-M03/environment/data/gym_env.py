from __future__ import annotations

import gymnasium as gym
from gymnasium import spaces
import numpy as np


class GymInspectorEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, max_steps: int = 6):
        super().__init__()

        if max_steps <= 0:
            raise ValueError("max_steps must be positive")

        self.max_steps = int(max_steps)

        self.observation_space = spaces.Dict(
            {
                "state": spaces.Box(
                    low=np.array([0], dtype=np.int32),
                    high=np.array([5], dtype=np.int32),
                    dtype=np.int32,
                ),
                "steps": spaces.Box(
                    low=np.array([0], dtype=np.int32),
                    high=np.array([self.max_steps], dtype=np.int32),
                    dtype=np.int32,
                ),
            }
        )

        self.action_space = spaces.Discrete(3)

        self.state = 0
        self.steps = 0

    def _get_observation(self):
        return {
            "state": np.array([self.state], dtype=np.int32),
            "steps": np.array([self.steps], dtype=np.int32),
        }

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.state = 0
        self.steps = 0

        observation = self._get_observation()
        info = {
            "target_state": 5,
            "max_steps": self.max_steps,
        }

        return observation, info

    def step(self, action):
        if not self.action_space.contains(action):
            raise ValueError(f"invalid action: {action!r}")

        action = int(action)
        self.steps += 1

        if action == 0:
            self.state = self.state
            reward = -0.1
        elif action == 1:
            self.state = min(self.state + 1, 5)
            reward = 0.4
        else:
            self.state = min(self.state + 2, 5)
            reward = 0.8

        terminated = self.state == 5
        truncated = self.steps >= self.max_steps and not terminated

        if terminated:
            reward += 1.5

        observation = self._get_observation()

        if not self.observation_space.contains(observation):
            raise RuntimeError("environment generated observation outside observation_space")

        info = {
            "target_state": 5,
            "state": self.state,
            "steps": self.steps,
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
