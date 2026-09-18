from __future__ import annotations

import gymnasium as gym
import numpy as np


class GymInspectorEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, max_steps: int = 7):
        super().__init__()

        self.max_steps = int(max_steps)

        self.observation_space = gym.spaces.Box(
            low=np.array([0], dtype=np.int32),
            high=np.array([7], dtype=np.int32),
            dtype=np.int32,
        )

        self.action_space = gym.spaces.Discrete(2)

        self.state = 0
        self.steps = 0

    def reset(self, *, seed: int | None = None, options=None):
        super().reset(seed=seed)

        self.state = 0
        self.steps = 0

        observation = np.array([self.state], dtype=np.int32)

        info = {
            "state": self.state,
            "steps": self.steps,
        }

        return observation, info

    def step(self, action):
        if not self.action_space.contains(action):
            raise ValueError("invalid action")

        self.steps += 1

        if int(action) == 1:
            self.state += 1
            reward = 1.0
        else:
            reward = 0.25

        terminated = self.state >= 7
        truncated = self.steps >= self.max_steps and not terminated

        observation = np.array([self.state], dtype=np.int32)

        if terminated:
            reward += 1.5

        info = {
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

    def render(self):
        return None

    def close(self):
        return None
