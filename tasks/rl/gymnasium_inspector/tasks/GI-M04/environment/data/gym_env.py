from __future__ import annotations

import gymnasium as gym
import numpy as np


class GymInspectorEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, max_steps: int = 6):
        super().__init__()

        self.max_steps = int(max_steps)

        self.observation_space = gym.spaces.Box(
            low=np.array([0], dtype=np.int32),
            high=np.array([6], dtype=np.int32),
            dtype=np.int32,
        )

        self.action_space = gym.spaces.Discrete(2)

        self.state = 0
        self.steps = 0

    def _observation(self):
        return np.array([self.state], dtype=np.int32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.state = 0
        self.steps = 0

        observation = self._observation()
        info = {
            "seeded": seed is not None,
            "state": int(self.state),
        }

        return observation, info

    def step(self, action):
        if not self.action_space.contains(action):
            raise ValueError("invalid action")

        self.steps += 1

        if int(action) == 1:
            self.state += 1
        else:
            self.state = max(0, self.state - 1)

        terminated = bool(self.state >= 6)
        truncated = bool(
            self.steps >= self.max_steps and not terminated
        )

        reward = float(
            1.0
            if int(action) == 1
            else -0.1
        )

        if terminated:
            reward += 1.0

        observation = self._observation()

        info = {
            "state": int(self.state),
            "steps": int(self.steps),
        }

        return (
            observation,
            float(reward),
            terminated,
            truncated,
            info,
        )

    def render(self):
        return None

    def close(self):
        return None
