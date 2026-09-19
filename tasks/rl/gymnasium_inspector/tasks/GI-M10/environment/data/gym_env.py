from __future__ import annotations

import gymnasium as gym
import numpy as np


class GymInspectorEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, max_steps: int = 8):
        super().__init__()

        self.max_steps = int(max_steps)

        self.observation_space = gym.spaces.Dict(
            {
                "state": gym.spaces.Box(
                    low=np.array([0], dtype=np.int32),
                    high=np.array([12], dtype=np.int32),
                    dtype=np.int32,
                ),
                "step": gym.spaces.Box(
                    low=np.array([0], dtype=np.int32),
                    high=np.array([8], dtype=np.int32),
                    dtype=np.int32,
                ),
            }
        )

        self.action_space = gym.spaces.Discrete(2)

        self.state = 0
        self.steps = 0

    def _observation(self):
        return {
            "state": np.array([self.state], dtype=np.int32),
            "step": np.array([self.steps], dtype=np.int32),
        }

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0

        if seed is None:
            self.state = 0
        else:
            self.state = int(seed % 3)

        observation = self._observation()

        info = {
            "seed": seed,
            "state": int(self.state),
        }

        return observation, info

    def step(self, action):
        if not self.action_space.contains(action):
            raise ValueError("invalid action")

        self.steps += 1

        if int(action) == 1:
            self.state += 1

        terminated = bool(self.state >= 6)
        truncated = bool(
            self.steps >= self.max_steps and not terminated
        )

        reward = float(
            1.0 if int(action) == 1 else 0.0
        )

        info = {
            "state": int(self.state),
            "step": int(self.steps),
        }

        return (
            self._observation(),
            reward,
            terminated,
            truncated,
            info,
        )

    def close(self):
        return None
