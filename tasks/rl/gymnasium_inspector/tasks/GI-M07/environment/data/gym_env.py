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
                    high=np.array([8], dtype=np.int32),
                    dtype=np.int32,
                ),
                "progress": gym.spaces.Discrete(9),
            }
        )

        self.action_space = gym.spaces.Discrete(2)

        self.state = 0
        self.steps = 0

    def _observation(self):
        return {
            "state": np.array([self.state], dtype=np.int32),
            "progress": int(self.state),
        }

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.state = 0
        self.steps = 0

        observation = self._observation()
        info = {
            "state": self.state,
            "steps": self.steps,
        }

        return observation, info

    def step(self, action):
        action = int(action)

        if not self.action_space.contains(action):
            raise ValueError("action outside action_space")

        self.steps += 1

        if action == 1:
            self.state += 1

        terminated = bool(self.state >= 8)
        truncated = bool(
            self.steps >= self.max_steps
            and not terminated
        )

        reward = float(
            1.0
            + (0.25 if action == 1 else 0.0)
            + (2.0 if terminated else 0.0)
            - (0.25 if truncated else 0.0)
        )

        info = {
            "state": self.state,
            "steps": self.steps,
            "action_valid": True,
        }

        observation = self._observation()

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
