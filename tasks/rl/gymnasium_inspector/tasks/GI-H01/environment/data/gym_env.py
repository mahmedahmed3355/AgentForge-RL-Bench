from __future__ import annotations

import gymnasium as gym
import numpy as np


class GymInspectorEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, max_steps: int = 6):
        super().__init__()

        self.max_steps = int(max_steps)

        self.observation_space = gym.spaces.Dict(
            {
                "position": gym.spaces.Box(
                    low=np.array([0], dtype=np.int32),
                    high=np.array([self.max_steps], dtype=np.int32),
                    dtype=np.int32,
                ),
                "step": gym.spaces.Box(
                    low=np.array([0], dtype=np.int32),
                    high=np.array([self.max_steps], dtype=np.int32),
                    dtype=np.int32,
                ),
            }
        )

        self.action_space = gym.spaces.Discrete(2)

        self.position = 0
        self.steps = 0
        self._episode_done = False

    def _observation(self):
        return {
            "position": np.array(
                [self.position],
                dtype=np.int32,
            ),
            "step": np.array(
                [self.steps],
                dtype=np.int32,
            ),
        }

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.position = 0
        self.steps = 0
        self._episode_done = False

        observation = self._observation()

        info = {
            "position": int(self.position),
            "step": int(self.steps),
        }

        return observation, info

    def step(self, action):
        if not self.action_space.contains(action):
            raise ValueError("action must belong to action_space")

        if self._episode_done:
            raise RuntimeError(
                "step() called after episode termination or truncation"
            )

        self.steps += 1

        if int(action) == 1:
            self.position += 1
        else:
            self.position = max(0, self.position - 1)

        terminated = bool(self.position >= self.max_steps)

        truncated = bool(
            self.steps >= self.max_steps
            and not terminated
        )

        if terminated:
            self.position = self.max_steps
            reward = 1.0 + 0.5 + 2.0
        elif truncated:
            reward = -1.0 + 0.25
        elif int(action) == 1:
            reward = 1.0 + 0.25
        else:
            reward = -0.25

        reward = float(reward)

        self._episode_done = bool(
            terminated or truncated
        )

        observation = self._observation()

        info = {
            "position": int(self.position),
            "step": int(self.steps),
            "episode_done": bool(self._episode_done),
        }

        return (
            observation,
            reward,
            bool(terminated),
            bool(truncated),
            info,
        )

    def render(self):
        return None

    def close(self):
        return None
