from __future__ import annotations

import gymnasium as gym
import numpy as np


class GymInspectorEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, max_steps: int = 7):
        super().__init__()

        self.max_steps = int(max_steps)

        self.observation_space = gym.spaces.Dict(
            {
                "position": gym.spaces.Box(
                    low=np.array([0], dtype=np.int32),
                    high=np.array([7], dtype=np.int32),
                    dtype=np.int32,
                ),
                "remaining": gym.spaces.Discrete(8),
            }
        )

        self.action_space = gym.spaces.Discrete(2)

        self.position = 0
        self.steps = 0

    def _observation(self):
        remaining = max(0, self.max_steps - self.steps)

        return {
            "position": np.array(
                [self.position],
                dtype=np.int32,
            ),
            "remaining": int(remaining),
        }

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.position = 0
        self.steps = 0

        return (
            self._observation(),
            {
                "position": self.position,
                "steps": self.steps,
            },
        )

    def step(self, action):
        action = int(action)

        if not self.action_space.contains(action):
            raise ValueError("action outside action_space")

        self.steps += 1

        if action == 1:
            self.position += 1

        terminated = bool(self.position >= 7)

        truncated = bool(
            self.steps >= self.max_steps
            and not terminated
        )

        progress_reward = 1.0 if action == 1 else 0.25
        completion_reward = 2.0 if terminated else 0.0
        truncation_penalty = -0.25 if truncated else 0.0

        reward = float(
            progress_reward
            + completion_reward
            + truncation_penalty
        )

        info = {
            "position": self.position,
            "steps": self.steps,
            "action_valid": True,
            "terminated": terminated,
            "truncated": truncated,
        }

        observation = self._observation()

        return (
            observation,
            reward,
            terminated,
            truncated,
            info,
        )

    def render(self):
        return None

    def close(self):
        return None
