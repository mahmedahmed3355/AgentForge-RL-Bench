from __future__ import annotations

import gymnasium as gym
import numpy as np


class GymInspectorEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, max_steps: int = 8):
        super().__init__()
        self.max_steps = int(max_steps)

        self.observation_space = gym.spaces.Box(
            low=np.array([0], dtype=np.int32),
            high=np.array([self.max_steps], dtype=np.int32),
            dtype=np.int32,
        )
        self.action_space = gym.spaces.Discrete(2)

        self.position = 0
        self.steps = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.position = 0
        self.steps = 0

        observation = np.array([self.position], dtype=np.int32)
        info = {
            "position": self.position,
            "steps": self.steps,
        }

        return observation, info

    def step(self, action):
        if not self.action_space.contains(action):
            raise ValueError("action must belong to action_space")

        action = int(action)
        self.steps += 1

        if action == 1:
            self.position += 1

        terminated = self.position >= self.max_steps
        truncated = self.steps >= self.max_steps and not terminated

        reward = float(
            1.0
            if action == 1
            else -0.25
        )

        if terminated:
            reward += 1.5

        observation = np.array([self.position], dtype=np.int32)

        if not self.observation_space.contains(observation):
            raise RuntimeError("invalid observation")

        info = {
            "position": self.position,
            "steps": self.steps,
            "action": action,
        }

        return (
            observation,
            float(reward),
            bool(terminated),
            bool(truncated),
            info,
        )
