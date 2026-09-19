from __future__ import annotations

import gymnasium as gym
import numpy as np

from environment.env import CudaH01Environment, TOOLS


class CudaH01GymEnv(gym.Env):
    """Gymnasium adapter using the same underlying CUDA-H01 semantics."""

    metadata = {"render_modes": []}

    def __init__(self, seed: int = 0, max_steps: int = 100):
        super().__init__()
        self.engine = CudaH01Environment(seed=seed, max_steps=max_steps)
        self.action_space = gym.spaces.Discrete(len(TOOLS))
        self.observation_space = gym.spaces.Dict({
            "step": gym.spaces.Discrete(max_steps + 1),
            "terminal": gym.spaces.Discrete(2),
            "success": gym.spaces.Discrete(2),
        })

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.engine.seed = int(seed)
        obs = self.engine.reset()
        return {
            "step": int(obs["step"]),
            "terminal": int(obs["terminal"]),
            "success": int(obs["success"]),
        }, obs

    def step(self, action):
        tool = TOOLS[int(action)]
        transition = self.engine.step({"tool": tool})
        obs = transition.observation
        return (
            {
                "step": int(obs["step"]),
                "terminal": int(obs["terminal"]),
                "success": int(obs["success"]),
            },
            float(transition.reward.total),
            bool(transition.done and obs["success"]),
            bool(transition.done and not obs["success"]),
            transition.info,
        )

    def close(self):
        self.engine.close()

