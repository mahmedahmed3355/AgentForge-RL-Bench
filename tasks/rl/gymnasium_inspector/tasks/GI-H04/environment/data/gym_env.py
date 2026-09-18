from __future__ import annotations

import gymnasium as gym
import numpy as np


class DependencyGraphEnv(gym.Env):
    """
    GI-H04: Dependency Graph Completion.

    Public dependency graph:
        0 -> 2 -> 4
        1 -> 3 -> 4
        4 -> 5

    Valid topological completion:
        0, 1, 2, 3, 4, 5

    The environment exposes the current completed-node state while
    requiring the agent to respect direct and indirect prerequisites.
    """

    metadata = {"render_modes": []}

    def __init__(self, max_steps: int = 8):
        super().__init__()

        self.max_steps = int(max_steps)

        self.action_space = gym.spaces.Discrete(6)

        self.observation_space = gym.spaces.Dict(
            {
                "completed": gym.spaces.MultiBinary(6),
                "progress": gym.spaces.Discrete(7),
                "step": gym.spaces.Discrete(self.max_steps + 1),
            }
        )

        self._dependencies = {
            0: (),
            1: (),
            2: (0,),
            3: (1,),
            4: (2, 3),
            5: (4,),
        }

        self.completed = np.zeros(6, dtype=np.int8)
        self.step_count = 0
        self.np_random = None

    def _observation(self):
        return {
            "completed": self.completed.copy(),
            "progress": int(self.completed.sum()),
            "step": int(self.step_count),
        }

    def _info(self, stage: str, valid: bool = True):
        return {
            "stage": stage,
            "valid_action": bool(valid),
            "completed_count": int(self.completed.sum()),
            "remaining": int(6 - self.completed.sum()),
        }

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.completed = np.zeros(6, dtype=np.int8)
        self.step_count = 0

        observation = self._observation()
        info = self._info("inspect")

        return observation, info

    def step(self, action):
        action = int(action)

        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")

        self.step_count += 1

        # Completing an already completed node is invalid and does not
        # modify the state.
        if self.completed[action]:
            reward = -1.0
            valid = False

        else:
            prerequisites = self._dependencies[action]

            if all(self.completed[p] for p in prerequisites):
                self.completed[action] = 1
                reward = 1.0
                valid = True
            else:
                reward = -0.5
                valid = False

        completed_count = int(self.completed.sum())

        terminated = bool(completed_count == 6)
        truncated = bool(
            self.step_count >= self.max_steps and not terminated
        )

        if terminated:
            reward += 4.0
            stage = "finish"
        elif truncated:
            stage = "timeout"
        elif valid:
            stage = "execute"
        else:
            stage = "reassess"

        observation = self._observation()
        info = self._info(stage, valid)

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
