from __future__ import annotations

import gymnasium as gym
import numpy as np


class GymInspectorEnv(gym.Env):
    """
    GI-H02: Interacting State Variables.

    The task models two coupled state variables.  The second variable
    depends on the current value of the first, so treating them as
    independent can produce an invalid transition.
    """

    metadata = {"render_modes": []}

    def __init__(self, max_steps: int = 6):
        super().__init__()

        self.max_steps = int(max_steps)

        self.observation_space = gym.spaces.Dict(
            {
                "position": gym.spaces.Box(
                    low=0,
                    high=6,
                    shape=(1,),
                    dtype=np.int64,
                ),
                "energy": gym.spaces.Box(
                    low=0,
                    high=12,
                    shape=(1,),
                    dtype=np.int64,
                ),
            }
        )

        self.action_space = gym.spaces.Discrete(2)

        self.position = 0
        self.energy = 6
        self.step_count = 0

    def _observation(self):
        return {
            "position": np.asarray([self.position], dtype=np.int64),
            "energy": np.asarray([self.energy], dtype=np.int64),
        }

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.position = 0
        self.energy = 6
        self.step_count = 0

        return self._observation(), {
            "state_consistent": True,
            "step": 0,
        }

    def step(self, action):
        action = int(action)

        if not self.action_space.contains(action):
            raise ValueError("action is outside action_space")

        if self.position >= 6:
            return (
                self._observation(),
                0.0,
                True,
                False,
                {
                    "state_consistent": True,
                    "already_complete": True,
                    "step": self.step_count,
                },
            )

        self.step_count += 1

        # Coupled transition:
        # action 1 advances position and consumes energy.
        # action 0 preserves position and restores one unit of energy.
        if action == 1:
            self.position += 1
            self.energy -= 1
        else:
            self.energy = min(12, self.energy + 1)

        # Causal invariant:
        # energy + position must remain equal to the initial budget.
        # The oracle uses action=1, so this invariant remains exact.
        state_consistent = (
            0 <= self.position <= 6
            and 0 <= self.energy <= 12
            and self.energy + self.position == 6
        )

        terminated = bool(self.position >= 6 and state_consistent)
        truncated = bool(
            self.step_count >= self.max_steps and not terminated
        )

        reward = float(
            1.0
            + (1.0 if state_consistent else -2.0)
            + (2.0 if terminated else 0.0)
        )

        info = {
            "state_consistent": bool(state_consistent),
            "step": int(self.step_count),
            "position": int(self.position),
            "energy": int(self.energy),
        }

        return (
            self._observation(),
            float(reward),
            terminated,
            truncated,
            info,
        )
