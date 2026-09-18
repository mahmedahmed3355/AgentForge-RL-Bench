from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class ComposedInspectorEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()

        self.action_space = spaces.Discrete(4)

        self.observation_space = spaces.Dict(
            {
                "step": spaces.Discrete(16),
                "stage": spaces.Discrete(5),
                "progress": spaces.Discrete(5),
                "diagnosis": spaces.Discrete(4),
                "verified": spaces.Discrete(2),
                "query_count": spaces.Discrete(8),
            }
        )

        self.step_count = 0
        self.stage = 0
        self.progress = 0
        self.diagnosis = -1
        self.verified = False
        self.query_count = 0
        self.history_seen = False
        self.diagnostics_seen = False
        self.state_seen = False
        self.correlated = False
        self.acted = False
        self.terminated = False

        self._target = 2

    def _obs(self):
        return {
            "step": int(self.step_count),
            "stage": int(self.stage),
            "progress": int(self.progress),
            "diagnosis": int(self.diagnosis),
            "verified": int(self.verified),
            "query_count": int(self.query_count),
        }

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)

        self.step_count = 0
        self.stage = 0
        self.progress = 0
        self.diagnosis = -1
        self.verified = False
        self.query_count = 0
        self.history_seen = False
        self.diagnostics_seen = False
        self.state_seen = False
        self.correlated = False
        self.acted = False
        self.terminated = False

        return self._obs(), {
            "inspector_contract": True,
            "public_information": True,
        }

    def step(self, action: int):
        if self.terminated:
            raise RuntimeError("step() called after termination")

        action = int(action)
        self.step_count += 1
        reward = 0.0
        event = "invalid"

        if action == 0:
            self.stage = 0
            self.query_count += 1

            if self.query_count == 1:
                self.state_seen = True
                event = "state_inspected"
                reward = 0.5
            elif self.query_count == 2:
                self.history_seen = True
                event = "history_inspected"
                reward = 0.5
            elif self.query_count == 3:
                self.diagnostics_seen = True
                event = "diagnostics_inspected"
                reward = 0.5
            else:
                event = "unnecessary_inspection"
                reward = -0.5

        elif action == 1:
            self.stage = 1

            if self.state_seen and self.history_seen and self.diagnostics_seen:
                self.correlated = True
                self.diagnosis = self._target
                self.progress = 1
                event = "minimum_sufficient_correlation"
                reward = 2.0
            else:
                event = "incomplete_correlation"
                reward = -1.0

        elif action == 2:
            self.stage = 3

            if self.correlated and self.diagnosis == self._target:
                self.acted = True
                self.progress = 2
                event = "correct_action"
                reward = 2.5
            else:
                event = "wrong_inference"
                reward = -2.0

        elif action == 3:
            self.stage = 4

            if self.acted and self.correlated and self.diagnosis == self._target:
                self.verified = True
                self.progress = 4
                self.terminated = True
                event = "terminal_verified"
                reward = 5.0
            else:
                event = "terminal_rejected"
                reward = -2.0

        else:
            reward = -2.0

        terminated = bool(self.terminated)
        truncated = False

        info = {
            "event": event,
            "stage": int(self.stage),
            "state_seen": bool(self.state_seen),
            "history_seen": bool(self.history_seen),
            "diagnostics_seen": bool(self.diagnostics_seen),
            "correlated": bool(self.correlated),
            "diagnosis": int(self.diagnosis),
            "verified": bool(self.verified),
            "query_count": int(self.query_count),
            "public_information_only": True,
            "valid_action": event not in {
                "wrong_inference",
                "terminal_rejected",
                "incomplete_correlation",
                "unnecessary_inspection",
            },
        }

        return self._obs(), float(reward), terminated, truncated, info


__all__ = ["ComposedInspectorEnv"]
