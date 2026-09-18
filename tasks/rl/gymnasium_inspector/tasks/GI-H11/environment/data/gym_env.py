from __future__ import annotations

import gymnasium as gym
from gymnasium import spaces


class UnseenParameterCompositionEnv(gym.Env):
    """
    GI-H11:
    Unseen Parameter Composition.

    The agent must derive behavior from public constraints rather than
    memorizing a previously seen configuration.
    """

    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()

        # Publicly inspectable parameter composition.
        self.alpha = 3
        self.beta = 4
        self.limit = 12

        # Invariant:
        # result = alpha * 2 + beta
        self.expected_result = self.alpha * 2 + self.beta

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Dict(
            {
                "alpha": spaces.Discrete(16),
                "beta": spaces.Discrete(16),
                "limit": spaces.Discrete(32),
                "result": spaces.Discrete(32),
                "inspected": spaces.Discrete(2),
                "planned": spaces.Discrete(2),
                "executed": spaces.Discrete(2),
                "verified": spaces.Discrete(2),
                "constraint_ok": spaces.Discrete(2),
            }
        )

        self.reset()

    def _get_obs(self):
        return {
            "alpha": int(self.alpha),
            "beta": int(self.beta),
            "limit": int(self.limit),
            "result": int(self.result),
            "inspected": int(self.inspected),
            "planned": int(self.planned),
            "executed": int(self.executed),
            "verified": int(self.verified),
            "constraint_ok": int(self.constraint_ok),
        }

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.result = 0
        self.inspected = False
        self.planned = False
        self.executed = False
        self.verified = False
        self.constraint_ok = False

        self.step_count = 0

        return self._get_obs(), {
            "event": "reset",
            "parameters": {
                "alpha": self.alpha,
                "beta": self.beta,
                "limit": self.limit,
            },
        }

    def step(self, action: int):
        self.step_count += 1

        reward = 0.0
        terminated = False
        truncated = False
        event = "noop"

        if action == 0:
            self.inspected = True
            reward = 0.5
            event = "parameters_inspected"

        elif action == 1:
            if not self.inspected:
                reward = -1.0
                event = "plan_without_inspection"
            else:
                self.planned = True
                reward = 1.5
                event = "invariant_plan_derived"

        elif action == 2:
            if not self.planned:
                reward = -1.0
                event = "execute_without_plan"
            else:
                candidate = self.alpha * 2 + self.beta

                if candidate > self.limit:
                    reward = -2.0
                    self.constraint_ok = False
                    event = "constraint_violation"
                else:
                    self.result = candidate
                    self.executed = True
                    self.constraint_ok = True
                    reward = 2.5
                    event = "rule_consistent_execution"

        elif action == 3:
            if (
                self.inspected
                and self.planned
                and self.executed
                and self.constraint_ok
                and self.result == self.expected_result
            ):
                self.verified = True
                reward = 5.0
                terminated = True
                event = "terminal_verified"
            else:
                reward = -1.5
                event = "invalid_finish"

        else:
            reward = -1.0
            event = "invalid_action"

        return (
            self._get_obs(),
            reward,
            terminated,
            truncated,
            {
                "event": event,
                "expected_result": self.expected_result,
                "result": self.result,
                "constraint_ok": self.constraint_ok,
                "inspected": self.inspected,
                "planned": self.planned,
                "executed": self.executed,
                "verified": self.verified,
            },
        )
