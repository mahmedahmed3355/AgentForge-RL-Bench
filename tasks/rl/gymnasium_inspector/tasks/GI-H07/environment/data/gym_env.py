from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class MultiConditionVerificationEnv(gym.Env):
    """
    GI-H07 environment.

    The task deliberately separates:
      - visible conditions,
      - dependency repair,
      - validation,
      - revalidation.

    Terminal success is only valid after all terminal invariants are true
    and a validation action has occurred after the final state-changing
    resolve action.
    """

    metadata = {"render_modes": []}

    INSPECT = 0
    VALIDATE = 1
    RESOLVE = 2
    FINISH = 3

    def __init__(self, render_mode=None):
        super().__init__()

        self.render_mode = render_mode

        self.action_space = spaces.Discrete(4)

        # [base_ready, dependency_ready, integrity_ok,
        #  validated, revalidated, step]
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0, 0], dtype=np.int32),
            high=np.array([1, 1, 1, 1, 1, 20], dtype=np.int32),
            dtype=np.int32,
        )

        self.base_ready = True
        self.dependency_ready = False
        self.integrity_ok = True
        self.validated = False
        self.revalidated = False
        self.step_count = 0
        self.terminated = False
        self.truncated = False
        self.history: list[dict[str, Any]] = []

    def _observation(self):
        return np.array(
            [
                int(self.base_ready),
                int(self.dependency_ready),
                int(self.integrity_ok),
                int(self.validated),
                int(self.revalidated),
                self.step_count,
            ],
            dtype=np.int32,
        )

    def _requirements_valid(self) -> bool:
        return (
            self.base_ready
            and self.dependency_ready
            and self.integrity_ok
        )

    def _info(self, reward_components, event):
        return {
            "base_ready": self.base_ready,
            "dependency_ready": self.dependency_ready,
            "integrity_ok": self.integrity_ok,
            "validated": self.validated,
            "revalidated": self.revalidated,
            "stage": self._stage(),
            "event": event,
            "reward_components": dict(reward_components),
            "requirements_valid": self._requirements_valid(),
        }

    def _stage(self):
        if self.terminated:
            return "finish"
        if self.revalidated:
            return "finish"
        if self.dependency_ready:
            return "revalidate"
        if self.validated:
            return "resolve"
        return "inspect"

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.base_ready = True
        self.dependency_ready = False
        self.integrity_ok = True
        self.validated = False
        self.revalidated = False
        self.step_count = 0
        self.terminated = False
        self.truncated = False
        self.history = []

        observation = self._observation()
        info = {
            "base_ready": True,
            "dependency_ready": False,
            "integrity_ok": True,
            "validated": False,
            "revalidated": False,
            "stage": "inspect",
            "event": "reset",
            "requirements_valid": False,
        }

        return observation, info

    def step(self, action):
        if self.terminated or self.truncated:
            raise RuntimeError("step() called after episode termination")

        action = int(action)
        if not self.action_space.contains(action):
            raise ValueError(f"invalid action: {action}")

        self.step_count += 1

        reward = 0.0
        components = {
            "progress": 0.0,
            "correctness": 0.0,
            "terminal": 0.0,
            "penalty": 0.0,
        }

        event = "no_op"
        terminated = False

        if action == self.INSPECT:
            reward = 0.5
            components["progress"] = 0.5
            event = "inspection"

        elif action == self.VALIDATE:
            if self._requirements_valid():
                reward = 1.5
                components["correctness"] = 1.5
                self.validated = True
                event = "validated_requirements"
            else:
                reward = 1.5
                components["correctness"] = 1.5
                self.validated = True
                event = "validated_missing_dependency"

            # Validation before repair is never sufficient for terminal
            # success. A later resolve invalidates the previous validation.
            self.revalidated = False

        elif action == self.RESOLVE:
            if not self.validated:
                reward = -1.0
                components["penalty"] = -1.0
                event = "resolve_without_validation"
            else:
                self.dependency_ready = True

                # Resolving the dependency exposes the requirement that
                # the complete state must be validated again.
                self.integrity_ok = True
                self.revalidated = False

                reward = 2.5
                components["progress"] = 1.5
                components["correctness"] = 1.0
                event = "dependency_resolved"

        elif action == self.FINISH:
            if self._requirements_valid() and self.revalidated:
                reward = 5.0
                components["terminal"] = 5.0
                event = "terminal_verified"
                terminated = True
                self.terminated = True
            else:
                reward = -1.0
                components["penalty"] = -1.0
                event = "terminal_rejected"

        info = self._info(components, event)

        record = {
            "action": action,
            "step": self.step_count,
            "base_ready": self.base_ready,
            "dependency_ready": self.dependency_ready,
            "integrity_ok": self.integrity_ok,
            "validated": self.validated,
            "revalidated": self.revalidated,
            "requirements_valid": self._requirements_valid(),
            "event": event,
            "reward": float(reward),
            "terminated": terminated,
            "truncated": False,
        }

        # A validation performed after dependency resolution is the
        # revalidation boundary.
        if action == self.VALIDATE and self.dependency_ready:
            self.revalidated = True
            record["revalidated"] = True
            info["revalidated"] = True
            info["stage"] = "finish"
            info["event"] = "revalidated_all_conditions"

        self.history.append(record)

        return (
            self._observation(),
            float(reward),
            terminated,
            False,
            info,
        )

    def render(self):
        return self._observation().copy()

    def close(self):
        return None
