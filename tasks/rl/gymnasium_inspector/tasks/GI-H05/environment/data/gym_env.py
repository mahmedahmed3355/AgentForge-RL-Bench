from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class AmbiguousFailureDiagnosisEnv(gym.Env):
    """
    Deterministic diagnosis environment.

    The visible symptom is compatible with multiple failure modes.
    Diagnostic evidence identifies the active root cause.
    The valid trajectory is:

        inspect -> diagnose(correct root) -> recover -> finish
    """

    metadata = {"render_modes": []}

    ACTION_INSPECT = 0
    ACTION_DIAGNOSE = 1
    ACTION_RECOVER = 2
    ACTION_FINISH = 3

    ROOT_FAILURE = 2
    SYMPTOM = 1
    DIAGNOSTIC_EVIDENCE = 7

    def __init__(self, render_mode=None):
        super().__init__()

        self.action_space = spaces.Discrete(4)

        # [stage, symptom, evidence, recovered]
        self.observation_space = spaces.MultiDiscrete(
            np.array([5, 3, 10, 2], dtype=np.int64)
        )

        self.render_mode = render_mode

        self.stage = 0
        self.symptom = self.SYMPTOM
        self.evidence = 0
        self.diagnosis = -1
        self.recovered = 0
        self.step_count = 0
        self.terminated = False
        self.truncated = False

        self.max_steps = 8

    def _observation(self):
        return np.asarray(
            [
                self.stage,
                self.symptom,
                self.evidence,
                self.recovered,
            ],
            dtype=np.int64,
        )

    def _info(self):
        stages = {
            0: "inspect",
            1: "compare",
            2: "diagnose",
            3: "recover",
            4: "verify",
        }

        return {
            "stage": stages.get(self.stage, "unknown"),
            "root_failure": self.ROOT_FAILURE,
            "symptom": self.symptom,
            "diagnostic_evidence": self.evidence,
            "diagnosis": self.diagnosis,
            "recovered": bool(self.recovered),
            "step_count": self.step_count,
        }

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.stage = 0
        self.symptom = self.SYMPTOM
        self.evidence = 0
        self.diagnosis = -1
        self.recovered = 0
        self.step_count = 0
        self.terminated = False
        self.truncated = False

        return self._observation(), self._info()

    def step(self, action):
        if not self.action_space.contains(action):
            raise ValueError("action outside action_space")

        if self.terminated or self.truncated:
            raise RuntimeError("step() called after episode ended")

        self.step_count += 1

        reward = 0.0

        if action == self.ACTION_INSPECT:
            self.evidence = self.DIAGNOSTIC_EVIDENCE
            self.stage = 1
            reward = 0.5

        elif action == self.ACTION_DIAGNOSE:
            if self.evidence == self.DIAGNOSTIC_EVIDENCE:
                self.diagnosis = self.ROOT_FAILURE
                self.stage = 2
                reward = 2.5
            else:
                self.diagnosis = 0
                reward = -1.0

        elif action == self.ACTION_RECOVER:
            if self.diagnosis == self.ROOT_FAILURE:
                self.recovered = 1
                self.stage = 3
                reward = 2.5
            else:
                reward = -1.5

        elif action == self.ACTION_FINISH:
            if self.recovered == 1 and self.diagnosis == self.ROOT_FAILURE:
                self.stage = 4
                self.terminated = True
                reward = 4.0
            else:
                reward = -1.0

        if not self.terminated and self.step_count >= self.max_steps:
            self.truncated = True

        if self.terminated and self.truncated:
            raise AssertionError("terminated and truncated cannot both be true")

        observation = self._observation()
        info = self._info()

        return (
            observation,
            float(reward),
            bool(self.terminated),
            bool(self.truncated),
            info,
        )
