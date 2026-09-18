import numpy as np
import gymnasium as gym
from gymnasium import spaces

class ProgressPreservingRecoveryEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0], dtype=np.int32),
            high=np.array([4, 1, 1, 6], dtype=np.int32),
            dtype=np.int32,
        )
        self.progress = 0
        self.failed = False
        self.recovered = False
        self.step_count = 0

    def _obs(self):
        if self.failed and not self.recovered:
            stage = 2
        elif self.recovered and self.progress < 4:
            stage = 3
        elif self.progress >= 4:
            stage = 4
        elif self.step_count == 0:
            stage = 0
        else:
            stage = 1
        return np.array([self.progress, int(self.failed), int(self.recovered), stage], dtype=np.int32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.progress = 0
        self.failed = False
        self.recovered = False
        self.step_count = 0
        return self._obs(), {
            "progress": 0,
            "failed": False,
            "recovered": False,
            "stage": "inspect",
        }

    def step(self, action):
        action = int(action)
        reward = 0.0
        terminated = False
        truncated = False
        valid_action = True
        event = "none"

        if self.step_count >= 8:
            return self._obs(), 0.0, False, True, {
                "progress": self.progress,
                "failed": self.failed,
                "recovered": self.recovered,
                "stage": "timeout",
                "valid_action": False,
            }

        if action == 0:
            reward = 0.5
            event = "inspection"
        elif action == 1:
            if self.failed and not self.recovered:
                reward = -1.0
                valid_action = False
                event = "continue_blocked"
            elif self.progress < 2:
                self.progress += 1
                reward = 1.0
                event = "milestone"
                if self.progress == 2:
                    self.failed = True
                    event = "failure_after_milestone"
            elif self.recovered and self.progress < 4:
                self.progress += 1
                reward = 1.0
                event = "milestone"
            else:
                reward = -0.5
                valid_action = False
                event = "no_progress"
        elif action == 2:
            if self.failed and not self.recovered:
                self.recovered = True
                reward = 2.5
                event = "recovered_preserving_progress"
            else:
                reward = -1.0
                valid_action = False
                event = "invalid_recovery"
        elif action == 3:
            if self.progress == 4 and self.recovered:
                reward = 4.0
                terminated = True
                event = "terminal_success"
            else:
                reward = -2.0
                valid_action = False
                event = "premature_finish"
        else:
            reward = -1.0
            valid_action = False
            event = "invalid_action"

        self.step_count += 1

        if self.failed and not self.recovered:
            stage = "diagnose"
        elif self.recovered and self.progress < 4:
            stage = "continue"
        elif self.progress >= 4 and not terminated:
            stage = "verify"
        elif terminated:
            stage = "finish"
        elif self.progress < 2:
            stage = "inspect"
        else:
            stage = "protect_progress"

        info = {
            "progress": self.progress,
            "failed": self.failed,
            "recovered": self.recovered,
            "stage": stage,
            "event": event,
            "valid_action": valid_action,
            "step": self.step_count,
        }
        return self._obs(), reward, terminated, truncated, info
