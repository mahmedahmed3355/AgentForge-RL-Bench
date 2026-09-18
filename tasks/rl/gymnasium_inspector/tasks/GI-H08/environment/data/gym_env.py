from __future__ import annotations

import gymnasium as gym
import numpy as np


class ActionEfficientCompletionEnv(gym.Env):
    """
    GI-H08: Action Efficient Completion.

    Several trajectories can reach the terminal state, but unnecessary
    actions consume the efficiency budget. The intended reasoning task
    is to inspect the state/options, choose an efficient valid path,
    and verify before finishing.
    """

    metadata = {"render_modes": []}

    INSPECT = 0
    FAST_EXECUTE = 1
    SAFE_EXECUTE = 2
    WASTE_ACTION = 3
    FINISH = 4

    TARGET_PROGRESS = 4
    EFFICIENCY_BUDGET = 6

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode

        self.action_space = gym.spaces.Discrete(5)

        # [progress, spent_cost, inspected, verified]
        self.observation_space = gym.spaces.Box(
            low=np.array([0, 0, 0, 0], dtype=np.int32),
            high=np.array(
                [
                    self.TARGET_PROGRESS,
                    self.EFFICIENCY_BUDGET,
                    1,
                    1,
                ],
                dtype=np.int32,
            ),
            dtype=np.int32,
        )

        self.progress = 0
        self.spent_cost = 0
        self.inspected = False
        self.verified = False
        self.step_count = 0
        self.terminated = False
        self.truncated = False
        self.trajectory = []

    def _observation(self):
        return np.array(
            [
                self.progress,
                self.spent_cost,
                int(self.inspected),
                int(self.verified),
            ],
            dtype=np.int32,
        )

    def _info(self, stage, event, reward, valid=True):
        return {
            "stage": stage,
            "event": event,
            "progress": int(self.progress),
            "spent_cost": int(self.spent_cost),
            "inspected": bool(self.inspected),
            "verified": bool(self.verified),
            "valid_action": bool(valid),
            "reward": float(reward),
            "efficiency_budget": self.EFFICIENCY_BUDGET,
            "remaining_budget": max(
                0, self.EFFICIENCY_BUDGET - self.spent_cost
            ),
        }

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.progress = 0
        self.spent_cost = 0
        self.inspected = False
        self.verified = False
        self.step_count = 0
        self.terminated = False
        self.truncated = False
        self.trajectory = []

        return self._observation(), {
            "stage": "inspect",
            "event": "reset",
            "progress": 0,
            "spent_cost": 0,
            "inspected": False,
            "verified": False,
        }

    def step(self, action):
        if self.terminated or self.truncated:
            raise RuntimeError("step() called after episode termination")

        action = int(action)
        self.step_count += 1

        reward = 0.0
        valid = True
        stage = "execute"
        event = "noop"

        if action == self.INSPECT:
            self.inspected = True
            reward = 0.5
            stage = "inspect"
            event = "inspection"

        elif action == self.FAST_EXECUTE:
            if not self.inspected:
                valid = False
                reward = -1.0
                event = "execute_before_inspection"
            elif self.progress >= self.TARGET_PROGRESS:
                valid = False
                reward = -0.5
                event = "already_complete"
            elif self.spent_cost + 2 > self.EFFICIENCY_BUDGET:
                valid = False
                reward = -2.0
                event = "efficiency_budget_exceeded"
            else:
                self.progress = min(
                    self.TARGET_PROGRESS,
                    self.progress + 2,
                )
                self.spent_cost += 2
                reward = 2.0
                event = "efficient_execution"

        elif action == self.SAFE_EXECUTE:
            if not self.inspected:
                valid = False
                reward = -1.0
                event = "execute_before_inspection"
            elif self.progress >= self.TARGET_PROGRESS:
                valid = False
                reward = -0.5
                event = "already_complete"
            elif self.spent_cost + 1 > self.EFFICIENCY_BUDGET:
                valid = False
                reward = -2.0
                event = "efficiency_budget_exceeded"
            else:
                self.progress = min(
                    self.TARGET_PROGRESS,
                    self.progress + 1,
                )
                self.spent_cost += 1
                reward = 1.0
                event = "valid_execution"

        elif action == self.WASTE_ACTION:
            if self.spent_cost + 1 > self.EFFICIENCY_BUDGET:
                valid = False
                reward = -2.0
                event = "waste_exceeds_budget"
            else:
                self.spent_cost += 1
                reward = -1.0
                event = "unnecessary_action"

        elif action == self.FINISH:
            stage = "verify"

            if self.progress == self.TARGET_PROGRESS:
                self.verified = True
                reward = 5.0
                event = "terminal_verified"
                self.terminated = True
            else:
                valid = False
                reward = -3.0
                event = "finish_before_completion"

        else:
            raise ValueError(f"Invalid action: {action}")

        if not self.terminated and self.spent_cost > self.EFFICIENCY_BUDGET:
            self.truncated = True
            event = "efficiency_timeout"
            reward -= 2.0

        info = self._info(
            stage=stage,
            event=event,
            reward=reward,
            valid=valid,
        )

        self.trajectory.append(
            {
                "action": action,
                "progress": self.progress,
                "spent_cost": self.spent_cost,
                "inspected": self.inspected,
                "verified": self.verified,
                "reward": float(reward),
                "event": event,
                "stage": stage,
                "valid_action": valid,
                "terminated": self.terminated,
                "truncated": self.truncated,
            }
        )

        return (
            self._observation(),
            float(reward),
            bool(self.terminated),
            bool(self.truncated),
            info,
        )
