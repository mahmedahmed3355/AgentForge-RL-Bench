from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np


class BranchingIrreversibleEnv(gym.Env):
    """
    GI-H03:
    Branching Irreversible Plan.

    The environment contains two branches:

    branch 0:
        commit cost = 3
        execution cost = 4
        completion value = 8
        globally valid

    branch 1:
        commit cost = 5
        execution cost = 6
        completion value = 8
        locally attractive but resource-infeasible

    The commitment is irreversible.
    """

    metadata = {"render_modes": []}

    INSPECT = 0
    COMMIT = 1
    EXECUTE = 2
    FINISH = 3

    def __init__(self, max_steps: int = 10):
        super().__init__()

        self.max_steps = int(max_steps)

        self.observation_space = gym.spaces.Dict(
            {
                "state": gym.spaces.Discrete(6),
                "resources": gym.spaces.Box(
                    low=np.array([0], dtype=np.int32),
                    high=np.array([10], dtype=np.int32),
                    dtype=np.int32,
                ),
                "committed": gym.spaces.Discrete(3),
                "executed": gym.spaces.Discrete(2),
                "step": gym.spaces.Discrete(self.max_steps + 1),
            }
        )

        self.action_space = gym.spaces.Discrete(4)

        self.state = 0
        self.resources = 10
        self.committed = 2
        self.executed = 0
        self.step_count = 0
        self.history: list[dict[str, Any]] = []
        self.np_random = None

    def _observation(self) -> dict[str, Any]:
        obs = {
            "state": int(self.state),
            "resources": np.array([self.resources], dtype=np.int32),
            "committed": int(self.committed),
            "executed": int(self.executed),
            "step": int(self.step_count),
        }

        assert self.observation_space.contains(obs)
        return obs

    def _record(
        self,
        action: int,
        reward: float,
        terminated: bool,
        truncated: bool,
    ) -> None:
        self.history.append(
            {
                "step": int(self.step_count),
                "action": int(action),
                "state": int(self.state),
                "resources": int(self.resources),
                "committed": int(self.committed),
                "executed": int(self.executed),
                "reward": float(reward),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
            }
        )

    def reset(self, *, seed: int | None = None, options=None):
        super().reset(seed=seed)

        self.state = 0
        self.resources = 10
        self.committed = 2
        self.executed = 0
        self.step_count = 0
        self.history = []

        info = {
            "stage": "inspect",
            "branches": {
                0: {
                    "commit_cost": 3,
                    "execution_cost": 4,
                    "valid": True,
                },
                1: {
                    "commit_cost": 5,
                    "execution_cost": 6,
                    "valid": False,
                },
            },
        }

        return self._observation(), info

    def step(self, action: int):
        if not self.action_space.contains(action):
            raise ValueError("action must belong to action_space")

        action = int(action)

        reward = 0.0
        terminated = False
        truncated = False
        info: dict[str, Any] = {}

        self.step_count += 1

        if action == self.INSPECT:
            self.state = 1
            reward = 0.5
            info["inspection"] = "branch_constraints_available"

        elif action == self.COMMIT:
            if self.state < 1:
                reward = -1.0
                info["error"] = "must_inspect_before_commit"

            elif self.committed != 2:
                reward = -1.0
                info["error"] = "commitment_is_irreversible"

            else:
                # Deterministic globally valid branch.
                self.committed = 0
                self.resources -= 3
                self.state = 2
                reward = 2.0
                info["committed_branch"] = 0
                info["irreversible"] = True

        elif action == self.EXECUTE:
            if self.committed == 2:
                reward = -1.0
                info["error"] = "must_commit_before_execute"

            elif self.executed:
                reward = -0.5
                info["error"] = "already_executed"

            elif self.resources < 4:
                reward = -2.0
                info["error"] = "resource_exhaustion"

            else:
                self.resources -= 4
                self.executed = 1
                self.state = 4
                reward = 3.0
                info["execution"] = "completed"

        elif action == self.FINISH:
            if self.committed == 0 and self.executed == 1:
                self.state = 5
                terminated = True
                reward = 4.0
                info["success"] = True
            else:
                reward = -1.0
                info["error"] = "plan_incomplete"

        if self.step_count >= self.max_steps and not terminated:
            truncated = True

        if terminated and truncated:
            raise RuntimeError("terminated and truncated cannot both be true")

        observation = self._observation()

        if not self.observation_space.contains(observation):
            raise RuntimeError("invalid observation")

        if not isinstance(reward, float):
            reward = float(reward)

        info["trajectory_length"] = len(self.history) + 1

        self._record(action, reward, terminated, truncated)

        info = dict(info) if isinstance(info, dict) else {}

        info["stage"] = (

            "inspect"

            if self.step_count == 1

            else info.get("stage", "execute")

        )

        return observation, float(reward), bool(terminated), bool(truncated), info

    def close(self):
        return None
