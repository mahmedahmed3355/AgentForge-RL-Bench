from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from environment.data.gym_env import AmbiguousFailureDiagnosisEnv


def verify():
    env = AmbiguousFailureDiagnosisEnv()

    obs, info = env.reset(seed=123)

    if not env.observation_space.contains(obs):
        return {"verified": False, "reason": "invalid_reset_observation"}

    expected = [0, 1, 2, 3]

    for action in expected:
        obs, reward, terminated, truncated, info = env.step(action)

        if not env.observation_space.contains(obs):
            return {"verified": False, "reason": "invalid_observation"}

        if not isinstance(reward, float):
            return {"verified": False, "reason": "reward_not_float"}

        if not isinstance(info, dict):
            return {"verified": False, "reason": "info_not_dict"}

        if terminated and truncated:
            return {"verified": False, "reason": "dual_terminal_flags"}

    verified = (
        terminated is True
        and truncated is False
        and info["diagnosis"] == env.ROOT_FAILURE
        and info["recovered"] is True
        and info["stage"] == "verify"
    )

    return {"verified": bool(verified)}


if __name__ == "__main__":
    print(verify())
