from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "environment" / "data"

sys.path.insert(0, str(DATA))

from gym_env import GymInspectorEnv


def verify() -> bool:
    env = GymInspectorEnv(max_steps=7)

    observation, info = env.reset(seed=123)

    if not isinstance(info, dict):
        return False

    if not env.observation_space.contains(observation):
        return False

    for action in [1, 1, 1, 1, 1, 1, 1]:
        if not env.action_space.contains(action):
            return False

        observation, reward, terminated, truncated, info = env.step(action)

        if not env.observation_space.contains(observation):
            return False

        if not isinstance(reward, float):
            return False

        if not isinstance(terminated, bool):
            return False

        if not isinstance(truncated, bool):
            return False

        if not isinstance(info, dict):
            return False

        if terminated and truncated:
            return False

        if terminated or truncated:
            break

    env.close()

    return bool(terminated and not truncated)


if __name__ == "__main__":
    result = verify()
    print({"verified": result})
    raise SystemExit(0 if result else 1)
