from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "environment" / "data"

sys.path.insert(0, str(DATA))

from gym_env import GymInspectorEnv


def verify() -> bool:
    env = GymInspectorEnv(max_steps=6)

    try:
        observation, info = env.reset(seed=123)

        if not isinstance(info, dict):
            return False

        if not env.observation_space.contains(observation):
            return False

        if not isinstance(observation, type(env._observation())):
            return False

        terminated = False
        truncated = False

        for action in [1, 1, 1, 1, 1, 1]:
            if not env.action_space.contains(action):
                return False

            (
                observation,
                reward,
                terminated,
                truncated,
                info,
            ) = env.step(action)

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

        return bool(terminated and not truncated)

    finally:
        env.close()


def main() -> int:
    result = {"verified": bool(verify())}

    print(json.dumps(result, indent=2))

    return 0 if result["verified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
