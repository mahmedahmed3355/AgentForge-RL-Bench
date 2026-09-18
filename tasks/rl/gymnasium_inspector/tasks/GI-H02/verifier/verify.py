from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "environment" / "data"

sys.path.insert(0, str(DATA))

from gym_env import GymInspectorEnv


def main() -> int:
    env = GymInspectorEnv(max_steps=6)

    observation, info = env.reset(seed=123)

    verified = True

    if not isinstance(info, dict):
        verified = False

    for action in [1, 1, 1, 1, 1, 1]:
        if not env.action_space.contains(action):
            verified = False
            break

        observation, reward, terminated, truncated, info = env.step(action)

        if not env.observation_space.contains(observation):
            verified = False

        if not isinstance(reward, float):
            verified = False

        if not isinstance(terminated, bool):
            verified = False

        if not isinstance(truncated, bool):
            verified = False

        if not isinstance(info, dict):
            verified = False

        if terminated and truncated:
            verified = False

        if terminated or truncated:
            break

    print({"verified": bool(verified)})

    env.close()
    return 0 if verified else 1


if __name__ == "__main__":
    raise SystemExit(main())
