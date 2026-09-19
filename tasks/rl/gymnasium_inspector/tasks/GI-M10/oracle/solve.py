from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "environment" / "data"

sys.path.insert(0, str(DATA))

from gym_env import GymInspectorEnv


def main() -> int:
    env = GymInspectorEnv(max_steps=8)

    observation, info = env.reset(seed=123)

    total_reward = 0.0
    trajectory = []

    terminated = False
    truncated = False

    actions = [1, 1, 1, 1, 1, 1]

    for action in actions:
        (
            observation,
            reward,
            terminated,
            truncated,
            info,
        ) = env.step(action)

        total_reward += float(reward)

        trajectory.append(
            {
                "action": int(action),
                "state": int(observation["state"][0]),
                "step": int(observation["step"][0]),
                "reward": float(reward),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
            }
        )

        if terminated or truncated:
            break

    observation_valid = bool(
        env.observation_space.contains(observation)
    )

    result = {
        "success": bool(terminated and not truncated),
        "steps": len(trajectory),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "final_state": int(observation["state"][0]),
        "reward_total": float(total_reward),
        "observation_valid": observation_valid,
        "trajectory_length": len(trajectory),
    }

    print(json.dumps(result, indent=2))

    env.close()

    return 0 if result["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
