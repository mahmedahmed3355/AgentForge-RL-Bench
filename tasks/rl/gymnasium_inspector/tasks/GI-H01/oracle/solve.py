from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "environment" / "data"

sys.path.insert(0, str(DATA))

from gym_env import GymInspectorEnv


def main() -> int:
    env = GymInspectorEnv(max_steps=6)

    observation, info = env.reset(seed=123)

    actions = [1, 1, 1, 1, 1, 1]

    rewards = []
    terminated = False
    truncated = False
    trajectory = []

    for action in actions:
        (
            observation,
            reward,
            terminated,
            truncated,
            info,
        ) = env.step(action)

        rewards.append(float(reward))

        trajectory.append(
            {
                "action": int(action),
                "step": int(info["step"]),
                "position": int(info["position"]),
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
        "success": bool(
            terminated and not truncated
        ),
        "steps": len(rewards),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "final_position": int(
            observation["position"][0]
        ),
        "final_step": int(
            observation["step"][0]
        ),
        "reward_total": float(sum(rewards)),
        "observation_valid": observation_valid,
        "trajectory_length": len(trajectory),
    }

    print(json.dumps(result, indent=2))

    env.close()

    return 0 if (
        result["success"]
        and result["observation_valid"]
        and result["trajectory_length"] == 6
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
