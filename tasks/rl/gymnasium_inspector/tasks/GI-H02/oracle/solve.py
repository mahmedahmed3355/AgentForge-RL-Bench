from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "environment" / "data"

sys.path.insert(0, str(DATA))

from gym_env import GymInspectorEnv


def main() -> int:
    env = GymInspectorEnv(max_steps=6)

    observation, info = env.reset(seed=123)

    rewards = []
    terminated = False
    truncated = False
    trajectory = []

    for action in [1, 1, 1, 1, 1, 1]:
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
                "position": int(observation["position"][0]),
                "energy": int(observation["energy"][0]),
                "reward": float(reward),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
            }
        )

        if terminated or truncated:
            break

    observation_valid = bool(env.observation_space.contains(observation))

    result = {
        "success": bool(terminated and not truncated),
        "steps": len(rewards),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "final_position": int(observation["position"][0]),
        "final_energy": int(observation["energy"][0]),
        "reward_total": float(sum(rewards)),
        "observation_valid": observation_valid,
        "trajectory_length": len(trajectory),
    }

    output = ROOT / "oracle_result.json"
    output.write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(result, indent=2))

    env.close()

    return 0 if result["success"] and observation_valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
