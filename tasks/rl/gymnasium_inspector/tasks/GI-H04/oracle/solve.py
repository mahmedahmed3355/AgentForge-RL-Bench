from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "environment" / "data"

sys.path.insert(0, str(DATA))

from gym_env import DependencyGraphEnv


def main() -> int:
    env = DependencyGraphEnv(max_steps=8)

    observation, info = env.reset(seed=123)

    # Deterministic topological ordering.
    actions = [0, 1, 2, 3, 4, 5]

    rewards = []
    trajectory = []

    terminated = False
    truncated = False

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
                "progress": int(observation["progress"]),
                "step": int(observation["step"]),
                "reward": float(reward),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "valid_action": bool(info["valid_action"]),
            }
        )

        if terminated or truncated:
            break

    observation_valid = bool(
        env.observation_space.contains(observation)
    )

    result = {
        "success": bool(terminated and not truncated),
        "steps": len(rewards),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "final_progress": int(observation["progress"]),
        "reward_total": float(sum(rewards)),
        "observation_valid": observation_valid,
        "trajectory_length": len(trajectory),
        "trajectory": trajectory,
    }

    output = ROOT / "oracle_result.json"

    output.write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )

    # The stdout contract is intentional: tests consume the JSON result.
    print(json.dumps(result, indent=2))

    env.close()

    return 0 if result["success"] and observation_valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
