from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "environment" / "data"

sys.path.insert(0, str(DATA))

from gym_env import BranchingIrreversibleEnv


def main() -> int:
    env = BranchingIrreversibleEnv(max_steps=10)

    observation, info = env.reset(seed=123)

    # Globally valid plan:
    # inspect -> commit branch 0 -> execute -> finish
    actions = [0, 1, 2, 3]

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
                "state": int(observation["state"]),
                "resources": int(observation["resources"][0]),
                "reward": float(reward),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
            }
        )

        if terminated or truncated:
            break

    observation_valid = env.observation_space.contains(observation)

    result = {
        "success": bool(terminated and not truncated),
        "steps": len(rewards),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "final_state": int(observation["state"]),
        "final_resources": int(observation["resources"][0]),
        "reward_total": float(sum(rewards)),
        "observation_valid": bool(observation_valid),
        "trajectory_length": len(trajectory),
        "trajectory": trajectory,
    }

    print(json.dumps(result, indent=2))

    output = ROOT / "oracle_result.json"
    output.write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )

    env.close()

    return 0 if result["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
