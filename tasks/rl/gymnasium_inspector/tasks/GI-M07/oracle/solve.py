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

    rewards = []
    terminated = False
    truncated = False

    for action in [1, 1, 1, 1, 1, 1, 1, 1]:
        (
            observation,
            reward,
            terminated,
            truncated,
            info,
        ) = env.step(action)

        rewards.append(float(reward))

        if terminated or truncated:
            break

    result = {
        "success": bool(terminated and not truncated),
        "steps": len(rewards),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "final_state": int(observation["state"][0]),
        "reward_total": float(sum(rewards)),
        "observation_valid": bool(env.observation_space.contains(observation)),
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
