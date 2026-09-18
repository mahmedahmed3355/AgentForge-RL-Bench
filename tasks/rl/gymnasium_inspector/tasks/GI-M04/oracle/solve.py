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

    try:
        observation, info = env.reset(seed=123)

        actions = [1, 1, 1, 1, 1, 1]

        rewards = []
        terminated = False
        truncated = False

        observation_valid = env.observation_space.contains(
            observation
        )

        for action in actions:
            if not env.action_space.contains(action):
                raise RuntimeError("oracle selected invalid action")

            (
                observation,
                reward,
                terminated,
                truncated,
                info,
            ) = env.step(action)

            rewards.append(float(reward))

            observation_valid = bool(
                observation_valid
                and env.observation_space.contains(observation)
            )

            if terminated or truncated:
                break

        final_state = int(observation[0])

        result = {
            "success": bool(terminated and not truncated),
            "steps": len(rewards),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "final_state": final_state,
            "reward_total": float(sum(rewards)),
            "observation_valid": bool(observation_valid),
        }

        output = ROOT / "oracle_result.json"

        output.write_text(
            json.dumps(result, indent=2),
            encoding="utf-8",
        )

        print(json.dumps(result, indent=2))

        return 0 if result["success"] else 1

    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
