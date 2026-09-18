from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from environment.data.gym_env import AmbiguousFailureDiagnosisEnv


def main():
    env = AmbiguousFailureDiagnosisEnv()

    obs, info = env.reset(seed=123)

    trajectory = []
    reward_total = 0.0

    actions = [0, 1, 2, 3]

    for action in actions:
        obs, reward, terminated, truncated, info = env.step(action)

        trajectory.append(
            {
                "action": action,
                "stage": info["stage"],
                "diagnosis": info["diagnosis"],
                "recovered": info["recovered"],
                "reward": reward,
                "terminated": terminated,
                "truncated": truncated,
            }
        )

        reward_total += float(reward)

    observation_valid = env.observation_space.contains(obs)

    payload = {
        "success": bool(
            terminated
            and not truncated
            and info["diagnosis"] == env.ROOT_FAILURE
            and info["recovered"] is True
        ),
        "steps": len(trajectory),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "final_stage": info["stage"],
        "final_diagnosis": info["diagnosis"],
        "final_recovered": info["recovered"],
        "reward_total": reward_total,
        "observation_valid": bool(observation_valid),
        "trajectory_length": len(trajectory),
        "trajectory": trajectory,
    }

    print(json.dumps(payload, indent=2))

    if not payload["success"]:
        return 1

    if not payload["observation_valid"]:
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
