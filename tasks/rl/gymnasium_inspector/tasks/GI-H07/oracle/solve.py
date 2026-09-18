from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from environment.data.gym_env import MultiConditionVerificationEnv


def main():
    env = MultiConditionVerificationEnv()
    obs, info = env.reset(seed=123)

    # Inspect -> validate the initially incomplete requirements ->
    # resolve the dependency -> revalidate -> finish.
    actions = [0, 1, 2, 1, 3]

    trajectory = []
    reward_total = 0.0

    for action in actions:
        obs, reward, terminated, truncated, info = env.step(action)
        reward_total += reward

        trajectory.append(
            {
                "action": action,
                "step": len(trajectory) + 1,
                "base_ready": bool(info["base_ready"]),
                "dependency_ready": bool(info["dependency_ready"]),
                "integrity_ok": bool(info["integrity_ok"]),
                "validated": bool(info["validated"]),
                "revalidated": bool(info["revalidated"]),
                "requirements_valid": bool(info["requirements_valid"]),
                "event": info["event"],
                "stage": info["stage"],
                "reward": float(reward),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
            }
        )

        if terminated or truncated:
            break

    result = {
        "success": bool(
            env.terminated
            and env.base_ready
            and env.dependency_ready
            and env.integrity_ok
            and env.revalidated
        ),
        "steps": len(trajectory),
        "terminated": bool(env.terminated),
        "truncated": bool(env.truncated),
        "final_base_ready": bool(env.base_ready),
        "final_dependency_ready": bool(env.dependency_ready),
        "final_integrity_ok": bool(env.integrity_ok),
        "final_revalidated": bool(env.revalidated),
        "reward_total": float(reward_total),
        "observation_valid": bool(env.observation_space.contains(obs)),
        "trajectory_length": len(trajectory),
        "trajectory": trajectory,
    }

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
