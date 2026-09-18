from __future__ import annotations

import json

from pathlib import Path
import sys

TASK_ROOT = Path(__file__).resolve().parents[1]
if str(TASK_ROOT) not in sys.path:
    sys.path.insert(0, str(TASK_ROOT))

from environment.data.gym_env import ComposedInspectorEnv


def main():
    env = ComposedInspectorEnv()
    obs, info = env.reset(seed=123)

    trajectory = []
    total_reward = 0.0

    # Three complementary public views are required.
    actions = [0, 0, 0, 1, 2, 3]

    for action in actions:
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)

        trajectory.append(
            {
                "action": int(action),
                "step": int(obs["step"]),
                "progress": int(obs["progress"]),
                "diagnosis": int(obs["diagnosis"]),
                "query_count": int(obs["query_count"]),
                "correlated": bool(info["correlated"]),
                "verified": bool(info["verified"]),
                "event": info["event"],
                "stage": int(info["stage"]),
                "reward": float(reward),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
            }
        )

        if terminated or truncated:
            break

    result = {
        "success": bool(env.terminated and env.verified),
        "steps": len(trajectory),
        "terminated": bool(env.terminated),
        "truncated": False,
        "final_progress": int(env.progress),
        "final_diagnosis": int(env.diagnosis),
        "final_query_count": int(env.query_count),
        "final_correlated": bool(env.correlated),
        "final_verified": bool(env.verified),
        "reward_total": float(total_reward),
        "observation_valid": True,
        "trajectory_length": len(trajectory),
        "trajectory": trajectory,
    }

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
