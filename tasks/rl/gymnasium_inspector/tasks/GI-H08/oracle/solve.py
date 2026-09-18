from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from environment.data.gym_env import ActionEfficientCompletionEnv


def main():
    env = ActionEfficientCompletionEnv()
    obs, info = env.reset(seed=123)

    trajectory = []
    reward_total = 0.0

    # Efficient valid plan:
    # inspect -> fast execute -> fast execute -> finish
    actions = [
        env.INSPECT,
        env.FAST_EXECUTE,
        env.FAST_EXECUTE,
        env.FINISH,
    ]

    for action in actions:
        obs, reward, terminated, truncated, info = env.step(action)
        reward_total += reward

        trajectory.append(
            {
                "action": int(action),
                "progress": int(env.progress),
                "spent_cost": int(env.spent_cost),
                "inspected": bool(env.inspected),
                "verified": bool(env.verified),
                "reward": float(reward),
                "event": info["event"],
                "stage": info["stage"],
                "valid_action": bool(info["valid_action"]),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
            }
        )

        if terminated or truncated:
            break

    result = {
        "success": bool(
            env.terminated
            and env.verified
            and env.progress == env.TARGET_PROGRESS
            and env.spent_cost <= env.EFFICIENCY_BUDGET
        ),
        "steps": len(trajectory),
        "terminated": bool(env.terminated),
        "truncated": bool(env.truncated),
        "final_progress": int(env.progress),
        "final_spent_cost": int(env.spent_cost),
        "final_verified": bool(env.verified),
        "reward_total": float(reward_total),
        "observation_valid": bool(
            env.observation_space.contains(obs)
        ),
        "trajectory_length": len(trajectory),
        "trajectory": trajectory,
    }

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
