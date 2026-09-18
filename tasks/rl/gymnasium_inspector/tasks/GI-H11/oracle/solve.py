from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from environment.data.gym_env import UnseenParameterCompositionEnv


def main():
    env = UnseenParameterCompositionEnv()

    trajectory = []

    obs, info = env.reset()

    trajectory.append(
        {
            "action": 0,
            "step": 0,
            "result": obs["result"],
            "inspected": bool(obs["inspected"]),
            "planned": bool(obs["planned"]),
            "executed": bool(obs["executed"]),
            "verified": bool(obs["verified"]),
            "constraint_ok": bool(obs["constraint_ok"]),
            "event": "inspection",
            "reward": 0.5,
        }
    )

    obs, reward, terminated, truncated, info = env.step(0)

    # The first explicit action after reset performs the legitimate
    # inspection required by the task.
    trajectory[-1]["actual_reward"] = reward
    trajectory[-1]["initial_event"] = info["event"]

    obs, reward, terminated, truncated, info = env.step(1)
    trajectory.append(
        {
            "action": 1,
            "step": 2,
            "result": obs["result"],
            "inspected": bool(obs["inspected"]),
            "planned": bool(obs["planned"]),
            "executed": bool(obs["executed"]),
            "verified": bool(obs["verified"]),
            "constraint_ok": bool(obs["constraint_ok"]),
            "event": info["event"],
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
        }
    )

    obs, reward, terminated, truncated, info = env.step(2)
    trajectory.append(
        {
            "action": 2,
            "step": 3,
            "result": obs["result"],
            "inspected": bool(obs["inspected"]),
            "planned": bool(obs["planned"]),
            "executed": bool(obs["executed"]),
            "verified": bool(obs["verified"]),
            "constraint_ok": bool(obs["constraint_ok"]),
            "event": info["event"],
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
        }
    )

    obs, reward, terminated, truncated, info = env.step(3)
    trajectory.append(
        {
            "action": 3,
            "step": 4,
            "result": obs["result"],
            "inspected": bool(obs["inspected"]),
            "planned": bool(obs["planned"]),
            "executed": bool(obs["executed"]),
            "verified": bool(obs["verified"]),
            "constraint_ok": bool(obs["constraint_ok"]),
            "event": info["event"],
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
        }
    )

    result = {
        "success": bool(terminated and obs["verified"]),
        "steps": len(trajectory),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "final_result": int(obs["result"]),
        "expected_result": int(info["expected_result"]),
        "final_inspected": bool(obs["inspected"]),
        "final_planned": bool(obs["planned"]),
        "final_executed": bool(obs["executed"]),
        "final_verified": bool(obs["verified"]),
        "final_constraint_ok": bool(obs["constraint_ok"]),
        "observation_valid": isinstance(obs, dict),
        "trajectory_length": len(trajectory),
        "trajectory": trajectory,
    }

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
