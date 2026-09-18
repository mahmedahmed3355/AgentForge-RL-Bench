from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from environment.data.gym_env import ActionEfficientCompletionEnv


def verify(result: dict | None = None):
    if result is None:
        return {"verified": False}

    trajectory = result.get("trajectory")
    if not isinstance(trajectory, list) or not trajectory:
        return {"verified": False}

    if not result.get("success"):
        return {"verified": False}

    if not result.get("terminated"):
        return {"verified": False}

    if result.get("truncated"):
        return {"verified": False}

    if result.get("final_progress") != 4:
        return {"verified": False}

    if result.get("final_spent_cost", 999) > 6:
        return {"verified": False}

    if not result.get("final_verified"):
        return {"verified": False}

    if trajectory[0].get("action") != 0:
        return {"verified": False}

    if trajectory[-1].get("action") != 4:
        return {"verified": False}

    if any(not item.get("valid_action", False) for item in trajectory):
        return {"verified": False}

    progress = 0
    spent_cost = 0
    inspected = False

    for item in trajectory:
        action = item["action"]

        if action == 0:
            inspected = True

        elif action == 1:
            if not inspected:
                return {"verified": False}
            progress = min(4, progress + 2)
            spent_cost += 2

        elif action == 2:
            if not inspected:
                return {"verified": False}
            progress = min(4, progress + 1)
            spent_cost += 1

        elif action == 3:
            spent_cost += 1

        elif action == 4:
            if progress != 4:
                return {"verified": False}

        else:
            return {"verified": False}

        if spent_cost > 6:
            return {"verified": False}

    if progress != result["final_progress"]:
        return {"verified": False}

    if spent_cost != result["final_spent_cost"]:
        return {"verified": False}

    return {"verified": True}
