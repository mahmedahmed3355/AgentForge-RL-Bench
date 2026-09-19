from __future__ import annotations

import json
import sys
from pathlib import Path

TASK_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = TASK_ROOT.parents[5]

# Allow imports from the task itself.
if str(TASK_ROOT) not in sys.path:
    sys.path.insert(0, str(TASK_ROOT))

# Allow imports from the project root.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from environment.env import GiVh01Environment
from oracle.oracle import Oracle


class GI_VH01_Oracle(Oracle):
    """Executable reference runner for GI-VH01."""

    def select_action(self, observation):
        if isinstance(observation, dict):
            available = observation.get("available_actions")

            if isinstance(available, (list, tuple)) and available:
                return available[0]

        return "inspect"


def main() -> int:
    environment = GiVh01Environment()
    oracle = GI_VH01_Oracle()

    try:
        trajectory = oracle.solve(environment)

        result = {
            "success": False,
            "task_id": "GI-VH01",
            "trajectory": trajectory,
            "steps": len(trajectory),
            "final_state": getattr(environment, "_state", {}),
        }

        print(json.dumps(result, indent=2, default=str))
        return 0

    finally:
        close = getattr(environment, "close", None)

        if callable(close):
            close()


if __name__ == "__main__":
    raise SystemExit(main())
