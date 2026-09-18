from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


EXPECTED_ACTIONS = [0, 1, 2, 3, 4, 5]


def verify() -> bool:
    result_path = ROOT / "oracle_result.json"

    if not result_path.exists():
        return False

    payload = json.loads(
        result_path.read_text(encoding="utf-8")
    )

    if payload.get("success") is not True:
        return False

    if payload.get("terminated") is not True:
        return False

    if payload.get("truncated") is not False:
        return False

    if payload.get("final_progress") != 6:
        return False

    if payload.get("observation_valid") is not True:
        return False

    trajectory = payload.get("trajectory")

    if not isinstance(trajectory, list):
        return False

    actions = [entry.get("action") for entry in trajectory]

    if actions != EXPECTED_ACTIONS:
        return False

    progress = [
        entry.get("progress")
        for entry in trajectory
    ]

    if progress != [1, 2, 3, 4, 5, 6]:
        return False

    # Every intermediate action must be valid.
    for entry in trajectory:
        if entry.get("valid_action") is not True:
            return False

    return True


if __name__ == "__main__":
    print({"verified": verify()})
    raise SystemExit(0 if verify() else 1)
