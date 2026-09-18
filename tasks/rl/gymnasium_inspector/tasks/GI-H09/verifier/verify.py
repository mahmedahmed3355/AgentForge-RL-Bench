from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def verify_result(result: dict) -> bool:
    trajectory = result.get("trajectory", [])

    if not result.get("success"):
        return False

    if not result.get("terminated"):
        return False

    if not result.get("final_verified"):
        return False

    if result.get("final_diagnosis") != 2:
        return False

    if result.get("final_correlated") is not True:
        return False

    events = [item.get("event") for item in trajectory]

    required = [
        "state_inspected",
        "history_inspected",
        "diagnostics_inspected",
        "minimum_sufficient_correlation",
        "correct_action",
        "terminal_verified",
    ]

    if events != required:
        return False

    if any(item.get("query_count", 0) > 3 for item in trajectory):
        return False

    return True


def main():
    import subprocess
    import sys

    proc = subprocess.run(
        [sys.executable, str(ROOT / "oracle" / "solve.py")],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    if proc.returncode != 0:
        print({"verified": False})
        return

    result = json.loads(proc.stdout)
    print({"verified": verify_result(result)})


if __name__ == "__main__":
    main()
