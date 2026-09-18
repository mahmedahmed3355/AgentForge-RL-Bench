from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def verify() -> bool:
    result = subprocess.run(
        [sys.executable, str(ROOT / "oracle" / "solve.py")],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        return False

    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError:
        return False

    trajectory = data.get("trajectory", [])

    return bool(
        data.get("success") is True
        and data.get("terminated") is True
        and data.get("final_progress") == 4
        and data.get("final_checkpoint_progress") == 2
        and data.get("final_resumed") is True
        and data.get("final_committed") is True
        and data.get("final_finished") is True
        and data.get("duplicate_transition") is False
        and data.get("stale_state") is False
        and len(trajectory) == 5
        and any(
            item.get("event") == "checkpoint_resumed"
            for item in trajectory
        )
    )


if __name__ == "__main__":
    print({"verified": verify()})
