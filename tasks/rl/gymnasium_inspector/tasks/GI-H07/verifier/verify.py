from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
ORACLE = ROOT / "oracle" / "solve.py"


def verify():
    proc = subprocess.run(
        [sys.executable, str(ORACLE)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    if proc.returncode != 0:
        return {"verified": False, "reason": "oracle_failed"}

    try:
        result = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return {"verified": False, "reason": "invalid_oracle_json"}

    trajectory = result.get("trajectory", [])

    if not trajectory:
        return {"verified": False, "reason": "empty_trajectory"}

    # Independently reconstruct terminal validity from the trajectory.
    final = trajectory[-1]

    required = (
        final.get("base_ready") is True
        and final.get("dependency_ready") is True
        and final.get("integrity_ok") is True
        and final.get("revalidated") is True
        and final.get("requirements_valid") is True
    )

    if not required:
        return {"verified": False, "reason": "terminal_invariants_failed"}

    if final.get("event") != "terminal_verified":
        return {"verified": False, "reason": "missing_terminal_verification"}

    # The trajectory must contain the dependency repair boundary.
    if not any(
        item.get("event") == "dependency_resolved"
        for item in trajectory
    ):
        return {"verified": False, "reason": "missing_resolution"}

    # Revalidation must happen after resolution.
    resolved_index = next(
        i for i, item in enumerate(trajectory)
        if item.get("event") == "dependency_resolved"
    )

    revalidated_after_resolution = any(
        item.get("revalidated") is True
        for item in trajectory[resolved_index + 1 :]
    )

    if not revalidated_after_resolution:
        return {"verified": False, "reason": "missing_revalidation_boundary"}

    if not result.get("success"):
        return {"verified": False, "reason": "oracle_claimed_failure"}

    if not result.get("terminated"):
        return {"verified": False, "reason": "not_terminated"}

    return {"verified": True}


if __name__ == "__main__":
    print(verify())
