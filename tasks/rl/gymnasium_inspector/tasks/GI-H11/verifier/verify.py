from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

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
        data = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return {"verified": False, "reason": "invalid_oracle_json"}

    trajectory = data.get("trajectory", [])

    if not data.get("success"):
        return {"verified": False, "reason": "oracle_not_successful"}

    if not data.get("terminated"):
        return {"verified": False, "reason": "not_terminated"}

    if data.get("final_result") != data.get("expected_result"):
        return {"verified": False, "reason": "rule_mismatch"}

    if not data.get("final_inspected"):
        return {"verified": False, "reason": "not_inspected"}

    if not data.get("final_planned"):
        return {"verified": False, "reason": "not_planned"}

    if not data.get("final_executed"):
        return {"verified": False, "reason": "not_executed"}

    if not data.get("final_constraint_ok"):
        return {"verified": False, "reason": "constraint_violation"}

    if not data.get("final_verified"):
        return {"verified": False, "reason": "not_verified"}

    events = [x.get("event") for x in trajectory]

    if "invariant_plan_derived" not in events:
        return {"verified": False, "reason": "missing_rule_derivation"}

    if "rule_consistent_execution" not in events:
        return {"verified": False, "reason": "missing_rule_consistent_execution"}

    if "terminal_verified" not in events:
        return {"verified": False, "reason": "missing_terminal_verification"}

    return {"verified": True}


if __name__ == "__main__":
    print(verify())
