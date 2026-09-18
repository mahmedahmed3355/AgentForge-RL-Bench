import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def verify():
    r = subprocess.run(
        [sys.executable, str(ROOT / "oracle" / "solve.py")],
        cwd=ROOT, capture_output=True, text=True, check=False
    )
    if r.returncode != 0:
        return {"verified": False, "reason": "oracle_failed"}
    try:
        p = json.loads(r.stdout)
    except json.JSONDecodeError:
        return {"verified": False, "reason": "invalid_oracle_json"}
    t = p.get("trajectory", [])
    recovery_index = next((i for i, x in enumerate(t) if x.get("recovered")), None)
    preserved = (
        recovery_index is not None and recovery_index > 0
        and t[recovery_index - 1]["progress"] == 2
        and t[recovery_index]["progress"] == 2
    )
    verified = (
        p.get("success") is True
        and p.get("terminated") is True
        and p.get("truncated") is False
        and p.get("final_progress") == 4
        and p.get("final_recovered") is True
        and p.get("observation_valid") is True
        and preserved
    )
    return {"verified": verified}

if __name__ == "__main__":
    print(verify())
