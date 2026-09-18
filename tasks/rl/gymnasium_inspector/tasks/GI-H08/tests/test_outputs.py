import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_oracle_output_contract():
    result = subprocess.run(
        [sys.executable, str(ROOT / "oracle" / "solve.py")],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )

    payload = json.loads(result.stdout)

    required = {
        "success",
        "steps",
        "terminated",
        "truncated",
        "final_progress",
        "final_spent_cost",
        "final_verified",
        "reward_total",
        "observation_valid",
        "trajectory_length",
        "trajectory",
    }

    assert required.issubset(payload)
    assert payload["trajectory_length"] == len(payload["trajectory"])
