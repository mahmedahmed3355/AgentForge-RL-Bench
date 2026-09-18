import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_oracle_output_schema():
    p = subprocess.run(
        [sys.executable, str(ROOT / "oracle" / "solve.py")],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    data = json.loads(p.stdout)

    required = {
        "success",
        "steps",
        "terminated",
        "truncated",
        "final_result",
        "expected_result",
        "final_inspected",
        "final_planned",
        "final_executed",
        "final_verified",
        "final_constraint_ok",
        "trajectory_length",
        "trajectory",
    }

    assert required.issubset(data.keys())
    assert isinstance(data["trajectory"], list)
    assert data["trajectory_length"] == len(data["trajectory"])
