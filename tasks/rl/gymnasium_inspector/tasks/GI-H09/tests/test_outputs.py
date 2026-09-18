import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_oracle_output_shape():
    p = subprocess.run(
        [sys.executable, str(ROOT / "oracle" / "solve.py")],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    payload = json.loads(p.stdout)

    required = {
        "success",
        "steps",
        "terminated",
        "truncated",
        "final_progress",
        "final_diagnosis",
        "final_query_count",
        "final_correlated",
        "final_verified",
        "reward_total",
        "trajectory_length",
        "trajectory",
    }

    assert required.issubset(payload)
    assert payload["trajectory_length"] == len(payload["trajectory"])
