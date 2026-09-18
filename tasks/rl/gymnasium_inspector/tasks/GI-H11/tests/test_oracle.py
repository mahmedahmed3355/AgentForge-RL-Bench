import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_oracle_success():
    p = subprocess.run(
        [sys.executable, str(ROOT / "oracle" / "solve.py")],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    assert p.returncode == 0

    data = json.loads(p.stdout)

    assert data["success"] is True
    assert data["terminated"] is True
    assert data["final_result"] == data["expected_result"]
    assert data["final_verified"] is True
