import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_oracle_success():
    result = subprocess.run(
        [sys.executable, str(ROOT / "oracle" / "solve.py")],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )

    payload = json.loads(result.stdout)

    assert payload["success"] is True
    assert payload["terminated"] is True
    assert payload["truncated"] is False
    assert payload["final_progress"] == 4
    assert payload["final_verified"] is True
    assert payload["final_spent_cost"] <= 6
