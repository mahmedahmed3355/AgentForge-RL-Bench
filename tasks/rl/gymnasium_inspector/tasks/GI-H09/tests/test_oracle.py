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
        check=False,
    )

    assert result.returncode == 0

    payload = json.loads(result.stdout)

    assert payload["success"] is True
    assert payload["terminated"] is True
    assert payload["final_diagnosis"] == 2
    assert payload["final_correlated"] is True
    assert payload["final_verified"] is True
