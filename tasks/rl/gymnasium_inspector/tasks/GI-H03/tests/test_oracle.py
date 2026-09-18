import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_oracle_completes():
    result = subprocess.run(
        [sys.executable, str(ROOT / "oracle" / "solve.py")],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0

    payload = json.loads(result.stdout)

    assert payload["success"] is True
    assert payload["terminated"] is True
    assert payload["truncated"] is False
    assert payload["final_state"] == 5
    assert payload["final_resources"] == 3
    assert payload["observation_valid"] is True
