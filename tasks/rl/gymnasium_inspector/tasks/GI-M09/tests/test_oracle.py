from pathlib import Path
import json
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def test_oracle_completes():
    result = subprocess.run(
        [sys.executable, str(ROOT / "oracle" / "solve.py")],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert '"success": true' in result.stdout.lower()

    payload = json.loads(result.stdout)

    assert payload["success"] is True
    assert payload["terminated"] is True
    assert payload["truncated"] is False
    assert payload["final_position"] == 8
    assert payload["observation_valid"] is True
