import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_trajectory_has_terminal_verified_event():
    result = subprocess.run(
        [sys.executable, str(ROOT / "oracle" / "solve.py")],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )

    payload = json.loads(result.stdout)
    trajectory = payload["trajectory"]

    assert any(
        item.get("event") == "terminal_verified"
        for item in trajectory
    )


def test_trajectory_progress_is_monotonic():
    result = subprocess.run(
        [sys.executable, str(ROOT / "oracle" / "solve.py")],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )

    payload = json.loads(result.stdout)
    progress_values = [
        item["progress"]
        for item in payload["trajectory"]
    ]

    assert progress_values == sorted(progress_values)
