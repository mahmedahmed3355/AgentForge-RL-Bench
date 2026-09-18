import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_trajectory_contains_composed_queries():
    p = subprocess.run(
        [sys.executable, str(ROOT / "oracle" / "solve.py")],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    payload = json.loads(p.stdout)
    events = [x["event"] for x in payload["trajectory"]]

    assert events[:3] == [
        "state_inspected",
        "history_inspected",
        "diagnostics_inspected",
    ]


def test_terminal_event_is_last():
    p = subprocess.run(
        [sys.executable, str(ROOT / "oracle" / "solve.py")],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    payload = json.loads(p.stdout)

    assert payload["trajectory"][-1]["event"] == "terminal_verified"
