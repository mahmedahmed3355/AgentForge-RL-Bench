import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_trajectory_contains_generalization_events():
    p = subprocess.run(
        [sys.executable, str(ROOT / "oracle" / "solve.py")],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    data = json.loads(p.stdout)
    events = [x.get("event") for x in data["trajectory"]]

    assert "invariant_plan_derived" in events
    assert "rule_consistent_execution" in events
    assert "terminal_verified" in events


def test_terminal_is_last_event():
    p = subprocess.run(
        [sys.executable, str(ROOT / "oracle" / "solve.py")],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    data = json.loads(p.stdout)

    assert data["trajectory"][-1]["event"] == "terminal_verified"
    assert data["trajectory"][-1]["terminated"] is True
