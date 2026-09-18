import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_resolution_event_exists():
    proc = subprocess.run(
        [sys.executable, str(ROOT / "oracle" / "solve.py")],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    payload = json.loads(proc.stdout)

    assert any(
        item["event"] == "dependency_resolved"
        for item in payload["trajectory"]
    )


def test_revalidation_event_exists():
    proc = subprocess.run(
        [sys.executable, str(ROOT / "oracle" / "solve.py")],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    payload = json.loads(proc.stdout)

    assert any(
        item["event"] == "revalidated_all_conditions"
        for item in payload["trajectory"]
    )


def test_terminal_event_is_last():
    proc = subprocess.run(
        [sys.executable, str(ROOT / "oracle" / "solve.py")],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    payload = json.loads(proc.stdout)

    assert payload["trajectory"][-1]["event"] == "terminal_verified"
