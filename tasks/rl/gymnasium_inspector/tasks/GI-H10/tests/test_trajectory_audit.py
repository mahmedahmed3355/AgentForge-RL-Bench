from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def get_data():
    p = subprocess.run(
        [sys.executable, str(ROOT / "oracle" / "solve.py")],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    return json.loads(p.stdout)


def test_recovery_event_exists():
    data = get_data()
    assert any(
        x["event"] == "checkpoint_resumed"
        for x in data["trajectory"]
    )


def test_progress_never_drops_after_resume():
    data = get_data()
    progress = [x["progress"] for x in data["trajectory"]]
    assert progress[1] == 2
    assert progress == sorted(progress)


def test_resume_occurs_once():
    data = get_data()
    assert sum(
        x["event"] == "checkpoint_resumed"
        for x in data["trajectory"]
    ) == 1


def test_terminal_is_last():
    data = get_data()
    assert data["trajectory"][-1]["terminated"] is True
