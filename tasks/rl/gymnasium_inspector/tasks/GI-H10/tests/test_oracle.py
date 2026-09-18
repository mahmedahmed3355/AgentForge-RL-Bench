from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def run_oracle():
    p = subprocess.run(
        [sys.executable, str(ROOT / "oracle" / "solve.py")],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert p.returncode == 0
    return json.loads(p.stdout)


def test_oracle_success():
    data = run_oracle()

    assert data["success"] is True
    assert data["terminated"] is True
    assert data["final_progress"] == 4
    assert data["final_checkpoint_progress"] == 2
    assert data["final_resumed"] is True


def test_oracle_preserves_checkpoint_boundary():
    data = run_oracle()

    assert data["final_checkpoint_progress"] == 2
    assert data["duplicate_transition"] is False
    assert data["stale_state"] is False


def test_oracle_has_resume_event():
    data = run_oracle()

    assert any(
        x["event"] == "checkpoint_resumed"
        for x in data["trajectory"]
    )


def test_oracle_trajectory_length():
    data = run_oracle()

    assert data["trajectory_length"] == 5


def test_oracle_does_not_replay_resume():
    data = run_oracle()

    assert sum(
        1 for x in data["trajectory"]
        if x["event"] == "checkpoint_resumed"
    ) == 1
