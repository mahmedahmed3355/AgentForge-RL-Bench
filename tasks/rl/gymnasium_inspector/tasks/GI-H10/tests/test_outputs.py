from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def test_oracle_output_is_json():
    p = subprocess.run(
        [sys.executable, str(ROOT / "oracle" / "solve.py")],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    data = json.loads(p.stdout)
    assert isinstance(data, dict)


def test_output_contains_required_fields():
    p = subprocess.run(
        [sys.executable, str(ROOT / "oracle" / "solve.py")],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    data = json.loads(p.stdout)

    required = {
        "success",
        "steps",
        "terminated",
        "truncated",
        "final_progress",
        "final_checkpoint_progress",
        "final_resumed",
        "final_committed",
        "final_finished",
        "duplicate_transition",
        "stale_state",
        "reward_total",
        "trajectory",
    }

    assert required.issubset(data)


def test_success_output():
    p = subprocess.run(
        [sys.executable, str(ROOT / "oracle" / "solve.py")],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    data = json.loads(p.stdout)

    assert data["success"] is True
    assert data["reward_total"] > 0
