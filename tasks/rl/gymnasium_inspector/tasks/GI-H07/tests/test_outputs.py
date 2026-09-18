import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_oracle_output_is_json():
    proc = subprocess.run(
        [sys.executable, str(ROOT / "oracle" / "solve.py")],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    assert proc.returncode == 0
    payload = json.loads(proc.stdout)
    assert isinstance(payload, dict)


def test_required_output_fields():
    proc = subprocess.run(
        [sys.executable, str(ROOT / "oracle" / "solve.py")],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    payload = json.loads(proc.stdout)

    required = {
        "success",
        "steps",
        "terminated",
        "truncated",
        "reward_total",
        "trajectory_length",
        "trajectory",
    }

    assert required.issubset(payload)


def test_trajectory_records_are_dicts():
    proc = subprocess.run(
        [sys.executable, str(ROOT / "oracle" / "solve.py")],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    payload = json.loads(proc.stdout)

    assert payload["trajectory"]
    assert all(isinstance(x, dict) for x in payload["trajectory"])
