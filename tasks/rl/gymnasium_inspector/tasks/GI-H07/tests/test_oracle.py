import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run_oracle():
    proc = subprocess.run(
        [sys.executable, str(ROOT / "oracle" / "solve.py")],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout)


def test_oracle_success():
    result = run_oracle()
    assert result["success"] is True
    assert result["terminated"] is True
    assert result["truncated"] is False


def test_oracle_terminal_invariants():
    result = run_oracle()

    assert result["final_base_ready"] is True
    assert result["final_dependency_ready"] is True
    assert result["final_integrity_ok"] is True
    assert result["final_revalidated"] is True


def test_oracle_observation_valid():
    result = run_oracle()
    assert result["observation_valid"] is True


def test_oracle_expected_trajectory_length():
    result = run_oracle()
    assert result["trajectory_length"] == 5


def test_oracle_contains_resolution():
    result = run_oracle()

    assert any(
        item["event"] == "dependency_resolved"
        for item in result["trajectory"]
    )


def test_oracle_contains_revalidation():
    result = run_oracle()

    assert any(
        item["event"] == "revalidated_all_conditions"
        for item in result["trajectory"]
    )


def test_oracle_final_event():
    result = run_oracle()
    assert result["trajectory"][-1]["event"] == "terminal_verified"
