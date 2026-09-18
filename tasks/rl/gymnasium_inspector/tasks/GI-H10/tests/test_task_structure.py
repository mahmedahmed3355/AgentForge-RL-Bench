from __future__ import annotations

from pathlib import Path
import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_required_files_exist():
    required = [
        "instruction.md",
        "task.yaml",
        "environment/data/gym_env.py",
        "oracle/solve.py",
        "verifier/verify.py",
        "tests/test_gym_env.py",
        "tests/test_oracle.py",
        "tests/test_outputs.py",
        "tests/test_multi_reward.py",
        "tests/test_hidden_boundary.py",
        "tests/test_trajectory_audit.py",
        "tests/test_anti_cheat.py",
        "tests/test_task_structure.py",
    ]

    for rel in required:
        assert (ROOT / rel).is_file(), rel


def test_task_contract():
    data = yaml.safe_load(
        (ROOT / "task.yaml").read_text(encoding="utf-8")
    )

    assert data["task_id"] == "GI-H10"
    assert data["family"] == "persistence"
    assert data["primary_capability"] == "persistence"
    assert data["difficulty"] == "hard"
    assert data["hidden_evaluation"] == "vary_interruption_boundary"


def test_terminal_contract():
    data = yaml.safe_load(
        (ROOT / "task.yaml").read_text(encoding="utf-8")
    )

    terminal = data["terminal_conditions"]

    assert terminal["progress"] == 4
    assert terminal["checkpoint_progress"] == 2
    assert terminal["resumed"] is True
    assert terminal["finished"] is True
