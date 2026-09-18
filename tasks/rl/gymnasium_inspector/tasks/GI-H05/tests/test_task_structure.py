from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_required_files():
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

    for relative in required:
        assert (ROOT / relative).is_file(), relative


def test_task_yaml_contract():
    data = yaml.safe_load((ROOT / "task.yaml").read_text())

    required = [
        "name",
        "title",
        "difficulty",
        "family",
        "primary_capability",
        "objective",
        "agent_interface",
        "environment",
        "evaluation",
        "oracle",
        "reward",
        "verifier",
    ]

    for key in required:
        assert key in data


def test_task_identity():
    data = yaml.safe_load((ROOT / "task.yaml").read_text())

    assert data["name"] == "GI-H05"
    assert data["difficulty"] == "hard"
    assert data["family"] == "diagnosis"
