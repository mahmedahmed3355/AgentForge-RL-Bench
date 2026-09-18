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

    for item in required:
        assert (ROOT / item).exists(), item


def test_task_contract_fields():
    payload = yaml.safe_load(
        (ROOT / "task.yaml").read_text(encoding="utf-8")
    )

    required = [
        "agent_interface",
        "environment",
        "evaluation",
        "name",
        "objective",
        "oracle",
        "reward",
        "verifier",
    ]

    for field in required:
        assert field in payload
