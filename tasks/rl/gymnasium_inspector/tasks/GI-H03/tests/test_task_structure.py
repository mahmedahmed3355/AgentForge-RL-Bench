from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


REQUIRED = [
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


def test_required_components_exist():
    for relative in REQUIRED:
        assert (ROOT / relative).exists(), relative


def test_task_contract():
    data = yaml.safe_load((ROOT / "task.yaml").read_text(encoding="utf-8"))

    required = {
        "task_id",
        "version",
        "domain",
        "difficulty",
        "name",
        "objective",
        "agent_interface",
        "environment",
        "evaluation",
        "oracle",
        "reward",
        "verifier",
    }

    assert required.issubset(data.keys())
    assert data["task_id"] == "GI-H03"
    assert data["difficulty"] == "hard"


def test_multi_reward_contract():
    data = yaml.safe_load((ROOT / "task.yaml").read_text(encoding="utf-8"))

    components = data["reward"]["components"]

    assert components["planning"] is True
    assert components["correctness"] is True
    assert components["efficiency"] is True
    assert components["terminal_success"] is True
    assert components["penalties"] is True
