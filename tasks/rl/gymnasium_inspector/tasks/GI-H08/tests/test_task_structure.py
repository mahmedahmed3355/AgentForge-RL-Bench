from pathlib import Path
import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_required_structure():
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
        assert (ROOT / item).is_file(), item


def test_task_contract_identity():
    data = yaml.safe_load(
        (ROOT / "task.yaml").read_text(encoding="utf-8")
    )

    assert data["id"] == "GI-H08"
    assert data["difficulty"] == "hard"
    assert data["family"] == "optimization"
    assert data["primary_capability"] == "optimization"
