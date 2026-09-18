from pathlib import Path
import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_required_files_exist():
    files = [
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

    for name in files:
        assert (ROOT / name).is_file(), name


def test_task_contract():
    data = yaml.safe_load((ROOT / "task.yaml").read_text())

    assert data["task_id"] == "GI-H11"
    assert data["family"] == "generalization"
    assert data["primary_capability"] == "generalization"
    assert data["environment"]["class"] == "UnseenParameterCompositionEnv"
