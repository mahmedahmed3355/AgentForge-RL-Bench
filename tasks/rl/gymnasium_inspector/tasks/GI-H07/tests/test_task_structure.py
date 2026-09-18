from pathlib import Path
import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_required_directories():
    for name in ("environment", "oracle", "verifier", "tests"):
        assert (ROOT / name).is_dir()


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

    for name in required:
        assert (ROOT / name).is_file(), name


def test_task_contract():
    data = yaml.safe_load(
        (ROOT / "task.yaml").read_text(encoding="utf-8")
    )

    assert data["task_id"] == "GI-H07"
    assert data["difficulty"] == "hard"
    assert data["family"] == "verification"
    assert data["primary_capability"] == "verification"
