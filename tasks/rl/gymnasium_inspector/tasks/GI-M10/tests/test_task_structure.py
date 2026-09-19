from pathlib import Path
import yaml


def test_task_structure():
    root = Path(__file__).resolve().parents[1]

    required = [
        root / "instruction.md",
        root / "task.yaml",
        root / "environment" / "data" / "gym_env.py",
        root / "oracle" / "solve.py",
        root / "verifier" / "verify.py",
        root / "tests" / "test_gym_env.py",
        root / "tests" / "test_oracle.py",
        root / "tests" / "test_multi_reward.py",
        root / "tests" / "test_hidden_boundary.py",
        root / "tests" / "test_trajectory_audit.py",
        root / "tests" / "test_anti_cheat.py",
    ]

    for path in required:
        assert path.exists(), str(path)

    payload = yaml.safe_load(
        (root / "task.yaml").read_text(
            encoding="utf-8"
        )
    )

    assert payload["task_id"] == "GI-M10"
    assert payload["difficulty"] == "medium"
    assert payload["reward"]["type"] == "multi_reward"
    assert payload["oracle"]["deterministic"] is True
    assert payload["verifier"]["hidden"] is True
