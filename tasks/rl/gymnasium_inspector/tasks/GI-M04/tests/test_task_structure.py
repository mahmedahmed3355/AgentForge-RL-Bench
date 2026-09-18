from pathlib import Path


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
    ]

    for relative in required:
        assert (ROOT / relative).exists()
