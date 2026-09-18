from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_required_task_structure():
    required = [
        "README.md",
        "instruction.md",
        "task.toml",
        "environment/Dockerfile",
        "environment/requirements.txt",
        "environment/data/gym_env.py",
        "oracle/solve.py",
        "verifier/verify.py",
        "tests/test_outputs.py",
        "tests/test_gym_env.py",
        "tests/test_oracle.py",
        "tests/test_verifier.py",
        "tests/test_anti_cheat.py",
        "tests/test_multi_reward.py",
        "tests/test_trajectory_audit.py",
        "tests/test_hidden_boundary.py",
    ]

    for relative in required:
        assert (ROOT / relative).exists(), relative
