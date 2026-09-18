from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_required_components_exist():
    required = [
        "instruction.md",
        "environment/data/gym_env.py",
        "oracle/solve.py",
        "verifier/verify.py",
        "tests/test_gym_env.py",
        "tests/test_outputs.py",
        "tests/test_multi_reward.py",
        "tests/test_hidden_boundary.py",
        "tests/test_trajectory_audit.py",
        "tests/test_anti_cheat.py",
        "tests/test_oracle.py",
    ]

    for relative_path in required:
        assert (ROOT / relative_path).is_file(), relative_path
