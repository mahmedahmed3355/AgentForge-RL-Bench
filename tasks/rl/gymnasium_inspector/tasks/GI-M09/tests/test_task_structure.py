from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_required_structure():
    required = [
        ROOT / "instruction.md",
        ROOT / "environment" / "data" / "gym_env.py",
        ROOT / "oracle" / "solve.py",
        ROOT / "verifier" / "verify.py",
        ROOT / "tests" / "test_gym_env.py",
        ROOT / "tests" / "test_oracle.py",
        ROOT / "tests" / "test_outputs.py",
        ROOT / "tests" / "test_multi_reward.py",
        ROOT / "tests" / "test_hidden_boundary.py",
        ROOT / "tests" / "test_trajectory_audit.py",
        ROOT / "tests" / "test_anti_cheat.py",
    ]

    for path in required:
        assert path.exists(), path
