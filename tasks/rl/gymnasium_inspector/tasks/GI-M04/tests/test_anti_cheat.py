from pathlib import Path
import hashlib


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_protected_test_files_exist():
    required = [
        ROOT / "tests" / "test_gym_env.py",
        ROOT / "tests" / "test_oracle.py",
        ROOT / "tests" / "test_outputs.py",
        ROOT / "tests" / "test_multi_reward.py",
        ROOT / "tests" / "test_hidden_boundary.py",
        ROOT / "tests" / "test_trajectory_audit.py",
    ]

    for path in required:
        assert path.exists()
        assert path.stat().st_size > 0


def test_oracle_and_verifier_are_separate():
    oracle = ROOT / "oracle" / "solve.py"
    verifier = ROOT / "verifier" / "verify.py"

    assert oracle.exists()
    assert verifier.exists()
    assert sha256(oracle) != sha256(verifier)
