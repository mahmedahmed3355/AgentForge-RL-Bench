from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_verifier_boundary_exists():
    verifier = ROOT / "verifier" / "verify.py"

    assert verifier.exists()
    assert verifier.stat().st_size > 0


def test_oracle_is_separate_from_verifier():
    oracle = (ROOT / "oracle" / "solve.py").read_text(encoding="utf-8")
    verifier = (ROOT / "verifier" / "verify.py").read_text(encoding="utf-8")

    assert "from gym_env import BranchingIrreversibleEnv" in oracle
    assert "from gym_env import BranchingIrreversibleEnv" in verifier
