from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_solution_does_not_replace_tests():
    tests = ROOT / "tests"

    assert tests.is_dir()
    assert len(list(tests.glob("test_*.py"))) >= 7


def test_oracle_is_separate_from_environment():
    oracle = ROOT / "oracle" / "solve.py"
    environment = ROOT / "environment" / "data" / "gym_env.py"

    assert oracle.exists()
    assert environment.exists()
    assert oracle.resolve() != environment.resolve()


def test_verifier_is_separate():
    verifier = ROOT / "verifier" / "verify.py"
    assert verifier.exists()
