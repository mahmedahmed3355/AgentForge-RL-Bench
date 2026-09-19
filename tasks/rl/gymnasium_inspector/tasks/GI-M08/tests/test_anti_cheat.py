from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_test_suite_is_present():
    tests = ROOT / "tests"

    assert tests.is_dir()
    assert len(list(tests.glob("test_*.py"))) >= 7


def test_oracle_isolated():
    oracle = ROOT / "oracle" / "solve.py"
    environment = ROOT / "environment" / "data" / "gym_env.py"

    assert oracle.exists()
    assert environment.exists()
    assert oracle.resolve() != environment.resolve()


def test_verifier_isolated():
    verifier = ROOT / "verifier" / "verify.py"

    assert verifier.exists()
    assert verifier.resolve() != (
        ROOT / "environment" / "data" / "gym_env.py"
    ).resolve()
