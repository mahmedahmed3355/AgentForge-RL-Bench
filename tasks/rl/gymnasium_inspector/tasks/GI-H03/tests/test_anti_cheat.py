from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_verifier_does_not_import_oracle():
    text = (ROOT / "verifier" / "verify.py").read_text(encoding="utf-8")

    assert "oracle.solve" not in text
    assert "oracle_result.json" not in text


def test_environment_does_not_depend_on_oracle():
    text = (ROOT / "environment" / "data" / "gym_env.py").read_text(
        encoding="utf-8"
    )

    assert "oracle_result.json" not in text
    assert "oracle.solve" not in text


def test_solution_is_not_required_for_environment():
    text = (ROOT / "environment" / "data" / "gym_env.py").read_text(
        encoding="utf-8"
    )

    assert "solution/" not in text
