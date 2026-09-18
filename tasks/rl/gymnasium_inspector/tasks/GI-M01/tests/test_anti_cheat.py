from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_candidate_does_not_reference_oracle():
    candidate = (
        ROOT
        / "environment"
        / "data"
        / "gym_env.py"
    ).read_text(encoding="utf-8")

    forbidden = [
        "oracle_result",
        "verifier/verify.py",
        "../oracle",
        "../verifier",
    ]

    for marker in forbidden:
        assert marker not in candidate


def test_candidate_isolated_from_tests():
    candidate = (
        ROOT
        / "environment"
        / "data"
        / "gym_env.py"
    ).read_text(encoding="utf-8")

    assert "tests/test_" not in candidate
