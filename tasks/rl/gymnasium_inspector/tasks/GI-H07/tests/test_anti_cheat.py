import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_verifier_does_not_import_oracle_result_artifact():
    source = (ROOT / "verifier" / "verify.py").read_text(encoding="utf-8")
    assert "oracle_result.json" not in source


def test_verifier_contains_independent_terminal_checks():
    source = (ROOT / "verifier" / "verify.py").read_text(encoding="utf-8")

    for token in (
        "base_ready",
        "dependency_ready",
        "integrity_ok",
        "revalidated",
        "terminal_verified",
    ):
        assert token in source


def test_oracle_is_valid_python():
    source = (ROOT / "oracle" / "solve.py").read_text(encoding="utf-8")
    ast.parse(source)
