from pathlib import Path
import hashlib

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_oracle_and_verifier_are_separate():
    oracle = ROOT / "oracle" / "solve.py"
    verifier = ROOT / "verifier" / "verify.py"

    assert oracle.is_file()
    assert verifier.is_file()
    assert sha256(oracle) != sha256(verifier)


def test_verifier_does_not_depend_on_oracle_result_file():
    verifier_text = (ROOT / "verifier" / "verify.py").read_text(
        encoding="utf-8"
    )

    assert "oracle_result.json" not in verifier_text
