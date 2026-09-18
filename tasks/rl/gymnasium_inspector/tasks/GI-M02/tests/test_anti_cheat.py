from pathlib import Path
import hashlib

ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_protected_oracle_and_verifier_are_present():
    oracle = ROOT / "oracle" / "solve.py"
    verifier = ROOT / "verifier" / "verify.py"

    assert oracle.is_file()
    assert verifier.is_file()
    assert oracle.stat().st_size > 0
    assert verifier.stat().st_size > 0


def test_oracle_and_verifier_are_distinct_artifacts():
    oracle = ROOT / "oracle" / "solve.py"
    verifier = ROOT / "verifier" / "verify.py"

    assert digest(oracle) != digest(verifier)
