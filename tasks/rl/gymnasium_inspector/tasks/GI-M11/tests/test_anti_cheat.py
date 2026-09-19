from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_protected_components_exist():
    assert (ROOT / "instruction.md").exists()
    assert (ROOT / "oracle" / "solve.py").exists()
    assert (ROOT / "verifier" / "verify.py").exists()


def test_oracle_is_not_empty():
    content = (ROOT / "oracle" / "solve.py").read_text(encoding="utf-8")
    assert len(content.strip()) > 100


def test_verifier_is_not_empty():
    content = (ROOT / "verifier" / "verify.py").read_text(encoding="utf-8")
    assert len(content.strip()) > 100
