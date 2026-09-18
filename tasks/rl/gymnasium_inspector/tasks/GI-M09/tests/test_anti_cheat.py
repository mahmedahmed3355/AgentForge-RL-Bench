from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_verifier_and_tests_are_separate():
    verifier = ROOT / "verifier" / "verify.py"
    tests = ROOT / "tests"

    assert verifier.exists()
    assert tests.exists()
    assert verifier.parent != tests


def test_instruction_does_not_embed_solution():
    text = (ROOT / "instruction.md").read_text(encoding="utf-8").lower()

    assert "do not modify the verifier or tests" in text
    assert "gyminspectorenv(max_steps=8)" not in text
