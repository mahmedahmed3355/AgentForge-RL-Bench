from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_protected_components_exist():
    assert (ROOT / "instruction.md").exists()
    assert (ROOT / "oracle" / "solve.py").exists()
    assert (ROOT / "verifier" / "verify.py").exists()


def test_oracle_has_independent_execution_logic():
    content = (
        ROOT / "oracle" / "solve.py"
    ).read_text(encoding="utf-8")

    assert "GymInspectorEnv" in content
    assert "success" in content
    assert "trajectory_length" in content


def test_verifier_has_independent_checks():
    content = (
        ROOT / "verifier" / "verify.py"
    ).read_text(encoding="utf-8")

    assert "observation_space.contains" in content
    assert "terminated" in content
    assert "truncated" in content
    assert "verified" in content
