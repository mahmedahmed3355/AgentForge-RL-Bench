from pathlib import Path


def test_anti_cheat_files():
    root = Path(__file__).resolve().parents[1]

    env_source = (
        root / "environment" / "data" / "gym_env.py"
    ).read_text(encoding="utf-8")

    assert "pytest" not in env_source.lower()
    assert "test_" not in env_source.lower()
    assert "verifier" not in env_source.lower()
    assert "oracle" not in env_source.lower()
