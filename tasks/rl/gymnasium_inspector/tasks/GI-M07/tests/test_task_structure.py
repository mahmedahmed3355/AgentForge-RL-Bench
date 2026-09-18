from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_required_files_exist():
    required = [
        "instruction.md",
        "environment/data/gym_env.py",
        "oracle/solve.py",
        "verifier/verify.py",
        "tests/test_task_structure.py",
    ]

    for relative in required:
        assert (ROOT / relative).exists(), relative
