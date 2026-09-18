from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_no_hidden_reference_files_in_candidate_environment():
    environment = ROOT / "environment"

    forbidden_names = {
        "oracle_result.json",
        "hidden.json",
        "secret.json",
    }

    for path in environment.rglob("*"):
        if path.is_file():
            assert path.name not in forbidden_names
