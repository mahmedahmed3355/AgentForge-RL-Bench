import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_oracle_output_exists_after_oracle_run():
    import subprocess
    import sys

    subprocess.run(
        [sys.executable, str(ROOT / "oracle" / "solve.py")],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    output = ROOT / "oracle_result.json"

    assert output.exists()

    payload = json.loads(
        output.read_text(encoding="utf-8")
    )

    assert payload["success"] is True
    assert payload["observation_valid"] is True
