from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def test_oracle_exists():
    assert (ROOT / "oracle" / "solve.py").exists()


def test_oracle_completes():
    result = subprocess.run(
        [sys.executable, str(ROOT / "oracle" / "solve.py")],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert '"success": true' in result.stdout.lower()
