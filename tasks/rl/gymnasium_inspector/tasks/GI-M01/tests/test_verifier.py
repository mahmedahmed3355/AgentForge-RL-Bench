from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def test_verifier_exists():
    assert (ROOT / "verifier" / "verify.py").exists()


def test_verifier_passes_reference_environment():
    result = subprocess.run(
        [sys.executable, str(ROOT / "verifier" / "verify.py")],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert '"passed": true' in result.stdout.lower()
