import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def test_oracle_completes():
    r = subprocess.run([sys.executable, str(ROOT/"oracle"/"solve.py")],
                       cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0
    p = json.loads(r.stdout)
    assert p["success"] is True
    assert p["terminated"] is True
    assert p["truncated"] is False
    assert p["final_progress"] == 4
    assert p["final_recovered"] is True
    assert p["observation_valid"] is True
