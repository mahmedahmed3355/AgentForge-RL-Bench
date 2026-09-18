import json
import subprocess
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]

def test_recovery_trajectory_boundary():
    r = subprocess.run([sys.executable, str(ROOT/"oracle"/"solve.py")],
                       cwd=ROOT, capture_output=True, text=True)
    p = json.loads(r.stdout); t = p["trajectory"]
    assert any(x["event"] == "recovered_preserving_progress" for x in t) or any(x["recovered"] for x in t)
    i = next(i for i,x in enumerate(t) if x["recovered"])
    assert t[i-1]["progress"] == t[i]["progress"] == 2

def test_terminal_last():
    r = subprocess.run([sys.executable, str(ROOT/"oracle"/"solve.py")],
                       cwd=ROOT, capture_output=True, text=True)
    p = json.loads(r.stdout)
    assert p["trajectory"][-1]["terminated"] is True
