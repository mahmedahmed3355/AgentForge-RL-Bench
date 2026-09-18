import json
import subprocess
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]

def test_output_schema():
    r = subprocess.run([sys.executable, str(ROOT/"oracle"/"solve.py")],
                       cwd=ROOT, capture_output=True, text=True)
    p = json.loads(r.stdout)
    for k in ["success","steps","terminated","truncated","final_progress",
              "final_failed","final_recovered","reward_total",
              "observation_valid","trajectory_length","trajectory"]:
        assert k in p
    assert p["trajectory_length"] == len(p["trajectory"])
