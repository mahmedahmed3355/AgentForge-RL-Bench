from pathlib import Path
import yaml
ROOT = Path(__file__).resolve().parents[1]

def test_required_structure():
    for x in [
        "instruction.md","task.yaml","environment/data/gym_env.py",
        "oracle/solve.py","verifier/verify.py",
        "tests/test_gym_env.py","tests/test_oracle.py","tests/test_outputs.py",
        "tests/test_multi_reward.py","tests/test_hidden_boundary.py",
        "tests/test_trajectory_audit.py","tests/test_anti_cheat.py",
        "tests/test_task_structure.py"
    ]:
        assert (ROOT/x).is_file(), x

def test_identity():
    d = yaml.safe_load((ROOT/"task.yaml").read_text())
    assert d["id"] == "GI-H06"
    assert d["difficulty"] == "hard"
