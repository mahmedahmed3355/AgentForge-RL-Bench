import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_efficiency_boundary():
    result = subprocess.run(
        [sys.executable, str(ROOT / "oracle" / "solve.py")],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )

    payload = json.loads(result.stdout)

    assert payload["final_spent_cost"] <= 6
    assert payload["final_spent_cost"] >= 0


def test_finish_is_terminal_only_after_valid_progress():
    from environment.data.gym_env import ActionEfficientCompletionEnv

    env = ActionEfficientCompletionEnv()
    env.reset(seed=123)
    env.step(env.INSPECT)
    env.step(env.FAST_EXECUTE)
    env.step(env.FAST_EXECUTE)

    _, _, terminated, truncated, info = env.step(
        env.FINISH
    )

    assert terminated is True
    assert truncated is False
    assert info["valid_action"] is True
