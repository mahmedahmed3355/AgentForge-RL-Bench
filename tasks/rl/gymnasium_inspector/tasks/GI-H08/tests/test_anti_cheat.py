import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_oracle_does_not_report_unearned_progress():
    result = subprocess.run(
        [sys.executable, str(ROOT / "oracle" / "solve.py")],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )

    payload = json.loads(result.stdout)

    progress = 0

    for item in payload["trajectory"]:
        action = item["action"]

        if action == 1:
            progress = min(4, progress + 2)
        elif action == 2:
            progress = min(4, progress + 1)

        assert item["progress"] == progress


def test_terminal_cannot_claim_success_without_budget():
    from environment.data.gym_env import ActionEfficientCompletionEnv

    env = ActionEfficientCompletionEnv()
    env.reset(seed=123)

    env.inspected = True
    env.progress = 4
    env.spent_cost = 6

    _, reward, terminated, truncated, info = env.step(
        env.FINISH
    )

    assert terminated is True
    assert truncated is False
    assert reward == 5.0
    assert info["valid_action"] is True
