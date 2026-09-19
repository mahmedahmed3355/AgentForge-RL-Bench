import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "environment" / "data"

sys.path.insert(0, str(DATA))

from gym_env import GymInspectorEnv


def test_trajectory_has_consistent_step_count():
    env = GymInspectorEnv(max_steps=4)
    env.reset(seed=99)

    records = []

    for action in [1, 1, 1, 1]:
        result = env.step(action)
        records.append(result)

        if result[2] or result[3]:
            break

    assert len(records) == 4
    assert records[-1][2] is True
    assert records[-1][3] is False

    env.close()
