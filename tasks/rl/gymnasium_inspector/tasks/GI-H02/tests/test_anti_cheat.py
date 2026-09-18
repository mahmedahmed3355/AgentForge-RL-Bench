import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "environment" / "data"
sys.path.insert(0, str(DATA))

from gym_env import GymInspectorEnv


def test_state_cannot_report_invalid_action_as_valid():
    env = GymInspectorEnv()

    env.reset(seed=123)

    assert env.action_space.contains(0)
    assert env.action_space.contains(1)
    assert not env.action_space.contains(99)
