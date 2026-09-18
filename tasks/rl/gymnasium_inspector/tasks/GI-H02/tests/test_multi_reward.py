import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "environment" / "data"
sys.path.insert(0, str(DATA))

from gym_env import GymInspectorEnv


def test_reward_is_float_and_positive_on_valid_progress():
    env = GymInspectorEnv()
    env.reset(seed=123)

    _, reward, _, _, info = env.step(1)

    assert isinstance(reward, float)
    assert info["state_consistent"] is True
    assert reward > 0.0
