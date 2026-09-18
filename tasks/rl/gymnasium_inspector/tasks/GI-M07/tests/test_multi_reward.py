from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "environment" / "data"

sys.path.insert(0, str(DATA))

from gym_env import GymInspectorEnv


def test_reward_is_composed_from_multiple_components():
    env = GymInspectorEnv()

    env.reset(seed=123)

    _, reward_continue, _, _, _ = env.step(1)

    env.reset(seed=123)

    _, reward_idle, _, _, _ = env.step(0)

    assert isinstance(reward_continue, float)
    assert isinstance(reward_idle, float)
    assert reward_continue != reward_idle

    env.close()
