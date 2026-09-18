from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "environment" / "data"

sys.path.insert(0, str(DATA))

from gym_env import GymInspectorEnv


def test_multi_reward_components():
    env = GymInspectorEnv(max_steps=8)
    env.reset(seed=123)

    _, reward_move, _, _, _ = env.step(1)
    assert isinstance(reward_move, float)

    _, reward_idle, _, _, _ = env.step(0)
    assert isinstance(reward_idle, float)
    assert reward_move != reward_idle

    env.close()
