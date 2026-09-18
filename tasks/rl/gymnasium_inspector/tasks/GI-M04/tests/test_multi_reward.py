from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "environment" / "data"

sys.path.insert(0, str(DATA))

from gym_env import GymInspectorEnv


def test_reward_is_float():
    env = GymInspectorEnv()

    env.reset(seed=123)

    _, reward, _, _, _ = env.step(1)

    assert isinstance(reward, float)

    env.close()


def test_reward_changes_with_action():
    env = GymInspectorEnv()

    env.reset(seed=123)
    _, reward_forward, _, _, _ = env.step(1)

    env.reset(seed=123)
    _, reward_backward, _, _, _ = env.step(0)

    assert reward_forward != reward_backward

    env.close()
