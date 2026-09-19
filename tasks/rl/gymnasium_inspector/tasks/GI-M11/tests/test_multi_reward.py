import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "environment" / "data"

sys.path.insert(0, str(DATA))

from gym_env import GymInspectorEnv


def test_reward_changes_with_action():
    env = GymInspectorEnv(max_steps=5)

    env.reset(seed=1)
    _, reward_forward, _, _, _ = env.step(1)

    env.reset(seed=1)
    _, reward_backward, _, _, _ = env.step(0)

    assert reward_forward != reward_backward
    assert isinstance(reward_forward, float)
    assert isinstance(reward_backward, float)

    env.close()


def test_terminal_reward_component():
    env = GymInspectorEnv(max_steps=2)
    env.reset(seed=1)

    _, reward_1, terminated_1, truncated_1, _ = env.step(1)
    _, reward_2, terminated_2, truncated_2, _ = env.step(1)

    assert terminated_1 is False
    assert truncated_1 is False
    assert terminated_2 is True
    assert truncated_2 is False
    assert reward_2 > reward_1

    env.close()
