import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "environment" / "data"

sys.path.insert(0, str(DATA))

from gym_env import DependencyGraphEnv


def test_reward_progress_component():
    env = DependencyGraphEnv()
    env.reset(seed=123)

    _, reward, _, _, info = env.step(0)

    assert reward == 1.0
    assert info["stage"] == "execute"
    assert info["completed_count"] == 1


def test_terminal_reward_component():
    env = DependencyGraphEnv(max_steps=8)
    env.reset(seed=123)

    for action in [0, 1, 2, 3, 4]:
        env.step(action)

    _, reward, terminated, truncated, info = env.step(5)

    assert terminated is True
    assert truncated is False
    assert reward == 5.0
    assert info["stage"] == "finish"
