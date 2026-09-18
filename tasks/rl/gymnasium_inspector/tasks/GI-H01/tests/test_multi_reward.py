import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "environment" / "data"

sys.path.insert(0, str(DATA))

from gym_env import GymInspectorEnv


def test_progress_reward_differs_by_action():
    env = GymInspectorEnv(max_steps=6)

    env.reset(seed=1)
    _, forward, _, _, _ = env.step(1)

    env.reset(seed=1)
    _, backward, _, _, _ = env.step(0)

    assert forward != backward

    env.close()


def test_completion_reward_is_distinct():
    env = GymInspectorEnv(max_steps=2)

    env.reset(seed=1)

    _, first, terminated, truncated, _ = env.step(1)

    assert terminated is False
    assert truncated is False

    _, final, terminated, truncated, _ = env.step(1)

    assert terminated is True
    assert truncated is False
    assert final > first

    env.close()


def test_truncation_reward_is_distinct():
    env = GymInspectorEnv(max_steps=3)

    env.reset(seed=1)

    env.step(0)
    env.step(0)

    _, reward, terminated, truncated, _ = env.step(0)

    assert terminated is False
    assert truncated is True
    assert isinstance(reward, float)

    env.close()
