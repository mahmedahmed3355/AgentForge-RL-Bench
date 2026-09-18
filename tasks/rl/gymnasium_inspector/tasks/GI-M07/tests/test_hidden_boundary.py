from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "environment" / "data"

sys.path.insert(0, str(DATA))

from gym_env import GymInspectorEnv


def test_success_boundary():
    env = GymInspectorEnv(max_steps=8)

    env.reset(seed=123)

    for _ in range(7):
        observation, reward, terminated, truncated, info = env.step(1)
        assert not terminated
        assert not truncated

    observation, reward, terminated, truncated, info = env.step(1)

    assert terminated is True
    assert truncated is False

    env.close()


def test_horizon_boundary():
    env = GymInspectorEnv(max_steps=3)

    env.reset(seed=123)

    env.step(0)
    env.step(0)
    observation, reward, terminated, truncated, info = env.step(0)

    assert terminated is False
    assert truncated is True
    assert not (terminated and truncated)

    env.close()
