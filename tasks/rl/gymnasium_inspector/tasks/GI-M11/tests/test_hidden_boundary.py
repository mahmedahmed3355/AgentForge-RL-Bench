import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "environment" / "data"

sys.path.insert(0, str(DATA))

from gym_env import GymInspectorEnv


def test_horizon_boundary():
    env = GymInspectorEnv(max_steps=3)
    env.reset(seed=1)

    env.step(0)
    env.step(0)

    observation, reward, terminated, truncated, info = env.step(0)

    assert terminated is False
    assert truncated is True
    assert not (terminated and truncated)
    assert env.observation_space.contains(observation)

    env.close()


def test_success_boundary():
    env = GymInspectorEnv(max_steps=3)
    env.reset(seed=1)

    for _ in range(3):
        observation, reward, terminated, truncated, info = env.step(1)

    assert terminated is True
    assert truncated is False
    assert not (terminated and truncated)

    env.close()
