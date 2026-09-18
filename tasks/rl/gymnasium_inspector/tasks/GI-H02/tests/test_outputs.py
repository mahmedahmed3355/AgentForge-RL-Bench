import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "environment" / "data"
sys.path.insert(0, str(DATA))

from gym_env import GymInspectorEnv


def test_observation_membership():
    env = GymInspectorEnv()
    observation, _ = env.reset(seed=123)

    assert env.observation_space.contains(observation)

    observation, reward, terminated, truncated, info = env.step(1)

    assert env.observation_space.contains(observation)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)
