from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "environment" / "data"

sys.path.insert(0, str(DATA))

from gym_env import GymInspectorEnv


def test_observation_space_membership():
    env = GymInspectorEnv()

    observation, _ = env.reset(seed=123)
    assert env.observation_space.contains(observation)

    for _ in range(3):
        observation, _, _, _, _ = env.step(1)
        assert env.observation_space.contains(observation)

    env.close()


def test_reward_is_float():
    env = GymInspectorEnv()

    env.reset(seed=123)
    _, reward, _, _, _ = env.step(1)

    assert isinstance(reward, float)

    env.close()
