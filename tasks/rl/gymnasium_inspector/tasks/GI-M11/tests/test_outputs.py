import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "environment" / "data"

sys.path.insert(0, str(DATA))

from gym_env import GymInspectorEnv


def test_observation_dtype():
    env = GymInspectorEnv()
    observation, _ = env.reset(seed=7)

    assert isinstance(observation, np.ndarray)
    assert observation.dtype == np.int32
    assert env.observation_space.contains(observation)

    env.close()


def test_reward_is_float():
    env = GymInspectorEnv()
    env.reset(seed=7)

    _, reward, _, _, _ = env.step(1)

    assert type(reward) is float

    env.close()
