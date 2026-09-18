from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "environment" / "data"

sys.path.insert(0, str(DATA))

from gym_env import GymInspectorEnv


def test_observation_space_membership():
    env = GymInspectorEnv(max_steps=8)
    observation, _ = env.reset(seed=123)

    assert env.observation_space.contains(observation)

    for _ in range(4):
        observation, _, _, _, _ = env.step(1)
        assert env.observation_space.contains(observation)

    env.close()


def test_observation_dtype():
    env = GymInspectorEnv(max_steps=8)
    observation, _ = env.reset(seed=123)
    assert isinstance(observation, np.ndarray)
    assert observation.dtype == np.int32
    env.close()
