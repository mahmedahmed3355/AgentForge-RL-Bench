from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "environment" / "data"

sys.path.insert(0, str(DATA))

from gym_env import GymInspectorEnv


def test_reset_observation_is_valid():
    env = GymInspectorEnv()

    observation, _ = env.reset(seed=7)

    assert env.observation_space.contains(observation)


def test_step_observation_is_valid():
    env = GymInspectorEnv()

    env.reset(seed=7)

    observation, reward, terminated, truncated, info = env.step(1)

    assert isinstance(observation, np.ndarray)
    assert observation.dtype == np.int32
    assert env.observation_space.contains(observation)
