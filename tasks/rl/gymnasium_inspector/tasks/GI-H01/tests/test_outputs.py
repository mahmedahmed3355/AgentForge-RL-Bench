import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "environment" / "data"

sys.path.insert(0, str(DATA))

from gym_env import GymInspectorEnv


def test_observation_membership():
    env = GymInspectorEnv(max_steps=6)

    observation, _ = env.reset(seed=7)

    assert env.observation_space.contains(observation)

    for _ in range(3):
        observation, _, terminated, truncated, _ = env.step(1)
        assert env.observation_space.contains(observation)

        if terminated or truncated:
            break

    env.close()


def test_reward_is_python_float():
    env = GymInspectorEnv()

    env.reset(seed=7)

    _, reward, _, _, _ = env.step(1)

    assert type(reward) is float

    env.close()


def test_observation_components_are_numpy_arrays():
    env = GymInspectorEnv()

    observation, _ = env.reset(seed=7)

    assert isinstance(
        observation["position"],
        np.ndarray,
    )
    assert isinstance(
        observation["step"],
        np.ndarray,
    )

    env.close()
