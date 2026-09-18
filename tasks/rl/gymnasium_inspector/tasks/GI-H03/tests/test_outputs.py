from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "environment" / "data"))

from gym_env import BranchingIrreversibleEnv


def test_reward_is_float():
    env = BranchingIrreversibleEnv()
    env.reset(seed=1)

    _, reward, _, _, _ = env.step(0)

    assert type(reward) is float


def test_flags_are_booleans():
    env = BranchingIrreversibleEnv()
    env.reset(seed=1)

    _, _, terminated, truncated, _ = env.step(0)

    assert type(terminated) is bool
    assert type(truncated) is bool


def test_info_is_dictionary():
    env = BranchingIrreversibleEnv()
    env.reset(seed=1)

    _, _, _, _, info = env.step(0)

    assert isinstance(info, dict)


def test_observation_numpy_resource_type():
    env = BranchingIrreversibleEnv()
    observation, _ = env.reset(seed=1)

    assert isinstance(observation["resources"], np.ndarray)
