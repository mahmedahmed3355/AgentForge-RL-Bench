from pathlib import Path
import sys

import gymnasium as gym
from gymnasium.utils.env_checker import check_env

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "environment" / "data"
sys.path.insert(0, str(DATA))

from gym_env import GymInspectorEnv


def test_gymnasium_checker():
    env = GymInspectorEnv(max_steps=6)
    check_env(env)
    env.close()


def test_reset_contract():
    env = GymInspectorEnv()
    result = env.reset(seed=1)

    assert isinstance(result, tuple)
    assert len(result) == 2

    observation, info = result

    assert env.observation_space.contains(observation)
    assert isinstance(info, dict)


def test_step_contract():
    env = GymInspectorEnv()
    env.reset(seed=1)

    result = env.step(1)

    assert isinstance(result, tuple)
    assert len(result) == 5

    observation, reward, terminated, truncated, info = result

    assert env.observation_space.contains(observation)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)
    assert not (terminated and truncated)
