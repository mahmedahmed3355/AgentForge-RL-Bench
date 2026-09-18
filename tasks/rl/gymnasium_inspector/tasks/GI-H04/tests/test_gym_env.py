from pathlib import Path

import numpy as np
import sys

import gymnasium as gym

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "environment" / "data"

sys.path.insert(0, str(DATA))

from gym_env import DependencyGraphEnv


def test_reset_contract():
    env = DependencyGraphEnv()

    observation, info = env.reset(seed=123)

    assert isinstance(observation, dict)
    assert isinstance(info, dict)
    assert env.observation_space.contains(observation)


def test_step_contract():
    env = DependencyGraphEnv()

    env.reset(seed=123)

    result = env.step(0)

    assert len(result) == 5

    observation, reward, terminated, truncated, info = result

    assert env.observation_space.contains(observation)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)


def test_gymnasium_checker():
    from gymnasium.utils.env_checker import check_env

    env = DependencyGraphEnv()
    check_env(env, skip_render_check=True)


def test_action_space():
    env = DependencyGraphEnv()

    for action in range(6):
        assert env.action_space.contains(action)


def test_seed_determinism():
    env1 = DependencyGraphEnv()
    env2 = DependencyGraphEnv()

    obs1, info1 = env1.reset(seed=123)
    obs2, info2 = env2.reset(seed=123)

    assert np.array_equal(obs1["completed"], obs2["completed"])
    assert obs1["progress"] == obs2["progress"]
    assert obs1["step"] == obs2["step"]
    assert info1 == info2

    for action in [0, 1, 2, 3, 4, 5]:
        r1 = env1.step(action)
        r2 = env2.step(action)

        obs_a, reward_a, terminated_a, truncated_a, info_a = r1
        obs_b, reward_b, terminated_b, truncated_b, info_b = r2

        assert np.array_equal(obs_a["completed"], obs_b["completed"])
        assert obs_a["progress"] == obs_b["progress"]
        assert obs_a["step"] == obs_b["step"]
        assert reward_a == reward_b
        assert terminated_a == terminated_b
        assert truncated_a == truncated_b
        assert info_a == info_b
