from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium.utils import env_checker

from environment.data.gym_env import AmbiguousFailureDiagnosisEnv


def test_reset_contract():
    env = AmbiguousFailureDiagnosisEnv()
    result = env.reset(seed=123)

    assert isinstance(result, tuple)
    assert len(result) == 2

    obs, info = result
    assert env.observation_space.contains(obs)
    assert isinstance(info, dict)


def test_step_contract():
    env = AmbiguousFailureDiagnosisEnv()
    env.reset(seed=123)

    result = env.step(0)

    assert isinstance(result, tuple)
    assert len(result) == 5

    obs, reward, terminated, truncated, info = result

    assert env.observation_space.contains(obs)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)


def test_gymnasium_checker():
    env = AmbiguousFailureDiagnosisEnv()
    gym.utils.env_checker.check_env(env, skip_render_check=True)


def test_seed_determinism():
    env1 = AmbiguousFailureDiagnosisEnv()
    env2 = AmbiguousFailureDiagnosisEnv()

    obs1, info1 = env1.reset(seed=123)
    obs2, info2 = env2.reset(seed=123)

    assert np.array_equal(obs1, obs2)
    assert info1 == info2


def test_different_seed_preserves_contract():
    env = AmbiguousFailureDiagnosisEnv()
    obs, info = env.reset(seed=999)

    assert env.observation_space.contains(obs)
    assert isinstance(info, dict)


def test_action_space():
    env = AmbiguousFailureDiagnosisEnv()

    for action in range(env.action_space.n):
        assert env.action_space.contains(action)


def test_successful_completion():
    env = AmbiguousFailureDiagnosisEnv()
    env.reset(seed=123)

    for action in [0, 1, 2, 3]:
        obs, reward, terminated, truncated, info = env.step(action)

    assert terminated is True
    assert truncated is False
    assert info["diagnosis"] == env.ROOT_FAILURE
    assert info["recovered"] is True
    assert info["stage"] == "verify"


def test_terminal_flags_are_exclusive():
    env = AmbiguousFailureDiagnosisEnv()
    env.reset(seed=123)

    for action in [0, 1, 2, 3]:
        _, _, terminated, truncated, _ = env.step(action)
        assert not (terminated and truncated)


def test_horizon_truncation():
    env = AmbiguousFailureDiagnosisEnv()
    env.reset(seed=123)

    for _ in range(env.max_steps):
        _, _, terminated, truncated, _ = env.step(0)

    assert terminated is False
    assert truncated is True


def test_observations_remain_valid():
    env = AmbiguousFailureDiagnosisEnv()
    obs, _ = env.reset(seed=123)

    assert env.observation_space.contains(obs)

    for action in [0, 1, 2, 3]:
        if not env.terminated:
            obs, _, _, _, _ = env.step(action)
            assert env.observation_space.contains(obs)
