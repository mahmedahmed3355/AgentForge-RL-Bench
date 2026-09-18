from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "environment" / "data"

sys.path.insert(0, str(DATA))

from gym_env import GymInspectorEnv


def test_environment_exists():
    assert (DATA / "gym_env.py").exists()


def test_environment_creation():
    env = GymInspectorEnv()
    try:
        assert env.action_space.n == 3
        assert "position" in env.observation_space.spaces
        assert "remaining" in env.observation_space.spaces
    finally:
        env.close()


def test_reset_contract():
    env = GymInspectorEnv()
    try:
        observation, info = env.reset(seed=1)

        assert isinstance(observation, dict)
        assert isinstance(info, dict)
        assert env.observation_space.contains(observation)
        assert info["step"] == 0
    finally:
        env.close()


def test_step_contract():
    env = GymInspectorEnv()
    try:
        env.reset(seed=1)

        observation, reward, terminated, truncated, info = env.step(1)

        assert env.observation_space.contains(observation)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
    finally:
        env.close()


def test_success_trajectory():
    env = GymInspectorEnv(max_steps=4)
    try:
        env.reset(seed=7)

        for _ in range(4):
            observation, reward, terminated, truncated, info = env.step(1)

        assert terminated is True
        assert truncated is False
        assert observation["position"][0] == np.float32(4.0)
    finally:
        env.close()


def test_reset_isolation():
    env = GymInspectorEnv(max_steps=4)
    try:
        env.reset(seed=10)
        env.step(1)
        env.step(1)

        observation, info = env.reset(seed=10)

        assert observation["position"][0] == np.float32(0.0)
        assert observation["remaining"][0] == np.float32(4.0)
        assert info["step"] == 0
    finally:
        env.close()


def test_seeded_reset_determinism():
    env = GymInspectorEnv()

    try:
        first, _ = env.reset(seed=123)
        second, _ = env.reset(seed=123)

        assert first.keys() == second.keys()

        for key in first:
            assert np.array_equal(
                first[key],
                second[key],
            )
    finally:
        env.close()
