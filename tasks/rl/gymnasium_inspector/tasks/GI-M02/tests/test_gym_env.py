from pathlib import Path
import sys

from gymnasium.utils.env_checker import check_env

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "environment" / "data"
sys.path.insert(0, str(DATA))

from gym_env import GymInspectorEnv


def test_gymnasium_checker():
    env = GymInspectorEnv(max_steps=5)
    check_env(env)
    env.close()


def test_reset_contract():
    env = GymInspectorEnv(max_steps=5)
    observation, info = env.reset(seed=123)

    assert env.observation_space.contains(observation)
    assert isinstance(info, dict)

    env.close()


def test_step_contract():
    env = GymInspectorEnv(max_steps=5)
    env.reset(seed=123)

    observation, reward, terminated, truncated, info = env.step(1)

    assert env.observation_space.contains(observation)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)

    env.close()


def test_success_is_terminated_not_truncated():
    env = GymInspectorEnv(max_steps=4)
    env.reset(seed=123)

    for _ in range(4):
        observation, reward, terminated, truncated, info = env.step(1)

    assert terminated is True
    assert truncated is False

    env.close()


def test_horizon_is_truncation():
    env = GymInspectorEnv(max_steps=3)
    env.reset(seed=123)

    observation, reward, terminated, truncated, info = env.step(0)
    observation, reward, terminated, truncated, info = env.step(0)
    observation, reward, terminated, truncated, info = env.step(0)

    assert terminated is False
    assert truncated is True

    env.close()


def test_terminal_flags_are_mutually_exclusive():
    env = GymInspectorEnv(max_steps=4)
    env.reset(seed=123)

    for _ in range(4):
        observation, reward, terminated, truncated, info = env.step(1)

    assert not (terminated and truncated)

    env.close()
