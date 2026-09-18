from pathlib import Path
import sys

from gymnasium.utils.env_checker import check_env

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "environment" / "data"

sys.path.insert(0, str(DATA))

from gym_env import GymInspectorEnv


def test_gymnasium_checker():
    env = GymInspectorEnv(max_steps=8)
    check_env(env)
    env.close()


def test_reset_contract():
    env = GymInspectorEnv(max_steps=8)
    result = env.reset(seed=123)
    assert isinstance(result, tuple)
    assert len(result) == 2
    observation, info = result
    assert env.observation_space.contains(observation)
    assert isinstance(info, dict)
    env.close()


def test_step_contract():
    env = GymInspectorEnv(max_steps=8)
    env.reset(seed=123)
    result = env.step(1)
    assert isinstance(result, tuple)
    assert len(result) == 5
    observation, reward, terminated, truncated, info = result
    assert env.observation_space.contains(observation)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)
    env.close()
