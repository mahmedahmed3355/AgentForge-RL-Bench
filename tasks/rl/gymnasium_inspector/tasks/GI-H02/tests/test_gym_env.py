import sys
from pathlib import Path

from gymnasium.utils.env_checker import check_env

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "environment" / "data"
sys.path.insert(0, str(DATA))

from gym_env import GymInspectorEnv


def test_gymnasium_checker():
    env = GymInspectorEnv(max_steps=6)
    check_env(env, skip_render_check=True)
    env.close()


def test_reset_contract():
    env = GymInspectorEnv()
    result = env.reset(seed=123)
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert env.observation_space.contains(result[0])
    assert isinstance(result[1], dict)


def test_step_contract():
    env = GymInspectorEnv()
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


def test_success_boundary():
    env = GymInspectorEnv(max_steps=6)
    env.reset(seed=123)

    for _ in range(6):
        result = env.step(1)

    _, _, terminated, truncated, _ = result

    assert terminated is True
    assert truncated is False
    assert not (terminated and truncated)


def test_horizon_boundary():
    env = GymInspectorEnv(max_steps=2)
    env.reset(seed=123)

    env.step(0)
    _, _, terminated, truncated, _ = env.step(0)

    assert terminated is False
    assert truncated is True
    assert not (terminated and truncated)


def test_action_space():
    env = GymInspectorEnv()
    assert env.action_space.contains(0)
    assert env.action_space.contains(1)
    assert not env.action_space.contains(2)


def test_deterministic_seed():
    env1 = GymInspectorEnv()
    env2 = GymInspectorEnv()

    o1, i1 = env1.reset(seed=42)
    o2, i2 = env2.reset(seed=42)

    assert o1.keys() == o2.keys()
    assert o1["position"].tolist() == o2["position"].tolist()
    assert o1["energy"].tolist() == o2["energy"].tolist()
    assert i1 == i2
