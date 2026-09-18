from pathlib import Path
import sys

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

    result = env.reset(seed=123)

    assert isinstance(result, tuple)
    assert len(result) == 2

    observation, info = result

    assert env.observation_space.contains(observation)
    assert isinstance(info, dict)

    env.close()


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

    env.close()


def test_action_space_contract():
    env = GymInspectorEnv()

    env.reset(seed=123)

    for action in range(env.action_space.n):
        result = env.step(action)
        assert env.observation_space.contains(result[0])

    env.close()


def test_success_terminates():
    env = GymInspectorEnv(max_steps=6)

    env.reset(seed=123)

    result = None

    for _ in range(6):
        result = env.step(1)
        if result[2] or result[3]:
            break

    assert result is not None
    assert result[2] is True
    assert result[3] is False

    env.close()


def test_no_dual_terminal_flags():
    env = GymInspectorEnv(max_steps=6)

    env.reset(seed=123)

    for _ in range(6):
        _, _, terminated, truncated, _ = env.step(1)
        assert not (terminated and truncated)

        if terminated or truncated:
            break

    env.close()


def test_seed_determinism():
    env = GymInspectorEnv()

    obs1, info1 = env.reset(seed=123)
    obs2, info2 = env.reset(seed=123)

    assert obs1.tolist() == obs2.tolist()
    assert info1 == info2

    env.close()
