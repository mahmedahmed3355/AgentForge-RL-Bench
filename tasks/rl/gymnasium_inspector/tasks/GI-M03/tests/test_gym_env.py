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


def test_reset_observation_belongs_to_space():
    env = GymInspectorEnv(max_steps=6)
    observation, info = env.reset(seed=123)

    assert env.observation_space.contains(observation)
    assert isinstance(info, dict)

    env.close()


def test_all_declared_actions_are_accepted():
    env = GymInspectorEnv(max_steps=10)
    env.reset(seed=123)

    for action in range(env.action_space.n):
        assert env.action_space.contains(action)

        observation, reward, terminated, truncated, info = env.step(action)

        assert env.observation_space.contains(observation)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

        if terminated or truncated:
            env.reset(seed=123)

    env.close()


def test_invalid_action_is_rejected():
    env = GymInspectorEnv(max_steps=6)
    env.reset(seed=123)

    try:
        env.step(99)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid action was silently accepted")

    env.close()


def test_step_observations_stay_inside_space():
    env = GymInspectorEnv(max_steps=6)
    env.reset(seed=123)

    for action in [0, 1, 2, 2, 1]:
        observation, _, terminated, truncated, _ = env.step(action)

        assert env.observation_space.contains(observation)

        if terminated or truncated:
            break

    env.close()
