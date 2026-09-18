import numpy as np
import gymnasium as gym
from gymnasium.utils.env_checker import check_env

from environment.data.gym_env import MultiConditionVerificationEnv


def test_reset_contract():
    env = MultiConditionVerificationEnv()
    obs, info = env.reset(seed=123)

    assert env.observation_space.contains(obs)
    assert isinstance(info, dict)
    assert info["stage"] == "inspect"


def test_action_space():
    env = MultiConditionVerificationEnv()
    assert env.action_space.n == 4


def test_observation_shape():
    env = MultiConditionVerificationEnv()
    obs, _ = env.reset()
    assert obs.shape == (6,)


def test_initial_conditions():
    env = MultiConditionVerificationEnv()
    _, info = env.reset()

    assert info["base_ready"] is True
    assert info["dependency_ready"] is False
    assert info["integrity_ok"] is True
    assert info["revalidated"] is False


def test_seed_determinism():
    env1 = MultiConditionVerificationEnv()
    env2 = MultiConditionVerificationEnv()

    obs1, info1 = env1.reset(seed=123)
    obs2, info2 = env2.reset(seed=123)

    assert np.array_equal(obs1, obs2)
    assert info1 == info2


def test_gymnasium_checker():
    env = MultiConditionVerificationEnv()
    check_env(env, skip_render_check=True)


def test_inspect_reward():
    env = MultiConditionVerificationEnv()
    env.reset(seed=123)

    _, reward, terminated, truncated, info = env.step(0)

    assert reward == 0.5
    assert terminated is False
    assert truncated is False
    assert info["event"] == "inspection"


def test_validate_before_resolution():
    env = MultiConditionVerificationEnv()
    env.reset(seed=123)
    env.step(0)

    _, reward, _, _, info = env.step(1)

    assert reward == 1.5
    assert info["validated"] is True
    assert info["dependency_ready"] is False
    assert info["revalidated"] is False


def test_resolution_requires_validation():
    env = MultiConditionVerificationEnv()
    env.reset(seed=123)

    _, reward, _, _, info = env.step(2)

    assert reward == -1.0
    assert info["dependency_ready"] is False


def test_resolution_preserves_base_state():
    env = MultiConditionVerificationEnv()
    env.reset(seed=123)
    env.step(1)
    env.step(2)

    assert env.base_ready is True
    assert env.dependency_ready is True
    assert env.integrity_ok is True


def test_revalidation_boundary():
    env = MultiConditionVerificationEnv()
    env.reset(seed=123)
    env.step(1)
    env.step(2)

    assert env.revalidated is False

    _, _, _, _, info = env.step(1)

    assert env.revalidated is True
    assert info["event"] == "revalidated_all_conditions"


def test_finish_requires_revalidation():
    env = MultiConditionVerificationEnv()
    env.reset(seed=123)
    env.step(1)
    env.step(2)

    _, reward, terminated, _, info = env.step(3)

    assert reward == -1.0
    assert terminated is False
    assert info["event"] == "terminal_rejected"


def test_finish_after_revalidation():
    env = MultiConditionVerificationEnv()
    env.reset(seed=123)
    env.step(1)
    env.step(2)
    env.step(1)

    _, reward, terminated, truncated, info = env.step(3)

    assert reward == 5.0
    assert terminated is True
    assert truncated is False
    assert info["event"] == "terminal_verified"


def test_step_after_terminal_rejected():
    env = MultiConditionVerificationEnv()
    env.reset(seed=123)
    env.step(1)
    env.step(2)
    env.step(1)
    env.step(3)

    try:
        env.step(0)
    except RuntimeError:
        pass
    else:
        raise AssertionError("step after terminal must raise RuntimeError")
