import numpy as np
import gymnasium as gym

from environment.data.gym_env import ActionEfficientCompletionEnv


def test_reset_contract():
    env = ActionEfficientCompletionEnv()
    obs, info = env.reset(seed=123)

    assert env.observation_space.contains(obs)
    assert isinstance(info, dict)
    assert info["stage"] == "inspect"


def test_gymnasium_checker():
    from gymnasium.utils.env_checker import check_env

    env = ActionEfficientCompletionEnv()
    check_env(env, skip_render_check=True)


def test_seed_determinism():
    env1 = ActionEfficientCompletionEnv()
    env2 = ActionEfficientCompletionEnv()

    obs1, info1 = env1.reset(seed=123)
    obs2, info2 = env2.reset(seed=123)

    assert np.array_equal(obs1, obs2)
    assert info1 == info2


def test_fast_action_requires_inspection():
    env = ActionEfficientCompletionEnv()
    env.reset(seed=123)

    _, reward, _, _, info = env.step(
        env.FAST_EXECUTE
    )

    assert reward < 0
    assert info["valid_action"] is False


def test_finish_requires_completion():
    env = ActionEfficientCompletionEnv()
    env.reset(seed=123)
    env.step(env.INSPECT)

    _, reward, terminated, truncated, info = env.step(
        env.FINISH
    )

    assert reward < 0
    assert not terminated
    assert not truncated
    assert info["event"] == "finish_before_completion"


def test_fast_execution_progress():
    env = ActionEfficientCompletionEnv()
    env.reset(seed=123)
    env.step(env.INSPECT)

    obs, reward, _, _, info = env.step(
        env.FAST_EXECUTE
    )

    assert obs[0] == 2
    assert obs[1] == 2
    assert reward == 2.0
    assert info["valid_action"] is True
