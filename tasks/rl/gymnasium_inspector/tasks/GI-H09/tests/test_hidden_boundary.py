from environment.data.gym_env import ComposedInspectorEnv


def test_single_view_is_insufficient():
    env = ComposedInspectorEnv()
    env.reset(seed=123)

    env.step(0)

    _, _, _, _, info = env.step(1)

    assert info["correlated"] is False


def test_all_public_views_enable_correlation():
    env = ComposedInspectorEnv()
    env.reset(seed=123)

    env.step(0)
    env.step(0)
    env.step(0)

    _, _, _, _, info = env.step(1)

    assert info["correlated"] is True
    assert info["diagnosis"] == 2
