from environment.data.gym_env import ComposedInspectorEnv


def test_inspection_reward():
    env = ComposedInspectorEnv()
    env.reset(seed=123)

    _, reward, _, _, _ = env.step(0)

    assert reward == 0.5


def test_correlation_reward():
    env = ComposedInspectorEnv()
    env.reset(seed=123)

    env.step(0)
    env.step(0)
    env.step(0)

    _, reward, _, _, info = env.step(1)

    assert reward == 2.0
    assert info["correlated"] is True


def test_terminal_reward():
    env = ComposedInspectorEnv()
    env.reset(seed=123)

    for action in [0, 0, 0, 1, 2]:
        env.step(action)

    _, reward, terminated, _, _ = env.step(3)

    assert reward == 5.0
    assert terminated is True
