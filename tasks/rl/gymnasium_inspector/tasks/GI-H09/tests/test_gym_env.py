from environment.data.gym_env import ComposedInspectorEnv


def test_reset_contract():
    env = ComposedInspectorEnv()
    obs, info = env.reset(seed=123)

    assert isinstance(obs, dict)
    assert info["inspector_contract"] is True
    assert info["public_information"] is True


def test_seed_determinism():
    env1 = ComposedInspectorEnv()
    env2 = ComposedInspectorEnv()

    obs1, _ = env1.reset(seed=123)
    obs2, _ = env2.reset(seed=123)

    assert obs1 == obs2


def test_inspection_sequence():
    env = ComposedInspectorEnv()
    env.reset(seed=123)

    _, r1, _, _, i1 = env.step(0)
    _, r2, _, _, i2 = env.step(0)
    _, r3, _, _, i3 = env.step(0)

    assert r1 == 0.5
    assert r2 == 0.5
    assert r3 == 0.5
    assert i1["event"] == "state_inspected"
    assert i2["event"] == "history_inspected"
    assert i3["event"] == "diagnostics_inspected"


def test_incomplete_correlation_rejected():
    env = ComposedInspectorEnv()
    env.reset(seed=123)

    env.step(0)
    _, reward, _, _, info = env.step(1)

    assert reward < 0
    assert info["correlated"] is False


def test_terminal_requires_valid_action():
    env = ComposedInspectorEnv()
    env.reset(seed=123)

    for action in [0, 0, 0, 1, 2]:
        env.step(action)

    _, reward, terminated, _, info = env.step(3)

    assert terminated is True
    assert reward == 5.0
    assert info["event"] == "terminal_verified"
