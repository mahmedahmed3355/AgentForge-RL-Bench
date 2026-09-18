from environment.data.gym_env import MultiConditionVerificationEnv


def test_inspection_progress_component():
    env = MultiConditionVerificationEnv()
    env.reset(seed=123)

    _, reward, _, _, info = env.step(0)

    assert reward == 0.5
    assert info["reward_components"]["progress"] == 0.5


def test_validation_correctness_component():
    env = MultiConditionVerificationEnv()
    env.reset(seed=123)

    _, reward, _, _, info = env.step(1)

    assert reward == 1.5
    assert info["reward_components"]["correctness"] == 1.5


def test_resolution_has_multiple_components():
    env = MultiConditionVerificationEnv()
    env.reset(seed=123)
    env.step(1)

    _, reward, _, _, info = env.step(2)

    assert reward == 2.5
    assert info["reward_components"]["progress"] == 1.5
    assert info["reward_components"]["correctness"] == 1.0


def test_terminal_component():
    env = MultiConditionVerificationEnv()
    env.reset(seed=123)
    env.step(1)
    env.step(2)
    env.step(1)

    _, reward, _, _, info = env.step(3)

    assert reward == 5.0
    assert info["reward_components"]["terminal"] == 5.0


def test_penalty_component():
    env = MultiConditionVerificationEnv()
    env.reset(seed=123)

    _, reward, _, _, info = env.step(3)

    assert reward == -1.0
    assert info["reward_components"]["penalty"] == -1.0
