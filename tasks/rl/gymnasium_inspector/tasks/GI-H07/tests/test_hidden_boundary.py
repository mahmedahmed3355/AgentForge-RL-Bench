from environment.data.gym_env import MultiConditionVerificationEnv


def test_superficial_success_is_not_terminal():
    env = MultiConditionVerificationEnv()
    env.reset(seed=123)

    env.step(1)
    env.step(2)

    assert env.dependency_ready is True
    assert env.integrity_ok is True
    assert env.revalidated is False

    _, reward, terminated, _, info = env.step(3)

    assert reward == -1.0
    assert terminated is False
    assert info["event"] == "terminal_rejected"


def test_revalidation_changes_terminal_boundary():
    env = MultiConditionVerificationEnv()
    env.reset(seed=123)

    env.step(1)
    env.step(2)
    env.step(1)

    assert env.revalidated is True

    _, reward, terminated, _, _ = env.step(3)

    assert reward == 5.0
    assert terminated is True
