from environment.data.gym_env import UnseenParameterCompositionEnv


def test_execution_requires_publicly_derived_plan():
    env = UnseenParameterCompositionEnv()
    env.reset()

    env.step(0)
    _, reward, _, _, info = env.step(2)

    assert reward < 0
    assert info["event"] == "execute_without_plan"


def test_constraint_violation_is_rejected():
    env = UnseenParameterCompositionEnv()
    env.limit = env.expected_result - 1
    env.reset()

    env.step(0)
    env.step(1)
    _, reward, _, _, info = env.step(2)

    assert reward < 0
    assert info["event"] == "constraint_violation"
