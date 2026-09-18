from environment.data.gym_env import UnseenParameterCompositionEnv


def test_constraint_boundary_is_valid():
    env = UnseenParameterCompositionEnv()
    env.limit = env.expected_result
    env.reset()

    env.step(0)
    env.step(1)
    obs, reward, _, _, info = env.step(2)

    assert reward > 0
    assert obs["constraint_ok"] == 1
    assert info["event"] == "rule_consistent_execution"


def test_constraint_below_result_rejects_execution():
    env = UnseenParameterCompositionEnv()
    env.limit = env.expected_result - 1
    env.reset()

    env.step(0)
    env.step(1)
    obs, reward, _, _, info = env.step(2)

    assert reward < 0
    assert obs["constraint_ok"] == 0
    assert info["event"] == "constraint_violation"
