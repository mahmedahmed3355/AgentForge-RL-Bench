from environment.data.gym_env import ActionEfficientCompletionEnv


def test_progress_reward_component():
    env = ActionEfficientCompletionEnv()
    env.reset(seed=123)
    env.step(env.INSPECT)

    _, reward, _, _, info = env.step(
        env.FAST_EXECUTE
    )

    assert reward == 2.0
    assert info["progress"] == 2


def test_efficiency_budget_is_tracked():
    env = ActionEfficientCompletionEnv()
    env.reset(seed=123)
    env.step(env.INSPECT)
    env.step(env.FAST_EXECUTE)

    assert env.spent_cost == 2
    assert env.EFFICIENCY_BUDGET == 6


def test_terminal_reward():
    env = ActionEfficientCompletionEnv()
    env.reset(seed=123)
    env.step(env.INSPECT)
    env.step(env.FAST_EXECUTE)
    env.step(env.FAST_EXECUTE)

    _, reward, terminated, truncated, info = env.step(
        env.FINISH
    )

    assert reward == 5.0
    assert terminated is True
    assert truncated is False
    assert info["event"] == "terminal_verified"
