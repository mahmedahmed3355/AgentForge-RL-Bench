from environment.data.gym_env import UnseenParameterCompositionEnv


def test_positive_reward_progression():
    env = UnseenParameterCompositionEnv()
    env.reset()

    _, r1, _, _, _ = env.step(0)
    _, r2, _, _, _ = env.step(1)
    _, r3, _, _, _ = env.step(2)
    _, r4, terminated, _, _ = env.step(3)

    assert r1 > 0
    assert r2 > 0
    assert r3 > 0
    assert r4 > 0
    assert terminated


def test_invalid_action_is_penalized():
    env = UnseenParameterCompositionEnv()
    env.reset()

    _, reward, _, _, info = env.step(99)

    assert reward < 0
    assert info["event"] == "invalid_action"
