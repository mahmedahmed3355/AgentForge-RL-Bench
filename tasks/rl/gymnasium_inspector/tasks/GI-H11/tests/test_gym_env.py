from environment.data.gym_env import UnseenParameterCompositionEnv


def test_reset_observation_contract():
    env = UnseenParameterCompositionEnv()
    obs, info = env.reset()

    assert isinstance(obs, dict)
    assert obs["inspected"] == 0
    assert obs["planned"] == 0
    assert obs["executed"] == 0
    assert obs["verified"] == 0
    assert info["event"] == "reset"


def test_inspection_is_required():
    env = UnseenParameterCompositionEnv()
    env.reset()

    _, reward, _, _, info = env.step(1)

    assert reward < 0
    assert info["event"] == "plan_without_inspection"


def test_rule_consistent_execution():
    env = UnseenParameterCompositionEnv()
    env.reset()

    env.step(0)
    env.step(1)
    obs, reward, terminated, _, info = env.step(2)

    assert reward > 0
    assert not terminated
    assert obs["executed"] == 1
    assert obs["constraint_ok"] == 1
    assert info["event"] == "rule_consistent_execution"


def test_terminal_verification():
    env = UnseenParameterCompositionEnv()
    env.reset()

    env.step(0)
    env.step(1)
    env.step(2)

    obs, reward, terminated, _, info = env.step(3)

    assert reward > 0
    assert terminated
    assert obs["verified"] == 1
    assert info["event"] == "terminal_verified"


def test_wrong_finish_is_rejected():
    env = UnseenParameterCompositionEnv()
    env.reset()

    _, reward, terminated, _, info = env.step(3)

    assert reward < 0
    assert not terminated
    assert info["event"] == "invalid_finish"
