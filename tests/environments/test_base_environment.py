from agentforge.environments import (
    BaseEnvironment,
    EnvironmentStep,
    MockBackendEnvironment,
)


def test_mock_backend_implements_environment_contract():
    env = MockBackendEnvironment()

    assert isinstance(env, BaseEnvironment)

    observation = env.reset()

    assert observation == {
        "state": "initial",
        "step": 0,
    }

    env.close()


def test_environment_step_returns_observation_reward_and_done():
    env = MockBackendEnvironment()

    env.reset()

    result = env.step("work")

    assert isinstance(result, EnvironmentStep)
    assert result.observation["state"] == "working"
    assert result.observation["step"] == 1
    assert result.reward.total == 0.1
    assert result.done is False

    env.close()


def test_environment_can_finish_with_solve_action():
    env = MockBackendEnvironment()

    env.reset()

    result = env.step("solve")

    assert result.observation["state"] == "solved"
    assert result.reward.progress == 0.2
    assert result.reward.correctness == 0.5
    assert result.reward.terminal == 1.0
    assert result.reward.total == 1.7
    assert result.done is True

    env.close()


def test_environment_rejects_steps_after_max_steps():
    env = MockBackendEnvironment(max_steps=1)

    env.reset()

    result = env.step("work")

    assert result.done is True

    try:
        env.step("work")
        raised = False
    except RuntimeError:
        raised = True

    assert raised is True

    env.close()
