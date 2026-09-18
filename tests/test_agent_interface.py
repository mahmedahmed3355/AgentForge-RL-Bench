from agentforge.agents import BaseAgent, TestAgent


def test_test_agent_implements_base_agent():
    agent = TestAgent()

    assert isinstance(agent, BaseAgent)


def test_agent_reset_and_act():
    agent = TestAgent()

    agent.reset()

    action = agent.act(
        {"state": "start"}
    )

    assert action["type"] == "test_action"
    assert action["step"] == 1
    assert action["observation"] == {"state": "start"}
    assert agent.step_count == 1


def test_agent_update():
    agent = TestAgent()

    agent.reset()

    action = agent.act(
        {"state": "start"}
    )

    agent.update(
        observation={"state": "start"},
        action=action,
        reward=1.5,
        next_observation={"state": "done"},
        done=True,
    )

    assert agent.update_count == 1


def test_agent_checkpoint_round_trip(tmp_path):
    agent = TestAgent()

    agent.reset()
    agent.act({"state": "one"})
    agent.act({"state": "two"})

    agent.update(
        observation={"state": "one"},
        action={"type": "test_action"},
        reward=1.0,
        next_observation={"state": "two"},
        done=False,
    )

    checkpoint = tmp_path / "agent.json"

    agent.save_checkpoint(str(checkpoint))

    restored = TestAgent()
    restored.load_checkpoint(str(checkpoint))

    assert restored.step_count == 2
    assert restored.update_count == 1


def test_agent_can_continue_after_resume(tmp_path):
    agent = TestAgent()

    agent.act({"state": "start"})
    checkpoint = tmp_path / "agent.json"

    agent.save_checkpoint(str(checkpoint))

    restored = TestAgent()
    restored.load_checkpoint(str(checkpoint))

    action = restored.act(
        {"state": "resumed"}
    )

    assert action["step"] == 2
    assert restored.step_count == 2
