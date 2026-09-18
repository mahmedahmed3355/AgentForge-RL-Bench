import json

from agentforge.agents import AgentAdapter


class ExternalMockAgent:
    def __init__(self):
        self.reset_calls = 0
        self.actions = []
        self.updates = []

    def reset(self):
        self.reset_calls += 1

    def act(self, observation):
        action = {
            "type": "external_action",
            "state": observation["state"],
        }
        self.actions.append(action)
        return action

    def update(
        self,
        observation,
        action,
        reward,
        next_observation,
        done,
    ):
        self.updates.append(
            {
                "observation": observation,
                "action": action,
                "reward": reward,
                "next_observation": next_observation,
                "done": done,
            }
        )


class CheckpointableExternalAgent(ExternalMockAgent):
    def __init__(self):
        super().__init__()
        self.state = {"updates": 0}
        self.saved_path = None
        self.loaded_path = None

    def update(
        self,
        observation,
        action,
        reward,
        next_observation,
        done,
    ):
        super().update(
            observation,
            action,
            reward,
            next_observation,
            done,
        )
        self.state["updates"] += 1

    def save_checkpoint(self, path):
        self.saved_path = path
        path.write_text(
            json.dumps(self.state),
            encoding="utf-8",
        )
        return path

    def load_checkpoint(self, path):
        self.loaded_path = path
        self.state = json.loads(
            path.read_text(encoding="utf-8")
        )


def test_external_agent_adapter_reset():
    external = ExternalMockAgent()
    adapter = AgentAdapter(external)

    adapter.reset()

    assert external.reset_calls == 1


def test_external_agent_adapter_forwards_action():
    external = ExternalMockAgent()
    adapter = AgentAdapter(external)

    observation = {"state": "start"}

    action = adapter.act(observation)

    assert action == {
        "type": "external_action",
        "state": "start",
    }

    assert external.actions == [action]


def test_external_agent_adapter_forwards_update():
    external = ExternalMockAgent()
    adapter = AgentAdapter(external)

    observation = {"state": "start"}
    action = {"type": "work"}
    next_observation = {"state": "done"}

    adapter.update(
        observation=observation,
        action=action,
        reward=1.5,
        next_observation=next_observation,
        done=True,
    )

    assert len(external.updates) == 1

    update = external.updates[0]

    assert update["observation"] == observation
    assert update["action"] == action
    assert update["reward"] == 1.5
    assert update["next_observation"] == next_observation
    assert update["done"] is True


def test_adapter_preserves_external_agent_identity():
    external = ExternalMockAgent()
    adapter = AgentAdapter(external)

    assert adapter.external_agent is external


def test_adapter_supports_fallback_checkpoint(tmp_path):
    external = ExternalMockAgent()
    adapter = AgentAdapter(external)

    adapter._checkpoint_state = {
        "training_step": 42,
        "policy_version": "v1",
    }

    path = adapter.save_checkpoint(
        tmp_path / "adapter.json"
    )

    assert path.exists()

    restored = AgentAdapter(
        ExternalMockAgent()
    )

    restored.load_checkpoint(path)

    assert restored._checkpoint_state == {
        "training_step": 42,
        "policy_version": "v1",
    }


def test_adapter_delegates_checkpoint_to_external_agent(tmp_path):
    external = CheckpointableExternalAgent()
    adapter = AgentAdapter(external)

    external.state["updates"] = 17

    path = adapter.save_checkpoint(
        tmp_path / "external.json"
    )

    assert external.saved_path == path
    assert json.loads(
        path.read_text(encoding="utf-8")
    )["updates"] == 17


def test_adapter_delegates_checkpoint_load(tmp_path):
    external = CheckpointableExternalAgent()
    adapter = AgentAdapter(external)

    external.state["updates"] = 23

    path = adapter.save_checkpoint(
        tmp_path / "external.json"
    )

    external.state["updates"] = 0

    adapter.load_checkpoint(path)

    assert external.loaded_path == path
    assert external.state["updates"] == 23
