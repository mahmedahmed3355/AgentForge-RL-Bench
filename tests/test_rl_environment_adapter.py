import pytest

from agentforge.core import RewardBreakdown
from agentforge.environments import (
    BaseEnvironment,
    EnvironmentStep,
    RLEnvironmentAdapter,
)


class DeterministicRLEnvironment(BaseEnvironment):
    def __init__(self):
        self.step_count = 0
        self.closed = False

    def reset(self):
        self.step_count = 0
        return {"state": "start"}

    def step(self, action):
        self.step_count += 1

        done = self.step_count >= 2

        return EnvironmentStep(
            observation={
                "state": f"step-{self.step_count}",
            },
            reward=RewardBreakdown(
                progress=0.5,
            ),
            done=done,
            info={
                "action_type": action["type"],
                "step": self.step_count,
            },
        )

    def close(self):
        self.closed = True


def test_reset_returns_initial_observation():
    environment = DeterministicRLEnvironment()
    adapter = RLEnvironmentAdapter(environment)

    observation, info = adapter.reset()
    assert info == {}

    assert observation == {
        "state": "start",
    }

    assert adapter.terminated is False
    assert adapter.truncated is False


def test_step_returns_gymnasium_style_transition():
    environment = DeterministicRLEnvironment()
    adapter = RLEnvironmentAdapter(environment)

    adapter.reset()

    observation, reward, terminated, truncated, info = (
        adapter.step({"type": "work"})
    )

    assert observation == {
        "state": "step-1",
    }

    assert reward == 0.5
    assert terminated is False
    assert truncated is False

    assert info == {
        "action_type": "work",
        "step": 1,
    }


def test_terminal_transition_sets_terminated():
    environment = DeterministicRLEnvironment()
    adapter = RLEnvironmentAdapter(environment)

    adapter.reset()

    adapter.step({"type": "work"})

    observation, reward, terminated, truncated, info = (
        adapter.step({"type": "finish"})
    )

    assert observation == {
        "state": "step-2",
    }

    assert reward == 0.5
    assert terminated is True
    assert truncated is False

    assert adapter.terminated is True
    assert adapter.truncated is False


def test_step_after_termination_requires_reset():
    environment = DeterministicRLEnvironment()
    adapter = RLEnvironmentAdapter(environment)

    adapter.reset()

    adapter.step({"type": "work"})
    adapter.step({"type": "finish"})

    with pytest.raises(RuntimeError):
        adapter.step({"type": "invalid"})


def test_reset_reopens_episode_after_termination():
    environment = DeterministicRLEnvironment()
    adapter = RLEnvironmentAdapter(environment)

    adapter.reset()

    adapter.step({"type": "work"})
    adapter.step({"type": "finish"})

    observation, info = adapter.reset()
    assert info == {}

    assert observation == {
        "state": "start",
    }

    assert adapter.terminated is False
    assert adapter.truncated is False


def test_close_delegates_to_environment():
    environment = DeterministicRLEnvironment()
    adapter = RLEnvironmentAdapter(environment)

    adapter.close()

    assert environment.closed is True


def test_adapter_preserves_wrapped_environment():
    environment = DeterministicRLEnvironment()
    adapter = RLEnvironmentAdapter(environment)

    assert adapter.environment is environment
