from agentforge.agents import TestAgent
from agentforge.core.agent_executor import AgentEpisodeExecutor
from agentforge.environments import BaseEnvironment, EnvironmentStep
from agentforge.core import RewardBreakdown


class DeterministicEnvironment(BaseEnvironment):
    """Small deterministic environment for integration testing."""

    def __init__(self):
        self.step_count = 0

    def reset(self):
        self.step_count = 0
        return {"state": "start"}

    def step(self, action):
        self.step_count += 1

        done = self.step_count >= 3

        return EnvironmentStep(
            observation={
                "state": f"step-{self.step_count}"
            },
            reward=RewardBreakdown(
                progress=1.0,
            ),
            done=done,
            info={
                "step": self.step_count,
            },
        )

    def close(self):
        pass


def test_agent_runs_inside_environment():
    agent = TestAgent()
    environment = DeterministicEnvironment()

    executor = AgentEpisodeExecutor(
        agent=agent,
        environment=environment,
    )

    result = executor.run(
        episode_id="episode-001",
        task_id="backend-001",
        max_steps=10,
    )

    assert result.episode_id == "episode-001"
    assert result.task_id == "backend-001"

    assert result.steps == 3
    assert result.done is True

    assert result.trajectory_length == 3
    assert result.total_reward == 3.0


def test_agent_receives_updates():
    agent = TestAgent()
    environment = DeterministicEnvironment()

    executor = AgentEpisodeExecutor(
        agent=agent,
        environment=environment,
    )

    executor.run(
        episode_id="episode-002",
        task_id="backend-002",
        max_steps=10,
    )

    assert agent.step_count == 3
    assert agent.update_count == 3


def test_max_steps_bounds_episode():
    agent = TestAgent()
    environment = DeterministicEnvironment()

    executor = AgentEpisodeExecutor(
        agent=agent,
        environment=environment,
    )

    result = executor.run(
        episode_id="episode-003",
        task_id="backend-003",
        max_steps=2,
    )

    assert result.steps == 2
    assert result.done is False
    assert result.trajectory_length == 2
    assert result.total_reward == 2.0
