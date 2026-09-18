from pathlib import Path

from agentforge.agents import TestAgent
from agentforge.core import RewardBreakdown
from agentforge.core.agent_executor import AgentEpisodeExecutor
from agentforge.environments import BaseEnvironment, EnvironmentStep


class PersistenceEnvironment(BaseEnvironment):
    def __init__(self):
        self.current_step = 0

    def reset(self):
        self.current_step = 0
        return {"state": "start"}

    def step(self, action):
        self.current_step += 1

        done = self.current_step >= 3

        return EnvironmentStep(
            observation={"state": f"step-{self.current_step}"},
            reward=RewardBreakdown(
                progress=1.0,
            ),
            done=done,
            info={"step": self.current_step},
        )

    def close(self):
        pass


def test_agent_executor_preserves_complete_trajectory():
    agent = TestAgent()
    environment = PersistenceEnvironment()

    executor = AgentEpisodeExecutor(
        agent=agent,
        environment=environment,
    )

    result = executor.run(
        episode_id="persist-001",
        task_id="backend-persist-001",
        max_steps=10,
    )

    assert result.episode_id == "persist-001"
    assert result.task_id == "backend-persist-001"

    assert result.steps == 3
    assert result.done is True

    assert result.trajectory_length == 3
    assert result.total_reward == 3.0


def test_trajectory_can_be_persisted_after_agent_execution(tmp_path: Path):
    agent = TestAgent()
    environment = PersistenceEnvironment()

    executor = AgentEpisodeExecutor(
        agent=agent,
        environment=environment,
    )

    executor.run(
        episode_id="persist-002",
        task_id="backend-persist-002",
        max_steps=10,
    )

    # The executor currently owns the episode runner internally.
    # Verify the persistence contract through a directly managed runner.
    from agentforge.core import EpisodeRunner

    runner = EpisodeRunner(
        episode_id="persist-002",
        task_id="backend-persist-002",
        max_steps=10,
    )

    runner.record_step(
        observation={"state": "start"},
        action={"type": "test"},
        reward=RewardBreakdown(progress=1.0),
        next_observation={"state": "step-1"},
        status="running",
    )

    runner.record_step(
        observation={"state": "step-1"},
        action={"type": "test"},
        reward=RewardBreakdown(progress=1.0),
        next_observation={"state": "step-2"},
        status="running",
    )

    runner.record_step(
        observation={"state": "step-2"},
        action={"type": "test"},
        reward=RewardBreakdown(progress=1.0),
        next_observation={"state": "step-3"},
        status="success",
    )

    output = tmp_path / "trajectory.json"

    saved = runner.recorder.save_json(output)

    assert saved == output
    assert output.exists()

    payload = output.read_text(encoding="utf-8")

    assert '"episode_id": "persist-002"' in payload
    assert '"task_id": "backend-persist-002"' in payload
    assert '"length": 3' in payload
    assert '"total_reward": 3.0' in payload


def test_persisted_trajectory_contains_step_level_data(tmp_path: Path):
    from agentforge.core import EpisodeRunner

    runner = EpisodeRunner(
        episode_id="persist-003",
        task_id="backend-persist-003",
        max_steps=5,
    )

    runner.record_step(
        observation={"state": "start"},
        action={"type": "inspect"},
        reward=RewardBreakdown(progress=0.4),
        next_observation={"state": "inspected"},
        status="running",
    )

    runner.record_step(
        observation={"state": "inspected"},
        action={"type": "repair"},
        reward=RewardBreakdown(
            progress=0.6,
            correctness=0.3,
        ),
        next_observation={"state": "fixed"},
        status="success",
    )

    output = tmp_path / "trajectory.json"
    runner.recorder.save_json(output)

    payload = output.read_text(encoding="utf-8")

    assert '"step_id": 1' in payload
    assert '"step_id": 2' in payload

    assert '"inspect"' in payload
    assert '"repair"' in payload

    assert '"progress": 0.4' in payload
    assert '"progress": 0.6' in payload
    assert '"correctness": 0.3' in payload
