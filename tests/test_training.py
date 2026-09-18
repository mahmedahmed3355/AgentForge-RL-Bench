import pytest

from agentforge.agents import TestAgent
from agentforge.core import RewardBreakdown
from agentforge.core.training import TrainingLoop
from agentforge.environments import BaseEnvironment, EnvironmentStep


class TrainingEnvironment(BaseEnvironment):
    def __init__(self):
        self.current_step = 0
        self.closed = False

    def reset(self):
        self.current_step = 0
        self.closed = False
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
        self.closed = True


def test_training_loop_runs_multiple_episodes():
    loop = TrainingLoop(
        agent=TestAgent(),
        environment_factory=TrainingEnvironment,
    )

    result = loop.run(
        run_id="run-001",
        task_id="backend-001",
        episodes=3,
        max_steps=10,
    )

    assert result.run_id == "run-001"
    assert result.episodes_completed == 3

    assert result.total_steps == 9
    assert result.total_reward == 9.0

    assert result.successful_episodes == 3
    assert result.success_rate == 1.0

    assert len(result.episode_metrics) == 3


def test_training_episode_ids_are_deterministic():
    loop = TrainingLoop(
        agent=TestAgent(),
        environment_factory=TrainingEnvironment,
    )

    result = loop.run(
        run_id="run-002",
        task_id="backend-002",
        episodes=3,
        max_steps=10,
    )

    assert [
        item.episode_id
        for item in result.episode_metrics
    ] == [
        "run-002-episode-0001",
        "run-002-episode-0002",
        "run-002-episode-0003",
    ]


def test_training_metrics_are_recorded_per_episode():
    loop = TrainingLoop(
        agent=TestAgent(),
        environment_factory=TrainingEnvironment,
    )

    result = loop.run(
        run_id="run-003",
        task_id="backend-003",
        episodes=2,
        max_steps=10,
    )

    for metric in result.episode_metrics:
        assert metric.steps == 3
        assert metric.total_reward == 3.0
        assert metric.success is True


def test_training_zero_episodes():
    loop = TrainingLoop(
        agent=TestAgent(),
        environment_factory=TrainingEnvironment,
    )

    result = loop.run(
        run_id="run-004",
        task_id="backend-004",
        episodes=0,
        max_steps=10,
    )

    assert result.episodes_completed == 0
    assert result.total_steps == 0
    assert result.total_reward == 0.0
    assert result.successful_episodes == 0
    assert result.success_rate == 0.0
    assert result.episode_metrics == []


def test_training_rejects_invalid_configuration():
    loop = TrainingLoop(
        agent=TestAgent(),
        environment_factory=TrainingEnvironment,
    )

    with pytest.raises(ValueError):
        loop.run(
            run_id="run-005",
            task_id="backend-005",
            episodes=-1,
            max_steps=10,
        )

    with pytest.raises(ValueError):
        loop.run(
            run_id="run-006",
            task_id="backend-006",
            episodes=1,
            max_steps=0,
        )
