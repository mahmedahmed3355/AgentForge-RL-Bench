import pytest

from agentforge.agents.test_agent import TestAgent
from agentforge.core import RewardBreakdown
from agentforge.environments import (
    BaseEnvironment,
    EnvironmentStep,
)
from agentforge.training import (
    TrainingEnvironmentRunner,
    TrainingTask,
)


class TrainingTestEnvironment(BaseEnvironment):
    def __init__(self, task_id):
        self.task_id = task_id
        self.current_step = 0
        self.closed = False

    def reset(self):
        self.current_step = 0
        return {
            "task": self.task_id,
            "step": 0,
        }

    def step(self, action):
        self.current_step += 1

        return EnvironmentStep(
            observation={
                "task": self.task_id,
                "step": self.current_step,
            },
            reward=RewardBreakdown(
                progress=0.1,
            ),
            done=self.current_step >= 3,
            info={
                "task_id": self.task_id,
            },
        )

    def close(self):
        self.closed = True


def make_task(task_id):
    return TrainingTask(
        task_id=task_id,
        environment_factory=lambda: TrainingTestEnvironment(
            task_id
        ),
        metadata={
            "split": "train",
        },
    )


def test_runner_requires_training_tasks():
    with pytest.raises(ValueError):
        TrainingEnvironmentRunner([])


def test_runner_rejects_duplicate_task_ids():
    with pytest.raises(ValueError):
        TrainingEnvironmentRunner(
            [
                make_task("backend-001"),
                make_task("backend-001"),
            ]
        )


def test_task_ids_are_deterministic():
    runner = TrainingEnvironmentRunner(
        [
            make_task("backend-001"),
            make_task("backend-002"),
            make_task("backend-003"),
        ]
    )

    assert runner.task_ids == (
        "backend-001",
        "backend-002",
        "backend-003",
    )


def test_get_task_returns_registered_task():
    runner = TrainingEnvironmentRunner(
        [make_task("backend-001")]
    )

    task = runner.get_task("backend-001")

    assert task.task_id == "backend-001"
    assert task.metadata["split"] == "train"


def test_unknown_task_is_rejected():
    runner = TrainingEnvironmentRunner(
        [make_task("backend-001")]
    )

    with pytest.raises(KeyError):
        runner.get_task("backend-999")


def test_create_environment_returns_fresh_instance():
    runner = TrainingEnvironmentRunner(
        [make_task("backend-001")]
    )

    first = runner.create_environment(
        "backend-001"
    )
    second = runner.create_environment(
        "backend-001"
    )

    assert first is not second
    assert first.task_id == "backend-001"
    assert second.task_id == "backend-001"


def test_run_episode_executes_agent_inside_task():
    runner = TrainingEnvironmentRunner(
        [make_task("backend-001")]
    )

    agent = TestAgent()

    result = runner.run_episode(
        agent=agent,
        task_id="backend-001",
        episode_id="train-episode-001",
        max_steps=10,
    )

    assert result.episode_id == "train-episode-001"
    assert result.task_id == "backend-001"
    assert result.steps == 3
    assert result.done is True
    assert result.trajectory_length == 3


def test_run_episode_rejects_invalid_max_steps():
    runner = TrainingEnvironmentRunner(
        [make_task("backend-001")]
    )

    with pytest.raises(ValueError):
        runner.run_episode(
            agent=TestAgent(),
            task_id="backend-001",
            episode_id="train-episode-001",
            max_steps=0,
        )


def test_training_environment_is_closed_after_episode():
    created = []

    def factory():
        environment = TrainingTestEnvironment(
            "backend-closed"
        )
        created.append(environment)
        return environment

    runner = TrainingEnvironmentRunner(
        [
            TrainingTask(
                task_id="backend-closed",
                environment_factory=factory,
                metadata={"split": "train"},
            )
        ]
    )

    runner.run_episode(
        agent=TestAgent(),
        task_id="backend-closed",
        episode_id="train-episode-002",
        max_steps=10,
    )

    assert len(created) == 1
    assert created[0].closed is True
