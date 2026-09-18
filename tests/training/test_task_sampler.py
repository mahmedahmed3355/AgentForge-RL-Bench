import pytest

from agentforge.environments import (
    BaseEnvironment,
    EnvironmentStep,
)
from agentforge.core import RewardBreakdown
from agentforge.training import TrainingTask
from agentforge.training.task_sampler import (
    TaskSampler,
)


class DummyEnvironment(BaseEnvironment):
    def reset(self):
        return {}

    def step(self, action):
        return EnvironmentStep(
            observation={},
            reward=RewardBreakdown(),
            done=True,
            info={},
        )

    def close(self):
        pass


def make_task(task_id):
    return TrainingTask(
        task_id=task_id,
        environment_factory=DummyEnvironment,
        metadata={"split": "train"},
    )


def test_sampler_requires_tasks():
    with pytest.raises(ValueError):
        TaskSampler([])


def test_sampler_rejects_duplicate_task_ids():
    with pytest.raises(ValueError):
        TaskSampler(
            [
                make_task("backend-001"),
                make_task("backend-001"),
            ]
        )


def test_first_episode_selects_first_task():
    sampler = TaskSampler(
        [
            make_task("backend-001"),
            make_task("backend-002"),
            make_task("backend-003"),
        ]
    )

    sample = sampler.sample(1)

    assert sample.task_id == "backend-001"
    assert sample.episode_index == 1
    assert sample.selection_index == 0


def test_sampling_is_round_robin():
    sampler = TaskSampler(
        [
            make_task("backend-001"),
            make_task("backend-002"),
            make_task("backend-003"),
        ]
    )

    samples = [
        sampler.sample(index)
        for index in range(1, 8)
    ]

    assert [
        sample.task_id
        for sample in samples
    ] == [
        "backend-001",
        "backend-002",
        "backend-003",
        "backend-001",
        "backend-002",
        "backend-003",
        "backend-001",
    ]


def test_selection_index_is_zero_based():
    sampler = TaskSampler(
        [
            make_task("backend-001"),
            make_task("backend-002"),
        ]
    )

    assert sampler.sample(1).selection_index == 0
    assert sampler.sample(2).selection_index == 1
    assert sampler.sample(3).selection_index == 0


def test_sequence_is_deterministic():
    sampler = TaskSampler(
        [
            make_task("backend-001"),
            make_task("backend-002"),
        ]
    )

    sequence = sampler.sequence(
        start_episode=4,
        count=5,
    )

    assert [
        sample.task_id
        for sample in sequence
    ] == [
        "backend-002",
        "backend-001",
        "backend-002",
        "backend-001",
        "backend-002",
    ]

    assert [
        sample.episode_index
        for sample in sequence
    ] == [4, 5, 6, 7, 8]


def test_sequence_can_be_empty():
    sampler = TaskSampler(
        [make_task("backend-001")]
    )

    assert sampler.sequence(
        start_episode=1,
        count=0,
    ) == ()


def test_invalid_episode_is_rejected():
    sampler = TaskSampler(
        [make_task("backend-001")]
    )

    with pytest.raises(ValueError):
        sampler.sample(0)


def test_invalid_sequence_start_is_rejected():
    sampler = TaskSampler(
        [make_task("backend-001")]
    )

    with pytest.raises(ValueError):
        sampler.sequence(
            start_episode=0,
            count=1,
        )


def test_invalid_sequence_count_is_rejected():
    sampler = TaskSampler(
        [make_task("backend-001")]
    )

    with pytest.raises(ValueError):
        sampler.sequence(
            start_episode=1,
            count=-1,
        )


def test_same_task_set_produces_same_sequence():
    tasks = [
        make_task("backend-001"),
        make_task("backend-002"),
        make_task("backend-003"),
    ]

    sampler_a = TaskSampler(tasks)
    sampler_b = TaskSampler(tasks)

    sequence_a = sampler_a.sequence(
        start_episode=10,
        count=12,
    )

    sequence_b = sampler_b.sequence(
        start_episode=10,
        count=12,
    )

    assert sequence_a == sequence_b
