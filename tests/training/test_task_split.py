import pytest

from agentforge.environments import (
    BaseEnvironment,
    EnvironmentStep,
)
from agentforge.core import RewardBreakdown
from agentforge.training import TrainingTask
from agentforge.training.task_split import (
    HeldOutEvaluationPool,
    TaskSplit,
    build_task_split,
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


def make_task(task_id, split):
    return TrainingTask(
        task_id=task_id,
        environment_factory=DummyEnvironment,
        metadata={"split": split},
    )


def test_split_requires_training_tasks():
    with pytest.raises(ValueError):
        build_task_split(
            [],
            [make_task("eval-001", "eval")],
        )


def test_split_requires_evaluation_tasks():
    with pytest.raises(ValueError):
        build_task_split(
            [make_task("train-001", "train")],
            [],
        )


def test_split_rejects_overlap():
    with pytest.raises(ValueError):
        build_task_split(
            [make_task("shared-001", "train")],
            [make_task("shared-001", "eval")],
        )


def test_split_preserves_declared_order():
    split = build_task_split(
        [
            make_task("train-001", "train"),
            make_task("train-002", "train"),
        ],
        [
            make_task("eval-001", "eval"),
            make_task("eval-002", "eval"),
        ],
    )

    assert split.training_task_ids == (
        "train-001",
        "train-002",
    )

    assert split.evaluation_task_ids == (
        "eval-001",
        "eval-002",
    )


def test_training_task_lookup_stays_in_training_pool():
    split = TaskSplit(
        training_tasks=(
            make_task("train-001", "train"),
        ),
        evaluation_tasks=(
            make_task("eval-001", "eval"),
        ),
    )

    assert (
        split.get_training_task("train-001").task_id
        == "train-001"
    )

    with pytest.raises(KeyError):
        split.get_training_task("eval-001")


def test_evaluation_task_lookup_stays_in_evaluation_pool():
    split = TaskSplit(
        training_tasks=(
            make_task("train-001", "train"),
        ),
        evaluation_tasks=(
            make_task("eval-001", "eval"),
        ),
    )

    assert (
        split.get_evaluation_task("eval-001").task_id
        == "eval-001"
    )

    with pytest.raises(KeyError):
        split.get_evaluation_task("train-001")


def test_held_out_pool_requires_tasks():
    with pytest.raises(ValueError):
        HeldOutEvaluationPool([])


def test_held_out_pool_rejects_duplicate_ids():
    with pytest.raises(ValueError):
        HeldOutEvaluationPool(
            [
                make_task("eval-001", "eval"),
                make_task("eval-001", "eval"),
            ]
        )


def test_held_out_pool_preserves_order():
    pool = HeldOutEvaluationPool(
        [
            make_task("eval-001", "eval"),
            make_task("eval-002", "eval"),
            make_task("eval-003", "eval"),
        ]
    )

    assert pool.task_ids == (
        "eval-001",
        "eval-002",
        "eval-003",
    )


def test_held_out_pool_lookup():
    pool = HeldOutEvaluationPool(
        [
            make_task("eval-001", "eval"),
        ]
    )

    task = pool.get("eval-001")

    assert task.task_id == "eval-001"
    assert task.metadata["split"] == "eval"


def test_held_out_pool_rejects_unknown_task():
    pool = HeldOutEvaluationPool(
        [
            make_task("eval-001", "eval"),
        ]
    )

    with pytest.raises(KeyError):
        pool.get("train-001")


def test_held_out_pool_returns_immutable_tuple():
    pool = HeldOutEvaluationPool(
        [
            make_task("eval-001", "eval"),
        ]
    )

    tasks = pool.as_tuple()

    assert isinstance(tasks, tuple)
    assert len(tasks) == 1


def test_split_does_not_mutate_input_collections():
    training = [
        make_task("train-001", "train"),
    ]

    evaluation = [
        make_task("eval-001", "eval"),
    ]

    split = build_task_split(
        training,
        evaluation,
    )

    training.append(
        make_task("train-002", "train")
    )

    evaluation.append(
        make_task("eval-002", "eval")
    )

    assert split.training_task_ids == (
        "train-001",
    )

    assert split.evaluation_task_ids == (
        "eval-001",
    )


def test_training_and_evaluation_ids_are_disjoint():
    split = build_task_split(
        [
            make_task("train-001", "train"),
            make_task("train-002", "train"),
        ],
        [
            make_task("eval-001", "eval"),
            make_task("eval-002", "eval"),
        ],
    )

    assert set(
        split.training_task_ids
    ).isdisjoint(
        split.evaluation_task_ids
    )
