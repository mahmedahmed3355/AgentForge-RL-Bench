import pytest

from agentforge.agents.test_agent import TestAgent
from agentforge.core import (
    RewardBreakdown,
)
from agentforge.environments import (
    BaseEnvironment,
    EnvironmentStep,
)
from agentforge.evaluation import (
    EvaluationRunner,
)
from agentforge.training import (
    TrainingTask,
)
from agentforge.training.task_split import (
    HeldOutEvaluationPool,
)


class EvaluationEnvironment(BaseEnvironment):
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
                progress=0.5,
            ),
            done=self.current_step >= 2,
            info={
                "task_id": self.task_id,
            },
        )

    def close(self):
        self.closed = True


def make_task(task_id):
    return TrainingTask(
        task_id=task_id,
        environment_factory=lambda: (
            EvaluationEnvironment(task_id)
        ),
        metadata={
            "split": "eval",
        },
    )


def make_runner():
    pool = HeldOutEvaluationPool(
        [
            make_task("eval-001"),
            make_task("eval-002"),
            make_task("eval-003"),
        ]
    )

    return EvaluationRunner(pool)


def test_baseline_evaluation_runs_all_held_out_tasks():
    runner = make_runner()

    result = runner.run(
        agent=TestAgent(),
        evaluation_id="eval-run-001",
        phase="baseline",
        max_steps=10,
    )

    assert result.evaluation_id == "eval-run-001"
    assert result.phase == "baseline"

    assert result.task_count == 3
    assert result.successful_tasks == 3
    assert result.failed_tasks == 0
    assert result.pass_rate == 1.0
    assert result.average_steps == 2.0
    assert result.total_reward == 3.0


def test_after_training_phase_is_recorded():
    runner = make_runner()

    result = runner.run(
        agent=TestAgent(),
        evaluation_id="eval-run-002",
        phase="after_training",
        max_steps=10,
    )

    assert result.phase == "after_training"

    for task_result in result.task_results:
        assert (
            task_result.metadata["phase"]
            == "after_training"
        )

        assert (
            task_result.metadata["evaluation_id"]
            == "eval-run-002"
        )


def test_evaluation_preserves_task_order():
    runner = make_runner()

    result = runner.run(
        agent=TestAgent(),
        evaluation_id="eval-run-003",
        phase="baseline",
        max_steps=10,
    )

    assert [
        item.task_id
        for item in result.task_results
    ] == [
        "eval-001",
        "eval-002",
        "eval-003",
    ]


def test_episode_ids_are_unique():
    runner = make_runner()

    result = runner.run(
        agent=TestAgent(),
        evaluation_id="eval-run-004",
        phase="baseline",
        max_steps=10,
    )

    episode_ids = [
        item.episode_id
        for item in result.task_results
    ]

    assert len(episode_ids) == 3
    assert len(set(episode_ids)) == 3


def test_failure_modes_are_aggregated():
    runner = make_runner()

    result = runner.run(
        agent=TestAgent(),
        evaluation_id="eval-run-005",
        phase="baseline",
        max_steps=1,
    )

    assert result.task_count == 3
    assert result.successful_tasks == 0
    assert result.failed_tasks == 3
    assert result.pass_rate == 0.0

    assert result.failure_modes == {
        "incomplete": 3,
    }


def test_to_dict_contains_complete_evaluation():
    runner = make_runner()

    result = runner.run(
        agent=TestAgent(),
        evaluation_id="eval-run-006",
        phase="baseline",
        max_steps=10,
    )

    data = result.to_dict()

    assert data["evaluation_id"] == "eval-run-006"
    assert data["phase"] == "baseline"
    assert data["task_count"] == 3
    assert data["successful_tasks"] == 3
    assert data["failed_tasks"] == 0
    assert data["pass_rate"] == 1.0
    assert data["average_steps"] == 2.0
    assert data["total_reward"] == 3.0

    assert len(data["tasks"]) == 3


def test_task_result_contains_trace_identity():
    runner = make_runner()

    result = runner.run(
        agent=TestAgent(),
        evaluation_id="eval-run-007",
        phase="after_training",
        max_steps=10,
    )

    task_result = result.task_results[0]

    assert task_result.task_id == "eval-001"
    assert task_result.episode_id.startswith(
        "eval-run-007-after_training"
    )
    assert task_result.success is True
    assert task_result.steps == 2
    assert task_result.total_reward == 1.0
    assert task_result.final_status == "success"
    assert task_result.failure_mode is None


def test_invalid_phase_is_rejected():
    runner = make_runner()

    with pytest.raises(ValueError):
        runner.run(
            agent=TestAgent(),
            evaluation_id="eval-run-008",
            phase="training",
            max_steps=10,
        )


def test_invalid_max_steps_is_rejected():
    runner = make_runner()

    with pytest.raises(ValueError):
        runner.run(
            agent=TestAgent(),
            evaluation_id="eval-run-009",
            phase="baseline",
            max_steps=0,
        )


def test_evaluation_uses_fresh_environment_per_task():
    created = []

    def factory():
        environment = EvaluationEnvironment(
            f"fresh-{len(created)}"
        )
        created.append(environment)
        return environment

    pool = HeldOutEvaluationPool(
        [
            TrainingTask(
                task_id="eval-fresh-001",
                environment_factory=factory,
                metadata={"split": "eval"},
            ),
            TrainingTask(
                task_id="eval-fresh-002",
                environment_factory=factory,
                metadata={"split": "eval"},
            ),
        ]
    )

    runner = EvaluationRunner(pool)

    runner.run(
        agent=TestAgent(),
        evaluation_id="eval-run-010",
        phase="baseline",
        max_steps=10,
    )

    assert len(created) == 2
    assert created[0] is not created[1]
    assert created[0].closed is True
    assert created[1].closed is True
