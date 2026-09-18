from agentforge.agents.test_agent import TestAgent
from agentforge.core import RewardBreakdown
from agentforge.core.orchestrator import (
    ExperimentOrchestrator,
)
from agentforge.environments import (
    BaseEnvironment,
    EnvironmentStep,
)
from agentforge.evaluation import EvaluationRunner
from agentforge.training import TrainingTask
from agentforge.training.task_split import (
    HeldOutEvaluationPool,
)


class OrchestratorEnvironment(BaseEnvironment):
    def __init__(self, task_id):
        self.task_id = task_id
        self.step_number = 0
        self.closed = False

    def reset(self):
        self.step_number = 0
        return {
            "task": self.task_id,
            "step": 0,
        }

    def step(self, action):
        self.step_number += 1

        return EnvironmentStep(
            observation={
                "task": self.task_id,
                "step": self.step_number,
            },
            reward=RewardBreakdown(
                progress=0.5,
            ),
            done=self.step_number >= 2,
            info={},
        )

    def close(self):
        self.closed = True


def make_runner():
    tasks = [
        TrainingTask(
            task_id=f"heldout-{index}",
            environment_factory=lambda task_id=f"heldout-{index}": (
                OrchestratorEnvironment(task_id)
            ),
            metadata={"split": "eval"},
        )
        for index in range(1, 4)
    ]

    pool = HeldOutEvaluationPool(tasks)

    return EvaluationRunner(pool)


def test_orchestrator_runs_baseline_then_training_then_after():
    runner = make_runner()
    orchestrator = ExperimentOrchestrator(runner)

    events = []

    def train(agent):
        events.append("training")

    result = orchestrator.run(
        experiment_id="experiment-001",
        agent=TestAgent(),
        max_steps=10,
        train=train,
    )

    assert result.experiment_id == "experiment-001"
    assert events == ["training"]

    assert result.baseline.phase == "baseline"
    assert (
        result.after_training.phase
        == "after_training"
    )


def test_orchestrator_evaluates_same_task_count_before_and_after():
    orchestrator = ExperimentOrchestrator(
        make_runner()
    )

    result = orchestrator.run(
        experiment_id="experiment-002",
        agent=TestAgent(),
        max_steps=10,
        train=lambda agent: None,
    )

    assert result.baseline.task_count == 3
    assert result.after_training.task_count == 3


def test_orchestrator_produces_learning_delta():
    orchestrator = ExperimentOrchestrator(
        make_runner()
    )

    result = orchestrator.run(
        experiment_id="experiment-003",
        agent=TestAgent(),
        max_steps=10,
        train=lambda agent: None,
    )

    assert result.learning_delta.pass_rate_delta == 0.0
    assert result.learning_delta.average_steps_delta == 0.0
    assert result.learning_delta.total_reward_delta == 0.0
    assert (
        result.learning_delta.successful_tasks_delta
        == 0
    )
    assert (
        result.learning_delta.failed_tasks_delta
        == 0
    )


def test_orchestrator_serialization_contains_all_layers():
    orchestrator = ExperimentOrchestrator(
        make_runner()
    )

    result = orchestrator.run(
        experiment_id="experiment-004",
        agent=TestAgent(),
        max_steps=10,
        train=lambda agent: None,
    )

    data = result.to_dict()

    assert data["experiment_id"] == "experiment-004"
    assert data["baseline"]["phase"] == "baseline"
    assert (
        data["after_training"]["phase"]
        == "after_training"
    )
    assert "learning_delta" in data
    assert "pass_rate_delta" in data["learning_delta"]


def test_training_occurs_between_evaluations():
    runner = make_runner()
    orchestrator = ExperimentOrchestrator(runner)

    phases = []

    original_run = runner.run

    def wrapped_run(**kwargs):
        phases.append(kwargs["phase"])
        return original_run(**kwargs)

    runner.run = wrapped_run

    orchestrator.run(
        experiment_id="experiment-005",
        agent=TestAgent(),
        max_steps=10,
        train=lambda agent: phases.append(
            "training"
        ),
    )

    assert phases == [
        "baseline",
        "training",
        "after_training",
    ]
