import json

from agentforge.agents.test_agent import TestAgent
from agentforge.core import (
    ExperimentArtifactStore,
    ExperimentOrchestrator,
    RewardBreakdown,
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


class ArtifactEnvironment(BaseEnvironment):
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


def make_result():
    tasks = [
        TrainingTask(
            task_id=f"artifact-{index}",
            environment_factory=lambda task_id=f"artifact-{index}": (
                ArtifactEnvironment(task_id)
            ),
            metadata={
                "split": "eval",
            },
        )
        for index in range(1, 3)
    ]

    pool = HeldOutEvaluationPool(tasks)
    runner = EvaluationRunner(pool)
    orchestrator = ExperimentOrchestrator(runner)

    return orchestrator.run(
        experiment_id="artifact-experiment-001",
        agent=TestAgent(),
        max_steps=10,
        train=lambda agent: None,
    )


def test_artifact_store_creates_all_layers(tmp_path):
    result = make_result()

    store = ExperimentArtifactStore(tmp_path)

    directory = store.save(
        result,
        config={
            "domain": "backend",
            "seed": 42,
        },
    )

    expected = {
        "config.json",
        "baseline.json",
        "after_training.json",
        "learning_delta.json",
        "experiment_result.json",
    }

    assert {
        path.name
        for path in directory.iterdir()
    } == expected


def test_artifact_store_round_trip(tmp_path):
    result = make_result()

    store = ExperimentArtifactStore(tmp_path)

    store.save(
        result,
        config={
            "domain": "backend",
            "seed": 42,
        },
    )

    loaded = store.load(
        "artifact-experiment-001"
    )

    assert loaded["config"] == {
        "domain": "backend",
        "seed": 42,
    }

    assert (
        loaded["baseline"]["phase"]
        == "baseline"
    )

    assert (
        loaded["after_training"]["phase"]
        == "after_training"
    )

    assert (
        loaded["learning_delta"]["pass_rate_delta"]
        == 0.0
    )


def test_experiment_result_artifact_matches_result(
    tmp_path,
):
    result = make_result()

    store = ExperimentArtifactStore(tmp_path)
    store.save(result)

    loaded = store.load(
        "artifact-experiment-001"
    )

    assert (
        loaded["experiment_result"]
        == result.to_dict()
    )


def test_artifacts_are_valid_json(tmp_path):
    result = make_result()

    store = ExperimentArtifactStore(tmp_path)

    directory = store.save(result)

    for path in directory.glob("*.json"):
        payload = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )

        assert isinstance(payload, dict)


def test_missing_experiment_raises(tmp_path):
    store = ExperimentArtifactStore(tmp_path)

    try:
        store.load("does-not-exist")
    except FileNotFoundError as exc:
        assert "does-not-exist" in str(exc)
    else:
        raise AssertionError(
            "Expected FileNotFoundError"
        )


def test_custom_artifact_root_is_respected(tmp_path):
    root = tmp_path / "custom-runs"

    result = make_result()

    store = ExperimentArtifactStore(root)
    directory = store.save(result)

    assert directory == (
        root / "artifact-experiment-001"
    )

    assert (
        directory
        / "experiment_result.json"
    ).exists()


def test_config_defaults_to_empty_object(tmp_path):
    result = make_result()

    store = ExperimentArtifactStore(tmp_path)
    directory = store.save(result)

    config = json.loads(
        (
            directory / "config.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    assert config == {}
