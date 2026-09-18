import pytest

from agentforge.core.evaluation import (
    EvaluationComparator,
    EvaluationMetrics,
)
from agentforge.core.failure_analysis import FailureModeAnalyzer
from agentforge.core.report import ExperimentReport


def build_report():
    baseline = EvaluationMetrics(
        pass_rate=0.20,
        average_steps=80.0,
        total_reward=40.0,
        successful_tasks=2,
        failed_tasks=8,
        failure_modes={
            "timeout": 5,
            "wrong_output": 3,
        },
    )

    after_training = EvaluationMetrics(
        pass_rate=0.70,
        average_steps=45.0,
        total_reward=140.0,
        successful_tasks=7,
        failed_tasks=3,
        failure_modes={
            "timeout": 1,
            "wrong_output": 2,
        },
    )

    delta = EvaluationComparator.compare(
        baseline,
        after_training,
    )

    failure_analysis = FailureModeAnalyzer.analyze(
        baseline,
        after_training,
    )

    return ExperimentReport(
        experiment_id="exp-001",
        name="Backend Learning Experiment",
        domain="backend",
        status="completed",
        baseline=baseline,
        after_training=after_training,
        learning_delta=delta,
        failure_analysis=failure_analysis,
        training_runs=[
            {
                "run_id": "run-training-001",
                "episodes": 50,
                "status": "completed",
            }
        ],
        metadata={
            "seed": 42,
            "agent": "test-agent",
            "environment_version": "0.1.0",
        },
    )


def test_report_contains_all_experiment_layers():
    report = build_report()

    data = report.to_dict()

    assert data["schema_version"] == "1.0"

    assert data["experiment"]["experiment_id"] == "exp-001"
    assert data["experiment"]["domain"] == "backend"
    assert data["experiment"]["status"] == "completed"

    assert data["baseline"]["pass_rate"] == 0.20
    assert data["after_training"]["pass_rate"] == 0.70

    assert data["learning_delta"]["pass_rate_delta"] == pytest.approx(0.5)

    assert (
        data["failure_analysis"]
        ["after_training_total_failures"]
        == 3
    )

    assert len(data["training_runs"]) == 1

    assert data["metadata"]["seed"] == 42


def test_report_save_and_load(tmp_path):
    report = build_report()

    path = report.save(
        tmp_path / "experiment_report.json"
    )

    assert path.exists()

    restored = ExperimentReport.load(path)

    assert restored.experiment_id == "exp-001"
    assert restored.name == "Backend Learning Experiment"
    assert restored.domain == "backend"
    assert restored.status == "completed"

    assert restored.baseline.pass_rate == 0.20
    assert restored.after_training.pass_rate == 0.70

    assert restored.learning_delta.total_reward_delta == 100.0

    assert (
        restored.failure_analysis
        .after_training_total_failures
        == 3
    )

    assert restored.training_runs[0]["run_id"] == (
        "run-training-001"
    )

    assert restored.metadata["seed"] == 42


def test_report_json_contains_required_sections(tmp_path):
    report = build_report()

    path = report.save(
        tmp_path / "experiment_report.json"
    )

    content = path.read_text(encoding="utf-8")

    assert '"schema_version": "1.0"' in content
    assert '"experiment"' in content
    assert '"baseline"' in content
    assert '"after_training"' in content
    assert '"learning_delta"' in content
    assert '"failure_analysis"' in content
    assert '"training_runs"' in content
    assert '"metadata"' in content
