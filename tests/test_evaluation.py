import pytest

from agentforge.core.evaluation import (
    EvaluationComparator,
    EvaluationMetrics,
)


def test_evaluation_metrics():
    metrics = EvaluationMetrics(
        pass_rate=0.60,
        average_steps=50.0,
        total_reward=120.0,
        successful_tasks=6,
        failed_tasks=4,
        failure_modes={
            "timeout": 2,
            "wrong_output": 2,
        },
    )

    assert metrics.total_tasks == 10
    assert metrics.pass_rate == 0.60
    assert metrics.average_steps == 50.0
    assert metrics.successful_tasks == 6
    assert metrics.failed_tasks == 4
    assert metrics.failure_modes["timeout"] == 2


def test_before_after_comparison():
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

    assert delta.pass_rate_delta == pytest.approx(0.50)
    assert delta.average_steps_delta == pytest.approx(-35.0)
    assert delta.total_reward_delta == pytest.approx(100.0)
    assert delta.successful_tasks_delta == 5
    assert delta.failed_tasks_delta == -5

    assert delta.baseline.pass_rate == 0.20
    assert delta.after_training.pass_rate == 0.70


def test_comparison_can_be_saved(tmp_path):
    baseline = EvaluationMetrics(
        pass_rate=0.30,
        average_steps=70.0,
        total_reward=50.0,
        successful_tasks=3,
        failed_tasks=7,
    )

    after_training = EvaluationMetrics(
        pass_rate=0.60,
        average_steps=55.0,
        total_reward=100.0,
        successful_tasks=6,
        failed_tasks=4,
    )

    delta = EvaluationComparator.compare(
        baseline,
        after_training,
    )

    path = EvaluationComparator.save(
        delta,
        tmp_path / "learning_delta.json",
    )

    assert path.exists()

    content = path.read_text(encoding="utf-8")

    assert '"pass_rate_delta": 0.3' in content
    assert '"average_steps_delta": -15.0' in content
    assert '"total_reward_delta": 50.0' in content
