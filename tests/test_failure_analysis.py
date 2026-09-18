from agentforge.core.evaluation import EvaluationMetrics
from agentforge.core.failure_analysis import FailureModeAnalyzer


def test_failure_modes_are_compared():
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

    analysis = FailureModeAnalyzer.analyze(
        baseline,
        after_training,
    )

    assert analysis.baseline_total_failures == 8
    assert analysis.after_training_total_failures == 3

    timeout = next(
        item
        for item in analysis.mode_deltas
        if item.mode == "timeout"
    )

    wrong_output = next(
        item
        for item in analysis.mode_deltas
        if item.mode == "wrong_output"
    )

    assert timeout.baseline_count == 5
    assert timeout.after_training_count == 1
    assert timeout.delta == -4

    assert wrong_output.baseline_count == 3
    assert wrong_output.after_training_count == 2
    assert wrong_output.delta == -1


def test_new_failure_mode_is_detected():
    baseline = EvaluationMetrics(
        pass_rate=0.50,
        average_steps=50.0,
        total_reward=80.0,
        successful_tasks=5,
        failed_tasks=5,
        failure_modes={
            "timeout": 5,
        },
    )

    after_training = EvaluationMetrics(
        pass_rate=0.60,
        average_steps=45.0,
        total_reward=100.0,
        successful_tasks=6,
        failed_tasks=4,
        failure_modes={
            "timeout": 2,
            "wrong_output": 2,
        },
    )

    analysis = FailureModeAnalyzer.analyze(
        baseline,
        after_training,
    )

    modes = {
        item.mode: item
        for item in analysis.mode_deltas
    }

    assert modes["timeout"].delta == -3
    assert modes["wrong_output"].baseline_count == 0
    assert modes["wrong_output"].after_training_count == 2
    assert modes["wrong_output"].delta == 2


def test_failure_analysis_can_be_saved(tmp_path):
    baseline = EvaluationMetrics(
        pass_rate=0.30,
        average_steps=70.0,
        total_reward=50.0,
        successful_tasks=3,
        failed_tasks=7,
        failure_modes={
            "timeout": 4,
            "wrong_output": 3,
        },
    )

    after_training = EvaluationMetrics(
        pass_rate=0.60,
        average_steps=55.0,
        total_reward=100.0,
        successful_tasks=6,
        failed_tasks=4,
        failure_modes={
            "timeout": 1,
            "wrong_output": 3,
        },
    )

    analysis = FailureModeAnalyzer.analyze(
        baseline,
        after_training,
    )

    path = FailureModeAnalyzer.save(
        analysis,
        tmp_path / "failure_analysis.json",
    )

    assert path.exists()

    content = path.read_text(encoding="utf-8")

    assert '"baseline_total_failures": 7' in content
    assert '"after_training_total_failures": 4' in content
    assert '"mode": "timeout"' in content
    assert '"delta": -3' in content


def test_failure_mode_union_handles_missing_modes():
    baseline = EvaluationMetrics(
        pass_rate=0.50,
        average_steps=50.0,
        total_reward=50.0,
        successful_tasks=5,
        failed_tasks=5,
        failure_modes={
            "timeout": 5,
        },
    )

    after_training = EvaluationMetrics(
        pass_rate=0.50,
        average_steps=50.0,
        total_reward=50.0,
        successful_tasks=5,
        failed_tasks=5,
        failure_modes={
            "syntax_error": 5,
        },
    )

    analysis = FailureModeAnalyzer.analyze(
        baseline,
        after_training,
    )

    modes = {
        item.mode: item
        for item in analysis.mode_deltas
    }

    assert modes["timeout"].baseline_count == 5
    assert modes["timeout"].after_training_count == 0

    assert modes["syntax_error"].baseline_count == 0
    assert modes["syntax_error"].after_training_count == 5
