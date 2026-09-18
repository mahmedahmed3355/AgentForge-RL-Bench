from agentforge.core.experiment import ExperimentManager


def test_experiment_tracks_runs():
    experiment = ExperimentManager(
        experiment_id="exp-001",
        name="Backend Learning Experiment",
        domain="backend",
        description="Before and after RL evaluation.",
    )

    experiment.start()

    experiment.add_run(
        run_id="run-baseline",
        phase="baseline",
        metrics={
            "pass_rate": 0.20,
            "avg_steps": 85,
        },
    )

    experiment.add_run(
        run_id="run-training",
        phase="training",
        metrics={
            "episodes": 50,
            "total_reward": 125.0,
        },
    )

    experiment.add_run(
        run_id="run-evaluation",
        phase="evaluation",
        metrics={
            "pass_rate": 0.60,
            "avg_steps": 51,
        },
    )

    assert experiment.status == "running"
    assert experiment.run_count == 3

    assert len(experiment.runs_for_phase("baseline")) == 1
    assert len(experiment.runs_for_phase("training")) == 1
    assert len(experiment.runs_for_phase("evaluation")) == 1

    assert experiment.runs_for_phase("baseline")[0]["metrics"]["pass_rate"] == 0.20
    assert experiment.runs_for_phase("evaluation")[0]["metrics"]["pass_rate"] == 0.60


def test_experiment_rejects_invalid_phase():
    experiment = ExperimentManager(
        experiment_id="exp-002",
        name="Invalid Phase Test",
        domain="backend",
    )

    try:
        experiment.add_run(
            run_id="run-invalid",
            phase="invalid",
        )
    except ValueError as exc:
        assert "phase" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def test_experiment_save_and_load(tmp_path):
    experiment = ExperimentManager(
        experiment_id="exp-003",
        name="Persistence Test",
        domain="backend",
        description="Checkpoint experiment.",
    )

    experiment.start()

    experiment.add_run(
        run_id="run-001",
        phase="baseline",
        metrics={"pass_rate": 0.40},
    )

    experiment.add_run(
        run_id="run-002",
        phase="evaluation",
        metrics={"pass_rate": 0.80},
    )

    experiment.complete()

    path = experiment.save(
        tmp_path / "experiment.json"
    )

    restored = ExperimentManager.load(path)

    assert restored.experiment_id == "exp-003"
    assert restored.name == "Persistence Test"
    assert restored.domain == "backend"
    assert restored.description == "Checkpoint experiment."

    assert restored.status == "completed"
    assert restored.run_count == 2

    assert restored.runs_for_phase("baseline")[0]["metrics"]["pass_rate"] == 0.40
    assert restored.runs_for_phase("evaluation")[0]["metrics"]["pass_rate"] == 0.80
