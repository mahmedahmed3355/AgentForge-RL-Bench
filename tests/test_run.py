from agentforge.core import EpisodeRunner, RewardBreakdown
from agentforge.core.run import RunManager


def build_result(
    episode_id: str,
    task_id: str,
    success: bool,
    reward: float,
):
    runner = EpisodeRunner(
        episode_id=episode_id,
        task_id=task_id,
    )

    runner.record_step(
        observation={"state": "start"},
        action="solve" if success else "work",
        reward=RewardBreakdown(
            progress=reward,
        ),
        next_observation={"state": "done"},
        status="success" if success else "failure",
    )

    return runner.result()


def test_run_manager_tracks_episodes():
    run = RunManager(
        run_id="run-001",
        domain="backend",
        agent_name="test-agent",
        config={"max_steps": 100},
    )

    run.start()

    run.record_episode(
        build_result(
            "episode-001",
            "backend-001",
            True,
            1.0,
        )
    )

    run.record_episode(
        build_result(
            "episode-002",
            "backend-002",
            False,
            0.5,
        )
    )

    assert run.status == "running"
    assert run.episode_count == 2
    assert run.successful_episodes == 1
    assert run.success_rate == 0.5
    assert run.total_reward == 1.5


def test_run_manager_save_and_load(tmp_path):
    run = RunManager(
        run_id="run-002",
        domain="backend",
        agent_name="qwen-test",
        config={
            "max_steps": 100,
            "training_tasks": 5,
        },
    )

    run.start()

    run.record_episode(
        build_result(
            "episode-003",
            "backend-003",
            True,
            1.2,
        )
    )

    run.complete()

    path = run.save(tmp_path / "run.json")

    restored = RunManager.load(path)

    assert restored.run_id == "run-002"
    assert restored.domain == "backend"
    assert restored.agent_name == "qwen-test"
    assert restored.config["training_tasks"] == 5

    assert restored.status == "completed"
    assert restored.episode_count == 1
    assert restored.successful_episodes == 1
    assert restored.success_rate == 1.0
    assert restored.total_reward == 1.2


def test_empty_run_has_zero_metrics():
    run = RunManager(
        run_id="run-003",
        domain="backend",
        agent_name="test-agent",
    )

    assert run.episode_count == 0
    assert run.successful_episodes == 0
    assert run.success_rate == 0.0
    assert run.total_reward == 0.0
