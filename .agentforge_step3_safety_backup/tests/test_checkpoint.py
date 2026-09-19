from agentforge.core.checkpoint import EpisodeCheckpoint
from agentforge.core.episode import EpisodeRunner
from agentforge.core import RewardBreakdown


def test_checkpoint_save_and_load(tmp_path):
    runner = EpisodeRunner(
        episode_id="checkpoint-001",
        task_id="backend-001",
        max_steps=100,
        metadata={"training_run": "run-001"},
    )

    runner.record_step(
        observation={"state": "start"},
        action={"type": "inspect"},
        reward=RewardBreakdown(progress=0.2),
        next_observation={"state": "inspected"},
        status="running",
    )

    runner.record_step(
        observation={"state": "inspected"},
        action={"type": "work"},
        reward=RewardBreakdown(
            progress=0.3,
            correctness=0.2,
        ),
        next_observation={"state": "working"},
        status="running",
    )

    path = EpisodeCheckpoint.save(
        runner,
        tmp_path / "checkpoint.json",
    )

    restored = EpisodeCheckpoint.load(path)

    assert restored.episode_id == "checkpoint-001"
    assert restored.task_id == "backend-001"
    assert restored.max_steps == 100
    assert restored.metadata == {"training_run": "run-001"}

    assert restored.step_count == 2
    assert restored.done is False
    assert restored.recorder.trajectory.total_reward == 0.7

    assert restored.recorder.trajectory.steps[0].action == {
        "type": "inspect"
    }


def test_checkpoint_preserves_finished_episode(tmp_path):
    runner = EpisodeRunner(
        episode_id="checkpoint-002",
        task_id="backend-002",
    )

    runner.record_step(
        observation={"state": "start"},
        action="solve",
        reward=RewardBreakdown(
            correctness=1.0,
            terminal=1.0,
        ),
        next_observation={"state": "solved"},
        status="success",
    )

    path = EpisodeCheckpoint.save(
        runner,
        tmp_path / "finished.json",
    )

    restored = EpisodeCheckpoint.load(path)

    assert restored.done is True
    assert restored.result().success is True
    assert restored.result().final_status == "success"
    assert restored.step_count == 1
