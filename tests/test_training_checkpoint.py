import json

import pytest

from agentforge.core.training import (
    TrainingEpisodeMetrics,
    TrainingRunResult,
)
from agentforge.core.training_checkpoint import TrainingCheckpoint


def make_training_result():
    return TrainingRunResult(
        run_id="run-checkpoint-001",
        episodes_completed=2,
        total_steps=6,
        total_reward=6.0,
        successful_episodes=2,
        episode_metrics=[
            TrainingEpisodeMetrics(
                episode_id="run-checkpoint-001-episode-0001",
                task_id="backend-001",
                steps=3,
                total_reward=3.0,
                success=True,
            ),
            TrainingEpisodeMetrics(
                episode_id="run-checkpoint-001-episode-0002",
                task_id="backend-001",
                steps=3,
                total_reward=3.0,
                success=True,
            ),
        ],
    )


def test_training_checkpoint_save_and_load(tmp_path):
    result = make_training_result()

    path = TrainingCheckpoint.save(
        result,
        tmp_path / "training.json",
        status="running",
    )

    restored, status = TrainingCheckpoint.load(path)

    assert status == "running"

    assert restored.run_id == result.run_id
    assert restored.episodes_completed == 2
    assert restored.total_steps == 6
    assert restored.total_reward == 6.0
    assert restored.successful_episodes == 2

    assert len(restored.episode_metrics) == 2

    assert (
        restored.episode_metrics[0].episode_id
        == "run-checkpoint-001-episode-0001"
    )

    assert restored.episode_metrics[1].total_reward == 3.0


def test_training_checkpoint_contains_schema_version(tmp_path):
    result = make_training_result()

    path = TrainingCheckpoint.save(
        result,
        tmp_path / "training.json",
    )

    payload = json.loads(
        path.read_text(encoding="utf-8")
    )

    assert payload["schema_version"] == "1.0"
    assert payload["status"] == "running"


def test_training_checkpoint_completed_status(tmp_path):
    result = make_training_result()

    path = TrainingCheckpoint.save(
        result,
        tmp_path / "completed.json",
        status="completed",
    )

    restored, status = TrainingCheckpoint.load(path)

    assert status == "completed"
    assert restored.run_id == "run-checkpoint-001"


def test_training_checkpoint_rejects_invalid_status(tmp_path):
    result = make_training_result()

    with pytest.raises(ValueError):
        TrainingCheckpoint.save(
            result,
            tmp_path / "invalid.json",
            status="paused",
        )


def test_training_checkpoint_rejects_unknown_schema(tmp_path):
    result = make_training_result()

    path = TrainingCheckpoint.save(
        result,
        tmp_path / "invalid-schema.json",
    )

    payload = json.loads(
        path.read_text(encoding="utf-8")
    )

    payload["schema_version"] = "999.0"

    path.write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        TrainingCheckpoint.load(path)
