import pytest

from agentforge.core.training import (
    TrainingEpisodeMetrics,
    TrainingRunResult,
)
from agentforge.core.training_resume import (
    TrainingResumeManager,
)


def make_result():
    return TrainingRunResult(
        run_id="resume-run-001",
        episodes_completed=3,
        total_steps=9,
        total_reward=9.0,
        successful_episodes=2,
        episode_metrics=[
            TrainingEpisodeMetrics(
                episode_id="resume-run-001-episode-0001",
                task_id="backend-001",
                steps=3,
                total_reward=3.0,
                success=True,
            ),
            TrainingEpisodeMetrics(
                episode_id="resume-run-001-episode-0002",
                task_id="backend-001",
                steps=3,
                total_reward=3.0,
                success=False,
            ),
            TrainingEpisodeMetrics(
                episode_id="resume-run-001-episode-0003",
                task_id="backend-001",
                steps=3,
                total_reward=3.0,
                success=True,
            ),
        ],
    )


def test_save_returns_next_episode(tmp_path):
    manager = TrainingResumeManager()
    result = make_result()

    state = manager.save(
        result,
        tmp_path / "resume.json",
    )

    assert state.run_id == "resume-run-001"
    assert state.episodes_completed == 3
    assert state.next_episode_index == 4
    assert state.total_steps == 9
    assert state.total_reward == 9.0
    assert state.successful_episodes == 2
    assert state.status == "running"


def test_load_restores_next_episode(tmp_path):
    manager = TrainingResumeManager()
    result = make_result()

    path = tmp_path / "resume.json"

    manager.save(
        result,
        path,
    )

    restored, state = manager.load(path)

    assert restored.run_id == "resume-run-001"
    assert restored.episodes_completed == 3
    assert len(restored.episode_metrics) == 3

    assert state.next_episode_index == 4
    assert state.episodes_completed == 3


def test_episode_id_is_deterministic():
    manager = TrainingResumeManager()

    assert (
        manager.episode_id(
            "resume-run-001",
            4,
        )
        == "resume-run-001-episode-0004"
    )


def test_episode_id_rejects_zero():
    with pytest.raises(ValueError):
        TrainingResumeManager.episode_id(
            "resume-run-001",
            0,
        )


def test_validate_resume_allows_continuation():
    manager = TrainingResumeManager()
    result = make_result()

    state = manager.save(
        result,
        "/tmp/agentforge-resume-validation.json",
    )

    manager.validate_resume(
        state,
        requested_total_episodes=10,
    )


def test_validate_resume_rejects_already_completed():
    manager = TrainingResumeManager()
    result = make_result()

    state = manager.save(
        result,
        "/tmp/agentforge-resume-completed.json",
        status="completed",
    )

    with pytest.raises(RuntimeError):
        manager.validate_resume(
            state,
            requested_total_episodes=10,
        )


def test_validate_resume_rejects_smaller_target():
    manager = TrainingResumeManager()
    result = make_result()

    state = manager.save(
        result,
        "/tmp/agentforge-resume-target.json",
    )

    with pytest.raises(ValueError):
        manager.validate_resume(
            state,
            requested_total_episodes=2,
        )


def test_merge_episode_continues_from_checkpoint():
    manager = TrainingResumeManager()
    result = make_result()

    merged = manager.merge_episode(
        result,
        steps=4,
        reward=4.5,
        success=True,
        episode_id=manager.episode_id(
            result.run_id,
            4,
        ),
        task_id="backend-002",
    )

    assert merged.episodes_completed == 4
    assert merged.total_steps == 13
    assert merged.total_reward == 13.5
    assert merged.successful_episodes == 3

    assert len(merged.episode_metrics) == 4

    assert (
        merged.episode_metrics[-1].episode_id
        == "resume-run-001-episode-0004"
    )

    assert (
        merged.episode_metrics[-1].total_reward
        == 4.5
    )


def test_merge_does_not_modify_original_result():
    manager = TrainingResumeManager()
    result = make_result()

    merged = manager.merge_episode(
        result,
        steps=2,
        reward=2.0,
        success=False,
        episode_id=manager.episode_id(
            result.run_id,
            4,
        ),
        task_id="backend-003",
    )

    assert result.episodes_completed == 3
    assert len(result.episode_metrics) == 3

    assert merged.episodes_completed == 4
    assert len(merged.episode_metrics) == 4
