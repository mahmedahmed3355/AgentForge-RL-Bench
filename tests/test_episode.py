import pytest

from agentforge.core import RewardBreakdown
from agentforge.core.episode import EpisodeRunner


def test_episode_runner_tracks_steps_and_result():
    runner = EpisodeRunner(
        episode_id="episode-001",
        task_id="backend-001",
        max_steps=100,
    )

    runner.record_step(
        observation={"state": "start"},
        action={"type": "inspect"},
        reward=RewardBreakdown(progress=0.2),
        next_observation={"state": "inspected"},
    )

    runner.record_step(
        observation={"state": "inspected"},
        action={"type": "fix"},
        reward=RewardBreakdown(correctness=0.5, tests=0.3),
        next_observation={"state": "fixed"},
        status="success",
    )

    result = runner.result()

    assert runner.done is True
    assert result.success is True
    assert result.steps == 2
    assert result.final_status == "success"
    assert result.total_reward == 1.0


def test_episode_runner_enforces_max_steps():
    runner = EpisodeRunner(
        episode_id="episode-002",
        task_id="backend-002",
        max_steps=2,
    )

    for _ in range(2):
        runner.record_step(
            observation={},
            action={"type": "work"},
            reward=RewardBreakdown(progress=0.1),
            next_observation={},
        )

    result = runner.result()

    assert runner.done is True
    assert result.success is False
    assert result.steps == 2
    assert result.final_status == "timeout"


def test_episode_runner_rejects_step_after_finish():
    runner = EpisodeRunner(
        episode_id="episode-003",
        task_id="backend-003",
    )

    runner.record_step(
        observation={},
        action={"type": "finish"},
        reward=RewardBreakdown(terminal=1.0),
        next_observation={},
        status="success",
    )

    with pytest.raises(RuntimeError, match="episode has finished"):
        runner.record_step(
            observation={},
            action={"type": "invalid"},
            reward=RewardBreakdown(),
            next_observation={},
        )


def test_explicit_finish():
    runner = EpisodeRunner(
        episode_id="episode-004",
        task_id="backend-004",
    )

    result = runner.finish(success=False, status="failure")

    assert runner.done is True
    assert result.success is False
    assert result.steps == 0
    assert result.final_status == "failure"
