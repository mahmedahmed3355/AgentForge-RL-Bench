import json

from agentforge.core import RewardBreakdown
from agentforge.core.trajectory import TrajectoryRecorder


def test_trajectory_recorder_records_steps():
    recorder = TrajectoryRecorder(
        episode_id="episode-001",
        task_id="backend-001",
    )

    recorder.record_step(
        observation={"state": "start"},
        action={"type": "inspect"},
        reward=RewardBreakdown(progress=0.2),
        next_observation={"state": "inspected"},
        status="running",
    )

    recorder.record_step(
        observation={"state": "inspected"},
        action={"type": "fix"},
        reward=RewardBreakdown(correctness=0.5),
        next_observation={"state": "fixed"},
        status="success",
    )

    assert recorder.trajectory.length == 2
    assert recorder.trajectory.total_reward == 0.7
    assert recorder.trajectory.steps[0].step_id == 1
    assert recorder.trajectory.steps[1].step_id == 2


def test_trajectory_can_be_saved(tmp_path):
    recorder = TrajectoryRecorder(
        episode_id="episode-002",
        task_id="backend-002",
    )

    recorder.record_step(
        observation={"state": "start"},
        action={"type": "inspect"},
        reward=RewardBreakdown(progress=0.1),
        next_observation={"state": "done"},
        status="success",
    )

    output = recorder.save_json(tmp_path / "trajectory.json")

    assert output.exists()

    data = json.loads(output.read_text(encoding="utf-8"))

    assert data["episode_id"] == "episode-002"
    assert data["task_id"] == "backend-002"
    assert data["length"] == 1
    assert data["total_reward"] == 0.1
    assert data["steps"][0]["step_id"] == 1
    assert data["steps"][0]["reward"]["total"] == 0.1
