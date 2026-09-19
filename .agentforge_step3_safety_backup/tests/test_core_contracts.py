from agentforge.core import (
    EpisodeResult,
    EvaluationResult,
    RewardBreakdown,
    StepRecord,
    TaskSpec,
    Trajectory,
)


def test_reward_breakdown_total():
    reward = RewardBreakdown(
        progress=0.2,
        correctness=0.4,
        tests=0.3,
        efficiency=0.05,
        terminal=1.0,
        penalties=-0.1,
    )

    assert reward.total == 1.85


def test_trajectory_records_steps():
    trajectory = Trajectory(
        episode_id="episode-001",
        task_id="backend-001",
    )

    step = StepRecord(
        step_id=1,
        observation={"state": "start"},
        action={"type": "inspect"},
        reward=RewardBreakdown(progress=0.2),
        next_observation={"state": "inspected"},
        status="running",
    )

    trajectory.append(step)

    assert trajectory.length == 1
    assert trajectory.total_reward == 0.2


def test_evaluation_pass_rate():
    results = [
        EpisodeResult(
            episode_id="ep-1",
            task_id="task-1",
            success=True,
            steps=10,
            total_reward=2.0,
            final_status="success",
        ),
        EpisodeResult(
            episode_id="ep-2",
            task_id="task-2",
            success=False,
            steps=20,
            total_reward=0.5,
            final_status="failure",
        ),
    ]

    evaluation = EvaluationResult(
        evaluation_id="eval-001",
        agent_id="agent-001",
        split="eval",
        task_results=results,
    )

    assert evaluation.task_count == 2
    assert evaluation.passed_count == 1
    assert evaluation.pass_rate == 0.5


def test_task_spec():
    task = TaskSpec(
        task_id="backend-001",
        domain="backend",
        split="train",
        description="Fix a backend bug.",
        difficulty="medium",
    )

    assert task.task_id == "backend-001"
    assert task.domain == "backend"
    assert task.split == "train"
