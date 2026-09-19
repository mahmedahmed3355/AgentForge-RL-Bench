
from agentforge.core import (
    BranchEvent,
    DecisionEvent,
    DelayedConsequenceEvent,
    FailureEvent,
    RecoveryEvent,
    RewardBreakdown,
    TerminalOutcomeEvent,
    TrajectoryEventType,
    WrongBranchEvent,
)
from agentforge.core.trajectory import TrajectoryRecorder


def test_all_required_event_types_are_explicit():
    events = [
        DecisionEvent(step_id=1),
        BranchEvent(step_id=1, branch_id="A"),
        WrongBranchEvent(step_id=2, branch_id="B"),
        FailureEvent(step_id=3),
        RecoveryEvent(step_id=4),
        DelayedConsequenceEvent(
            step_id=7,
            related_step_id=2,
            consequence_delay=5,
        ),
        TerminalOutcomeEvent(step_id=8),
    ]

    assert [
        event.event_type
        for event in events
    ] == [
        TrajectoryEventType.DECISION,
        TrajectoryEventType.BRANCH,
        TrajectoryEventType.WRONG_BRANCH,
        TrajectoryEventType.FAILURE,
        TrajectoryEventType.RECOVERY,
        TrajectoryEventType.DELAYED_CONSEQUENCE,
        TrajectoryEventType.TERMINAL_OUTCOME,
    ]


def test_recorder_preserves_typed_events():
    recorder = TrajectoryRecorder(
        episode_id="ep-1",
        task_id="task-1",
    )

    recorder.record_step(
        observation={"state": 0},
        action="choose",
        reward=RewardBreakdown(),
        next_observation={"state": 1},
        status="running",
        events=[
            DecisionEvent(step_id=1),
            BranchEvent(
                step_id=1,
                branch_id="A",
            ),
        ],
    )

    assert recorder.trajectory.steps[0].events[0].event_type == (
        TrajectoryEventType.DECISION
    )


def test_recorder_rejects_event_for_wrong_step():
    recorder = TrajectoryRecorder(
        episode_id="ep-1",
        task_id="task-1",
    )

    try:
        recorder.record_step(
            observation={},
            action={},
            reward=RewardBreakdown(),
            next_observation={},
            status="running",
            events=[DecisionEvent(step_id=99)],
        )
    except ValueError as exc:
        assert "step_id" in str(exc)
    else:
        raise AssertionError(
            "Mismatched event step was accepted."
        )
