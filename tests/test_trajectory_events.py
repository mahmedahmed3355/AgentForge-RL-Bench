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


def test_all_core_event_types_are_explicitly_typed():
    events = [
        DecisionEvent(
            step_id=1,
            description="decision",
        ),
        BranchEvent(
            step_id=1,
            branch_id="branch-a",
        ),
        WrongBranchEvent(
            step_id=2,
            branch_id="branch-b",
        ),
        FailureEvent(
            step_id=3,
            severity="high",
        ),
        RecoveryEvent(
            step_id=4,
        ),
        DelayedConsequenceEvent(
            step_id=8,
            consequence_delay=4,
            related_step_id=4,
        ),
        TerminalOutcomeEvent(
            step_id=10,
            description="terminal outcome",
        ),
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


def test_recorder_rejects_event_from_another_step():
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
            events=[
                DecisionEvent(
                    step_id=99
                )
            ],
        )
    except ValueError as exc:
        assert "step_id" in str(exc)
    else:
        raise AssertionError(
            "Mismatched event step was accepted."
        )
