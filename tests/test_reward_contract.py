from agentforge.core import (
    RewardAttribution,
    RewardBreakdown,
    TrajectoryEventType,
)


def test_reward_has_recovery_and_backward_compatible_testing():
    reward = RewardBreakdown(
        progress=1.0,
        correctness=2.0,
        tests=3.0,
        recovery=4.0,
        efficiency=5.0,
        terminal=6.0,
        penalties=-1.0,
    )

    assert reward.testing == 3.0
    assert reward.total == 20.0


def test_reward_provenance_is_typed():
    attribution = RewardAttribution(
        source="hidden_verifier",
        component="correctness",
        value=1.0,
        step_id=7,
        event_type=TrajectoryEventType.DECISION,
        evidence="decision matched expected policy",
    )

    reward = RewardBreakdown(
        correctness=1.0,
        provenance=(attribution,),
    )

    assert reward.provenance[0] == attribution
    assert reward.total == 1.0
