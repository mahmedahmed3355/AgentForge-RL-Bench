from __future__ import annotations

from environment.data.gym_env import AmbiguousFailureDiagnosisEnv


def test_valid_trajectory_order():
    env = AmbiguousFailureDiagnosisEnv()
    env.reset(seed=123)

    observations = []

    for action in [0, 1, 2, 3]:
        obs, reward, terminated, truncated, info = env.step(action)
        observations.append(
            {
                "action": action,
                "stage": info["stage"],
                "diagnosis": info["diagnosis"],
                "recovered": info["recovered"],
                "terminated": terminated,
                "truncated": truncated,
            }
        )

    assert [item["action"] for item in observations] == [0, 1, 2, 3]
    assert observations[-1]["terminated"] is True
    assert observations[-1]["truncated"] is False


def test_recovery_precedes_finish():
    env = AmbiguousFailureDiagnosisEnv()
    env.reset(seed=123)

    env.step(0)
    env.step(1)

    _, _, terminated, _, info = env.step(3)

    assert terminated is False
    assert info["recovered"] is False
