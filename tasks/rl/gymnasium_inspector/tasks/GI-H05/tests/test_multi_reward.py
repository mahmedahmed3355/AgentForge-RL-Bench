from __future__ import annotations

from environment.data.gym_env import AmbiguousFailureDiagnosisEnv


def test_reward_inspection_component():
    env = AmbiguousFailureDiagnosisEnv()
    env.reset(seed=123)

    _, reward, _, _, info = env.step(0)

    assert reward == 0.5
    assert info["stage"] == "compare"


def test_reward_diagnosis_component():
    env = AmbiguousFailureDiagnosisEnv()
    env.reset(seed=123)
    env.step(0)

    _, reward, _, _, info = env.step(1)

    assert reward == 2.5
    assert info["diagnosis"] == env.ROOT_FAILURE


def test_reward_recovery_component():
    env = AmbiguousFailureDiagnosisEnv()
    env.reset(seed=123)
    env.step(0)
    env.step(1)

    _, reward, _, _, info = env.step(2)

    assert reward == 2.5
    assert info["recovered"] is True


def test_reward_terminal_component():
    env = AmbiguousFailureDiagnosisEnv()
    env.reset(seed=123)

    for action in [0, 1, 2]:
        env.step(action)

    _, reward, terminated, truncated, _ = env.step(3)

    assert reward == 4.0
    assert terminated is True
    assert truncated is False
