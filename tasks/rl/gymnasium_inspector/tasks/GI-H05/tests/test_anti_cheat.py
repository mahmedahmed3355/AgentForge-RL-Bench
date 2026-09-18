from __future__ import annotations

from environment.data.gym_env import AmbiguousFailureDiagnosisEnv


def test_root_failure_is_not_equal_to_visible_symptom():
    env = AmbiguousFailureDiagnosisEnv()

    assert env.ROOT_FAILURE != env.SYMPTOM


def test_correct_diagnosis_requires_inspection():
    env = AmbiguousFailureDiagnosisEnv()
    env.reset(seed=123)

    env.step(1)

    assert env.diagnosis != env.ROOT_FAILURE


def test_recovery_requires_correct_diagnosis():
    env = AmbiguousFailureDiagnosisEnv()
    env.reset(seed=123)

    env.step(0)

    _, reward, terminated, truncated, info = env.step(2)

    assert reward < 0
    assert terminated is False
    assert truncated is False
    assert info["recovered"] is False
