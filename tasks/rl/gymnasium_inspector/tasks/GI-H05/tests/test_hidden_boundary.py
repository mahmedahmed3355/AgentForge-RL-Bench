from __future__ import annotations

from environment.data.gym_env import AmbiguousFailureDiagnosisEnv


def test_symptom_is_not_root_failure():
    env = AmbiguousFailureDiagnosisEnv()

    assert env.SYMPTOM != env.ROOT_FAILURE


def test_diagnosis_requires_discriminating_evidence():
    env = AmbiguousFailureDiagnosisEnv()
    env.reset(seed=123)

    _, reward, terminated, truncated, info = env.step(1)

    assert reward < 0
    assert terminated is False
    assert truncated is False
    assert info["diagnosis"] != env.ROOT_FAILURE


def test_evidence_changes_after_inspection():
    env = AmbiguousFailureDiagnosisEnv()
    env.reset(seed=123)

    assert env.evidence == 0

    env.step(0)

    assert env.evidence == env.DIAGNOSTIC_EVIDENCE
