import pytest

from agentforge.core import (
    HiddenVerifier,
    TaskSpec,
    VerificationEvidence,
    VerificationResult,
)


def make_task():
    return TaskSpec(
        task_id="verify-test",
        domain="backend",
        split="eval",
        description="Verifier contract test.",
        difficulty="medium",
    )


def test_verification_result_supports_required_fields():
    result = VerificationResult(
        passed=False,
        score=0.25,
        failure_step=17,
        failure_mode="wrong_branch",
        violations=(
            "selected invalid branch",
        ),
        trajectory_evidence=(
            VerificationEvidence(
                kind="trajectory",
                description="wrong branch at step 17",
                step_id=17,
            ),
        ),
        terminal_state_evidence=(
            VerificationEvidence(
                kind="terminal",
                description="terminal state invalid",
            ),
        ),
    )

    assert result.passed is False
    assert result.score == 0.25
    assert result.failure_step == 17
    assert result.failure_mode == "wrong_branch"
    assert len(result.violations) == 1


def test_verification_result_rejects_invalid_score():
    with pytest.raises(ValueError):
        VerificationResult(
            passed=False,
            score=1.5,
        )


def test_hidden_verifier_is_an_explicit_interface():
    class ExampleVerifier(HiddenVerifier):
        def verify(
            self,
            *,
            task,
            trajectory,
            terminal_observation=None,
        ):
            return VerificationResult(
                passed=True,
                score=1.0,
            )

    verifier = ExampleVerifier()

    result = verifier.verify(
        task=make_task(),
        trajectory=None,
    )

    assert result.passed is True
    assert result.score == 1.0
