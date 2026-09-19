
import pytest

from agentforge.core import (
    HiddenVerifier,
    TaskSpec,
    VerificationEvidence,
    VerificationResult,
)


def make_task():
    return TaskSpec(
        task_id="verify-task",
        domain="backend",
        split="eval",
        description="Verifier contract test.",
        difficulty="medium",
    )


def test_verification_result_contains_required_fields():
    result = VerificationResult(
        passed=False,
        score=0.25,
        failure_step=17,
        failure_mode="wrong_branch",
        violations=("invalid branch",),
        trajectory_evidence=(
            VerificationEvidence(
                kind="trajectory",
                description="wrong branch",
                step_id=17,
            ),
        ),
        terminal_state_evidence=(
            VerificationEvidence(
                kind="terminal",
                description="invalid terminal state",
            ),
        ),
    )

    assert result.passed is False
    assert result.score == 0.25
    assert result.failure_step == 17
    assert result.failure_mode == "wrong_branch"
    assert result.violations == ("invalid branch",)


def test_invalid_verification_score_is_rejected():
    with pytest.raises(ValueError):
        VerificationResult(
            passed=False,
            score=1.5,
        )


def test_hidden_verifier_is_implementable():
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

    result = ExampleVerifier().verify(
        task=make_task(),
        trajectory=None,
    )

    assert result.passed is True
