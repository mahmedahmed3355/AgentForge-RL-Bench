"""Framework-level hidden verifier contracts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from .contracts import TaskSpec, Trajectory


@dataclass(frozen=True)
class VerificationEvidence:
    """Evidence supporting a verifier decision."""

    kind: str
    description: str
    step_id: int | None = None
    value: Any = None


@dataclass(frozen=True)
class VerificationResult:
    """Framework-neutral result of hidden task verification."""

    passed: bool
    score: float
    failure_step: int | None = None
    failure_mode: str | None = None
    violations: tuple[str, ...] = ()
    trajectory_evidence: tuple[
        VerificationEvidence, ...
    ] = ()
    terminal_state_evidence: tuple[
        VerificationEvidence, ...
    ] = ()

    def __post_init__(self) -> None:
        if not 0.0 <= self.score <= 1.0:
            raise ValueError(
                "Verification score must be between 0.0 and 1.0."
            )

        if self.passed and self.failure_step is not None:
            raise ValueError(
                "A passing verification cannot have failure_step."
            )


class HiddenVerifier(ABC):
    """Contract implemented by hidden task verifiers."""

    @abstractmethod
    def verify(
        self,
        *,
        task: TaskSpec,
        trajectory: Trajectory,
        terminal_observation: Any = None,
    ) -> VerificationResult:
        """Verify a completed episode without exposing task internals."""
        raise NotImplementedError
