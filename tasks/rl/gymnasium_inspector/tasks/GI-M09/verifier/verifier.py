from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class VerificationResult:
    passed: bool
    score: float
    reasons: tuple[str, ...]


class Verifier:
    """Independent verifier for GI-M09."""

    task_id = "GI-M09"

    def verify(
        self,
        trajectory: Any,
        final_state: Any,
    ) -> VerificationResult:
        reasons: list[str] = []

        if trajectory is None:
            reasons.append("missing_trajectory")

        if final_state is None:
            reasons.append("missing_final_state")

        if reasons:
            return VerificationResult(
                passed=False,
                score=0.0,
                reasons=tuple(reasons),
            )

        # TODO:
        # 1. Validate trajectory transitions.
        # 2. Validate required state invariants.
        # 3. Validate hidden/adversarial conditions.
        # 4. Detect reward-hacking shortcuts.
        # 5. Validate terminal goal state.

        return VerificationResult(
            passed=False,
            score=0.0,
            reasons=("verifier_not_implemented",),
        )
