from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class VerificationResult:
    passed: bool
    score: float
    reasons: tuple[str, ...]


class Verifier:
    task_id = "CUDA-H01"

    def verify(self, trajectory: Any, final_state: Any) -> VerificationResult:
        reasons: list[str] = []

        if not isinstance(trajectory, list):
            reasons.append("missing_trajectory")
        if not isinstance(final_state, dict):
            reasons.append("missing_final_state")
        if reasons:
            return VerificationResult(False, 0.0, tuple(reasons))

        tools = [
            item.get("action", {}).get("tool")
            for item in trajectory
            if isinstance(item, dict) and isinstance(item.get("action"), dict)
        ]

        required = {
            "inspect_device",
            "inspect_pipeline",
            "inspect_memory",
            "inspect_kernel",
            "benchmark_pipeline",
            "validate_output",
            "stress_test",
            "final_verify",
        }

        missing = sorted(required - set(tools))
        if missing:
            reasons.append("missing_tools:" + ",".join(missing))

        if not final_state.get("success"):
            reasons.append("final_success_false")
        if not final_state.get("terminal"):
            reasons.append("not_terminal")
        if final_state.get("stage") != "d2h":
            reasons.append("pipeline_incomplete")
        if final_state.get("recovery_count", 0) < 1:
            reasons.append("required_recovery_missing")
        if not final_state.get("validation_count", 0):
            reasons.append("no_validation")

        workload = final_state.get("workload", {})
        required_performance = float(workload.get("required_performance", 9999))
        config = final_state.get("config", {})

        score = 92.0
        if config.get("memory_strategy") == "pinned":
            score += 12
        if config.get("transfer_strategy") == "async":
            score += 10
        if config.get("streams") == 2:
            score += 8
        elif config.get("streams") == 3:
            score += 5
        if config.get("events"):
            score += 5
        if config.get("sync_strategy") == "event":
            score += 2
        if config.get("registers", 32) < 64:
            score += 12
        if (
            config.get("transfer_strategy") == "async"
            and config.get("streams", 1) > 1
            and not config.get("events")
        ):
            score -= 20
        if (
            config.get("transfer_strategy") == "async"
            and config.get("streams", 1) > 1
            and config.get("sync_strategy") == "device"
        ):
            score -= 15
        if (
            config.get("memory_strategy") == "pageable"
            and workload.get("memory_pressure", 0) >= 6
        ):
            score -= 8

        if score < required_performance:
            reasons.append("performance_target_not_met")

        if len(tools) and len(set(tools)) < 4:
            reasons.append("insufficient_behavioral_diversity")

        passed = not reasons
        return VerificationResult(
            passed=passed,
            score=1.0 if passed else 0.0,
            reasons=tuple(reasons),
        )

