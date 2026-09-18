"""Failure mode analysis for AgentForge-RL-Bench."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .evaluation import EvaluationMetrics


@dataclass
class FailureModeDelta:
    """Comparison of one failure mode before and after training."""

    mode: str
    baseline_count: int
    after_training_count: int

    @property
    def delta(self) -> int:
        """Return the count change."""

        return self.after_training_count - self.baseline_count


@dataclass
class FailureAnalysis:
    """Detailed failure-mode analysis."""

    baseline_failures: dict[str, int]
    after_training_failures: dict[str, int]
    mode_deltas: list[FailureModeDelta]

    @property
    def baseline_total_failures(self) -> int:
        """Return total baseline failures."""

        return sum(self.baseline_failures.values())

    @property
    def after_training_total_failures(self) -> int:
        """Return total post-training failures."""

        return sum(self.after_training_failures.values())

    def to_dict(self) -> dict[str, Any]:
        """Serialize analysis results."""

        return {
            "baseline_failures": self.baseline_failures,
            "after_training_failures": self.after_training_failures,
            "baseline_total_failures": self.baseline_total_failures,
            "after_training_total_failures": self.after_training_total_failures,
            "mode_deltas": [
                {
                    "mode": item.mode,
                    "baseline_count": item.baseline_count,
                    "after_training_count": item.after_training_count,
                    "delta": item.delta,
                }
                for item in self.mode_deltas
            ],
        }


class FailureModeAnalyzer:
    """Analyze failure modes across evaluations."""

    @staticmethod
    def analyze(
        baseline: EvaluationMetrics,
        after_training: EvaluationMetrics,
    ) -> FailureAnalysis:
        """Compare failure modes before and after training."""

        all_modes = sorted(
            set(baseline.failure_modes)
            | set(after_training.failure_modes)
        )

        deltas = [
            FailureModeDelta(
                mode=mode,
                baseline_count=baseline.failure_modes.get(mode, 0),
                after_training_count=after_training.failure_modes.get(
                    mode,
                    0,
                ),
            )
            for mode in all_modes
        ]

        return FailureAnalysis(
            baseline_failures=dict(baseline.failure_modes),
            after_training_failures=dict(
                after_training.failure_modes
            ),
            mode_deltas=deltas,
        )

    @staticmethod
    def save(
        analysis: FailureAnalysis,
        path: str | Path,
    ) -> Path:
        """Save failure analysis to JSON."""

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        output_path.write_text(
            json.dumps(
                analysis.to_dict(),
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        return output_path
