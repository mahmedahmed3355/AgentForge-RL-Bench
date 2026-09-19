"""Public API for AgentForge-RL-Bench core."""

from __future__ import annotations

from .contracts import (
    EpisodeResult,
    EvaluationResult,
    TaskSpec,
    RewardBreakdown,
    StepRecord,
    Trajectory,
)

# These names are exported lazily because some core modules import
# agentforge.environments, while environments import agentforge.core.
_LAZY_EXPORTS = {
    "EpisodeRunner": (".episode", "EpisodeRunner"),
    "TrajectoryRecorder": (".trajectory", "TrajectoryRecorder"),
    "EpisodeExecutor": (".executor", "EpisodeExecutor"),
    "EpisodeCheckpoint": (".checkpoint", "EpisodeCheckpoint"),
    "RunManager": (".run", "RunManager"),
    "EvaluationComparator": (
        ".evaluation",
        "EvaluationComparator",
    ),
    "EvaluationMetrics": (
        ".evaluation",
        "EvaluationMetrics",
    ),
    "LearningDelta": (
        ".evaluation",
        "LearningDelta",
    ),
    "FailureModeAnalyzer": (
        ".failure_analysis",
        "FailureModeAnalyzer",
    ),
    "ExperimentManager": (
        ".experiment",
        "ExperimentManager",
    ),
    "ExperimentReport": (
        ".report",
        "ExperimentReport",
    ),
    "ExperimentExecutionResult": (
        ".orchestrator",
        "ExperimentExecutionResult",
    ),
    "ExperimentOrchestrator": (
        ".orchestrator",
        "ExperimentOrchestrator",
    ),
    "ExperimentArtifactStore": (
        ".artifacts",
        "ExperimentArtifactStore",
    ),
}


def __getattr__(name: str):
    """Resolve optional/cyclic core exports lazily."""

    if name not in _LAZY_EXPORTS:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}"
        )

    module_name, attribute_name = _LAZY_EXPORTS[name]

    from importlib import import_module

    module = import_module(
        module_name,
        package=__name__,
    )

    value = getattr(module, attribute_name)

    globals()[name] = value

    return value


__all__ = [
    "EpisodeResult",
    "EvaluationResult",
    "TaskSpec",
    "RewardBreakdown",
    "StepRecord",
    "Trajectory",
    "EpisodeRunner",
    "TrajectoryRecorder",
    "EpisodeExecutor",
    "EpisodeCheckpoint",
    "RunManager",
    "EvaluationComparator",
    "EvaluationMetrics",
    "LearningDelta",
    "FailureModeAnalyzer",
    "ExperimentManager",
    "ExperimentReport",
    "ExperimentExecutionResult",
    "ExperimentOrchestrator",
    "ExperimentArtifactStore",
]
